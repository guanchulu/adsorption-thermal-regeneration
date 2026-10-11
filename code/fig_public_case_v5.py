# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "polars==2.0.0", "matplotlib==3.11.2", "pillow==12.3.0"]
# ///
# Run: uv run --project .. --frozen python code/fig_public_case_v5.py

import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from figure_style import BLUE, GRAY, INK, ORANGE, ROOT, panel, save, setup
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def uptake_panel(ax: Axes, path: pl.DataFrame) -> None:
    for selected, style in ((pl.col("temperature_K") <= 323, "-"), (pl.col("temperature_K") >= 323, "--")):
        part = path.filter(selected)
        ax.plot(part["temperature_K"], part["q_final_mol_kg"], color=BLUE, ls=style)
    ax.scatter(
        [298],
        [path["q_initial_mol_kg"][0]],
        marker="^",
        facecolors="white",
        edgecolors=BLUE,
        s=48,
        zorder=4,
        label=r"Initial loading at 0.15 bar",
    )
    for temperature in (303, 323):
        ax.axvline(temperature, color=GRAY, ls=":", lw=0.8, zorder=0)
    ax.text(0.46, 0.57, "Published DSL model\nat 0.10 bar", transform=ax.transAxes, color=BLUE, fontsize=8)
    ax.text(
        0.46,
        0.39,
        "Dotted lines: reported\n303 / 323 K measurements",
        transform=ax.transAxes,
        color=GRAY,
        fontsize=7.5,
    )
    ax.set(
        xlabel=r"Desorption temperature, $T_d$ (K)",
        ylabel=r"Loading, $q$ (mol kg$^{-1}$)",
        xlim=(294, 427),
        ylim=(0, 3.05),
        xticks=[300, 340, 380, 420],
    )
    ax.legend(loc="upper right", fontsize=7.5)
    panel(ax, "a", "Literature-parameter equilibrium")


def heat_panel(ax: Axes, path: pl.DataFrame) -> None:
    t = path["temperature_K"].to_numpy()
    dry = path["dry_heat_kJ_kg"].to_numpy()
    binding = path["binding_heat_kJ_kg"].to_numpy()
    total = path["path_heat_kJ_kg"].to_numpy()
    ax.fill_between(t, 0, dry, color=GRAY, alpha=0.55, label="Dry solid")
    ax.fill_between(t, dry, dry + binding, color=BLUE, alpha=0.85, label="Binding")
    ax.fill_between(t, dry + binding, total, color=ORANGE, label="Retained gas")
    ax.plot(t, total, color=INK, lw=0.7)
    ax.set(
        xlabel=r"Desorption temperature, $T_d$ (K)",
        ylabel=r"Path heat (kJ kg$^{-1}$)",
        xlim=(294, 427),
        ylim=(0, 245),
        xticks=[300, 340, 380, 420],
    )
    ax.legend(loc="upper left", fontsize=7.7)
    ax.annotate(
        r"Retained: 4.30 kJ kg$^{-1}$",
        xy=(421, 221),
        xytext=(342, 237),
        fontsize=7.2,
        arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.7},
    )
    panel(ax, "b", "Heat allocation along the path")


def budget_panel(ax: Axes, budgets: pl.DataFrame) -> None:
    ax.plot(
        budgets["budget_kJ_kg"], budgets["full_release_mol_kg"], color=BLUE, label="Complete heat balance"
    )
    ax.plot(
        budgets["budget_kJ_kg"],
        budgets["reduced_release_mol_kg"],
        color=ORANGE,
        ls="--",
        label="Omitting retained heat",
    )
    selected = budgets.filter(pl.col("budget_kJ_kg") == 50)
    q = float(selected["reduced_release_mol_kg"].item())
    ax.scatter([50], [q], marker="s", facecolors="white", edgecolors=ORANGE, s=25, zorder=4)
    ax.annotate(
        "50 kJ per kg budget:\nreduced target needs\n51.53 kJ per kg",
        xy=(50, q),
        xytext=(110, 0.48),
        fontsize=7.6,
        color=ORANGE,
        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.8},
    )
    ax.set(
        xlabel=r"Heat budget, $B$ (kJ kg$^{-1}$)",
        ylabel=r"Released CO$_2$ (mol kg$^{-1}$)",
        xlim=(0, 255),
        ylim=(0, 2.9),
        xticks=[0, 50, 100, 150, 200, 250],
    )
    ax.legend(loc="upper left", fontsize=7.7)
    panel(ax, "c", "Fixed-material budget consequence")


def comparator_panel(ax: Axes, data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    budgets, bounds = data
    levels = np.array([25.0, 50.0, 100.0])
    fixed = budgets.filter(pl.col("budget_kJ_kg").is_in(levels))["full_release_mol_kg"]
    lower = bounds["endpoint_lower_mol_kg"].to_numpy()
    upper = bounds["endpoint_upper_mol_kg"].to_numpy()
    positions = levels
    ax.scatter(positions, fixed, color=BLUE, s=38, marker="o", label="Fixed published model")
    ax.errorbar(
        positions,
        (lower + upper) / 2,
        yerr=(upper - lower) / 2,
        fmt="none",
        ecolor=ORANGE,
        elinewidth=3,
        capsize=6,
        capthick=1.2,
        label="Free-affinity family bounds",
    )
    ax.set(
        xticks=positions,
        xticklabels=["25", "50", "100"],
        xlim=(15, 110),
        ylim=(0, 2.8),
        xlabel=r"Heat budget, $B$ (kJ kg$^{-1}$)",
        ylabel=r"Released CO$_2$ (mol kg$^{-1}$)",
    )
    ax.legend(loc="upper left", fontsize=7.5)
    ax.text(
        0.03,
        0.61,
        r"Same $Q$ and $C$;" + "\ndifferent allowed affinities",
        transform=ax.transAxes,
        color=GRAY,
        fontsize=7.4,
    )
    panel(ax, "d", "Material and admissible-family limits")


def audit_figure(fig: Figure) -> None:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    outside: list[str] = []
    fonts: list[float] = []
    for ax in fig.axes:
        legend = ax.get_legend()
        legend_text = [] if legend is None else legend.get_texts()
        labels = [
            ax.title,
            ax.xaxis.label,
            ax.yaxis.label,
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
            *ax.texts,
            *legend_text,
        ]
        for label in labels:
            if not label.get_visible() or not label.get_text():
                continue
            fonts.append(float(label.get_fontsize()))
            box = label.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                outside.append(label.get_text())
    report = {
        "panels": len(fig.axes),
        "layout": "2x2",
        "outside_labels": outside,
        "minimum_font_pt": min(fonts),
        "interval_semantics": "deterministic optimization bounds; not CIs",
    }
    (ROOT / "qa/Fig6_public_case_layout.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if outside:
        raise RuntimeError("Figure6 public-case labels exceed canvas")


def main() -> None:
    setup()
    path = pl.read_csv(ROOT / "results/public_case_temperature_path.csv")
    budgets = pl.read_csv(ROOT / "results/public_case_budget_release.csv")
    bounds = pl.read_csv(ROOT / "results/public_case_resource_bounds.csv")
    profiles = []
    for name, frame in (("temperature_path", path), ("budget_release", budgets), ("resource_bounds", bounds)):
        profiles.append(
            {
                "name": name,
                "rows": frame.height,
                "columns": frame.width,
                "nulls": int(frame.null_count().sum_horizontal().item()),
            }
        )
    (ROOT / "qa/Fig6_public_case_data_profile.json").write_text(
        json.dumps(profiles, indent=2), encoding="utf-8"
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), layout="constrained")
    a, b, c, d = axes.flat
    uptake_panel(a, path)
    heat_panel(b, path)
    budget_panel(c, budgets)
    comparator_panel(d, (budgets, bounds))
    audit_figure(fig)
    save(fig, "Fig6_public_case")
    plt.close(fig)


if __name__ == "__main__":
    main()
