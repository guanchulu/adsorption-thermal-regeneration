# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2", "numpy==2.5.3", "polars==2.0.0"]
# ///
# Used by the reproducible figure scripts in this project.
import numpy as np
import polars as pl
from figure_style import BLUE, GRAY, INK, ORANGE, ROOT
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


def arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.5, color=color))


def bed(ax: Axes, x: float, y: float, filled: int = 8) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            1.2,
            1.45,
            boxstyle="round,pad=0.04,rounding_size=0.16",
            facecolor="#F2F5F6",
            edgecolor=INK,
            linewidth=1.2,
        )
    )
    for index in range(12):
        px, py = x + 0.22 + 0.38 * (index % 3), y + 0.23 + 0.31 * (index // 3)
        ax.add_patch(Circle((px, py), 0.12, facecolor="#CCD4D8", edgecolor="#929DA5", linewidth=0.5))
        if index < filled:
            ax.add_patch(
                Circle((px + 0.075, py + 0.07), 0.045, facecolor=BLUE, edgecolor="white", linewidth=0.3)
            )
    ax.plot([x + 0.18, x + 0.18], [y - 0.14, y], color=INK, lw=1)
    ax.plot([x + 1.02, x + 1.02], [y - 0.14, y], color=INK, lw=1)


def co2(ax: Axes, x: float, y: float, scale: float = 1.0) -> None:
    ax.plot([x - 0.2 * scale, x + 0.2 * scale], [y, y], color=INK, lw=1.5)
    for dx, color, radius in ((-0.2, ORANGE, 0.11), (0.0, INK, 0.09), (0.2, ORANGE, 0.11)):
        ax.add_patch(Circle((x + dx * scale, y), radius * scale, facecolor=color, edgecolor="white", lw=0.3))


def prepare(ax: Axes, title: str) -> None:
    ax.set(xlim=(0, 12), ylim=(0, 3.25))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, 3.12, "a", fontsize=12, weight="bold", va="top")
    ax.text(0.5, 3.12, title, fontsize=10, va="top")


def process_path(ax: Axes) -> None:
    prepare(ax, "A bed-level heat budget along a specified regeneration path")
    for x, filled, label in (
        (0.5, 10, r"Loaded: $T_a,f_a$"),
        (5.0, 7, r"Post-drop: $T_a,f_d$"),
        (9.5, 3, r"Regenerated: $T_d,f_d$"),
    ):
        bed(ax, x, 0.62, filled)
        ax.text(x + 0.6, 0.3, label, ha="center", fontsize=8)
    arrow(ax, (1.95, 1.32), (4.65, 1.32), INK)
    ax.text(3.3, 1.66, "Isothermal\ndepressurization", ha="center", fontsize=8)
    arrow(ax, (6.45, 1.32), (9.15, 1.32), INK)
    ax.text(7.8, 1.66, "Heating at\nfixed fugacity", ha="center", fontsize=8)
    for center in (5.6, 10.1):
        arrow(ax, (center, 2.12), (center + 1.0, 2.62), BLUE)
        ax.text(center + 1.05, 2.65, "Gas out", fontsize=7, va="center")
        arrow(ax, (center - 1.4, 0.68), (center - 0.58, 0.96), ORANGE)
        ax.text(center - 2.0, 0.73, "Heat in", color=ORANGE, fontsize=7, ha="center")
    ax.text(
        11.85, 0.02, "Occupancy schematic; pressure work outside the heat budget", fontsize=6.8, ha="right"
    )


def heat_terms(ax: Axes) -> None:
    prepare(ax, "Three physical contributions to the supplied regeneration heat")
    for center, filled, formula, label in (
        (2.0, 0, r"$C(T_d-T_a)$", "Dry-solid heating"),
        (6.0, 6, r"$Q\!\int E\delta\,d\mu$", "Desorption"),
        (10.0, 5, r"$Q\!\int\langle\theta\rangle c_{p,g}\,dT$", "Retained-gas heating"),
    ):
        bed(ax, center - 0.6, 1.02, filled)
        arrow(ax, (center - 1.4, 1.65), (center - 0.67, 1.65), ORANGE)
        ax.text(center, 0.59, formula, fontsize=10.5, ha="center")
        ax.text(center, 0.14, label, ha="center", fontsize=8)
    for x in (4.0, 8.0):
        ax.text(x, 1.7, "+", fontsize=16, ha="center", va="center")
    arrow(ax, (6.25, 2.5), (6.9, 2.75), BLUE)
    co2(ax, 7.18, 2.75, 0.65)


def screening_bed(ax: Axes) -> None:
    prepare(ax, "From a heat-limited bed to a required gas release")
    bed(ax, 4.0, 0.75, 7)
    arrow(ax, (1.0, 1.47), (3.9, 1.47), ORANGE)
    ax.text(2.35, 1.8, r"Available heat $B$", ha="center", fontsize=9, color=ORANGE)
    ax.text(2.35, 1.05, r"kJ per kg dry adsorbent", ha="center", fontsize=7.5)
    arrow(ax, (5.25, 1.47), (8.2, 1.47), BLUE)
    co2(ax, 8.8, 1.47, 1.2)
    ax.text(6.75, 1.8, r"Released amount $Qd$", ha="center", fontsize=9, color=BLUE)
    ax.text(6.75, 1.05, r"mol per kg dry adsorbent", ha="center", fontsize=7.5)
    ax.text(4.6, 0.28, r"Fixed $Q,C$; vary affinity and $T_d$", ha="center", fontsize=8)
    ax.text(10.1, 1.8, "Compare with\nrequired release", ha="center", fontsize=8)
    ax.text(10.1, 0.9, "Feasible / unresolved\n/ excluded", ha="center", fontsize=8)


def heat_allocation(ax: Axes) -> None:
    data = (
        pl.read_csv(ROOT / "results/path_brackets.csv")
        .filter((pl.col("capacity") == 4) & (pl.col("dry_cp") == 1) & (pl.col("cluster") == 1))
        .sort("budget")
    )
    baseline = np.zeros(data.height)
    for field, color, label in (
        ("witness_dry", GRAY, "Dry solid"),
        ("witness_binding", BLUE, "Binding"),
        ("witness_loading", ORANGE, "Retained gas"),
    ):
        values = data[field].to_numpy()
        ax.bar(np.arange(data.height), values, bottom=baseline, color=color, width=0.58, label=label)
        baseline += values
    ax.set(
        xticks=np.arange(data.height),
        xticklabels=[f"{x:g}" for x in data["budget"]],
        xlabel=r"Available heat, $B$ (kJ kg$^{-1}$)",
        ylabel=r"Path heat (kJ kg$^{-1}$)",
        ylim=(0, 235),
    )
    ax.legend(loc="upper left", fontsize=7)
    for index, row in enumerate(data.iter_rows(named=True)):
        ax.text(
            index,
            float(row["witness_path_heat"]) + 5,
            f"{float(row['witness_T']):.1f} K",
            ha="center",
            fontsize=7,
        )
    data.select("budget", "witness_T", "witness_dry", "witness_binding", "witness_loading").write_csv(
        ROOT / "results/figure2_heat_allocation.csv"
    )
