import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from figure_style import BLUE, GRAY, GREEN, ORANGE, ROOT, panel, save, setup
from matplotlib.lines import Line2D


def symmetry() -> None:
    profiles = pl.read_csv(ROOT / "results/symmetry_profiles.csv")
    data = pl.read_csv(ROOT / "results/symmetry_breaking.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), layout="constrained")
    for j, m in enumerate((1, 4)):
        ax = axes[0, j]
        p = profiles.filter(pl.col("m") == m)
        for name, baseline, color, ls, label in (
            ("heat_binding", "binding_symmetric", BLUE, "--", "Binding"),
            ("heat_loading", "loading_symmetric", ORANGE, "-.", "Retained gas"),
            ("heat_total", "heat_symmetric", GREEN, "-", "Total"),
        ):
            ax.plot(p["v"], p[name] - p[baseline], color=color, ls=ls, label=label)
        ax.axhline(0, color=GRAY, lw=0.6)
        ax.axvline(0.2, color=GRAY, lw=1, ls=":")
        ax.set(xlabel=r"Desorption occupation, $v$", ylabel=r"Heat change (kJ kg$^{-1}$)", xlim=(0.1, 0.3))
        ax.legend(loc="upper left", fontsize=7, ncol=1)
        panel(ax, "ab"[j], f"Opposing costs: m = {m}")
    for m in (1, 4):
        p = data.filter(pl.col("m") == m)
        for q, color, marker in ((2.0, BLUE, "o"), (4.0, GREEN, "s"), (8.0, ORANGE, "^")):
            sub = p.filter(pl.col("Q") == q)
            if q == 4.0:
                axes[1, 0].plot(
                    sub["delta"],
                    (sub["v_symmetric"] - sub["v_feasible_minimum"]),
                    color=BLUE if m == 1 else ORANGE,
                    marker="o" if m == 1 else "^",
                    ms=4,
                    ls="-" if m == 1 else "--",
                    label=f"m = {m}",
                )
            axes[1, 1].plot(
                sub["delta"],
                100 * sub["fraction_saved"],
                color=color,
                marker=marker,
                ms=4,
                ls="-" if m == 1 else "--",
            )
    axes[1, 0].set(xlabel=r"Target release, $\delta$", ylabel=r"Inventory shift, $v_0-v$", ylim=(0, None))
    axes[1, 0].legend(fontsize=8, loc="upper right")
    axes[1, 0].text(0.56, 0.052, "Independent of Q\nat fixed release", fontsize=8, color=GRAY)
    axes[1, 1].set(
        xlabel=r"Target release, $\delta$", ylabel="Reduction in gross path heat (%)", ylim=(0, 0.85)
    )
    axes[1, 1].text(0.57, 0.48, "Solid: m = 1\nDashed: m = 4", fontsize=8)
    axes[1, 1].legend(
        handles=[
            Line2D([], [], color=c, marker=mk, label=f"Q = {q}")
            for q, c, mk in ((2, BLUE, "o"), (4, GREEN, "s"), (8, ORANGE, "^"))
        ],
        loc="upper right",
        fontsize=7,
    )
    for ax, letter, title in (
        (axes[1, 0], "c", "Optimum shifts to less retained gas"),
        (axes[1, 1], "d", "A strict but modest benefit"),
    ):
        ax.set_xticks([0.4, 0.6, 0.8])
        panel(ax, letter, title)
    save(fig, "FigS2_inventory_asymmetry")
    plt.close(fig)


def brackets() -> None:
    data = pl.read_csv(ROOT / "results/path_brackets.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), layout="constrained")
    for j, m in enumerate((1, 4)):
        p = data.filter(pl.col("cluster") == m).sort("upper_release")
        x = np.arange(1, p.height + 1)
        ax = axes[0, j]
        ax.vlines(x, p["lower_release"], p["upper_release"], color=BLUE, lw=1.4, label="Analytical bracket")
        ax.scatter(x, p["upper_release"], color=BLUE, marker="_", s=40)
        ax.scatter(x, p["lower_release"], color=BLUE, marker="_", s=40)
        ax.scatter(
            x, p["witness_release"], s=10, color=ORANGE, marker="o", label="Path-feasible witness", zorder=3
        )
        ax.set(xlabel="Scenario rank within m", ylabel="Release fraction", ylim=(0, 1.03), xlim=(0.5, 27.5))
        ax.legend(loc="lower right", fontsize=7)
        panel(ax, "ab"[j], f"Constructive brackets: m = {m}")
    ax = axes[1, 0]
    for m, color, marker in ((1, BLUE, "o"), (4, ORANGE, "^")):
        p = data.filter(pl.col("cluster") == m)
        gap = 100 * (p["upper_release"] - p["witness_release"]) / p["upper_release"]
        ax.scatter(
            p["thermal_loading_ratio"],
            gap,
            color=color,
            marker=marker,
            s=25,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.3,
            label=f"m = {m}",
        )
    ax.set(
        xlabel=r"Loading heat scale, $Q c_{p,g}^{\max}/C$",
        ylabel="Upper-to-witness release gap (%)",
        ylim=(0, 8),
    )
    ax.legend(loc="upper left", fontsize=7)
    panel(ax, "c", "Gap depends on thermal loading")
    ax = axes[1, 1]
    sub = data.filter((pl.col("capacity") == 4) & (pl.col("dry_cp") == 1) & (pl.col("budget") == 100)).sort(
        "cluster"
    )
    y = np.array([0.0, 1.0])
    left = np.zeros(2)
    for field, color, label in (
        ("witness_dry", GRAY, "Dry solid"),
        ("witness_binding", BLUE, "Release enthalpy"),
        ("witness_loading", ORANGE, "Retained gas"),
    ):
        values = sub[field].to_numpy()
        ax.barh(y, values, left=left, color=color, height=0.42, label=label)
        left += values
    ax.set(
        yticks=y,
        yticklabels=["m = 1", "m = 4"],
        xlabel=r"Path heat (kJ kg$^{-1}$)",
        xlim=(0, 102),
        ylim=(-0.6, 1.6),
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=6.8, columnspacing=0.8)
    panel(ax, "d", "Heat allocation at B = 100")
    save(fig, "FigS3_reference_brackets")
    plt.close(fig)


if __name__ == "__main__":
    setup()
    symmetry()
    brackets()
