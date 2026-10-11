# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "matplotlib==3.11.2", "polars==2.0.0"]
# ///
# Run: uv run --no-sync --project .. python code/fig_decisions_v5.py
import matplotlib.pyplot as plt
import polars as pl
from figure_style import BLUE, GRAY, GREEN, ORANGE, ROOT, panel, save, setup


def main() -> None:
    setup()
    points = pl.read_csv(ROOT / "results/operating_map_points.csv")
    data = points.filter(pl.col("dry_cp_kJ_kg_K") == 1).sort("budget_kJ_kg")
    cells = pl.read_csv(ROOT / "results/operating_map_cells.csv")
    old = pl.read_csv(ROOT / "results/endpoint_brackets.csv").filter(pl.col("cluster") == 1)
    fig, array = plt.subplots(2, 2, figsize=(7.2, 5.35), layout="constrained")
    axes = list(array.flat)
    edges = [*cells["B_left_kJ_kg"].to_list(), float(cells["B_right_kJ_kg"][-1])]
    low, high = cells["feasible_floor_mol_kg"].to_numpy(), cells["upper_ceiling_mol_kg"].to_numpy()
    ax = axes[0]
    ax.stairs(low, edges, baseline=0, fill=True, color=GREEN, alpha=0.30, label="Feasible")
    ax.stairs(high, edges, baseline=low, fill=True, color=ORANGE, alpha=0.32, hatch="///", label="Unresolved")
    ax.stairs(high, edges, baseline=4, fill=True, color=GRAY, alpha=0.15, label="Excluded")
    ax.plot(data["budget_kJ_kg"], data["feasible_mol_kg"], ".", color=GREEN, ms=3)
    ax.plot(data["budget_kJ_kg"], data["refined_upper_mol_kg"], ".", color=BLUE, ms=3)
    ax.plot(data["budget_kJ_kg"], data["reduced_upper_mol_kg"], ":", color=GRAY, lw=1)
    ax.set(
        xlabel=r"Heat budget (kJ kg$^{-1}$)",
        ylabel=r"Required gas release (mol kg$^{-1}$)",
        xlim=(25, 200),
        ylim=(0, 4),
    )
    ax.legend(loc="upper left", fontsize=7)
    panel(ax, "a", "A heat-budget operating map")
    ax = axes[1]
    for capacity_heat, color, marker in ((0.5, ORANGE, "^"), (1.0, BLUE, "o"), (2.0, GREEN, "s")):
        sub = points.filter(
            (pl.col("dry_cp_kJ_kg_K") == capacity_heat) & pl.col("budget_kJ_kg").is_in([50.0, 100.0, 200.0])
        ).sort("budget_kJ_kg")
        x = sub["budget_kJ_kg"].to_numpy()
        lo, hi = sub["feasible_mol_kg"].to_numpy(), sub["refined_upper_mol_kg"].to_numpy()
        ax.vlines(x, lo, hi, color=color, lw=2)
        ax.plot(x, lo, marker=marker, color=color, ls="none", ms=4, label=f"C = {capacity_heat:g}")
        ax.plot(x, hi, marker="_", color=color, ls="none", ms=7)
    ax.set(
        xlabel=r"Heat budget (kJ kg$^{-1}$)",
        ylabel=r"Release bounds (mol kg$^{-1}$)",
        ylim=(0, 4),
        xlim=(30, 220),
        xticks=[50, 100, 200],
    )
    ax.legend(loc="upper left", fontsize=7)
    panel(ax, "b", "Dry heat capacity shifts the limit")
    ax = axes[2]
    ax.plot([0, 8], [0, 8], color=GRAY, ls=":", lw=1)
    ax.scatter(
        100 * old["old_relative_gap"],
        100 * old["new_relative_gap"],
        color=BLUE,
        s=22,
        edgecolor="white",
        lw=0.4,
    )
    ax.set(
        xlabel="Reduced upper-to-feasible gap (%)",
        ylabel="Refined upper-to-feasible gap (%)",
        xlim=(0, 8),
        ylim=(0, 8),
    )
    ax.text(0.5, 6.5, "27 independent-site settings\nMedian: 1.78 → 0.62%", fontsize=7)
    panel(ax, "c", "Less unresolved design space")
    row = data.filter(pl.col("budget_kJ_kg") == 100).row(0, named=True)
    lo, hi = float(row["feasible_mol_kg"]), float(row["refined_upper_mol_kg"])
    ax = axes[3]
    ax.axvspan(1.76, lo, color=GREEN, alpha=0.30)
    ax.axvspan(lo, hi, color=ORANGE, alpha=0.24, hatch="///")
    ax.axvspan(hi, 1.85, color=GRAY, alpha=0.15)
    ax.axvline(float(row["reduced_upper_mol_kg"]), color=GRAY, ls=":", lw=1)
    for x, y, color, text in (
        (1.78, 2, GREEN, "Feasible"),
        (1.808, 1, ORANGE, "Unresolved"),
        (1.82, 0, GRAY, "Excluded"),
    ):
        ax.scatter([x], [y], color=color, s=25)
        ax.annotate(text, (x, y), (0, 10), textcoords="offset points", ha="center", fontsize=7)
    ax.set(
        xlabel=r"Required release (mol kg$^{-1}$)",
        xlim=(1.76, 1.85),
        ylim=(-0.4, 2.65),
        yticks=[],
        xticks=[1.76, 1.80, 1.84],
    )
    panel(ax, "d", r"Three targets at B = 100")
    save(fig, "Fig4_budget_map")
    plt.close(fig)


if __name__ == "__main__":
    main()
