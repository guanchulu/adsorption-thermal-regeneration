# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "matplotlib==3.11.2", "polars==2.0.0"]
# ///
# Run: uv run --no-sync --project .. python code/fig_mechanism_v5.py
import matplotlib.pyplot as plt
import polars as pl
from figure_style import BLUE, GRAY, GREEN, ORANGE, ROOT, panel, save, setup
from process_art import heat_allocation


def main() -> None:
    setup()
    cases = pl.read_csv(ROOT / "results/path_brackets.csv").with_columns(
        (100 * (pl.col("upper_design_path_heat") / pl.col("budget") - 1)).alias("excess_pct")
    )
    profiles = pl.read_csv(ROOT / "results/symmetry_profiles.csv")
    fig, array = plt.subplots(2, 2, figsize=(7.2, 5.3), layout="constrained")
    axes = list(array.flat)
    for m, color, marker in ((1, BLUE, "o"), (4, ORANGE, "^")):
        sub = cases.filter(pl.col("cluster") == m)
        axes[0].scatter(
            sub["thermal_loading_ratio"],
            sub["excess_pct"],
            color=color,
            marker=marker,
            s=26,
            edgecolor="white",
            linewidth=0.4,
            label=f"m = {m}",
        )
    axes[0].set(
        xlabel=r"Loading heat scale, $Q c_{p,g}^{max}/C$", ylabel="Budget exceedance (%)", ylim=(0, 14)
    )
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].text(
        0.03, 0.69, "54 prescribed designs\n0.57–12.61% excess", transform=axes[0].transAxes, fontsize=7
    )
    panel(axes[0], "a", "Reduced designs exceed the budget")
    heat_allocation(axes[1])
    panel(axes[1], "b", "Feasible designs account for all heat")
    sub = profiles.filter(pl.col("m") == 1)
    for key, ref, color, style, label in (
        ("heat_binding", "binding_symmetric", BLUE, "--", "Binding"),
        ("heat_loading", "loading_symmetric", ORANGE, "-.", "Retained gas"),
        ("heat_total", "heat_symmetric", GREEN, "-", "Total"),
    ):
        axes[2].plot(sub["v"], sub[key] - sub[ref], color=color, ls=style, label=label)
    axes[2].axhline(0, color=GRAY, lw=0.6)
    axes[2].axvline(0.2, color=GRAY, ls=":", lw=1)
    axes[2].set(xlabel="Final occupation, v", ylabel=r"Heat change (kJ kg$^{-1}$)", xlim=(0.1, 0.3))
    axes[2].legend(loc="upper left", fontsize=7)
    axes[2].text(
        0.37, 0.57, "Equal fugacities: 0.10 bar", transform=axes[2].transAxes, fontsize=7, color=GRAY
    )
    panel(axes[2], "c", "Binding and inventory compete")
    temperatures = pl.read_csv(ROOT / "results/figure2_temperature.csv")
    for m, color, style in ((1, BLUE, "-"), (2, GREEN, "--"), (4, ORANGE, "-.")):
        sub = temperatures.filter(pl.col("m") == m).sort("Td")
        axes[3].plot(sub["Td"], sub["delta"], color=color, ls=style, label=f"m = {m}")
    ref = (298**2 + 100 * 298) ** 0.5
    axes[3].axvline(ref, color=GRAY, ls=":", lw=1)
    axes[3].set(
        xlabel="Desorption temperature (K)", ylabel="Reduced release limit", ylim=(0, 1), xlim=(298, 398)
    )
    axes[3].text(ref + 1.7, 0.1, "344.4 K", fontsize=7, rotation=90, color=GRAY)
    axes[3].legend(loc="upper right", fontsize=7)
    panel(axes[3], "d", "Temperature trades heat for release")
    save(fig, "Fig2_thermal_tradeoff_final")
    plt.close(fig)
    cases.write_csv(ROOT / "results/figure2_v5_excess.csv")


if __name__ == "__main__":
    main()
