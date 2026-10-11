import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from figure_style import BLUE, GRAY, GREEN, ORANGE, ROOT, panel, save, setup


def main() -> None:
    data = pl.read_csv(ROOT / "data/temperature_holdout_frozen.csv")
    pivot = data.pivot(on="model", index="group_key", values="test_normalized_rmse")
    thermal = pivot["thermal_langmuir"].to_numpy()
    constant = pivot["constant_temperature_langmuir"].to_numpy()
    baseline = pivot["pressurewise_logq_vanthoff"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), layout="constrained")
    for j, (other, title) in enumerate(
        ((constant, "Constant-T comparison"), (baseline, "Stronger baseline comparison"))
    ):
        ax = axes[j]
        lo = min(float(np.min(other)), float(np.min(thermal))) * 0.65
        hi = max(float(np.max(other)), float(np.max(thermal))) * 1.5
        ax.plot([lo, hi], [lo, hi], color=GRAY, ls="--", lw=1)
        ax.scatter(
            other, thermal, s=24, color=BLUE if j == 0 else GREEN, alpha=0.8, edgecolor="white", linewidth=0.4
        )
        ax.set(
            xscale="log",
            yscale="log",
            xlim=(lo, hi),
            ylim=(lo, hi),
            xlabel=("Constant-T Langmuir NRMSE" if j == 0 else "Pressurewise baseline NRMSE"),
            ylabel="Thermal Langmuir NRMSE",
        )
        panel(ax, "ab"[j], title)
    groups = data.filter(pl.col("model") == "thermal_langmuir")
    temps = groups["heldout_temperature_K"].to_numpy()
    axes[2].hist(temps, bins=np.arange(273, 454, 20), color=BLUE, edgecolor="white")
    axes[2].axvline(353, color=ORANGE, ls="--", lw=1.4)
    axes[2].set(xlabel="Held-out maximum temperature (K)", ylabel="Material-source groups")
    axes[2].text(
        0.97,
        0.95,
        "32 groups / 25 DOIs\n9 groups at ≥353 K",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=7.7,
    )
    panel(axes[2], "c", "Temperature coverage")
    save(fig, "FigS1_temperature_transfer")
    plt.close(fig)


if __name__ == "__main__":
    setup()
    main()
