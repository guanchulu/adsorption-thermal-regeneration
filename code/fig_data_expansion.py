import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from figure_style import BLUE, GRAY, GREEN, ORANGE, ROOT, panel, save, setup
from matplotlib.axes import Axes
from numpy.typing import NDArray


def ecdf(ax: Axes, values: NDArray[np.float64], label: str, color: str, style: str = "-") -> None:
    ordered = np.sort(values[np.isfinite(values)])
    ax.step(
        ordered,
        np.arange(1, len(ordered) + 1) / len(ordered),
        where="post",
        label=label,
        color=color,
        ls=style,
    )


def main_figure() -> None:
    data = pl.read_parquet(ROOT / "results/crafted_reference_metrics.parquet")
    co2 = data.filter(pl.col("gas") == "CO2")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), layout="constrained")
    for model, label, color, style in (
        ("thermal_langmuir", "Thermal single site", BLUE, "-"),
        ("fugacitywise_inverse_temperature", "Fugacitywise baseline", GREEN, "--"),
    ):
        part = co2.filter(pl.col("model") == model)
        ecdf(axes[0], part["uptake_nrmse"].to_numpy(), label, color, style)
    axes[0].set(xlabel="323 K uptake NRMSE", ylabel="Fraction of structures", ylim=(0, 1.02), xlim=(1e-4, 1))
    axes[0].set_xscale("log")
    axes[0].set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    axes[0].text(
        0.97,
        0.35,
        "1 baseline case > 1\n(max ≈ 39,260)",
        transform=axes[0].transAxes,
        ha="right",
        fontsize=6.7,
    )
    axes[0].legend(loc="lower right", fontsize=6.7)
    axes[0].text(
        0.04, 0.95, r"$\mathrm{CO_2}$" + "\nn = 1,352", transform=axes[0].transAxes, fontsize=7, va="top"
    )
    panel(axes[0], "a", "Temperature transfer")
    part = co2.filter(pl.col("model") == "thermal_langmuir")
    axes[1].scatter(
        part["uptake_nrmse"], part["common_heat_mae_kj_mol"], s=7, color=BLUE, alpha=0.4, edgecolors="none"
    )
    axes[1].set_xscale("symlog", linthresh=1e-4)
    axes[1].set_yscale("symlog", linthresh=0.01)
    axes[1].set(xlabel="Uptake NRMSE", ylabel=r"Heat-observable MAE (kJ mol$^{-1}$)")
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
    panel(axes[1], "b", "Uptake and heat errors")
    for gas, label, color, style in (
        ("CO2", r"$\mathrm{CO_2}$, n = 1,352", BLUE, "-"),
        ("N2", r"$\mathrm{N_2}$, n = 1,351", ORANGE, "--"),
    ):
        part = data.filter((pl.col("gas") == gas) & (pl.col("model") == "thermal_langmuir"))
        ecdf(axes[2], part["common_heat_mae_kj_mol"].to_numpy(), label, color, style)
    axes[2].set_xscale("symlog", linthresh=0.01)
    axes[2].set(
        xlabel=r"Heat-observable MAE (kJ mol$^{-1}$)", ylabel="Fraction of structures", ylim=(0, 1.02)
    )
    axes[2].legend(loc="upper left", fontsize=7)
    axes[2].set_xlim(left=0)
    panel(axes[2], "c", "Thermal checks for two gases")
    save(fig, "Fig5_public_thermal_benchmark")
    plt.close(fig)


def experiments() -> None:
    data = pl.read_csv(ROOT / "results/experimental_response_pairs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    for flag, color, marker, label in (
        (False, BLUE, "o", "No review flag"),
        (True, ORANGE, "x", "Pointwise review flag"),
    ):
        part = data.filter(pl.col("inversion_both_same_point_5pct") == flag)
        axes[0].scatter(
            part["temperature_span_K"],
            100 * part["mean_response_linear"],
            color=color,
            marker=marker,
            s=22,
            label=label,
            alpha=0.75,
        )
    axes[0].axhline(0, color=GRAY, lw=0.8, ls=":")
    axes[0].set(xlabel="Temperature separation (K)", ylabel="Mean normalized uptake decrease (%)")
    axes[0].legend(fontsize=7)
    panel(axes[0], "a", "122 literature-derived pairs")
    ordered = data.sort("max_interpolation_sensitivity")
    axes[1].scatter(np.arange(1, 123), 100 * ordered["max_interpolation_sensitivity"], color=GREEN, s=12)
    axes[1].set(xlabel="Pair rank", ylabel="Maximum interpolation sensitivity (%)", ylim=(0, 5.1))
    axes[1].text(
        0.04, 0.94, "72 source DOIs · 116 material IDs", transform=axes[1].transAxes, fontsize=8, va="top"
    )
    panel(axes[1], "b", "Linear-pressure vs log-pressure")
    save(fig, "FigS4_experimental_response")
    plt.close(fig)


def resolution() -> None:
    data = pl.read_csv(ROOT / "results/dictionary_resolution_cases.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    for ax, field, letter, title in (
        (axes[0], "train_nrmse", "a", "Training fit under refinement"),
        (axes[1], "heat_mae", "b", "Heat prediction under refinement"),
    ):
        x, y = data[f"coarse_{field}"].to_numpy(), data[f"fine_{field}"].to_numpy()
        ceiling = 1.1 * max(float(x.max()), float(y.max()))
        ax.plot([0, ceiling], [0, ceiling], color=GRAY, ls="--", lw=1)
        ax.scatter(x, y, color=BLUE, s=18, alpha=0.65, edgecolor="none")
        ax.set_xscale("symlog", linthresh=0.0001 if field == "train_nrmse" else 0.01)
        ax.set_yscale("symlog", linthresh=0.0001 if field == "train_nrmse" else 0.01)
        units = "NRMSE" if field == "train_nrmse" else r"MAE (kJ mol$^{-1}$)"
        ax.set(
            xlabel=f"525-kernel {units}", ylabel=f"2,009-kernel {units}", xlim=(0, ceiling), ylim=(0, ceiling)
        )
        panel(ax, letter, title)
    axes[1].text(0.04, 0.96, "99 matched pilot structures", transform=axes[1].transAxes, fontsize=8, va="top")
    save(fig, "FigS5_dictionary_resolution")
    plt.close(fig)


if __name__ == "__main__":
    setup()
    main_figure()
    experiments()
    resolution()
