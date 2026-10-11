# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2", "numpy==2.5.3", "polars==2.0.0", "scipy==1.18.1"]
# ///
# Run: uv run --frozen python code/fig_release_final.py
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from figure_style import BLUE, GRAY, GREEN, INK, ORANGE, ROOT, panel, save, setup
from process_art import process_path
from scipy.special import expit
from thermal_core import Cycle


def main() -> None:
    setup()
    fig = plt.figure(figsize=(7.2, 4.8), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.55])
    process_path(fig.add_subplot(grid[0, :]))
    axes = [fig.add_subplot(grid[1, i]) for i in range(3)]
    x = np.linspace(-5, 5, 501)
    ads, des = expit(x + 1), expit(x - 1)
    ax = axes[0]
    ax.plot(x, ads, color=BLUE, label="Adsorption")
    ax.plot(x, des, color=ORANGE, ls="--", label="Desorption")
    ax.plot(x, ads - des, color=INK, ls="-.", lw=1.3, label="Release")
    ax.axvline(0, color=GRAY, ls=":", lw=1)
    ax.set(xlabel="Common log-affinity shift", ylabel=r"Occupation / release fraction", ylim=(0, 1.04))
    ax.legend(loc="upper left", fontsize=6.9)
    panel(ax, "b", "Centering maximizes release")
    cycle = Cycle(298, 350, 0.15, 0.1)
    d = np.linspace(0, 0.95, 401)
    for m, color, style, label in (
        (1, BLUE, "-", "m = 1: independent"),
        (2, GREEN, "--", "m = 2: ideal cluster"),
        (4, ORANGE, "-.", "m = 4: ideal cluster"),
    ):
        energy = np.maximum(0, (4 * np.arctanh(d) / m - cycle.log_fugacity_ratio) / cycle.thermal_factor)
        axes[1].plot(d, energy, color=color, ls=style, label=label)
    axes[1].set(xlabel="Release fraction, d", ylabel=r"Minimum E (kJ mol$^{-1}$)", ylim=(0, None))
    axes[1].legend(loc="upper left", fontsize=6.7)
    panel(axes[1], "c", "Affinity cost across models")
    g = d * np.maximum(0, (4 * np.arctanh(d) - cycle.log_fugacity_ratio) / cycle.thermal_factor)
    dd = np.array([0.25, 0.8])
    gg = dd * (4 * np.arctanh(dd) - cycle.log_fugacity_ratio) / cycle.thermal_factor
    axes[2].plot(d, g, color=BLUE, label="Homogeneous envelope")
    axes[2].plot(dd, gg, "o--", color=ORANGE, ms=4, label="Two-site mixture")
    xm, ym = float(dd.mean()), float(gg.mean())
    gm = xm * (4 * np.arctanh(xm) - cycle.log_fugacity_ratio) / cycle.thermal_factor
    axes[2].annotate("", (xm, gm), (xm, ym), arrowprops={"arrowstyle": "<->", "color": ORANGE, "lw": 1.2})
    axes[2].set(xlabel="Mean release fraction", ylabel=r"Binding heat / Q (kJ mol$^{-1}$)", ylim=(0, None))
    axes[2].legend(loc="upper left", fontsize=6.7)
    panel(axes[2], "d", "Convex binding-heat cost")
    save(fig, "Fig1_release_final")
    plt.close(fig)
    pl.DataFrame(
        {"common_shift": x, "adsorption": ads, "desorption": des, "release_difference": ads - des}
    ).write_csv(ROOT / "results/figure1_final_centering.csv")


if __name__ == "__main__":
    main()
