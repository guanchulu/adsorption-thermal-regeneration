# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "matplotlib==3.11.2", "polars==2.0.0", "scipy==1.18.1"]
# ///
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from figure_style import BLUE, GRAY, GREEN, ORANGE, ROOT, panel, save, setup
from ideal_cluster_probe import cluster_bound
from process_art import heat_allocation, process_path
from scipy.special import expit
from thermal_core import Cycle, Resources


def envelope() -> None:
    fig = plt.figure(figsize=(7.2, 4.9), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.45])
    process_path(fig.add_subplot(grid[0, :]))
    axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    x = np.linspace(-5, 5, 501)
    ax = axes[0]
    ax.plot(x, expit(x + 1), color=BLUE, label="Adsorption")
    ax.plot(x, expit(x - 1), color=ORANGE, ls="--", label="Desorption")
    ax.fill_between(x, expit(x - 1), expit(x + 1), color=BLUE, alpha=0.10)
    ax.axvline(0, color=GRAY, ls=":", lw=1)
    ax.set(xlabel="Common log-affinity shift", ylabel=r"Occupation, $\theta$", ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=7)
    panel(ax, "b", "Centering maximizes release")
    cycle = Cycle(298, 350, 0.15, 0.1)
    d = np.linspace(0, 0.95, 401)
    rows = []
    for m, color, ls in ((1, BLUE, "-"), (2, GREEN, "--"), (4, ORANGE, "-.")):
        e = np.maximum(0, (4 * np.arctanh(d) / m - cycle.log_fugacity_ratio) / cycle.thermal_factor)
        axes[1].plot(d, e, color=color, ls=ls, label=f"m = {m}")
        rows.extend({"m": m, "delta": float(dd), "Emin": float(ee)} for dd, ee in zip(d, e, strict=True))
    axes[1].set(xlabel=r"Release fraction, $\delta$", ylabel=r"Minimum $E$ (kJ mol$^{-1}$)", ylim=(0, None))
    axes[1].legend(loc="upper left", fontsize=7)
    panel(axes[1], "c", "Affinity cost of release")
    g = d * np.maximum(0, (4 * np.arctanh(d) - cycle.log_fugacity_ratio) / cycle.thermal_factor)
    axes[2].plot(d, g, color=BLUE, label="Homogeneous envelope")
    dd = np.array([0.25, 0.8])
    gg = dd * np.maximum(0, (4 * np.arctanh(dd) - cycle.log_fugacity_ratio) / cycle.thermal_factor)
    axes[2].plot(dd, gg, "o--", color=ORANGE, ms=4, label="Two-state mixture")
    xm, ym = float(dd.mean()), float(gg.mean())
    gm = xm * (4 * np.arctanh(xm) - cycle.log_fugacity_ratio) / cycle.thermal_factor
    axes[2].annotate("", (xm, gm), (xm, ym), arrowprops={"arrowstyle": "<->", "color": ORANGE, "lw": 1.2})
    axes[2].set(
        xlabel=r"Mean release fraction, $\bar\delta$",
        ylabel=r"Binding heat / capacity (kJ mol$^{-1}$)",
        ylim=(0, None),
    )
    axes[2].legend(loc="upper left", fontsize=6.7)
    panel(axes[2], "d", "Mixing cannot lower the cost")
    save(fig, "Fig1_release_envelope")
    plt.close(fig)
    pl.DataFrame(rows).write_csv(ROOT / "results/figure1_envelope.csv")


def temperature() -> None:
    fig, array = plt.subplots(2, 2, figsize=(7.2, 5.25), layout="constrained")
    axes = list(array.flat)
    t = np.linspace(298.05, 397.9, 501)
    rows = []
    for m, color, ls in ((1, BLUE, "-"), (2, GREEN, "--"), (4, ORANGE, "-.")):
        ds = []
        for td in t:
            bound = cluster_bound(Cycle(298, float(td), 0.15, 0.1), Resources(4, 1, 100), m)
            ds.append(bound.utilization)
            rows.append(
                {"m": m, "Td": float(td), "delta": bound.utilization, "s": bound.entropy_loss_kj_mol_K}
            )
        axes[0].plot(t, ds, color=color, ls=ls, label=f"m = {m}")
    ref = np.sqrt(298**2 + 100 * 298)
    axes[0].axvline(ref, color=GRAY, lw=1, ls=":")
    axes[0].text(ref + 2, 0.13, "344.4 K", rotation=90, fontsize=8, color=GRAY)
    axes[0].set(
        xlabel=r"Desorption temperature, $T_d$ (K)",
        ylabel=r"Reduced release limit, $\delta_{\rm red}^*$",
        ylim=(0, 1),
    )
    axes[0].legend(loc="upper right", fontsize=7)
    panel(axes[0], "a", "One thermal reference")
    beta = np.linspace(0, 2, 401)
    axes[1].plot(beta, np.sqrt(1 + beta), color=BLUE, label="Uncapped")
    axes[1].plot(
        beta, np.minimum(np.sqrt(1 + beta), 1.35), color=ORANGE, ls="--", label=r"$T_{\max}/T_a=1.35$"
    )
    axes[1].set(
        xlabel=r"Relative budget, $B/(C T_a)$", ylabel=r"Reference ratio, $T_{\rm ref}/T_a$", ylim=(1, 1.8)
    )
    axes[1].legend(loc="upper left", fontsize=7)
    panel(axes[1], "b", "Dimensionless collapse")
    budget = np.linspace(5, 250, 200)
    for c, color, ls in ((0.5, ORANGE, "-."), (1, BLUE, "-"), (2, GREEN, "--")):
        td = np.minimum(np.sqrt(298**2 + budget * 298 / c), 423)
        axes[2].plot(budget, td, color=color, ls=ls, label=f"C = {c:g}")
    axes[2].axhline(423, color=GRAY, ls=":", lw=1)
    axes[2].set(xlabel=r"Heat budget, $B$ (kJ kg$^{-1}$)", ylabel=r"Capped reference (K)", ylim=(298, 433))
    axes[2].legend(loc="lower right", fontsize=7)
    panel(axes[2], "c", "Background heat sets the scale")
    heat_allocation(axes[3])
    panel(axes[3], "d", "Heat allocation in feasible beds")
    save(fig, "Fig2_thermal_matching")
    plt.close(fig)
    pl.DataFrame(rows).write_csv(ROOT / "results/figure2_temperature.csv")


if __name__ == "__main__":
    setup()
    envelope()
    temperature()
