# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "matplotlib==3.11.2", "polars==2.0.0", "scipy==1.18.1"]
# ///
# Run: uv run --no-sync --project .. python code/fig_framework_v5.py
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from endpoint_bounds import endpoint_cost, gas_heat_increment, log_odds
from figure_style import BLUE, GRAY, GREEN, ORANGE, ROOT, panel, save, setup
from path_model import Setting, Site, heat_path
from process_art import process_path
from scipy.special import expit
from thermal_core import R


def framework() -> None:
    row = (
        pl.read_csv(ROOT / "results/path_brackets.csv")
        .filter(
            (pl.col("capacity") == 4)
            & (pl.col("dry_cp") == 1)
            & (pl.col("cluster") == 1)
            & (pl.col("budget") == 100)
        )
        .row(0, named=True)
    )
    td, energy, entropy = (float(row[k]) for k in ("witness_T", "witness_E", "witness_s"))
    setting, site = Setting(), Site(energy, entropy)
    state = heat_path(setting, site, td)
    fig = plt.figure(figsize=(7.2, 4.9), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.5])
    process_path(fig.add_subplot(grid[0, :]))
    axes = [fig.add_subplot(grid[1, j]) for j in range(3)]
    ax = axes[0]
    ax.plot([298, 298, td], [0.15, 0.1, 0.1], "o-", color=BLUE, ms=4)
    ax.annotate("", (298, 0.118), (298, 0.143), arrowprops={"arrowstyle": "->", "color": BLUE})
    ax.annotate("", (330, 0.1), (309, 0.1), arrowprops={"arrowstyle": "->", "color": BLUE})
    ax.text(301, 0.147, "Loaded", fontsize=7)
    ax.text(301, 0.106, "Post-drop", fontsize=7)
    ax.text(td - 1, 0.093, "Regenerated", ha="right", fontsize=7)
    ax.set(xlabel="Bed temperature (K)", ylabel="Fugacity (bar)", xlim=(293, 347), ylim=(0.085, 0.158))
    ax.set_xticks([298, 320, 342])
    panel(ax, "b", "Two controlled steps")
    t = np.linspace(298, td, 180)
    theta = expit(np.log(0.1) + energy / (R * t) - entropy / R)
    ax = axes[1]
    ax.plot(t, 4 * theta, color=BLUE)
    ax.plot([298, 298], [4 * state.theta_ads, 4 * theta[0]], "o--", color=ORANGE, ms=4)
    ax.fill_between(t, 0, 4 * theta, color=BLUE, alpha=0.1)
    ax.text(310, 0.48, "Retained inventory", color=BLUE, fontsize=7)
    ax.set(xlabel="Bed temperature (K)", ylabel=r"Adsorbed gas (mol kg$^{-1}$)", ylim=(0, 3.3))
    ax.set_xticks([298, 320, 342])
    panel(ax, "c", "Release leaves gas behind")
    ax = axes[2]
    pressure_heat = 4 * energy * (state.theta_ads - state.theta_after_pressure)
    values = [
        (np.array([0, state.dry]), GRAY, "Dry solid"),
        (np.array([pressure_heat, state.binding - pressure_heat]), BLUE, "Binding"),
        (np.array([0, state.loading]), ORANGE, "Retained gas"),
    ]
    bottom = np.zeros(2)
    for val, color, label in values:
        ax.bar([0, 1], val, bottom=bottom, color=color, label=label, width=0.55)
        bottom += val
    ax.set(
        xticks=[0, 1],
        xticklabels=["Pressure\nstep", "Heating\nstep"],
        ylabel=r"Heat input (kJ kg$^{-1}$)",
        ylim=(0, 105),
    )
    ax.legend(fontsize=6.6, loc="upper left")
    panel(ax, "d", "Heat enters both steps")
    save(fig, "Fig1_physical_boundary")
    plt.close(fig)
    pl.DataFrame({"temperature_K": t, "retained_mol_kg": 4 * theta}).write_csv(
        ROOT / "results/figure1_v5_path.csv"
    )


def envelopes() -> None:
    fig, array = plt.subplots(2, 2, figsize=(7.2, 5.3), layout="constrained")
    axes = list(array.flat)
    a, j, ratio = (1 / 298 - 1 / 350) / R, gas_heat_increment(298, 350), float(np.log(1.5))
    d = np.linspace(0.001, 0.85, 180)
    reduced = d * np.maximum(0, (4 * np.arctanh(d) - ratio) / a)
    table = pl.read_csv(ROOT / "results/heat_interval_map.csv").sort("temperature_K", "release_fraction")
    temperatures = table["temperature_K"].unique().sort().to_numpy()
    releases = table["release_fraction"].unique().sort().to_numpy()
    values = table["relative_heat_gap_pct"].to_numpy().reshape(len(temperatures), len(releases))
    mesh = axes[0].pcolormesh(
        releases, temperatures, values, cmap="viridis", shading="nearest", rasterized=True, vmin=0
    )
    colorbar = fig.colorbar(mesh, ax=axes[0], orientation="vertical", pad=0.025, fraction=0.046)
    colorbar.set_label("Heat-bracket width (%)", fontsize=7.5)
    colorbar.ax.tick_params(labelsize=7)
    axes[0].plot([0.6], [350], marker="o", ms=5, mfc="none", mec="white", mew=1)
    axes[0].axhline(350, color="white", ls=":", lw=0.8)
    axes[0].text(0.1, 353, "c, d", color="white", fontsize=7)
    axes[0].set(
        xlabel="Release fraction, d", ylabel="Desorption temperature (K)", xlim=(0.05, 0.9), ylim=(310, 423)
    )
    panel(axes[0], "a", "Heat-bound tightness")
    selected = endpoint_cost(0.6, a, j, ratio, 1, upper=True)
    s = R * np.log(0.15) + selected.energy / 298 - R * log_odds(selected.v + 0.6)
    t = np.linspace(298, 350, 180)
    theta = expit(np.log(0.1) + selected.energy / (R * t) - s / R)
    ax = axes[1]
    ax.axhspan(selected.v, selected.upper_occupation, color=BLUE, alpha=0.1)
    ax.plot(t, theta, color=BLUE, label="Heating path")
    ax.axhline(selected.v, color=GREEN, ls="--", label="Final inventory, v")
    ax.axhline(selected.upper_occupation, color=ORANGE, ls="-.", label="Post-drop inventory, w")
    ax.set(xlabel="Bed temperature (K)", ylabel="Occupied capacity", ylim=(0, 1))
    ax.legend(loc="upper right", fontsize=7)
    panel(ax, "b", "Inventory bounds sensible heat")
    rows = []
    for delta, g in zip(d, reduced, strict=True):
        low = endpoint_cost(float(delta), a, j, ratio, 1, upper=False)
        high = endpoint_cost(float(delta), a, j, ratio, 1, upper=True)
        entropy = R * np.log(0.15) + high.energy / 298 - R * log_odds(high.v + delta)
        state = heat_path(Setting(), Site(high.energy, float(entropy)), 350)
        rows.append(
            {
                "d": float(delta),
                "reduced": float(g),
                "lower": low.value,
                "upper": high.value,
                "path": (state.binding + state.loading) / 4,
            }
        )
    data = pl.DataFrame(rows)
    for field, color, style, label in (
        ("reduced", GRAY, ":", "Binding only"),
        ("lower", GREEN, "--", "Lower cost"),
        ("path", BLUE, "-", "Constructed path"),
        ("upper", ORANGE, "-.", "Upper construction"),
    ):
        axes[2].plot(d, data[field], color=color, ls=style, label=label)
    axes[2].set(xlabel="Release fraction, d", ylabel=r"Variable heat / Q (kJ mol$^{-1}$)", ylim=(0, None))
    axes[2].legend(loc="upper left", fontsize=7)
    panel(axes[2], "c", "Bracketing the complete path")
    axes[3].fill_between(d, 0, j, color=GRAY, alpha=0.13, label="Full-loading estimate")
    axes[3].fill_between(
        d, data["lower"] - reduced, data["upper"] - reduced, color=BLUE, alpha=0.25, label="Endpoint interval"
    )
    axes[3].plot(d, data["path"] - reduced, color=BLUE, label="Constructed path")
    axes[3].set(xlabel="Release fraction, d", ylabel=r"Heat above minimum (kJ mol$^{-1}$)", ylim=(0, 2.15))
    axes[3].legend(loc="upper left", fontsize=7)
    panel(axes[3], "d", "A tighter inventory allowance")
    save(fig, "Fig3_endpoint_tightness_final")
    plt.close(fig)
    data.write_csv(ROOT / "results/figure3_v5_costs.csv")


if __name__ == "__main__":
    setup()
    envelopes()
