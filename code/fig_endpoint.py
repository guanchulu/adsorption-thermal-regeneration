import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from endpoint_bounds import endpoint_cost, gas_heat_increment, log_odds
from figure_style import BLUE, GRAY, GREEN, ORANGE, ROOT, panel, save, setup
from path_model import Setting, Site, heat_path
from process_art import heat_terms, screening_bed
from thermal_core import R


def bounds() -> None:
    fig = plt.figure(figsize=(7.2, 4.8), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.5])
    heat_terms(fig.add_subplot(grid[0, :]))
    axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    setting = Setting()
    td, delta = 350.0, 0.6
    a = (1 / 298 - 1 / td) / R
    j = gas_heat_increment(298, td)
    log_ratio = np.log(1.5)
    chosen = endpoint_cost(delta, a, j, float(log_ratio), 1, upper=True)
    entropy = R * np.log(0.15) + chosen.energy / 298 - R * log_odds(chosen.v + delta)
    temperatures = np.linspace(298, td, 200)
    theta = 1 / (1 + np.exp(-(np.log(0.1) + chosen.energy / (R * temperatures) - entropy / R)))
    ax = axes[0]
    ax.axhspan(chosen.v, chosen.upper_occupation, color=BLUE, alpha=0.10)
    ax.plot(temperatures, theta, color=BLUE, label="Actual heating path")
    ax.axhline(chosen.v, color=GREEN, ls="--", label=r"Residual $v$")
    ax.axhline(chosen.upper_occupation, color=ORANGE, ls="-.", label=r"After pressure drop $w$")
    ax.set(xlabel="Heating-path temperature (K)", ylabel="Occupied molecular capacity", ylim=(0, 1))
    ax.legend(loc="upper right", fontsize=6.7)
    panel(ax, "b", "Inventory during heating")
    rows = []
    for d in np.linspace(0.01, 0.82, 150):
        low = endpoint_cost(float(d), a, j, float(log_ratio), 1, upper=False)
        upper = endpoint_cost(float(d), a, j, float(log_ratio), 1, upper=True)
        s = R * np.log(0.15) + upper.energy / 298 - R * log_odds(upper.v + d)
        state = heat_path(setting, Site(upper.energy, float(s)), td)
        reduced = d * max(0.0, (4 * np.arctanh(d) - log_ratio) / a)
        actual = (state.binding + state.loading) / setting.capacity
        rows.append(
            {
                "d": float(d),
                "reduced": float(reduced),
                "endpoint_low": low.value,
                "endpoint_upper": upper.value,
                "path_homogeneous": actual,
                "old_upper_heat": float(reduced + j),
            }
        )
    data = pl.DataFrame(rows)
    data.write_csv(ROOT / "results/figure_endpoint_costs.csv")
    ax = axes[1]
    ax.plot(data["d"], data["reduced"], color=GRAY, ls=":", label="Binding-only envelope")
    ax.plot(data["d"], data["endpoint_low"], color=GREEN, ls="--", label="Endpoint lower")
    ax.plot(data["d"], data["path_homogeneous"], color=BLUE, label="Constructed path design")
    ax.plot(data["d"], data["endpoint_upper"], color=ORANGE, ls="-.", label="Constructive upper")
    ax.set(
        xlabel=r"Release fraction, $d$", ylabel=r"Variable heat / capacity (kJ mol$^{-1}$)", ylim=(0, None)
    )
    ax.legend(fontsize=6.5, loc="upper left")
    panel(ax, "c", "Convex heat envelopes")
    ax = axes[2]
    ax.fill_between(
        data["d"],
        0,
        data["old_upper_heat"] - data["reduced"],
        color=GRAY,
        alpha=0.15,
        label="Full-loading bracket",
    )
    ax.fill_between(
        data["d"],
        data["endpoint_low"] - data["reduced"],
        data["endpoint_upper"] - data["reduced"],
        color=BLUE,
        alpha=0.3,
        label="Endpoint bracket",
    )
    ax.plot(
        data["d"], data["path_homogeneous"] - data["reduced"], color=BLUE, label="Constructed path design"
    )
    ax.set(
        xlabel=r"Release fraction, $d$", ylabel=r"Heat above binding minimum (kJ mol$^{-1}$)", ylim=(0, 2.15)
    )
    ax.legend(fontsize=6.5, loc="upper right")
    panel(ax, "d", "Residual heat interval")
    save(fig, "Fig3_endpoint_bounds")
    plt.close(fig)


def decisions() -> None:
    data = (
        pl.read_csv(ROOT / "results/endpoint_brackets.csv").filter(pl.col("cluster") == 1).sort("new_upper")
    )
    fig = plt.figure(figsize=(7.2, 4.7), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.5])
    screening_bed(fig.add_subplot(grid[0, :]))
    axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    rank = np.arange(1, data.height + 1)
    ax = axes[0]
    ax.vlines(
        rank - 0.12, data["new_lower"], data["old_upper"], color=GRAY, alpha=0.7, lw=2, label="Reduced upper"
    )
    ax.vlines(rank + 0.12, data["new_lower"], data["new_upper"], color=BLUE, lw=2, label="Endpoint upper")
    ax.scatter(rank, data["new_lower"], color=ORANGE, s=8, zorder=3, label="Feasible design")
    ax.set(xlabel="Scenario rank (m = 1)", ylabel="Release fraction", ylim=(0, 1))
    ax.legend(fontsize=6.5, loc="lower right")
    panel(ax, "b", "Independent-site bounds")
    ax = axes[1]
    old, new = 100 * data["old_relative_gap"], 100 * data["new_relative_gap"]
    ax.plot([0, 8], [0, 8], color=GRAY, ls=":", lw=1)
    ax.scatter(old, new, color=BLUE, s=24, edgecolor="white", lw=0.4)
    ax.set(
        xlabel="Previous upper-to-feasible gap (%)",
        ylabel="Refined upper-to-feasible gap (%)",
        xlim=(0, 8),
        ylim=(0, 8),
    )
    ax.text(0.45, 6.9, "Median: 1.78 → 0.62%\nMaximum: 7.26 → 2.79%", fontsize=7)
    panel(ax, "c", "Upper-to-feasible gap")
    case = data.filter((pl.col("capacity") == 4) & (pl.col("dry_cp") == 1) & (pl.col("budget") == 100)).row(
        0, named=True
    )
    lower, upper = float(case["new_lower"]), float(case["new_upper"])
    ax = axes[2]
    ax.axvspan(0.44, lower, color=GREEN, alpha=0.18)
    ax.axvspan(lower, upper, color=ORANGE, alpha=0.18)
    ax.axvspan(upper, 0.462, color=GRAY, alpha=0.18)
    ax.axvline(lower, color=GREEN, ls="--", lw=1)
    ax.axvline(upper, color=BLUE, ls="--", lw=1)
    ax.axvline(float(case["old_upper"]), color=GRAY, ls=":", lw=1)
    targets = [0.445, 0.452, 0.455]
    ax.scatter(targets, [2, 1, 0], color=[GREEN, ORANGE, GRAY], s=32, zorder=5)
    for target, y, text in zip(targets, [2, 1, 0], ["Feasible", "Unresolved", "Excluded"], strict=True):
        ax.annotate(
            text,
            (target, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
        )
    ax.set(xlabel="Target release fraction", xlim=(0.44, 0.462), ylim=(-0.4, 2.6), yticks=[])
    ax.set_xticks([0.44, 0.45, 0.46])
    panel(ax, "d", "Screening outcomes")
    save(fig, "Fig4_decision_bounds")
    plt.close(fig)


if __name__ == "__main__":
    setup()
    bounds()
    decisions()
