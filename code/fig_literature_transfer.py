# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "polars==2.0.0", "matplotlib==3.11.2", "pillow==12.3.0"]
# ///
# Run: uv run --frozen python code/fig_literature_transfer.py
import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from figure_style import BLUE, GRAY, ORANGE, ROOT, panel, save, setup
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import StrMethodFormatter


def difference_axis(ax: Axes) -> None:
    ax.axhline(0, color=GRAY, lw=1, ls="--", zorder=1)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylim(-0.6, 0.06)
    ax.set_yticks([-0.3, -0.1, -0.03, -0.01, 0, 0.01, 0.03])
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.set_ylabel(r"Paired error difference, $\Delta$NRMSE")


def main() -> None:
    setup()
    data = pl.read_csv(ROOT / "results/nist_paired_differences.csv").sort("paired_difference")
    data = data.with_row_index("display_rank", offset=1)
    summary = pl.read_csv(ROOT / "results/nist_paired_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.65), layout="constrained")
    a, b, c, d = axes.flat
    a.plot([0.003, 0.8], [0.003, 0.8], color=GRAY, ls="--", lw=1, zorder=1)
    b.vlines(data["display_rank"], 0, data["paired_difference"], color=GRAY, lw=0.7, alpha=0.6)
    for high, color, marker, label in (
        (False, BLUE, "o", r"$T_{\rm test}<353$ K (n = 23)"),
        (True, ORANGE, "^", r"$T_{\rm test}\geq353$ K (n = 9)"),
    ):
        part = data.filter((pl.col("heldout_temperature_K") >= 353) == high)
        for ax, xfield, yfield in (
            (a, "baseline_nrmse", "thermal_nrmse"),
            (b, "display_rank", "paired_difference"),
            (c, "extrapolation_gap_K", "paired_difference"),
        ):
            ax.scatter(
                part[xfield],
                part[yfield],
                s=33,
                marker=marker,
                color=color,
                edgecolors="white",
                linewidths=0.4,
                alpha=0.9,
                label=label,
                zorder=3,
            )
    a.set(
        xscale="log",
        yscale="log",
        xlim=(0.003, 0.8),
        ylim=(0.003, 0.8),
        xticks=[0.01, 0.1],
        yticks=[0.01, 0.1],
        xlabel="Pressurewise baseline NRMSE",
        ylabel="Thermal Langmuir NRMSE",
    )
    a.legend(loc="upper left", fontsize=7, handletextpad=0.3, labelspacing=0.3)
    panel(a, "a", r"Paired $\mathrm{CO_2}$ predictions")
    difference_axis(b)
    b.set(xlabel="Group rank by paired difference", xlim=(0, 33), xticks=[1, 8, 16, 24, 32])
    b.text(0.45, 0.10, "20 better / 12 worse", transform=b.transAxes, fontsize=8)
    panel(b, "b", "Gains vary across source groups")
    difference_axis(c)
    c.set(
        xlabel=r"Extrapolation gap, $T_{\rm test}-T_{\rm train,max}$ (K)",
        xlim=(0, 55),
        xticks=[0, 10, 20, 30, 40, 50],
    )
    panel(c, "c", "Temperature distance and error")
    d.axvline(0, color=GRAY, lw=1, ls="--", zorder=1)
    labels = []
    for position, (key, label, color, marker) in enumerate(
        (
            ("all", "All groups", GRAY, "D"),
            ("heldout_below_353K", r"$T_{\rm test}<353$ K", BLUE, "o"),
            ("heldout_at_least_353K", r"$T_{\rm test}\geq353$ K", ORANGE, "^"),
        )
    ):
        row = summary.filter(pl.col("stratum") == key)
        lower = float(row["q1_difference"].item())
        center = float(row["median_difference"].item())
        upper = float(row["q3_difference"].item())
        d.plot([lower, upper], [position, position], color=color, lw=4, solid_capstyle="round")
        d.scatter(
            [center],
            [position],
            s=44,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        labels.append(f"{label}\nn = {row['n_groups'].item()}")
    d.set(
        yticks=[0, 1, 2],
        yticklabels=labels,
        ylim=(2.7, -0.7),
        xlim=(-0.045, 0.018),
        xlabel=r"Paired error difference, $\Delta$NRMSE",
    )
    d.set_xticks([-0.04, -0.02, 0, 0.01])
    d.text(0.03, 0.94, "Median and sample IQR", transform=d.transAxes, fontsize=7.5)
    panel(d, "d", "Descriptive temperature strata")
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    outside = []
    for index, ax in enumerate(axes.flat):
        for label in [
            ax.title,
            ax.xaxis.label,
            ax.yaxis.label,
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
            *ax.texts,
        ]:
            if not label.get_visible() or not label.get_text():
                continue
            box = label.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                outside.append({"panel": index, "text": label.get_text()})
    audit = {
        "panels": len(fig.axes),
        "rows": 2,
        "columns": 2,
        "groups": data.height,
        "source_dois": data["doi"].n_unique(),
        "minimum_positive_nrmse": float(np.min(data.select("thermal_nrmse", "baseline_nrmse").to_numpy())),
        "outside_labels": outside,
    }
    (ROOT / "qa/figure6_layout.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if outside:
        raise RuntimeError("Figure 6 contains labels outside the canvas")
    data.write_csv(ROOT / "results/figure6_literature_source.csv")
    save(fig, "Fig6_literature_temperature_transfer")
    plt.close(fig)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
