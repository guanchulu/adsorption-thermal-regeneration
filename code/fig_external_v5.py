# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2", "numpy==2.5.3", "polars==2.0.0", "pillow==12.3.0"]
# ///
# Run: uv run --project .. --no-sync --offline python code/fig_external_v5.py
"""External temperature/enthalpy validation with fixed quantile examples."""

import json
import warnings
from typing import Final

import matplotlib.pyplot as plt
import polars as pl
from fig_external_data_v5 import Case, prepare
from figure_style import BLUE, GRAY, GREEN, INK, ORANGE, ROOT, setup
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image

NAME: Final = "Fig5_external_validation"
TEMPERATURES: Final = ((273, BLUE, "o", "-"), (298, GREEN, "s", "-."), (323, ORANGE, "^", "--"))


class FigureLayoutError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Figure rendering audit failed; inspect layout JSON")


def isotherm(ax: Axes, case: Case) -> None:
    """Plot original pressure coordinates and fitted/predicted uptake."""
    for temperature, color, marker, style in TEMPERATURES:
        points = case.points.filter(
            (pl.col("temperature_K") == temperature)
            & ((pl.col("temperature_K") < 323) | pl.col("held_out_test"))
        )
        curve = case.curves.filter(pl.col("temperature_K") == temperature)
        ax.plot(curve["pressure_bar"], curve["predicted_uptake_mol_kg"], color=color, ls=style, lw=1.5)
        ax.scatter(
            points["pressure_bar"],
            points["uptake_mol_kg"],
            color=color,
            marker=marker,
            s=19,
            facecolors="white",
            linewidths=0.9,
            zorder=3,
        )
    ax.set(xscale="log", xlabel=r"Pressure, $P$ (bar)", ylabel=r"Uptake, $q$ (mol kg$^{-1}$)", ylim=(0, None))
    ax.set_xticks([0.001, 0.01, 0.1, 1.0])
    ax.text(
        0.03,
        0.96,
        f"{case.structure}\n323 K NRMSE = {case.uptake_error:.4f}",
        transform=ax.transAxes,
        va="top",
        fontsize=7.5,
    )


def thermal(ax: Axes, cases: list[Case]) -> None:
    """Plot fully withheld 323 K heat observables for the same two structures."""
    for case, color, marker, style in zip(cases, (BLUE, ORANGE), ("o", "s"), ("-", "--"), strict=True):
        held = case.heat
        ax.scatter(
            held["pressure_bar"],
            held["minus_reported_enthalpy_kj_mol"],
            color=color,
            marker=marker,
            s=25,
            facecolors="white",
            linewidths=1.0,
            label=case.structure,
            zorder=3,
        )
        ax.plot(held["pressure_bar"], held["predicted_heat_kj_mol"], color=color, ls=style, lw=1.5)
    ax.set(
        xscale="log",
        xlabel=r"Pressure, $P$ (bar)",
        ylabel=r"Heat observable (kJ mol$^{-1}$)",
        ylim=(26, 40.8),
    )
    ax.set_xticks([0.001, 0.01, 0.1, 1.0])
    ax.legend(loc="center left", fontsize=7)
    ax.text(
        0.97, 0.96, "323 K; all enthalpy withheld", transform=ax.transAxes, ha="right", va="top", fontsize=7
    )


def aggregate(ax: Axes, cohort: pl.DataFrame, cases: list[Case]) -> None:
    """Retain every reference structure and show sample-to-cohort context."""
    ax.scatter(
        cohort["uptake_nrmse"], cohort["common_heat_mae_kj_mol"], s=9, color=GRAY, alpha=0.45, linewidths=0
    )
    for case, color, marker in zip(cases, (BLUE, ORANGE), ("o", "s"), strict=True):
        ax.scatter(
            [case.uptake_error],
            [case.heat_error],
            s=45,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
        )
    ax.set(xscale="log", xlabel="323 K uptake NRMSE", ylabel=r"Heat-observable MAE (kJ mol$^{-1}$)")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xticks([0.001, 0.01, 0.1])
    ax.set_ylim(bottom=-0.004)
    ax.text(
        0.03,
        0.33,
        "1,352 structures\nMedian uptake NRMSE: 0.00324\nMedian heat MAE: 1.59 kJ mol$^{-1}$",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
    )


def audit(fig: Figure) -> None:
    """Record artist clipping and rendering warnings before final export."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
        fig.savefig(ROOT / "qa" / f"{NAME}_preview.png", dpi=180)
        fig.canvas.draw()
    canvas = fig.bbox
    outside: list[str] = []
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent()
            if (
                box.x0 < canvas.x0 - 1
                or box.y0 < canvas.y0 - 1
                or box.x1 > canvas.x1 + 1
                or box.y1 > canvas.y1 + 1
            ):
                outside.append(text.get_text())
    (ROOT / "qa" / f"{NAME}_layout.json").write_text(
        json.dumps(
            {
                "outside_canvas": outside,
                "warnings": [str(item.message) for item in caught],
                "size_inches": list(fig.get_size_inches()),
                "panel_count": len(fig.axes),
                "all_cohort_points_retained": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if outside or caught:
        raise FigureLayoutError


def main() -> None:
    setup()
    cohort, cases = prepare()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.86, hspace=0.48, wspace=0.34)
    isotherm(axes[0, 0], cases[0])
    isotherm(axes[0, 1], cases[1])
    thermal(axes[1, 0], cases)
    aggregate(axes[1, 1], cohort, cases)
    titles = (
        "Median-error structure",
        "90th-percentile-error structure",
        "Uptake fit does not fix enthalpy",
        "Cohort-wide thermal discrepancy",
    )
    for ax, letter, title in zip(axes.flat, "abcd", titles, strict=True):
        ax.set_title(title, loc="left", fontsize=9, pad=10)
        position = ax.get_position()
        fig.text(position.x0 - 0.06, position.y1 + 0.02, letter, weight="bold", fontsize=12)
        ax.tick_params(direction="out", length=3, width=0.7)
    fig.text(
        0.10,
        0.97,
        r"Public CO$_2$ isotherms and withheld adsorption enthalpy",
        color=INK,
        fontsize=10,
        weight="bold",
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            ls=style,
            markerfacecolor="white",
            lw=1.4,
            label=f"{temp} K " + ("training" if temp < 323 else "held out"),
        )
        for temp, color, marker, style in TEMPERATURES
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.09, 0.95), ncol=3, fontsize=7.5)
    fig.text(
        0.10,
        0.02,
        "CRAFTED2.0.1  |  DREIDING / DDEC  |  Symbols: GCMC  |  Lines: thermal single-site model",
        fontsize=7,
        color=GRAY,
    )
    audit(fig)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(ROOT / "figures" / f"{NAME}.{extension}", dpi=400)
    with Image.open(ROOT / "figures" / f"{NAME}.png") as source:
        source.convert("L").save(ROOT / "qa" / f"{NAME}_gray.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
