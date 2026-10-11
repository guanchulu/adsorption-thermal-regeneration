# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2", "pillow==12.3.0"]
# ///
from pathlib import Path
from typing import Final

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

ROOT: Final = Path(__file__).resolve().parents[1]
BLUE: Final = "#28658B"
ORANGE: Final = "#C26B35"
GREEN: Final = "#378878"
GRAY: Final = "#7E8790"
INK: Final = "#253340"


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.titlepad": 12,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": True,
        }
    )


def panel(ax: Axes, letter: str, title: str) -> None:
    ax.set_title(title, loc="left")
    ax.text(-0.19, 1.085, letter, transform=ax.transAxes, weight="bold", fontsize=12)
    ax.tick_params(direction="out", length=3, width=0.7)


def save(fig: Figure, name: str) -> None:
    fig.canvas.draw()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(ROOT / "figures" / f"{name}.{ext}", dpi=400, bbox_inches="tight", pad_inches=0.07)
    with Image.open(ROOT / "figures" / f"{name}.png") as source:
        source.convert("L").save(ROOT / "qa" / f"{name}_gray.png")
