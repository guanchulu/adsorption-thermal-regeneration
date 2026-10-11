# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "polars==2.0.0", "scipy==1.18.1"]
# ///
# Run: uv run --frozen python code/heat_interval_map.py
from dataclasses import asdict, dataclass
from math import log
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from endpoint_bounds import endpoint_cost, gas_heat_increment, log_odds
from path_model import Setting, Site, heat_path
from thermal_core import R

ROOT: Final = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class HeatInterval:
    temperature_K: float
    release_fraction: float
    lower_heat_kJ_kg: float
    upper_heat_kJ_kg: float
    constructed_heat_kJ_kg: float
    relative_heat_gap_pct: float


def heat_interval(setting: Setting, td: float, release: float) -> HeatInterval:
    a = (1 / setting.adsorption_K - 1 / td) / R
    j = gas_heat_increment(setting.adsorption_K, td)
    ratio = log(setting.fa_bar / setting.fd_bar)
    lower = endpoint_cost(release, a, j, ratio, setting.cluster, upper=False)
    upper = endpoint_cost(release, a, j, ratio, setting.cluster, upper=True)
    dry = setting.dry_cp * (td - setting.adsorption_K)
    h_low = dry + setting.capacity * lower.lower_support
    h_high = dry + setting.capacity * (upper.value + 1e-8 * (1 + abs(upper.value)))
    entropy = (
        R * log(setting.fa_bar)
        + upper.energy / setting.adsorption_K
        - R / setting.cluster * log_odds(upper.v + release)
    )
    state = heat_path(setting, Site(upper.energy, entropy, setting.cluster), td)
    if not h_low - 1e-7 <= state.total <= h_high + 1e-7:
        raise ArithmeticError("Direct path falls outside the endpoint heat interval")
    if abs(state.release - release) > 1e-9:
        raise ArithmeticError("Endpoint construction changed the specified release")
    return HeatInterval(td, release, h_low, h_high, state.total, 100 * (h_high - h_low) / h_high)


def main() -> None:
    setting = Setting()
    rows = [
        asdict(heat_interval(setting, float(td), float(release)))
        for td in np.linspace(310, 423, 61)
        for release in np.linspace(0.05, 0.9, 86)
    ]
    table = pl.DataFrame(rows)
    table.write_csv(ROOT / "results/heat_interval_map.csv")
    print(
        table.select(
            pl.len(),
            pl.col("relative_heat_gap_pct").min().alias("gap_min_pct"),
            pl.col("relative_heat_gap_pct").max().alias("gap_max_pct"),
        )
    )


if __name__ == "__main__":
    main()
