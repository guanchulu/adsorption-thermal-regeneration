# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest==9.1.1"]
# ///
# Run: uv run --frozen pytest tests/test_heat_interval_map.py
from dataclasses import replace
from math import sqrt

import pytest
from endpoint_bounds import gas_heat_increment
from heat_interval_map import heat_interval
from path_model import Setting


@pytest.mark.parametrize("temperature", [310.0, 350.0, 423.0])
@pytest.mark.parametrize("release", [0.05, 0.4, 0.9])
def test_direct_construction_is_inside_heat_interval(temperature: float, release: float) -> None:
    interval = heat_interval(Setting(), temperature, release)
    assert interval.lower_heat_kJ_kg <= interval.constructed_heat_kJ_kg + 1e-7
    assert interval.constructed_heat_kJ_kg <= interval.upper_heat_kJ_kg + 1e-7
    assert interval.relative_heat_gap_pct >= 0
    assert interval.upper_heat_kJ_kg >= temperature - 298


def test_zero_energy_construction_matches_pressure_step_occupancy() -> None:
    release = 0.05
    residual = ((1 - release) - sqrt((1 - release) ** 2 - 4 * release / 0.5)) / 2
    exact_heat = 52 + 4 * residual * gas_heat_increment(298, 350)
    interval = heat_interval(Setting(), 350.0, release)
    assert interval.constructed_heat_kJ_kg == pytest.approx(exact_heat, abs=1e-9)
    assert interval.lower_heat_kJ_kg < interval.constructed_heat_kJ_kg


def test_dry_heat_changes_both_endpoints_equally() -> None:
    base = heat_interval(Setting(), 350.0, 0.6)
    warm = heat_interval(replace(Setting(), dry_cp=2.0), 350.0, 0.6)
    assert warm.lower_heat_kJ_kg - base.lower_heat_kJ_kg == pytest.approx(52.0)
    assert warm.upper_heat_kJ_kg - base.upper_heat_kJ_kg == pytest.approx(52.0)
    assert warm.relative_heat_gap_pct < base.relative_heat_gap_pct
