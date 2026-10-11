# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "pytest==9.1.1", "pydantic==2.13.5"]
# ///
# Run: uv run --project .. --frozen pytest code/public_case_tests.py
# ruff: noqa: S101 -- this owned module contains pytest assertions.

from math import exp, isclose

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from fig_public_case_v5 import comparator_panel
from public_case_model import (
    ROOT,
    budget_temperature,
    case_heat,
    firstlaw_heat,
    independent_loading_heat,
    load_case,
    occupation,
)
from thermal_core import R


@pytest.mark.parametrize("temperature", [298.0, 303.0, 323.0, 373.0, 423.0])
def test_pressure_units_and_entropy(temperature: float) -> None:
    case = load_case()
    for b0, (_capacity, site) in zip((case.b01_per_kPa, case.b02_per_kPa), case.sites, strict=True):
        pressure_kPa = 15.0
        odds = b0 * pressure_kPa * exp(site.energy / (R * temperature))
        assert isclose(odds / (1 + odds), occupation(site, (temperature, 0.15)), rel_tol=1e-13)


@pytest.mark.parametrize("temperature", [298.0, 303.0, 323.0, 373.0, 423.0])
def test_open_system_firstlaw(temperature: float) -> None:
    case = load_case()
    direct = case_heat(case, temperature)
    assert isclose(direct.path_heat_kJ_kg, firstlaw_heat(case, temperature), abs_tol=1e-9)
    assert isclose(direct.retained_heat_kJ_kg, independent_loading_heat(case, temperature), abs_tol=1e-9)


def test_budget_inversion_and_pressure_leg() -> None:
    case = load_case()
    assert budget_temperature(case, 0.0, complete=True) is None
    for budget in (25.0, 50.0, 100.0):
        full_t = budget_temperature(case, budget, complete=True)
        reduced_t = budget_temperature(case, budget, complete=False)
        assert full_t is not None
        assert reduced_t is not None
        assert full_t < reduced_t
        assert isclose(case_heat(case, full_t).path_heat_kJ_kg, budget, abs_tol=1e-8)
        assert case_heat(case, reduced_t).path_heat_kJ_kg > budget
    heats = [case_heat(case, float(t)).path_heat_kJ_kg for t in np.linspace(298, 423, 251)]
    assert all(np.diff(heats) > 0)


def test_figure_comparator_accepts_float_budget_columns() -> None:

    budgets = pl.read_csv(ROOT / "results/public_case_budget_release.csv")
    bounds = pl.read_csv(ROOT / "results/public_case_resource_bounds.csv")
    fig, ax = plt.subplots()
    comparator_panel(ax, (budgets, bounds))
    assert len(ax.collections) >= 2
    plt.close(fig)
