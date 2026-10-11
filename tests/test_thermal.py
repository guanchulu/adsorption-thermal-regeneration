# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "pytest==9.1.1"]
# ///
# Run: uv run --frozen pytest tests/test_thermal.py
import numpy as np
import pytest
from thermal_core import (
    Cycle,
    R,
    Resources,
    difference,
    homogeneous_optimum,
    minimum_binding_heat,
    resource_bound,
)


@pytest.mark.parametrize(("pa", "pd"), [(0.15, 0.01), (0.15, 0.1), (0.0004, 0.1)])
def test_entropy_center_attains_fixed_energy_response_envelope(pa: float, pd: float) -> None:
    # Given a fixed energy above the positive-release threshold and unconstrained entropy.
    cycle = Cycle(298.0, 393.0, pa, pd)
    a = (1 / 298.0 - 1 / 393.0) / R
    energy = max(30.0, -np.log(pa / pd) / a + 10.0)
    entropy = R / 2 * np.log(pa * pd) + energy / 2 * (1 / 298.0 + 1 / 393.0)
    # When the occupation difference is evaluated at the centered entropy.
    value = difference(np.array([energy]), np.array([entropy]), cycle)[0]
    # Then it equals the analytic logistic separation bound.
    assert value == pytest.approx(np.tanh((a * energy + np.log(pa / pd)) / 4))


@pytest.mark.parametrize(("pa", "pd"), [(0.15, 0.01), (0.15, 0.1), (0.0004, 0.1)])
def test_random_mixtures_respect_binding_heat_bound(pa: float, pd: float) -> None:
    # Given distributions including possible negative net-release sites.
    rng = np.random.default_rng(20261007)
    cycle = Cycle(298.0, 393.0, pa, pd)
    energies = rng.uniform(0.0, 150.0, size=(100, 50))
    entropies = rng.uniform(0.01, 0.20, size=(100, 50))
    weights = rng.dirichlet(np.ones(50), size=100)
    # When actual signed releases and binding heats are averaged.
    delta = difference(energies, entropies, cycle)
    means = np.sum(weights * delta, axis=1)
    heats = np.sum(weights * energies * delta, axis=1)
    # Then every positive average release satisfies the derived bound.
    for u, heat in zip(means, heats, strict=True):
        if u > 0:
            assert heat >= minimum_binding_heat(float(u), cycle) - 1e-10


def test_budget_below_sensible_heating_is_temperature_infeasible() -> None:
    # Given a target temperature whose solid sensible heat exceeds the budget.
    cycle = Cycle(298.0, 398.0, 0.15, 0.01)
    # When the bound is evaluated without invented heat recovery.
    result = resource_bound(cycle, Resources(4.0, 1.0, 50.0))
    # Then the target temperature itself is flagged infeasible.
    assert not result.temperature_feasible


def test_resource_bound_attains_declared_heat_budget() -> None:
    # Given an entropy-relaxed independent-site benchmark at equal fugacity.
    cycle = Cycle(298.0, 353.0, 0.1, 0.1)
    resources = Resources(4.0, 1.0, 120.0)
    # When the sharp resource bound is solved.
    result = resource_bound(cycle, resources)
    actual = difference(np.array([result.energy_kj_mol]), np.array([result.entropy_loss_kj_mol_K]), cycle)[0]
    # Then its representative state attains both utilization and heat balance.
    assert actual == pytest.approx(result.utilization, abs=1e-10)
    assert result.sensible_heat_kj_kg + result.binding_heat_kj_kg == pytest.approx(120.0)


def test_bound_is_nondecreasing_in_heat_budget() -> None:
    # Given a fixed cycle and material capacity/heat capacity.
    cycle = Cycle(298.0, 353.0, 0.15, 0.1)
    # When additional heat is made available.
    values = [resource_bound(cycle, Resources(4.0, 1.0, b)).utilization for b in (60.0, 80.0, 120.0, 200.0)]
    # Then the optimistic releasable fraction cannot decrease.
    assert np.all(np.diff(values) >= 0)


def test_tiny_positive_release_peak_survives_saturated_tails() -> None:
    # Given a dilute-feed cycle whose broad high-energy probes round both coverages to one.

    cycle = Cycle(298.0, 353.0, 0.0004, 0.1)
    # When the continuous homogeneous optimum is located.
    result = homogeneous_optimum(cycle, 0.14, 0.1)
    # Then the resolved interior peak cannot be mistaken for zero at the threshold.
    assert 90.4 < result.energy < 90.6
    assert result.utilization > 1.30e-6


def test_joint_temperature_optimum_matches_independent_scan() -> None:
    # Given jointly adjustable entropy/enthalpy and a finite thermal budget.
    resources = Resources(4.0, 1.0, 100.0)
    expected = np.sqrt(298.0**2 + 100.0 * 298.0)
    # When the exact resource bound is evaluated across temperatures around the optimum.
    center = resource_bound(Cycle(298.0, expected, 0.15, 0.1), resources).utilization
    nearby = [
        resource_bound(Cycle(298.0, t, 0.15, 0.1), resources).utilization
        for t in np.linspace(expected - 20, expected + 20, 81)
    ]
    # Then none of the independently sampled temperatures exceeds the closed-form choice.
    assert center >= max(nearby) - 1e-12


@pytest.mark.parametrize("cap", [323.0, 353.0, 393.0])
def test_temperature_cap_requires_recomputed_resource_bound(cap: float) -> None:
    # Given a temperature ceiling on the jointly designed benchmark.
    resources = Resources(4.0, 1.0, 100.0)
    selected = min(np.sqrt(298.0**2 + 100.0 * 298.0), cap)
    # When the attainable bound is recomputed at the capped temperature.
    value = resource_bound(Cycle(298.0, selected, 0.15, 0.1), resources).utilization
    alternatives = [
        resource_bound(Cycle(298.0, float(t), 0.15, 0.1), resources).utilization
        for t in np.linspace(299.0, cap, 101)
    ]
    # Then the clipped optimum bounds all sampled feasible temperatures.
    assert value >= max(alternatives) - 1e-12
