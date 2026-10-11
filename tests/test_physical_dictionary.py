# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pytest"]
# ///
# Run: PYTHONPATH=code uv run --project .. --no-sync --offline pytest tests/test_physical_dictionary.py
"""Observable adsorption/enthalpy checks, not unique site-recovery claims."""
import numpy as np
import pytest
from physical_dictionary import (
    PhysicalDictionaryError,
    SiteDictionary,
    default_dictionary,
    fit_dictionary,
    predict_qst,
    predict_uptake,
)
from scipy.special import expit


def test_single_basis_enthalpy_and_unseen_temperature() -> None:
    # Given one physical Langmuir site and uptake observations at two temperatures.
    grid = SiteDictionary(np.array([35.0]), np.array([2.0]))
    p = np.tile(np.geomspace(0.001, 1, 20), 2)
    t = np.repeat([273.0, 298.0], 20)
    q = 4.0 * expit(np.log(p) + 2 + 35 / 0.008314462618 * (1 / t - 1 / 298))
    # When fitting uptake only and querying an unrevealed temperature.
    fit = fit_dictionary(p, t, q, dictionary=grid)
    held_p = np.geomspace(0.002, 1, 15)
    prediction = predict_uptake(fit, held_p, np.full(15, 323.0))
    heat = predict_qst(fit, held_p, np.full(15, 323.0))
    # Then uptake and analytic constant enthalpy agree.
    expected = 4 * expit(np.log(held_p) + 2 + 35 / 0.008314462618 * (1 / 323 - 1 / 298))
    np.testing.assert_allclose(prediction, expected, atol=1e-10)
    np.testing.assert_allclose(heat, 35, atol=1e-10)


def test_two_basis_exact_prediction_without_unique_energy_claim() -> None:
    # Given a two-site generating dictionary with different affinities and enthalpies.
    energy = np.array([20.0, 65.0])
    affinity = np.array([-1.0, 4.0])
    grid = SiteDictionary(energy, affinity)
    p = np.tile(np.geomspace(0.001, 1, 24), 2)
    t = np.repeat([273.0, 298.0], 24)
    theta = expit(np.log(p[:, None]) + affinity + energy / 0.008314462618 * (1 / t[:, None] - 1 / 298))
    q = theta @ np.array([3.0, 1.5])
    # When fitting only the uptake curves.
    fit = fit_dictionary(p, t, q, dictionary=grid)
    held_p = np.geomspace(0.001, 1, 19)
    pred = predict_uptake(fit, held_p, np.full(19, 323.0))
    # Then held-out uptake is recovered, without claiming identified capacities.
    held_theta = expit(np.log(held_p[:, None]) + affinity + energy / 0.008314462618 * (1 / 323 - 1 / 298))
    np.testing.assert_allclose(pred, held_theta @ np.array([3.0, 1.5]), atol=1e-9)
    assert fit.capacity_nonidentifiable
    assert fit.normalized_rmse < 1e-10


def test_saturation_denominator_returns_nan_not_artificial_heat() -> None:
    # Given a fitted strongly saturated site.
    grid = SiteDictionary(np.array([40.0]), np.array([40.0]))
    p = np.array([0.1, 1.0])
    t = np.array([298.0, 298.0])
    fit = fit_dictionary(p, t, np.array([2.0, 2.0]), dictionary=grid)
    # When the differential response is effectively zero.
    heat = predict_qst(fit, p, t)
    # Then an unavailable heat is represented explicitly.
    assert np.isnan(heat).all()


def test_default_grid_has_required_physical_coordinates() -> None:
    # Given the predefined thermodynamic dictionary.
    grid = default_dictionary()
    # When inspecting its independent axes.
    energies = np.unique(grid.energy_kj_mol)
    affinities = np.unique(grid.logb_ref)
    # Then the required 21 by 25 basis is present.
    np.testing.assert_array_equal(energies, np.arange(0, 101, 5))
    np.testing.assert_array_equal(affinities, np.arange(-10, 15))
    assert grid.energy_kj_mol.size == 525


def test_zero_training_signal_is_not_fabricated_capacity() -> None:
    # Given identically zero uptake with no train-peak normalization scale.
    p = np.array([0.1, 1.0])
    # When attempting a fit.
    with pytest.raises(PhysicalDictionaryError):
        fit_dictionary(p, np.array([273.0, 298.0]), np.zeros(2))
    # Then failure is explicit rather than yielding an arbitrary capacity.
