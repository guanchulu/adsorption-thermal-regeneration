# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "pytest==9.1.1"]
# ///
# Run: uv run --project .. --frozen pytest -c pyproject.toml tests/test_path_model.py
import numpy as np
import pytest
from path_model import Setting, Site, co2_cp, endpoint_site, heat_path


def test_nist_heat_capacity_reference_value() -> None:
    # Given NIST Shomate CO2 coefficients and298.15K.
    # When the gas heat capacity is evaluated in kJ/mol/K.
    value = co2_cp(np.array([298.15]))[0]
    # Then it agrees with the tabulated gas heat capacity scale, not J units.
    assert value == pytest.approx(0.0371299625, rel=1e-7)


def test_loading_sensible_heat_is_positive_and_bounded() -> None:
    # Given a pressure-first isothermal depressurization followed by heating.
    setting = Setting()
    site = Site(30.0, 0.075)
    # When the full declared path heat is evaluated.
    result = heat_path(setting, site, 353.0)
    ceiling = setting.capacity * (353.0 - 298.0) * float(co2_cp(np.array([353.0]))[0])
    # Then the loading correction lies within the occupation-independent ceiling.
    assert 0 < result.loading < ceiling
    assert result.total == pytest.approx(result.dry + result.binding + result.loading)
    assert result.theta_ads > result.theta_after_pressure > result.theta_des


@pytest.mark.parametrize("cluster", [1, 4])
def test_loaded_heat_breaks_symmetric_endpoint_optimum(cluster: int) -> None:
    # Given a fixed positive release in the active binding-heat region.
    setting = Setting(fa_bar=0.1, fd_bar=0.1, capacity=2.0, cluster=cluster)
    symmetric = endpoint_site(setting, 350.0, (0.8, 0.2))
    lower_inventory = endpoint_site(setting, 350.0, (0.7999, 0.1999))
    # When endpoints shift equally toward lower loading at fixed release.
    old = heat_path(setting, symmetric, 350.0)
    new = heat_path(setting, lower_inventory, 350.0)
    # Then binding heat increases but full path heat decreases.
    assert new.release == pytest.approx(old.release, abs=1e-12)
    assert new.binding > old.binding
    assert new.total < old.total


def test_nondepressurizing_path_is_not_silently_counted_as_gross_heat() -> None:
    # Given a fugacity increase, excluded from the declared gross-heat path.
    setting = Setting(fa_bar=0.01, fd_bar=0.1)
    # When a path calculation is requested.
    with pytest.raises(ValueError, match="fd<=fa"):
        heat_path(setting, Site(30.0, 0.075), 353.0)
    # Then it is rejected rather than crediting released compression heat.
