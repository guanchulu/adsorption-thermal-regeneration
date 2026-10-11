from math import log

import numpy as np
import pytest
from endpoint_bounds import constructive_site, endpoint_cost, gas_heat_increment, temperature_interval_upper
from path_model import Setting, Site, co2_cp, heat_path
from scipy.integrate import quad
from scipy.special import expit
from thermal_core import R


def test_integrated_gas_heat() -> None:
    def cp(t: float) -> float:
        return float(co2_cp(np.array([t]))[0])

    numerical = quad(cp, 298, 423)[0]
    assert gas_heat_increment(298, 423) == pytest.approx(numerical, abs=1e-12)


@pytest.mark.parametrize("m", [1, 4])
def test_endpoint_ensemble_bounds(m: int) -> None:
    setting = Setting(cluster=m)
    td = 350.0
    sites = [Site(18.0, 0.055, m), Site(35.0, 0.095, m), Site(50.0, 0.15, m)]
    weights = [0.2, 0.3, 0.5]
    states = [heat_path(setting, site, td) for site in sites]
    d = sum(weight * state.release for weight, state in zip(weights, states, strict=True))
    h = (
        sum(weight * (state.binding + state.loading) for weight, state in zip(weights, states, strict=True))
        / setting.capacity
    )
    a = (1 / 298 - 1 / td) / R
    j = gas_heat_increment(298, td)
    lower = endpoint_cost(d, a, j, log(1.5), m, upper=False)
    upper = endpoint_cost(d, a, j, log(1.5), m, upper=True)
    assert lower.lower_support <= h + 1e-10
    entropy = R * log(0.15) + upper.energy / 298 - R / m * log((upper.v + d) / (1 - upper.v - d))
    state = heat_path(setting, Site(upper.energy, entropy, m), td)
    assert state.release == pytest.approx(d, abs=1e-12)
    assert state.binding + state.loading <= setting.capacity * upper.value + 1e-9


@pytest.mark.parametrize("d", [0.02, 0.10, 0.40, 0.8])
def test_supporting_line_below_full_domain(d: float) -> None:
    a, j, log_ratio, m = 0.1, 2.0, log(1.5), 4
    result = endpoint_cost(d, a, j, log_ratio, m, upper=False)
    v = np.linspace(1e-7, 1 - d - 1e-7, 10001)
    energy = ((np.log((v + d) / (1 - v - d)) - np.log(v / (1 - v))) / m - log_ratio) / a
    f = np.maximum(0, d * energy) + j * v
    assert result.lower_support <= float(f.min()) + 1e-10
    assert result.energy >= 0


def test_constructive_heat_and_continuous_temperature_interval() -> None:
    setting = Setting()
    d, site = constructive_site(setting, 344.0)
    actual = heat_path(setting, site, 344.0)
    assert actual.total < setting.budget
    assert actual.release == pytest.approx(d, abs=1e-12)
    upper = temperature_interval_upper(setting, 340.0, 345.0)
    assert upper >= d
    assert expit(site.energy / (R * 344.0) - site.entropy / R + log(0.1)) < 1
