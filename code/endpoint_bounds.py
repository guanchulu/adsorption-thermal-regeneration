# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1"]
# ///
from dataclasses import dataclass
from math import atanh, expm1, log, log1p, sqrt, tanh

import numpy as np
from path_model import Setting, Site
from scipy.optimize import brentq
from thermal_core import R, ThermalInputError


@dataclass(frozen=True, slots=True)
class EndpointCost:
    value: float
    lower_support: float
    residual: float
    v: float
    energy: float
    upper_occupation: float


def gas_heat_increment(ta: float, td: float) -> float:
    if not 298 <= ta <= td <= 1200:
        raise ThermalInputError("Shomate temperature interval")

    def primitive(t: float) -> float:
        return 24.99735 * t + 55.18696 * t * t / 2 - 33.69137 * t**3 / 3 + 7.948387 * t**4 / 4 + 0.136638 / t

    return primitive(td / 1000) - primitive(ta / 1000)


def log_odds(p: float) -> float:
    return log(p) - log1p(-p)


def endpoint_cost(d: float, a: float, j: float, log_ratio: float, m: int, *, upper: bool) -> EndpointCost:
    if not 0 < d < 1 or a <= 0 or j < 0 or log_ratio < 0 or m < 1:
        raise ThermalInputError("endpoint envelope domain")
    v_max = (1 - d) / 2
    pressure_only = tanh(m * log_ratio / 4)
    at_boundary = d <= pressure_only and log_ratio > 0
    ratio_minus_one = expm1(m * log_ratio)
    if at_boundary:
        discriminant = max(0.0, (1 - d) ** 2 - 4 * d / ratio_minus_one)
        v_max = 2 * d / (ratio_minus_one * (1 - d + sqrt(discriminant)))

    def derivative(v: float) -> float:
        ads = v + d
        binding = d / (m * a) * (1 / (ads * (1 - ads)) - 1 / (v * (1 - v)))
        sensible = j * (1 + ratio_minus_one) / (1 + ratio_minus_one * (1 - ads)) ** 2 if upper else j
        return binding + sensible

    boundary_minimum = derivative(v_max) <= 0
    if boundary_minimum:
        v = v_max
    else:
        left = v_max * 1e-12
        root, _info = brentq(derivative, left, v_max, xtol=1e-30, rtol=np.float64(1e-13), full_output=True)
        v = float(root)
    ads = v + d
    energy = max(0.0, ((log_odds(ads) - log_odds(v)) / m - log_ratio) / a)
    occupancy = ads / (1 + ratio_minus_one * (1 - ads))
    value = d * energy + j * (occupancy if upper else v)
    gradient = 0.0 if boundary_minimum and at_boundary else derivative(v)
    support = value + min(-gradient * v, gradient * (1 - d - v))
    guard = 1e-8 * (1 + abs(value))
    reduced = d * max(0.0, (4 * atanh(d) / m - log_ratio) / a)
    return EndpointCost(
        value, max(0.0, support - guard, reduced - guard), abs(gradient), v, energy, occupancy
    )


def release_bound(
    a: float, j: float, log_ratio: float, m: int, available: float, *, upper_heat: bool
) -> float:
    if available == 0 and j == 0:
        return tanh(m * log_ratio / 4)
    if available <= 0:
        return 0.0
    lo, hi = 0.0, 1 - 1e-12
    ceiling = endpoint_cost(hi, a, j, log_ratio, m, upper=upper_heat)
    guarded_ceiling = ceiling.value + 1e-8 * (1 + ceiling.value) if upper_heat else ceiling.lower_support
    if guarded_ceiling <= available:
        return hi if upper_heat else 1.0
    for _ in range(37):
        mid = (lo + hi) / 2
        cost = endpoint_cost(mid, a, j, log_ratio, m, upper=upper_heat)
        heat = cost.value + 1e-8 * (1 + cost.value) if upper_heat else cost.lower_support
        if heat <= available:
            lo = mid
        else:
            hi = mid
    return lo if upper_heat else hi


def temperature_interval_upper(setting: Setting, low: float, high: float) -> float:
    available = (setting.budget - setting.dry_cp * (low - setting.adsorption_K)) / setting.capacity
    a = (1 / setting.adsorption_K - 1 / high) / R
    j = gas_heat_increment(setting.adsorption_K, low)
    return release_bound(
        a, j, log(setting.fa_bar / setting.fd_bar), setting.cluster, available, upper_heat=False
    )


def constructive_site(setting: Setting, td: float) -> tuple[float, Site]:
    available = (setting.budget - setting.dry_cp * (td - setting.adsorption_K)) / setting.capacity
    a = (1 / setting.adsorption_K - 1 / td) / R
    j = gas_heat_increment(setting.adsorption_K, td)
    log_ratio = log(setting.fa_bar / setting.fd_bar)
    d = release_bound(a, j, log_ratio, setting.cluster, available, upper_heat=True)
    if d <= 0:
        raise ThermalInputError("positive budget left for endpoint construction")
    cost = endpoint_cost(d, a, j, log_ratio, setting.cluster, upper=True)
    entropy = (
        R * log(setting.fa_bar)
        + cost.energy / setting.adsorption_K
        - R / setting.cluster * log_odds(cost.v + d)
    )
    return d, Site(cost.energy, entropy, setting.cluster)
