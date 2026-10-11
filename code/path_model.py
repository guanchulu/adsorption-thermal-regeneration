# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1"]
# ///
# Run: uv run --project .. --frozen pytest -c pyproject.toml tests/test_path_model.py
"""Explicit pressure-first, fixed-fugacity heating path with loading sensible heat."""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit
from thermal_core import R, ThermalInputError

Vector = NDArray[np.float64]
NODES: Final = tuple(np.asarray(a, dtype=float) for a in np.polynomial.legendre.leggauss(96))


@dataclass(frozen=True, slots=True)
class Site:
    energy: float
    entropy: float
    cluster: int = 1


@dataclass(frozen=True, slots=True)
class Setting:
    adsorption_K: float = 298.0
    fa_bar: float = 0.15
    fd_bar: float = 0.1
    capacity: float = 4.0
    dry_cp: float = 1.0
    budget: float = 100.0
    temperature_cap: float = 423.0
    cluster: int = 1


@dataclass(frozen=True, slots=True)
class Heat:
    release: float
    theta_ads: float
    theta_after_pressure: float
    theta_des: float
    dry: float
    binding: float
    loading: float
    total: float


def co2_cp(temperature: Vector) -> Vector:
    """NIST gas Shomate298-1200K, converted from J to kJ/mol/K."""
    if np.any((temperature < 298.0) | (temperature > 1200.0)):
        raise ThermalInputError("CO2 Shomate range298-1200K")
    t = temperature / 1000.0
    return np.asarray(
        (24.99735 + 55.18696 * t - 33.69137 * t * t + 7.948387 * t**3 - 0.136638 / t**2) / 1000.0, dtype=float
    )


def heat_path(setting: Setting, site: Site, desorption_K: float) -> Heat:
    """Gross path heat; only depressurization and monotone heating are included."""
    if setting.fd_bar > setting.fa_bar or setting.fd_bar <= 0 or site.energy < 0 or site.cluster < 1:
        raise ThermalInputError("fd<=fa, positive fugacity, nonnegative enthalpy, positive cluster size")
    if not setting.adsorption_K < desorption_K <= setting.temperature_cap:
        raise ThermalInputError("Ta<Td<=temperature cap")
    theta_ads = float(
        expit(
            site.cluster
            * (np.log(setting.fa_bar) + site.energy / (R * setting.adsorption_K) - site.entropy / R)
        )
    )
    theta_start = float(
        expit(
            site.cluster
            * (np.log(setting.fd_bar) + site.energy / (R * setting.adsorption_K) - site.entropy / R)
        )
    )
    theta_des = float(
        expit(site.cluster * (np.log(setting.fd_bar) + site.energy / (R * desorption_K) - site.entropy / R))
    )
    half = (desorption_K - setting.adsorption_K) / 2
    nodes, weights = NODES
    temperature = half * nodes + (desorption_K + setting.adsorption_K) / 2
    theta = expit(
        site.cluster * (np.log(setting.fd_bar) + site.energy / (R * temperature) - site.entropy / R)
    )
    loading = float(setting.capacity * half * np.dot(weights, theta * co2_cp(temperature)))
    dry = setting.dry_cp * (desorption_K - setting.adsorption_K)
    released = theta_ads - theta_des
    binding = setting.capacity * site.energy * released
    return Heat(released, theta_ads, theta_start, theta_des, dry, binding, loading, dry + binding + loading)


def endpoint_site(setting: Setting, temperature: float, endpoints: tuple[float, float]) -> Site:
    adsorption, desorption = endpoints
    if not 0 < desorption < adsorption < 1:
        raise ThermalInputError("0<theta_des<theta_ads<1")
    a = (1 / setting.adsorption_K - 1 / temperature) / R
    log_ratio = np.log(setting.fa_bar / setting.fd_bar)
    energy = float(((logit(adsorption) - logit(desorption)) / setting.cluster - log_ratio) / a)
    entropy = float(
        R * np.log(setting.fa_bar) + energy / setting.adsorption_K - R / setting.cluster * logit(adsorption)
    )
    return Site(energy, entropy, setting.cluster)
