# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1"]
# ///
# Run: uv run --frozen pytest tests/test_thermal.py
"""Independent-site temperature-swing responses and conditional heat bounds."""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import expit

R: Final = 0.008314462618
Vector = NDArray[np.float64]


class ThermalInputError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(f"Invalid thermal input: {detail}")


@dataclass(frozen=True, slots=True)
class Cycle:
    adsorption_K: float
    desorption_K: float
    adsorption_fugacity_bar: float
    desorption_fugacity_bar: float

    def __post_init__(self) -> None:
        values = (
            self.adsorption_K,
            self.desorption_K,
            self.adsorption_fugacity_bar,
            self.desorption_fugacity_bar,
        )
        if not np.isfinite(values).all() or min(values) <= 0 or self.desorption_K <= self.adsorption_K:
            raise ThermalInputError("positive finite states and desorption T>adsorption T")

    @property
    def thermal_factor(self) -> float:
        return (1 / self.adsorption_K - 1 / self.desorption_K) / R

    @property
    def log_fugacity_ratio(self) -> float:
        return float(np.log(self.adsorption_fugacity_bar / self.desorption_fugacity_bar))


@dataclass(frozen=True, slots=True)
class Resources:
    capacity_mol_kg: float
    heat_capacity_kj_kg_K: float
    heat_budget_kj_kg: float

    def __post_init__(self) -> None:
        vals = (self.capacity_mol_kg, self.heat_capacity_kj_kg_K, self.heat_budget_kj_kg)
        if not np.isfinite(vals).all() or self.capacity_mol_kg <= 0 or min(vals[1:]) < 0:
            raise ThermalInputError("positive capacity and nonnegative heat capacity/budget")


@dataclass(frozen=True, slots=True)
class ResourceBound:
    temperature_feasible: bool
    utilization: float
    energy_kj_mol: float
    entropy_loss_kj_mol_K: float
    sensible_heat_kj_kg: float
    binding_heat_kj_kg: float


@dataclass(frozen=True, slots=True)
class HomogeneousOptimum:
    energy: float
    utilization: float
    binding_heat: float


def difference(energy: Vector, entropy: Vector, cycle: Cycle) -> Vector:
    za = np.log(cycle.adsorption_fugacity_bar) + energy / (R * cycle.adsorption_K) - entropy / R
    zd = np.log(cycle.desorption_fugacity_bar) + energy / (R * cycle.desorption_K) - entropy / R
    large = np.maximum(za, zd)
    separation = np.abs(za - zd)
    with np.errstate(divide="ignore"):
        log_magnitude = large + np.log(-np.expm1(-separation)) - np.logaddexp(0.0, za) - np.logaddexp(0.0, zd)
    return np.asarray(np.sign(za - zd) * np.exp(log_magnitude), dtype=float)


def minimum_binding_heat(utilization: float, cycle: Cycle) -> float:
    """Optimistic normalized binding heat for a positive mean release fraction."""
    if not 0 <= utilization < 1:
        raise ThermalInputError("release fraction must lie in [0,1)")
    energy = max(0.0, (4 * np.arctanh(utilization) - cycle.log_fugacity_ratio) / cycle.thermal_factor)
    return float(utilization * energy)


def resource_bound(cycle: Cycle, resources: Resources) -> ResourceBound:
    sensible = resources.heat_capacity_kj_kg_K * (cycle.desorption_K - cycle.adsorption_K)
    available = resources.heat_budget_kj_kg - sensible
    if available < 0:
        return ResourceBound(
            temperature_feasible=False,
            utilization=0.0,
            energy_kj_mol=float("nan"),
            entropy_loss_kj_mol_K=float("nan"),
            sensible_heat_kj_kg=sensible,
            binding_heat_kj_kg=0.0,
        )
    pressure_only = max(0.0, float(np.tanh(cycle.log_fugacity_ratio / 4)))
    utilization = pressure_only
    if available > 0:
        upper = float(np.nextafter(1.0, 0.0))
        if resources.capacity_mol_kg * minimum_binding_heat(upper, cycle) < available:
            raise ThermalInputError("utilization closer to one than floating-point resolution")

        def budget_error(value: float) -> float:
            return resources.capacity_mol_kg * minimum_binding_heat(value, cycle) - available

        root, _info = brentq(budget_error, pressure_only, upper, xtol=5e-15, full_output=True)
        utilization = float(root)
    energy = max(0.0, float((4 * np.arctanh(utilization) - cycle.log_fugacity_ratio) / cycle.thermal_factor))
    entropy = R / 2 * np.log(cycle.adsorption_fugacity_bar * cycle.desorption_fugacity_bar) + energy / 2 * (
        1 / cycle.adsorption_K + 1 / cycle.desorption_K
    )
    return ResourceBound(
        temperature_feasible=True,
        utilization=utilization,
        energy_kj_mol=energy,
        entropy_loss_kj_mol_K=float(entropy),
        sensible_heat_kj_kg=sensible,
        binding_heat_kj_kg=resources.capacity_mol_kg * energy * utilization,
    )


def homogeneous_optimum(cycle: Cycle, entropy: float, normalized_budget: float) -> HomogeneousOptimum:
    """Continuous fixed-entropy comparator over the full efficient positive branch."""
    lower = max(0.0, -cycle.log_fugacity_ratio / cycle.thermal_factor)

    def response(energy: float) -> float:
        return float(difference(np.array([energy]), np.array([entropy]), cycle)[0])

    def log_slope(energy: float) -> float:
        alpha, beta = 1 / (R * cycle.adsorption_K), 1 / (R * cycle.desorption_K)
        za = np.log(cycle.adsorption_fugacity_bar) + alpha * energy - entropy / R
        zd = np.log(cycle.desorption_fugacity_bar) + beta * energy - entropy / R
        gap = cycle.thermal_factor * energy + cycle.log_fugacity_ratio
        if gap <= 0:
            return float("inf")
        return float(
            alpha * (1 - expit(za))
            - beta * expit(zd)
            + cycle.thermal_factor * np.exp(-gap) / (-np.expm1(-gap))
        )

    peak_energy = lower
    if log_slope(lower) > 0:
        right = max(200.0, lower + 200.0)
        while log_slope(right) > 0:
            right *= 2
        peak_root, _info = brentq(log_slope, lower, right, xtol=1e-11, full_output=True)
        peak_energy = float(peak_root)
    if peak_energy * response(peak_energy) <= normalized_budget:
        energy = peak_energy
    elif normalized_budget == 0:
        energy = lower
    else:

        def heat_error(value: float) -> float:
            return value * response(value) - normalized_budget

        root, _info = brentq(heat_error, lower, peak_energy, xtol=1e-11, full_output=True)
        energy = float(root)
    value = response(energy)
    return HomogeneousOptimum(energy, value, energy * value)


def optimal_joint_temperature(adsorption_K: float, resources: Resources) -> float:
    """Interior benchmark when entropy, enthalpy and temperature are co-designed."""
    if adsorption_K <= 0 or resources.heat_capacity_kj_kg_K <= 0 or resources.heat_budget_kj_kg <= 0:
        raise ThermalInputError("positive Ta, heat capacity and budget for finite joint optimum")
    return float(
        np.sqrt(
            adsorption_K**2 + resources.heat_budget_kj_kg * adsorption_K / resources.heat_capacity_kj_kg_K
        )
    )
