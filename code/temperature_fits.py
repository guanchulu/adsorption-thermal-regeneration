# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1"]
# ///
# Run through run_temperature_validation.py in the locked environment.
"""Matched constant-temperature and van't Hoff Langmuir fitting."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, least_squares
from scipy.special import expit, logit
from thermal_core import R

Vector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Observations:
    pressure: Vector
    temperature: Vector
    uptake: Vector
    weights: Vector
    scale: float
    reference_K: float


@dataclass(frozen=True, slots=True)
class Fit:
    parameters: Vector
    train_error: float
    converged: bool
    near_count: int
    near_energy_min: float
    near_energy_max: float
    near_capacity_min: float
    near_capacity_max: float
    jacobian_condition: float


def predict(parameters: Vector, observations: Observations) -> Vector:
    energy = float(parameters[2]) if len(parameters) == 3 else 0.0
    z = (
        np.log(observations.pressure)
        + parameters[1]
        + energy / R * (1 / observations.temperature - 1 / observations.reference_K)
    )
    return np.asarray(observations.scale * np.exp(parameters[0]) * expit(z), dtype=float)


def fit_model(observations: Observations, *, thermal: bool) -> Fit:
    trials: list[OptimizeResult] = []

    def residual(parameters: Vector) -> Vector:
        return (
            (predict(parameters, observations) - observations.uptake)
            / observations.scale
            * observations.weights
        )

    for capacity in (1.2, 3.0, 10.0):
        fraction = np.clip(np.median(observations.uptake) / observations.scale / capacity, 1e-5, 1 - 1e-5)
        logb = float(logit(fraction) - np.log(np.median(observations.pressure)))
        for energy in (15.0, 35.0, 70.0) if thermal else (0.0,):
            initial = (
                np.array([np.log(capacity), np.clip(logb, -19.0, 19.0), energy])
                if thermal
                else np.array([np.log(capacity), np.clip(logb, -19.0, 19.0)])
            )
            lower = np.array([0.0, -20.0, 0.0]) if thermal else np.array([0.0, -20.0])
            upper = np.array([np.log(30.0), 20.0, 120.0]) if thermal else np.array([np.log(30.0), 20.0])
            result = least_squares(
                residual, initial, bounds=(lower, upper), max_nfev=1500, ftol=1e-10, xtol=1e-10, gtol=1e-10
            )
            trials.append(result)
    trials.sort(key=lambda r: float(np.sum(np.asarray(r.fun, dtype=float) ** 2)))
    best = trials[0]
    error = float(np.linalg.norm(best.fun))
    near = [r for r in trials if np.linalg.norm(r.fun) <= max(1.05 * error, error + 0.005)]
    energies = [float(r.x[2]) if thermal else 0.0 for r in near]
    capacities = [float(observations.scale * np.exp(r.x[0])) for r in near]
    singular = np.linalg.svd(np.asarray(best.jac, dtype=float), compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-14 else float("inf")
    return Fit(
        np.asarray(best.x, dtype=float),
        error,
        bool(best.success),
        len(near),
        min(energies),
        max(energies),
        min(capacities),
        max(capacities),
        condition,
    )
