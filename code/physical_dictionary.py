# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
# Run through the parent benchmark in its locked environment.
"""Uptake-only nonnegative site fits; capacities/energies are not uniquely recovered."""
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls
from scipy.special import expit

type Vector = NDArray[np.float64]
R: Final = 0.008314462618
REFERENCE_K: Final = 298.0
RESPONSE_FLOOR: Final = 1e-12


class PhysicalDictionaryError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class SiteDictionary:
    energy_kj_mol: Vector
    logb_ref: Vector

    def __post_init__(self) -> None:
        if (self.energy_kj_mol.ndim != 1 or self.energy_kj_mol.size == 0
                or self.energy_kj_mol.shape != self.logb_ref.shape
                or not np.isfinite(self.energy_kj_mol).all()
                or not np.isfinite(self.logb_ref).all()
                or np.any(self.energy_kj_mol < 0)):
            raise PhysicalDictionaryError("Dictionary requires matching finite vectors and nonnegative energies")


@dataclass(frozen=True, slots=True)
class DictionaryFit:
    dictionary: SiteDictionary
    capacity_weights_mmol_g: Vector
    train_peak_mmol_g: float
    normalized_rmse: float
    weighted_training_error_mmol_g: float
    active_atoms: int
    total_capacity_mmol_g: float
    energy_boundary_fraction: float
    affinity_boundary_fraction: float
    max_iterations: int
    capacity_nonidentifiable: bool = True


def default_dictionary() -> SiteDictionary:
    energy, affinity = np.meshgrid(np.arange(0.0, 101.0, 5.0), np.arange(-10.0, 15.0), indexing="ij")
    return SiteDictionary(energy.ravel(), affinity.ravel())


def _coordinates(pressure_bar: Vector, temperature_k: Vector, dictionary: SiteDictionary) -> Vector:
    if (pressure_bar.ndim != 1 or pressure_bar.size == 0
            or pressure_bar.shape != temperature_k.shape
            or not np.isfinite(pressure_bar).all() or not np.isfinite(temperature_k).all()
            or np.any(pressure_bar <= 0) or np.any(temperature_k <= 0)):
        raise PhysicalDictionaryError("Pressure and temperature must be matching finite positive vectors")
    return np.asarray(
        np.log(pressure_bar[:, None]) + dictionary.logb_ref
        + dictionary.energy_kj_mol / R * (1 / temperature_k[:, None] - 1 / REFERENCE_K),
        dtype=np.float64,
    )


def fit_dictionary(
    pressure_bar: Vector,
    temperature_k: Vector,
    uptake_mmol_g: Vector,
    *,
    sample_weights: Vector | None = None,
    dictionary: SiteDictionary | None = None,
) -> DictionaryFit:
    """Minimize weighted squared uptake residuals divided by the training peak.

    Statistical sample weights sum to one; default is uniform per observation.
    Capacity sums and boundary masses diagnose the chosen representation, not
    independently measured saturation capacities or unique site-energy recovery.
    """
    grid = default_dictionary() if dictionary is None else dictionary
    theta = expit(_coordinates(pressure_bar, temperature_k, grid))
    if (uptake_mmol_g.shape != pressure_bar.shape or not np.isfinite(uptake_mmol_g).all()
            or np.any(uptake_mmol_g < 0)):
        raise PhysicalDictionaryError("Training uptake must be finite nonnegative and match the coordinates")
    weights = np.ones(pressure_bar.size) if sample_weights is None else np.asarray(sample_weights, dtype=float)
    if (weights.shape != pressure_bar.shape or not np.isfinite(weights).all()
            or np.any(weights < 0) or float(weights.sum()) <= 0):
        raise PhysicalDictionaryError("Sample weights must be matching finite nonnegative values with positive sum")
    used = weights > 0
    scale = float(uptake_mmol_g[used].max())
    if scale <= 0:
        raise PhysicalDictionaryError("A positive training uptake peak is required")
    factors = np.sqrt(weights[used] / weights[used].sum())
    matrix = theta[used] * factors[:, None]
    response = uptake_mmol_g[used] / scale * factors
    max_iterations = 3 * grid.energy_kj_mol.size
    try:
        normalized_capacity, error = nnls(matrix, response, maxiter=max_iterations)
    except RuntimeError as error:
        raise PhysicalDictionaryError(f"NNLS failed within {max_iterations} iterations: {error}") from error
    capacity = np.asarray(normalized_capacity * scale, dtype=np.float64)
    total = float(capacity.sum())
    energy_edge = (grid.energy_kj_mol == grid.energy_kj_mol.min()) | (grid.energy_kj_mol == grid.energy_kj_mol.max())
    affinity_edge = (grid.logb_ref == grid.logb_ref.min()) | (grid.logb_ref == grid.logb_ref.max())
    return DictionaryFit(
        grid, capacity, scale, float(error), float(error * scale),
        int(np.count_nonzero(capacity > 1e-10 * max(total, scale))), total,
        float(capacity[energy_edge].sum() / total) if total > 0 else 0.0,
        float(capacity[affinity_edge].sum() / total) if total > 0 else 0.0,
        max_iterations,
    )


def predict_uptake(fit: DictionaryFit, pressure_bar: Vector, temperature_k: Vector) -> Vector:
    theta = expit(_coordinates(pressure_bar, temperature_k, fit.dictionary))
    return np.asarray(theta @ fit.capacity_weights_mmol_g, dtype=np.float64)


def predict_qst(fit: DictionaryFit, pressure_bar: Vector, temperature_k: Vector) -> Vector:
    """Return positive isosteric heat in kJ/mol, NaN when response/capacity <=1e-12."""
    coordinate = _coordinates(pressure_bar, temperature_k, fit.dictionary)
    response = expit(coordinate) * expit(-coordinate)
    weighted = response * fit.capacity_weights_mmol_g
    denominator = weighted.sum(axis=1)
    numerator = weighted @ fit.dictionary.energy_kj_mol
    output = np.full(pressure_bar.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=output,
              where=denominator > RESPONSE_FLOOR * fit.total_capacity_mmol_g)
    return output
