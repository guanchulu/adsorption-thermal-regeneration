from dataclasses import dataclass

import numpy as np
from gas_fugacity import Gas, gas_fugacity
from numpy.typing import NDArray
from physical_dictionary import PhysicalDictionaryError, fit_dictionary, predict_qst, predict_uptake
from temperature_fits import Observations, fit_model, predict

type Vector = NDArray[np.float64]
type Row = dict[str, str | float | int | bool]


@dataclass(frozen=True, slots=True)
class Task:
    structure: str
    gas: str
    force_field: str
    charge: str
    pressure: Vector
    temperature: Vector
    uptake: Vector
    enthalpy: Vector
    uptake_error: Vector
    enthalpy_error: Vector


def evaluate(task: Task) -> tuple[list[Row], list[Row]]:
    meta: Row = {
        "structure_id": task.structure,
        "gas": task.gas,
        "force_field": task.force_field,
        "charge": task.charge,
    }
    valid = (
        np.isfinite(task.uptake)
        & (task.uptake >= 0)
        & np.isfinite(task.pressure)
        & (task.pressure > 0)
        & (task.pressure <= 1)
    )
    fugacity = gas_fugacity(task.pressure, task.temperature, Gas(task.gas)).fugacity_bar
    cold, warm = valid & (task.temperature == 273), valid & (task.temperature == 298)
    if min(int(cold.sum()), int(warm.sum())) < 6:
        return [{**meta, "model": "all", "status": "insufficient_training_support"}], []
    low = max(float(fugacity[cold].min()), float(fugacity[warm].min()))
    high = min(float(fugacity[cold].max()), float(fugacity[warm].max()))
    test_mask = valid & (task.temperature == 323) & (fugacity >= low) & (fugacity <= high)
    train_mask = cold | warm
    if int(test_mask.sum()) < 6:
        return [{**meta, "model": "all", "status": "insufficient_test_support"}], []
    scale = float(task.uptake[train_mask].max())
    if scale <= 1e-12:
        return [{**meta, "model": "all", "status": "zero_training_signal"}], []
    train_weights = np.where(task.temperature[train_mask] == 273, 1 / (2 * cold.sum()), 1 / (2 * warm.sum()))
    train = Observations(
        fugacity[train_mask],
        task.temperature[train_mask],
        task.uptake[train_mask],
        np.sqrt(train_weights),
        scale,
        298.0,
    )
    test = Observations(
        fugacity[test_mask],
        task.temperature[test_mask],
        task.uptake[test_mask],
        np.ones(test_mask.sum()),
        scale,
        298.0,
    )
    truth_h = -task.enthalpy[test_mask]
    metrics: list[Row] = []
    predictions: list[Row] = []

    def record(name: str, predicted_q: Vector, predicted_h: Vector, extras: Row) -> None:
        heat_mask = np.isfinite(predicted_h) & np.isfinite(truth_h)
        errors = predicted_q - test.uptake
        heat_mae = (
            float(np.mean(np.abs(predicted_h[heat_mask] - truth_h[heat_mask])))
            if heat_mask.any()
            else float("nan")
        )
        metrics.append(
            {
                **meta,
                "model": name,
                "status": "ok",
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "train_peak_mol_kg": scale,
                "uptake_nrmse": float(np.sqrt(np.mean(errors**2)) / scale),
                "uptake_nmae": float(np.mean(np.abs(errors)) / scale),
                "heat_mae_kj_mol": heat_mae,
                "heat_reference_points": int(np.isfinite(truth_h).sum()),
                "heat_predicted_points": int(heat_mask.sum()),
                "test_mean_reported_uptake_variability": float(np.nanmean(task.uptake_error[test_mask])),
                **extras,
            }
        )
        indices = np.flatnonzero(test_mask)
        for i, index in enumerate(indices):
            predictions.append(
                {
                    **meta,
                    "model": name,
                    "temperature_K": 323.0,
                    "pressure_bar": float(task.pressure[index]),
                    "fugacity_bar": float(fugacity[index]),
                    "observed_uptake_mol_kg": float(test.uptake[i]),
                    "predicted_uptake_mol_kg": float(predicted_q[i]),
                    "minus_reported_enthalpy_kj_mol": float(truth_h[i]),
                    "predicted_heat_kj_mol": float(predicted_h[i]),
                    "uptake_reported_error": float(task.uptake_error[index]),
                    "enthalpy_reported_error": float(task.enthalpy_error[index]),
                }
            )

    single = fit_model(train, thermal=True)
    single_q = predict(single.parameters, test)
    record(
        "thermal_langmuir",
        single_q,
        np.full(test.pressure.shape, float(single.parameters[2])),
        {
            "converged": single.converged,
            "train_nrmse": single.train_error,
            "effective_energy_kj_mol": float(single.parameters[2]),
            "capacity_mol_kg": float(scale * np.exp(single.parameters[0])),
            "jacobian_condition": single.jacobian_condition,
        },
    )
    try:
        fitted = fit_dictionary(train.pressure, train.temperature, train.uptake, sample_weights=train_weights)
        record(
            "independent_site_dictionary",
            predict_uptake(fitted, test.pressure, test.temperature),
            predict_qst(fitted, test.pressure, test.temperature),
            {
                "converged": True,
                "train_nrmse": fitted.normalized_rmse,
                "active_atoms": fitted.active_atoms,
                "capacity_mol_kg": fitted.total_capacity_mmol_g,
                "capacity_nonidentifiable": True,
                "energy_boundary_fraction": fitted.energy_boundary_fraction,
                "affinity_boundary_fraction": fitted.affinity_boundary_fraction,
            },
        )
    except PhysicalDictionaryError as error:
        metrics.append({**meta, "model": "independent_site_dictionary", "status": str(error)})
    samples = np.array([np.interp(test.pressure, fugacity[mask], task.uptake[mask]) for mask in (cold, warm)])
    matrix = np.column_stack((np.ones(2), 1 / np.array([273.0, 298.0]) - 1 / 298.0))
    beta = np.linalg.lstsq(matrix, np.log(np.maximum(samples, scale * 1e-6)), rcond=None)[0]
    baseline = np.exp(np.clip(np.array([1.0, 1 / 323.0 - 1 / 298.0]) @ beta, -700, 700))
    record(
        "fugacitywise_inverse_temperature",
        np.asarray(baseline, dtype=float),
        np.full(test.pressure.shape, np.nan),
        {"converged": True},
    )
    return metrics, predictions
