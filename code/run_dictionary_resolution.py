# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "polars"]
# ///
# Run: uv run --project .. --no-sync --offline python code/run_dictionary_resolution.py
"""Post-hoc nested-grid diagnostic on frozen pilot IDs, never model selection."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final

import numpy as np
import polars as pl
from crafted_benchmark import Task
from gas_fugacity import Gas, gas_fugacity
from numpy.typing import NDArray
from physical_dictionary import (
    DictionaryFit,
    SiteDictionary,
    default_dictionary,
    fit_dictionary,
    predict_qst,
    predict_uptake,
)

type Vector = NDArray[np.float64]
type Row = dict[str, str | float | int | bool]
ROOT: Final = Path(__file__).resolve().parents[1]
POINTS: Final = ROOT / "data/crafted_points.parquet"
PILOT: Final = ROOT / "results/crafted_pilot_metrics.parquet"
PRIMARY: Final = ROOT / "results/crafted_matched_metrics.parquet"


@dataclass(frozen=True, slots=True)
class Prepared:
    task: Task
    train_f: Vector
    train_t: Vector
    train_q: Vector
    weights: Vector
    test_f: Vector
    test_t: Vector
    test_q: Vector
    test_h: Vector
    scale: float


def prepare(task: Task) -> Prepared | None:
    """Mirror crafted_benchmark.evaluate masks and temperature-balanced weighting."""
    valid = np.isfinite(task.uptake) & (task.uptake >= 0) & np.isfinite(task.pressure) & (task.pressure > 0) & (task.pressure <= 1)
    fugacity = gas_fugacity(task.pressure, task.temperature, Gas(task.gas)).fugacity_bar
    cold, warm = valid & (task.temperature == 273), valid & (task.temperature == 298)
    if min(int(cold.sum()), int(warm.sum())) < 6:
        return None
    low = max(float(fugacity[cold].min()), float(fugacity[warm].min()))
    high = min(float(fugacity[cold].max()), float(fugacity[warm].max()))
    test = valid & (task.temperature == 323) & (fugacity >= low) & (fugacity <= high)
    train = cold | warm
    scale = float(task.uptake[train].max())
    if int(test.sum()) < 6 or scale <= 1e-12:
        return None
    weights = np.where(task.temperature[train] == 273, 1 / (2 * cold.sum()), 1 / (2 * warm.sum()))
    return Prepared(task, fugacity[train], task.temperature[train], task.uptake[train], weights,
                    fugacity[test], task.temperature[test], task.uptake[test], -task.enthalpy[test], scale)


def fit_grid(case: Prepared, grid: SiteDictionary) -> tuple[DictionaryFit, Vector, Vector, float]:
    started = perf_counter()
    fitted = fit_dictionary(case.train_f, case.train_t, case.train_q, sample_weights=case.weights, dictionary=grid)
    return fitted, predict_uptake(fitted, case.test_f, case.test_t), predict_qst(fitted, case.test_f, case.test_t), perf_counter() - started


def compare(case: Prepared, grids: tuple[SiteDictionary, SiteDictionary]) -> Row:
    coarse, fine = (fit_grid(case, grid) for grid in grids)
    common = np.isfinite(case.test_h) & np.isfinite(coarse[2]) & np.isfinite(fine[2])
    if fine[0].normalized_rmse > coarse[0].normalized_rmse + 1e-8:
        raise ArithmeticError("Nested-grid NNLS training residual increased beyond numerical tolerance")
    row: Row = {"structure_id": case.task.structure, "gas": case.task.gas,
                "force_field": case.task.force_field, "charge": case.task.charge,
                "n_train": case.train_q.size, "n_test": case.test_q.size,
                "common_heat_points": int(common.sum()), "scale": case.scale,
                "posthoc_diagnostic": True, "capacity_nonidentifiable": True}
    for name, result in zip(("coarse", "fine"), (coarse, fine), strict=True):
        fitted, q, heat, elapsed = result
        row.update({f"{name}_train_nrmse": fitted.normalized_rmse,
                    f"{name}_test_nrmse": float(np.sqrt(np.mean((q - case.test_q)**2)) / case.scale),
                    f"{name}_heat_mae": float(np.mean(np.abs(heat[common] - case.test_h[common]))) if common.any() else float("nan"),
                    f"{name}_capacity": fitted.total_capacity_mmol_g,
                    f"{name}_energy_boundary_fraction": fitted.energy_boundary_fraction,
                    f"{name}_affinity_boundary_fraction": fitted.affinity_boundary_fraction,
                    f"{name}_energy_boundary_capacity": fitted.energy_boundary_fraction * fitted.total_capacity_mmol_g,
                    f"{name}_affinity_boundary_capacity": fitted.affinity_boundary_fraction * fitted.total_capacity_mmol_g,
                    f"{name}_active_atoms": fitted.active_atoms, f"{name}_seconds": elapsed})
    return row


def load_tasks() -> tuple[list[Task], pl.DataFrame]:
    pilot = pl.read_parquet(PILOT)
    ids = sorted(pilot["structure_id"].unique().to_list())
    frozen_ids = (pl.scan_parquet(POINTS).filter(pl.col("pressure_bar") <= 1)
                  .select("structure_id").unique().collect()["structure_id"].to_list())
    selected = sorted(frozen_ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest())[:100]
    if ids != sorted(selected):
        raise ArithmeticError("Saved pilot IDs do not match the original hash-selected 100")
    points = (pl.scan_parquet(POINTS).filter(pl.col("structure_id").is_in(ids) & (pl.col("pressure_bar") <= 1)
              & (pl.col("gas") == "CO2") & (pl.col("force_field") == "DREIDING") & (pl.col("charge") == "DDEC"))
              .sort("structure_id", "temperature_K", "pressure_bar").collect())
    tasks = []
    for key, part in points.partition_by(["structure_id", "gas", "force_field", "charge"], as_dict=True).items():
        arrays = [np.asarray(part[col].to_numpy(), dtype=float) for col in
                  ("pressure_bar", "temperature_K", "uptake_mol_kg", "enthalpy_kj_mol", "uptake_reported_error", "enthalpy_reported_error")]
        tasks.append(Task(str(key[0]), str(key[1]), str(key[2]), str(key[3]),
                          arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5]))
    return tasks, pilot


def verify_primary(table: pl.DataFrame, pilot: pl.DataFrame) -> None:
    frozen = pilot.filter((pl.col("status") == "ok") & (pl.col("model") == "independent_site_dictionary"))
    joined = table.join(frozen, on="structure_id", suffix="_frozen")
    if len(joined) != len(table) or len(joined) != len(frozen):
        raise ArithmeticError("Eligible pilot membership changed")
    for current, original in (("coarse_train_nrmse", "train_nrmse"), ("coarse_test_nrmse", "uptake_nrmse"),
                              ("scale", "train_peak_mol_kg"), ("n_train", "n_train_frozen"), ("n_test", "n_test_frozen")):
        if not np.allclose(joined[current].to_numpy(), joined[original].to_numpy(), rtol=1e-10, atol=1e-10):
            raise ArithmeticError(f"Frozen pilot parity failed for {current}")


def main() -> None:
    primary_hash = hashlib.sha256(PRIMARY.read_bytes()).hexdigest()
    started = perf_counter()
    energy, affinity = np.meshgrid(np.arange(0.0, 100.1, 2.5), np.arange(-10.0, 14.1, 0.5), indexing="ij")
    grids = (default_dictionary(), SiteDictionary(energy.ravel(), affinity.ravel()))
    if not set(zip(grids[0].energy_kj_mol, grids[0].logb_ref, strict=True)) <= set(zip(grids[1].energy_kj_mol, grids[1].logb_ref, strict=True)):
        raise ArithmeticError("Dictionary nodes are not nested")
    tasks, pilot = load_tasks()
    rows, excluded = [], []
    for task in tasks:
        case = prepare(task)
        if case is None:
            excluded.append({"structure_id": task.structure, "status": "ineligible_under_frozen_masks"})
        else:
            rows.append(compare(case, grids))
    metrics = ("train_nrmse", "test_nrmse", "heat_mae", "capacity", "energy_boundary_fraction",
               "affinity_boundary_fraction", "energy_boundary_capacity", "affinity_boundary_capacity", "active_atoms")
    table = pl.DataFrame(rows).with_columns([(pl.col(f"fine_{name}") - pl.col(f"coarse_{name}")).alias(f"change_{name}") for name in metrics])
    verify_primary(table, pilot)
    if hashlib.sha256(PRIMARY.read_bytes()).hexdigest() != primary_hash:
        raise ArithmeticError("Primary benchmark changed during the diagnostic")
    table.write_parquet(ROOT / "results/dictionary_resolution_cases.parquet")
    table.write_csv(ROOT / "results/dictionary_resolution_cases.csv")
    pl.DataFrame(excluded).write_csv(ROOT / "results/dictionary_resolution_excluded.csv")
    summary = table.select(pl.len().alias("n"), *[
        pl.col(f"{grid}_{metric}").drop_nans().median().alias(f"median_{grid}_{metric}")
        for grid in ("coarse", "fine", "change") for metric in metrics], *[
        expression for metric in ("train_nrmse", "test_nrmse", "heat_mae") for expression in
        ((pl.col(f"change_{metric}") < -1e-8).sum().alias(f"improved_{metric}"),
         (pl.col(f"change_{metric}") > 1e-8).sum().alias(f"worsened_{metric}"),
         (pl.col(f"change_{metric}").abs() <= 1e-8).sum().alias(f"tied_{metric}"))])
    summary.write_csv(ROOT / "results/dictionary_resolution_summary.csv")
    metadata = {"scope": "posthoc_fixed_pilot_grid_diagnostic_not_model_selection", "attempted": len(tasks),
                "eligible": len(table), "excluded": excluded, "dictionary_atoms": [525, 2009],
                "primary_sha256_unchanged": primary_hash, "pilot_sha256": hashlib.sha256(PILOT.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "points_sha256": hashlib.sha256(POINTS.read_bytes()).hexdigest(),
                "dependency_sha256": {name: hashlib.sha256((ROOT / "code" / name).read_bytes()).hexdigest()
                                      for name in ("crafted_benchmark.py", "physical_dictionary.py", "gas_fugacity.py")},
                "elapsed_seconds": perf_counter() - started, "summary": summary.to_dicts()}
    (ROOT / "results/dictionary_resolution_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
