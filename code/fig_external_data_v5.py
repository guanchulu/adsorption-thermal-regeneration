# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "polars==2.0.0", "scipy==1.18.1"]
# ///
# Run via: uv run --project .. --no-sync --offline python code/fig_external_v5.py
"""Data and provenance for the external-validation figure."""

import hashlib
import json
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from figure_style import ROOT
from gas_fugacity import Gas, gas_fugacity
from temperature_fits import Observations, fit_model, predict

REFERENCE: Final = (
    (pl.col("gas") == "CO2") & (pl.col("force_field") == "DREIDING") & (pl.col("charge") == "DDEC")
)
MODEL: Final = "thermal_langmuir"


class ReproductionError(RuntimeError):
    def __init__(self, structure: str, discrepancy: float) -> None:
        super().__init__(f"Stored prediction mismatch for {structure}: {discrepancy}")


@dataclass(frozen=True, slots=True)
class Case:
    structure: str
    quantile: float
    rank: int
    uptake_error: float
    heat_error: float
    energy: float
    points: pl.DataFrame
    curves: pl.DataFrame
    heat: pl.DataFrame


def prepare() -> tuple[pl.DataFrame, list[Case]]:
    """Select fixed empirical ranks, refit training uptake, and export provenance."""
    cohort = (
        pl.read_parquet(ROOT / "results/crafted_reference_metrics.parquet")
        .filter(REFERENCE & (pl.col("model") == MODEL))
        .sort(["uptake_nrmse", "structure_id"])
    )
    raw = pl.scan_parquet(ROOT / "data/crafted_points.parquet")
    stored = pl.scan_parquet(ROOT / "results/crafted_full/predictions_*.parquet")
    cases: list[Case] = []
    checks: list[float] = []
    for quantile in (0.5, 0.9):
        rank = int(np.ceil(quantile * cohort.height))
        row = cohort.row(rank - 1, named=True)
        structure = str(row["structure_id"])
        points = (
            raw.filter(
                REFERENCE
                & (pl.col("structure_id") == structure)
                & (pl.col("pressure_bar") > 0)
                & (pl.col("pressure_bar") <= 1)
            )
            .sort(["temperature_K", "pressure_bar"])
            .collect()
        )
        pressure = np.asarray(points["pressure_bar"], dtype=float)
        temperature = np.asarray(points["temperature_K"], dtype=float)
        uptake = np.asarray(points["uptake_mol_kg"], dtype=float)
        fugacity = gas_fugacity(pressure, temperature, Gas.CO2).fugacity_bar
        cold, warm = temperature == 273, temperature == 298
        train_mask = cold | warm
        scale = float(uptake[train_mask].max())
        weights = np.sqrt(
            np.where(temperature[train_mask] == 273, 1 / (2 * cold.sum()), 1 / (2 * warm.sum()))
        )
        train = Observations(
            fugacity[train_mask], temperature[train_mask], uptake[train_mask], weights, scale, 298.0
        )
        fitted = fit_model(train, thermal=True)
        held = (
            stored.filter(REFERENCE & (pl.col("structure_id") == structure) & (pl.col("model") == MODEL))
            .sort("pressure_bar")
            .collect()
        )
        held_observations = Observations(
            np.asarray(held["fugacity_bar"], dtype=float),
            np.full(held.height, 323.0),
            np.asarray(held["observed_uptake_mol_kg"], dtype=float),
            np.ones(held.height),
            scale,
            298.0,
        )
        reproduced = predict(fitted.parameters, held_observations)
        difference = float(np.max(np.abs(reproduced - held["predicted_uptake_mol_kg"].to_numpy())))
        if difference > 1e-10:
            raise ReproductionError(structure, difference)
        checks.append(difference)
        held_pressure = np.asarray(held["pressure_bar"], dtype=float)
        support = (float(held_pressure.min()), float(held_pressure.max()))
        grids: list[pl.DataFrame] = []
        for temp in (273, 298, 323):
            lower, upper = support if temp == 323 else (float(pressure.min()), float(pressure.max()))
            grid = np.geomspace(lower, upper, 180)
            grid_temp = np.full(grid.size, float(temp))
            observation = Observations(
                gas_fugacity(grid, grid_temp, Gas.CO2).fugacity_bar,
                grid_temp,
                np.zeros(grid.size),
                np.ones(grid.size),
                scale,
                298.0,
            )
            grids.append(
                pl.DataFrame(
                    {
                        "structure_id": structure,
                        "temperature_K": temp,
                        "pressure_bar": grid,
                        "predicted_uptake_mol_kg": predict(fitted.parameters, observation),
                    }
                )
            )
        points = points.with_columns(
            pl.Series("fugacity_bar", fugacity),
            (
                (pl.col("temperature_K") == 323)
                & pl.col("pressure_bar").is_in(held["pressure_bar"].implode())
            ).alias("held_out_test"),
        )
        cases.append(
            Case(
                structure,
                quantile,
                rank,
                float(row["uptake_nrmse"]),
                float(row["common_heat_mae_kj_mol"]),
                float(fitted.parameters[2]),
                points,
                pl.concat(grids),
                held,
            )
        )
    pl.concat([case.points for case in cases]).write_csv(ROOT / "results/figure5_v5_raw_source_rows.csv")
    pl.concat([case.curves for case in cases]).write_csv(ROOT / "results/figure5_v5_fitted_curves.csv")
    pl.concat([case.heat for case in cases]).write_csv(ROOT / "results/figure5_v5_heldout_predictions.csv")
    cohort.select(
        "structure_id", "gas", "force_field", "charge", "uptake_nrmse", "common_heat_mae_kj_mol"
    ).write_csv(ROOT / "results/figure5_v5_cohort.csv")
    source_files = [
        ROOT / "data/crafted_points.parquet",
        ROOT / "results/crafted_reference_metrics.parquet",
        *sorted((ROOT / "results/crafted_full").glob("predictions_*.parquet")),
    ]
    manifest = {
        "source": "CRAFTED2.0.1, DOI 10.5281/zenodo.10120180; GCMC observations",
        "reference": "CO2 / DREIDING / DDEC / thermal_langmuir",
        "selection_rule": "Sort ascending by uptake_nrmse then structure_id; one-based rank ceil(p*n), p=0.5 and 0.9. No curve inspection enters selection.",
        "cohort_structures": cohort.height,
        "median_uptake_nrmse": float(np.median(cohort["uptake_nrmse"].to_numpy())),
        "median_heat_mae_kj_mol": float(np.median(cohort["common_heat_mae_kj_mol"].to_numpy())),
        "selected": [
            {
                "structure_id": c.structure,
                "quantile": c.quantile,
                "rank": c.rank,
                "uptake_nrmse": c.uptake_error,
                "heat_mae_kj_mol": c.heat_error,
                "predicted_heat_kj_mol": c.energy,
                "raw_rows": c.points.height,
                "heldout_points": c.heat.height,
            }
            for c in cases
        ],
        "max_refit_stored_discrepancy_mol_kg": max(checks),
        "interpretation": "Temperature extrapolation 298 to 323 K within training fugacity support; all enthalpy withheld. Not cycle heat validation. Supplied variability is retained without assigning SD/SEM/CI semantics.",
        "inputs": [
            {"file": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in source_files
        ],
    }
    (ROOT / "results/figure5_v5_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return cohort, cases
