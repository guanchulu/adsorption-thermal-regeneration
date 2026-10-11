# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "polars==2.0.0"]
# ///
# Run: uv run --frozen python code/normalized_heat_error.py
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

ROOT: Final = Path(__file__).resolve().parents[1]
MODEL: Final = "thermal_langmuir"
KEYS: Final = ["structure_id", "pressure_bar"]


def main() -> None:
    reference = pl.read_parquet(ROOT / "results/crafted_reference_metrics.parquet").filter(
        (pl.col("gas") == "CO2") & (pl.col("model") == MODEL)
    )
    points = (
        pl.scan_parquet(str(ROOT / "results/crafted_full/predictions_*.parquet"))
        .filter(
            (pl.col("gas") == "CO2")
            & (pl.col("force_field") == "DREIDING")
            & (pl.col("charge") == "DDEC")
            & pl.col("model").is_in([MODEL, "independent_site_dictionary"])
            & pl.col("predicted_heat_kj_mol").is_finite()
            & pl.col("minus_reported_enthalpy_kj_mol").is_finite()
        )
        .join(reference.select("structure_id").lazy(), on="structure_id", how="inner")
        .collect()
    )
    common = (
        points.group_by(KEYS)
        .agg(pl.col("model").n_unique().alias("models"))
        .filter(pl.col("models") == 2)
        .select(KEYS)
    )
    chosen = points.join(common, on=KEYS, how="inner").filter(pl.col("model") == MODEL)
    chosen = chosen.with_columns(
        (pl.col("predicted_heat_kj_mol") - pl.col("minus_reported_enthalpy_kj_mol")).abs().alias("abs_error"),
        pl.col("minus_reported_enthalpy_kj_mol").abs().alias("abs_reference"),
    )
    per_structure = (
        chosen.group_by("structure_id")
        .agg(
            pl.len().alias("n_heat_points"),
            pl.col("abs_error").mean().alias("heat_mae_kj_mol"),
            pl.col("abs_reference").mean().alias("reference_scale_kj_mol"),
        )
        .join(
            reference.select("structure_id", "common_heat_mae_kj_mol", "common_heat_points"),
            on="structure_id",
            how="inner",
        )
        .with_columns(
            pl.when(pl.col("reference_scale_kj_mol") > 0)
            .then(pl.col("heat_mae_kj_mol") / pl.col("reference_scale_kj_mol"))
            .otherwise(None)
            .alias("normalized_heat_mae")
        )
        .sort("structure_id")
    )
    mismatch = float(
        (per_structure["heat_mae_kj_mol"] - per_structure["common_heat_mae_kj_mol"]).abs().to_numpy().max()
    )
    if mismatch > 1e-10 or (per_structure["n_heat_points"] != per_structure["common_heat_points"]).any():
        raise ArithmeticError("Relative error support differs from the original absolute MAE")
    checks = []
    for group in chosen.partition_by("structure_id"):
        truth = group["minus_reported_enthalpy_kj_mol"].to_numpy()
        prediction = group["predicted_heat_kj_mol"].to_numpy()
        scale = float(np.mean(np.abs(truth)))
        if scale > 0:
            checks.append(float(np.mean(np.abs(prediction - truth))) / scale)
    valid = per_structure["normalized_heat_mae"].drop_nulls().to_numpy()
    if not np.allclose(np.sort(valid), np.sort(checks), rtol=1e-12, atol=1e-12):
        raise ArithmeticError("Independent normalized-error computation disagrees")
    matched = pl.read_parquet(ROOT / "results/crafted_matched_metrics.parquet").filter(
        (pl.col("gas") == "CO2") & (pl.col("model") == MODEL)
    )
    extra = sorted(set(matched["structure_id"].to_list()) - set(reference["structure_id"].to_list()))
    summary = {
        "cohort": "CO2 / DREIDING / DDEC / thermal_langmuir",
        "definition": "Per-structure MAE divided by mean absolute reference enthalpy on the identical finite common heat support; median across structures.",
        "structures": per_structure.height,
        "valid_relative_structures": len(valid),
        "zero_reference_scale_structures": per_structure.filter(pl.col("reference_scale_kj_mol") == 0)[
            "structure_id"
        ].to_list(),
        "median_normalized_mae_percent": 100 * float(np.median(valid)),
        "normalized_mae_percent_q25_q75": (100 * np.quantile(valid, [0.25, 0.75])).tolist(),
        "reference_scale_median_kj_mol": float(np.median(per_structure["reference_scale_kj_mol"].to_numpy())),
        "reference_scale_q25_q75": np.quantile(
            per_structure["reference_scale_kj_mol"].to_numpy(), [0.25, 0.75]
        ).tolist(),
        "original_absolute_mae_max_discrepancy": mismatch,
        "numpy_crosscheck": True,
        "extra_structure_ids_in_other_CO2_variants": extra,
    }
    per_structure.write_csv(ROOT / "results/normalized_heat_error.csv")
    (ROOT / "results/normalized_heat_error_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
