import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

ROOT: Final = Path(__file__).resolve().parents[1]
KEYS: Final = ["structure_id", "gas", "force_field", "charge"]


def main() -> None:
    metrics = pl.read_parquet(ROOT / "results/crafted_full_metrics.parquet")
    good = metrics.filter((pl.col("status") == "ok") & pl.col("converged").fill_null(value=False))
    cohort = (
        good.group_by(KEYS)
        .agg(pl.col("model").n_unique().alias("models"))
        .filter(pl.col("models") == 3)
        .select(KEYS)
    )
    matched = good.join(cohort, on=KEYS, how="inner")
    predictions = pl.scan_parquet(str(ROOT / "results/crafted_full/predictions_*.parquet")).join(
        cohort.lazy(), on=KEYS, how="inner"
    )
    heat_predictions = predictions.filter(
        pl.col("model").is_in(["thermal_langmuir", "independent_site_dictionary"])
    )
    finite = heat_predictions.filter(
        pl.col("predicted_heat_kj_mol").is_finite() & pl.col("minus_reported_enthalpy_kj_mol").is_finite()
    )
    common_points = (
        finite.group_by([*KEYS, "pressure_bar"])
        .agg(pl.col("model").n_unique().alias("models"))
        .filter(pl.col("models") == 2)
        .select([*KEYS, "pressure_bar"])
    )
    common = finite.join(common_points, on=[*KEYS, "pressure_bar"], how="inner")
    heat = (
        common.group_by([*KEYS, "model"])
        .agg(
            (pl.col("predicted_heat_kj_mol") - pl.col("minus_reported_enthalpy_kj_mol"))
            .abs()
            .mean()
            .alias("common_heat_mae_kj_mol"),
            pl.len().alias("common_heat_points"),
        )
        .collect()
    )
    matched = matched.join(heat, on=[*KEYS, "model"], how="left")
    matched.write_parquet(ROOT / "results/crafted_matched_metrics.parquet")
    summary = matched.group_by("gas", "force_field", "charge", "model").agg(
        pl.len().alias("groups"),
        pl.col("structure_id").n_unique().alias("structures"),
        pl.col("uptake_nrmse").median().alias("median_uptake_nrmse"),
        pl.col("uptake_nrmse").quantile(0.9).alias("p90_uptake_nrmse"),
        pl.col("common_heat_mae_kj_mol").median().alias("median_common_heat_mae"),
        pl.col("common_heat_points").sum().alias("common_heat_points"),
        pl.col("common_heat_points").is_not_null().sum().alias("groups_with_common_heat"),
    )
    summary.write_csv(ROOT / "results/crafted_matched_summary.csv")
    reference = matched.filter((pl.col("force_field") == "DREIDING") & (pl.col("charge") == "DDEC"))
    reference.write_parquet(ROOT / "results/crafted_reference_metrics.parquet")
    reference_summary = summary.filter(
        (pl.col("force_field") == "DREIDING") & (pl.col("charge") == "DDEC")
    ).sort("gas", "model")
    reference_summary.write_csv(ROOT / "results/crafted_reference_summary.csv")
    errors_by_structure = matched.group_by("structure_id", "gas", "model").agg(
        pl.len().alias("variants"),
        pl.col("uptake_nrmse").median().alias("median_variant_uptake_error"),
        pl.col("uptake_nrmse").min().alias("minimum_variant_error"),
        pl.col("uptake_nrmse").max().alias("maximum_variant_error"),
        pl.col("common_heat_mae_kj_mol").median().alias("median_variant_heat_error"),
    )
    errors_by_structure.write_parquet(ROOT / "results/crafted_structure_metrics.parquet")
    structure_summary = errors_by_structure.group_by("gas", "model").agg(
        pl.len().alias("structures"),
        pl.col("median_variant_uptake_error").median().alias("median_structure_uptake_error"),
        pl.col("median_variant_heat_error").median().alias("median_structure_heat_error"),
    )
    structure_summary.write_csv(ROOT / "results/crafted_structure_summary.csv")
    observed = pl.scan_parquet(ROOT / "data/crafted_points.parquet").filter(
        (pl.col("temperature_K") == 323) & (pl.col("pressure_bar") == 0.1)
    )
    sensitivity = (
        observed.group_by("structure_id", "gas")
        .agg(
            pl.col("uptake_mol_kg").min().alias("min_q"),
            pl.col("uptake_mol_kg").max().alias("max_q"),
            pl.col("uptake_mol_kg").median().alias("median_q"),
            pl.len().alias("variants"),
            pl.col("reported_minus_enthalpy_kj_mol")
            .filter(pl.col("reported_minus_enthalpy_kj_mol").is_finite())
            .min()
            .alias("min_heat"),
            pl.col("reported_minus_enthalpy_kj_mol")
            .filter(pl.col("reported_minus_enthalpy_kj_mol").is_finite())
            .max()
            .alias("max_heat"),
        )
        .with_columns(
            pl.when(pl.col("median_q") > 1e-12)
            .then((pl.col("max_q") - pl.col("min_q")) / pl.col("median_q"))
            .otherwise(None)
            .alias("relative_uptake_span")
        )
        .collect()
    )
    sensitivity.write_parquet(ROOT / "results/crafted_modeling_sensitivity.parquet")
    all_predictions = pl.scan_parquet(str(ROOT / "results/crafted_full/predictions_*.parquet"))
    prediction_count = all_predictions.select(pl.len()).collect().item()
    nonconverged = metrics.filter(
        (pl.col("status") == "ok") & ~pl.col("converged").fill_null(value=False)
    ).height
    counts = {
        "attempted_groups": 32568,
        "matched_success_groups": cohort.height,
        "matched_structure_ids": cohort["structure_id"].n_unique(),
        "nonconverged_model_rows": nonconverged,
        "successful_model_rows": good.height,
        "total_prediction_rows": prediction_count,
        "shared_heat_rows_two_models": common.select(pl.len()).collect().item(),
        "reference_summary": reference_summary.to_dicts(),
        "structure_summary": structure_summary.to_dicts(),
        "error_definition": "Uptake RMSE / training maximum; heat MAE is matched finite-point minus-reported-enthalpy error, kJ/mol.",
        "independence": "Variants and gases are grouped within source structure identifiers; no population confidence intervals.",
        "numpy_version": np.__version__,
    }
    (ROOT / "results/crafted_final_summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
