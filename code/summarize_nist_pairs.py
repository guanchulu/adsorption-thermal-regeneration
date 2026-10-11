# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "polars==2.0.0"]
# ///
# Run from the package root: uv run --frozen python code/summarize_nist_pairs.py


from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

ROOT: Final = Path(__file__).resolve().parents[1]


def main() -> None:
    data = pl.read_csv(ROOT / "results/temperature_holdout.csv")
    thermal = data.filter(pl.col("model") == "thermal_langmuir").select(
        "group_key",
        "doi",
        "heldout_temperature_K",
        "training_temperatures_K",
        pl.col("test_normalized_rmse").alias("thermal_nrmse"),
    )
    baseline = data.filter(pl.col("model") == "pressurewise_logq_vanthoff").select(
        "group_key",
        pl.col("test_normalized_rmse").alias("baseline_nrmse"),
    )
    pairs = (
        thermal.join(baseline, on="group_key", validate="1:1")
        .with_columns(
            (pl.col("thermal_nrmse") - pl.col("baseline_nrmse")).alias("paired_difference"),
            (
                pl.col("heldout_temperature_K")
                - pl.col("training_temperatures_K")
                .str.split("|")
                .list.eval(pl.element().cast(pl.Float64))
                .list.max()
            ).alias("extrapolation_gap_K"),
        )
        .sort("group_key")
    )
    pairs.write_csv(ROOT / "results/nist_paired_differences.csv")
    rows = []
    for label, part in (
        ("all", pairs),
        ("heldout_below_353K", pairs.filter(pl.col("heldout_temperature_K") < 353)),
        ("heldout_at_least_353K", pairs.filter(pl.col("heldout_temperature_K") >= 353)),
        ("extrapolation_gap_at_most_20K", pairs.filter(pl.col("extrapolation_gap_K") <= 20)),
        ("extrapolation_gap_above_20K", pairs.filter(pl.col("extrapolation_gap_K") > 20)),
    ):
        delta = part["paired_difference"].to_numpy()
        q1, median, q3 = np.quantile(delta, [0.25, 0.5, 0.75], method="linear")
        rows.append(
            {
                "stratum": label,
                "n_groups": part.height,
                "n_dois": part["doi"].n_unique(),
                "median_difference": float(median),
                "q1_difference": float(q1),
                "q3_difference": float(q3),
                "iqr_width": float(q3 - q1),
                "thermal_better": int(np.sum(delta < -1e-12)),
                "baseline_better": int(np.sum(delta > 1e-12)),
                "ties": int(np.sum(np.abs(delta) <= 1e-12)),
            }
        )
    summary = pl.DataFrame(rows)
    summary.write_csv(ROOT / "results/nist_paired_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
