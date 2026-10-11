# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "polars", "scipy"]
# ///
# Run: uv run --project .. --no-sync --offline python code/run_crafted_benchmark.py
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from crafted_benchmark import Task, evaluate

ROOT: Final = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    name = "pilot" if args.pilot else "full"
    out = ROOT/f"results/crafted_{name}"
    out.mkdir(exist_ok=True)
    table = pl.scan_parquet(ROOT/"data/crafted_points.parquet").filter(pl.col("pressure_bar")<=1).collect()
    if args.pilot:
        ids = sorted(table["structure_id"].unique().to_list(),key=lambda value:hashlib.sha256(value.encode()).hexdigest())[:100]
        table = table.filter(pl.col("structure_id").is_in(ids)&(pl.col("gas")=="CO2")&(pl.col("force_field")=="DREIDING")&(pl.col("charge")=="DDEC"))
    tasks: list[Task] = []
    for key,part in table.partition_by(["structure_id","gas","force_field","charge"],as_dict=True).items():
        sorted_part = part.sort("temperature_K","pressure_bar")
        arrays = [np.asarray(sorted_part[col].to_numpy(),dtype=float) for col in
                  ("pressure_bar","temperature_K","uptake_mol_kg","enthalpy_kj_mol","uptake_reported_error","enthalpy_reported_error")]
        tasks.append(Task(str(key[0]), str(key[1]), str(key[2]), str(key[3]),
                          arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5]))
    metrics, predictions = [], []
    started = time.monotonic()
    batch = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (metric_rows, prediction_rows) in enumerate(pool.map(evaluate,tasks,chunksize=20),1):
            metrics.extend(metric_rows)
            predictions.extend(prediction_rows)
            if index % 500 == 0 or index == len(tasks):
                pl.DataFrame(metrics,infer_schema_length=None).write_parquet(out/f"metrics_{batch:03}.parquet")
                pl.DataFrame(predictions,infer_schema_length=None).write_parquet(out/f"predictions_{batch:03}.parquet")
                metrics.clear()
                predictions.clear()
                batch += 1
                print(name,index,"/",len(tasks),"elapsed",round(time.monotonic()-started,1),flush=True)
    results = pl.concat([pl.read_parquet(p) for p in sorted(out.glob("metrics_*.parquet"))],how="diagonal_relaxed")
    results.write_parquet(ROOT/f"results/crafted_{name}_metrics.parquet")
    summary = results.filter(pl.col("status")=="ok").group_by("gas","force_field","charge","model").agg(
        pl.len().alias("groups"),pl.col("structure_id").n_unique().alias("structure_ids"),pl.col("uptake_nrmse").median().alias("median_uptake_nrmse"),
        pl.col("uptake_nrmse").quantile(.9).alias("p90_uptake_nrmse"),pl.col("heat_mae_kj_mol").drop_nans().median().alias("median_heat_mae_kj_mol"))
    summary.write_csv(ROOT/f"results/crafted_{name}_summary.csv")
    (ROOT/f"qa/crafted_{name}_run.json").write_text(json.dumps({"groups_attempted":len(tasks),"metric_rows":results.height,
       "status_counts":results.group_by("status").len().to_dicts(),"elapsed_seconds":time.monotonic()-started,
       "protocol":"qa/benchmark_protocol.txt","workers":args.workers},indent=2),encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
