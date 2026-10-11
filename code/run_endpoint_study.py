import heapq
import json
import time
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from endpoint_bounds import constructive_site, temperature_interval_upper
from path_model import Setting, Site, heat_path

ROOT: Final = Path(__file__).resolve().parents[1]


def endpoint_case(
    setting: Setting, references: list[float], resolution: float = 0.05
) -> dict[str, float | int]:
    partition = np.linspace(setting.adsorption_K, setting.temperature_cap, 17)
    intervals = [
        (-temperature_interval_upper(setting, float(low), float(high)), float(low), float(high))
        for low, high in pairwise(partition)
    ]
    heapq.heapify(intervals)
    refinements = 0
    temperatures = set(references)
    while intervals[0][2] - intervals[0][1] > resolution:
        _, low, high = heapq.heappop(intervals)
        mid = (low + high) / 2
        for left, right in ((low, mid), (mid, high)):
            heapq.heappush(intervals, (-temperature_interval_upper(setting, left, right), left, right))
        temperatures.add(mid)
        refinements += 1
    temperatures.update(float(t) for t in partition[1:])
    best, best_t = 0.0, 0.0
    best_site = Site(0.0, 0.0, setting.cluster)
    for td in sorted(temperatures):
        if setting.dry_cp * (td - setting.adsorption_K) >= setting.budget:
            continue
        d, site = constructive_site(setting, td)
        state = heat_path(setting, site, td)
        if state.total > setting.budget + 1e-7:
            raise ArithmeticError("Endpoint construction violates directly integrated heat")
        if d > best:
            best, best_t, best_site = d, td, site
    upper = -intervals[0][0]
    state = heat_path(setting, best_site, best_t)
    if best > upper + 1e-8:
        raise ArithmeticError("Constructive lower release exceeds endpoint upper")
    return {
        "endpoint_upper": upper,
        "endpoint_lower": best,
        "endpoint_T": best_t,
        "endpoint_E": best_site.energy,
        "endpoint_s": best_site.entropy,
        "endpoint_path_heat": state.total,
        "interval_count": len(intervals),
        "refinements": refinements,
        "maximum_bound_bin_width_K": intervals[0][2] - intervals[0][1],
        "active_bin_low": intervals[0][1],
        "active_bin_high": intervals[0][2],
        "temperature_resolution_K": resolution,
    }


def main() -> None:
    baseline = pl.read_csv(ROOT / "results/path_brackets.csv")
    rows = []
    start = time.monotonic()
    for index, row in enumerate(baseline.iter_rows(named=True), 1):
        setting = Setting(
            adsorption_K=float(row["adsorption_K"]),
            fa_bar=float(row["fa_bar"]),
            fd_bar=float(row["fd_bar"]),
            capacity=float(row["capacity"]),
            dry_cp=float(row["dry_cp"]),
            budget=float(row["budget"]),
            temperature_cap=float(row["temperature_cap"]),
            cluster=int(row["cluster"]),
        )
        result = endpoint_case(
            setting, [float(row["lower_T"]), float(row["upper_T"]), float(row["witness_T"])]
        )
        lower = max(float(row["witness_release"]), float(result["endpoint_lower"]))
        upper = min(float(row["upper_release"]), float(result["endpoint_upper"]))
        if lower > upper + 1e-7:
            raise ArithmeticError("Existing path witness exceeds refined upper bound")
        rows.append(
            {
                **asdict(setting),
                **result,
                "old_analytic_lower": row["lower_release"],
                "old_upper": row["upper_release"],
                "old_witness": row["witness_release"],
                "new_lower": lower,
                "new_upper": upper,
                "old_relative_gap": (row["upper_release"] - row["witness_release"]) / row["upper_release"],
                "new_relative_gap": (upper - lower) / upper,
                "upper_improvement": row["upper_release"] - upper,
            }
        )
        print(
            f"Endpoint case {index}/54: gap {(upper - lower) / upper:.5%}, intervals {result['interval_count']}",
            flush=True,
        )
    table = pl.DataFrame(rows)
    table.write_csv(ROOT / "results/endpoint_brackets.csv")
    summary = {}
    for m in (0, 1, 4):
        part = table if m == 0 else table.filter(pl.col("cluster") == m)
        summary[str(m)] = {
            "n": part.height,
            **{
                f"{field}_{stat}": float(getattr(part[field], stat)())
                for field in ("old_relative_gap", "new_relative_gap", "upper_improvement")
                for stat in ("min", "median", "max")
            },
        }
    summary["elapsed_seconds"] = time.monotonic() - start
    summary["scope"] = (
        "Analytical endpoint/temperature-bin inequalities; convex tangent lower estimates with 1e-8 relative-plus-absolute guard. Standard floating point, not formal interval arithmetic."
    )
    (ROOT / "results/endpoint_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
