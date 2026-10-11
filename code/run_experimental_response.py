# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "polars==2.0.0"]
# ///
# Run: uv run --project .. --no-sync --offline python code/run_experimental_response.py
"""Paired measured-temperature responses, separate from temperature holdout validation."""

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

ROOT: Final = Path(__file__).resolve().parents[1]
PARENT: Final = ROOT.parent
KEYS: Final = ["doi", "material_hash", "isotherm_type"]


@dataclass(frozen=True, slots=True)
class Supported:
    low: pl.DataFrame
    high: pl.DataFrame
    pressure_low: float
    pressure_high: float


@dataclass(frozen=True, slots=True)
class Choice:
    low_id: str
    high_id: str
    low_temperature: float
    high_temperature: float
    points: Supported


class ResponseAuditError(ArithmeticError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"Experimental response audit: {reason}")


def supported(low: pl.DataFrame, high: pl.DataFrame) -> Supported | None:
    p_low = max(0.001, float(low["pressure_bar"][0]), float(high["pressure_bar"][0]))
    p_high = min(1.0, float(low["pressure_bar"][-1]), float(high["pressure_bar"][-1]))
    if p_high / p_low < 5:
        return None
    low = low.filter(pl.col("pressure_bar").is_between(p_low, p_high))
    high = high.filter(pl.col("pressure_bar").is_between(p_low, p_high))
    if min(len(low), len(high)) < 6:
        return None
    high = high.filter(pl.col("pressure_bar").is_between(low["pressure_bar"][0], low["pressure_bar"][-1]))
    return Supported(low, high, p_low, p_high) if len(high) >= 6 else None


def choose_pair(group: pl.DataFrame, cache: dict[str, pl.DataFrame]) -> Choice | None:
    choices: list[Choice] = []
    for low, high in itertools.combinations(group.sort("temperature_K").iter_rows(named=True), 2):
        if high["temperature_K"] - low["temperature_K"] < 20:
            continue
        pair = supported(cache[low["curve_id"]], cache[high["curve_id"]])
        if pair is not None:
            choices.append(
                Choice(low["curve_id"], high["curve_id"], low["temperature_K"], high["temperature_K"], pair)
            )
    return (
        min(
            choices,
            key=lambda c: (
                c.low_temperature - c.high_temperature,
                -len(c.points.high),
                -len(c.points.low),
                c.low_temperature,
                c.high_temperature,
                c.low_id,
                c.high_id,
            ),
        )
        if choices
        else None
    )


def main() -> None:
    curves = pl.read_parquet(PARENT / "data/multitemperature_curves.parquet")
    source_points = pl.read_parquet(PARENT / "data/multitemperature_points.parquet")
    cache = {
        str(key[0]): part.unique("pressure_bar", keep="first").sort("pressure_bar")
        for key, part in source_points.partition_by("curve_id", as_dict=True).items()
    }
    metadata = {row["curve_id"]: row for row in curves.iter_rows(named=True)}
    selected = curves.sort(
        [*KEYS, "temperature_K", "n_points", "curve_id"], descending=[False, False, False, False, True, False]
    ).unique([*KEYS, "temperature_K"], keep="first", maintain_order=True)
    pair_rows, comparison_rows, original_rows = [], [], []
    for group in selected.partition_by(KEYS):
        choice = choose_pair(group, cache)
        if choice is None:
            continue
        low, high = metadata[choice.low_id], metadata[choice.high_id]
        group_key = "|".join(str(low[key]) for key in KEYS)
        pair_id = hashlib.sha256(group_key.encode()).hexdigest()[:20]
        p_low = choice.points.low["pressure_bar"].to_numpy()
        q_low = choice.points.low["uptake_mmol_g"].to_numpy()
        p_high = choice.points.high["pressure_bar"].to_numpy()
        q_high = choice.points.high["uptake_mmol_g"].to_numpy()
        scale = float(q_low.max())
        if scale <= 0:
            raise ResponseAuditError(f"Zero lower-temperature normalization: {group_key}")
        linear = np.interp(p_high, p_low, q_low)
        logarithmic = np.interp(np.log(p_high), np.log(p_low), q_low)
        response_linear, response_log = (linear - q_high) / scale, (logarithmic - q_high) / scale
        linear_flags, log_flags = response_linear < -0.05, response_log < -0.05
        insertion = np.searchsorted(p_low, p_high)
        exact = np.isclose(p_low[insertion], p_high, rtol=0, atol=1e-12)
        right = np.clip(insertion, 1, len(p_low) - 1)
        gaps = np.where(exact, 0.0, p_low[right] - p_low[right - 1])
        sensitivity = np.abs(linear - logarithmic) / scale
        pair_rows.append(
            {
                "pair_id": pair_id,
                "group_key": group_key,
                "doi": low["doi"],
                "material_hash": low["material_hash"],
                "material_name": low["material_name"],
                "reported_convention": low["isotherm_type"],
                "T1_K": choice.low_temperature,
                "T2_K": choice.high_temperature,
                "temperature_span_K": choice.high_temperature - choice.low_temperature,
                "low_curve_id": choice.low_id,
                "high_curve_id": choice.high_id,
                "low_archive_path": low["archive_path"],
                "high_archive_path": high["archive_path"],
                "low_raw_sha256": low["raw_sha256"],
                "high_raw_sha256": high["raw_sha256"],
                "low_original_uptake_unit": low["adsorption_units"],
                "high_original_uptake_unit": high["adsorption_units"],
                "low_stp_assumed": low["stp_22414_assumption"],
                "high_stp_assumed": high["stp_22414_assumption"],
                "common_pressure_low_bar": choice.points.pressure_low,
                "common_pressure_high_bar": choice.points.pressure_high,
                "test_pressure_low_bar": float(p_high.min()),
                "test_pressure_high_bar": float(p_high.max()),
                "n_low_original_points": len(p_low),
                "n_high_original_points": len(p_high),
                "low_temperature_peak_mmol_g": scale,
                "mean_response_linear": float(response_linear.mean()),
                "mean_response_log": float(response_log.mean()),
                "min_response_linear": float(response_linear.min()),
                "min_response_log": float(response_log.min()),
                "inversion_linear_5pct": bool(linear_flags.any()),
                "inversion_log_5pct": bool(log_flags.any()),
                "inversion_both_same_point_5pct": bool((linear_flags & log_flags).any()),
                "max_interpolation_sensitivity": float(sensitivity.max()),
                "mean_interpolation_sensitivity": float(sensitivity.mean()),
                "max_linear_bracket_bar": float(gaps.max()),
            }
        )
        comparison_rows.extend(
            {
                "pair_id": pair_id,
                "high_curve_id": choice.high_id,
                "high_original_point_index": int(point_index),
                "pressure_bar": float(p),
                "high_reported_uptake": float(q),
                "low_interp_linear": float(ql),
                "low_interp_log": float(qlog),
                "low_temperature_peak_mmol_g": scale,
                "normalized_response_linear": float(rl),
                "normalized_response_log": float(rlog),
                "inversion_linear_5pct": bool(il),
                "inversion_log_5pct": bool(ilog),
                "linear_bracket_bar": float(gap),
            }
            for point_index, p, q, ql, qlog, rl, rlog, il, ilog, gap in zip(
                choice.points.high["point_index"],
                p_high,
                q_high,
                linear,
                logarithmic,
                response_linear,
                response_log,
                linear_flags,
                log_flags,
                gaps,
                strict=True,
            )
        )
        for role, frame in (("low", choice.points.low), ("high", choice.points.high)):
            original_rows.extend(
                {"pair_id": pair_id, "role": role, **row} for row in frame.iter_rows(named=True)
            )
    pairs = pl.DataFrame(pair_rows).sort("doi", "material_hash", "reported_convention")
    comparisons = pl.DataFrame(comparison_rows).sort("pair_id", "pressure_bar")
    original = pl.DataFrame(original_rows).sort("pair_id", "role", "pressure_bar")
    actual = (len(pairs), pairs["doi"].n_unique(), pairs["material_hash"].n_unique())
    if actual != (122, 72, 116):
        raise ResponseAuditError(f"Expected cohort (122,72,116), got {actual}; no data written")
    for directory in ("data", "results", "qa"):
        (ROOT / directory).mkdir(exist_ok=True)
    pairs.write_parquet(ROOT / "data/experimental_pairs.parquet")
    original.write_parquet(ROOT / "data/experimental_pair_points.parquet")
    pairs.write_csv(ROOT / "results/experimental_response_pairs.csv")
    comparisons.write_parquet(ROOT / "results/experimental_response_points.parquet")
    summary = {
        "groups": len(pairs),
        "doi_labels": actual[1],
        "material_hashes": actual[2],
        "distinct_selected_curves": len(set(pairs["low_curve_id"]) | set(pairs["high_curve_id"])),
        "original_point_records": len(original),
        "high_temperature_comparison_points": len(comparisons),
        "highest_temperature_atleast353K": int((pairs["T2_K"] >= 353).sum()),
        "unknown_convention_groups": int((pairs["reported_convention"] == "").sum()),
        "pair_mean_response_linear_median": float(np.median(pairs["mean_response_linear"].to_numpy())),
        "pair_mean_response_log_median": float(np.median(pairs["mean_response_log"].to_numpy())),
        "inversion_pairs_linear": int(pairs["inversion_linear_5pct"].sum()),
        "inversion_pairs_log": int(pairs["inversion_log_5pct"].sum()),
        "inversion_pairs_both_same_point": int(pairs["inversion_both_same_point_5pct"].sum()),
        "pair_max_interpolation_sensitivity_median": float(
            np.median(pairs["max_interpolation_sensitivity"].to_numpy())
        ),
        "pair_max_interpolation_sensitivity_max": float(
            np.max(pairs["max_interpolation_sensitivity"].to_numpy())
        ),
    }
    (ROOT / "results/experimental_response_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    scope = {
        "selection": "One maximum-span pressure-supported pair per DOI/material/convention group; span>=20K; most source points per exactT, curve-ID tie; pair ties favor high-point count, low-point count, lowerT, upperT, IDs. No uptake-based pair selection.",
        "pressure": "Common original support intersected with0.001-1bar; span ratio>=5; at least6 retained originals per curve; highT nodes additionally inside retained lowT pressure support; no extrapolation.",
        "response": "(lowT interpolated uptake minus highT original uptake)/maximum lowT original uptake in the pair common pressure window; positive means lower uptake on heating.",
        "inversion": "HighT exceeds lowT interpolation by >5% lowT peak; a flag only. Source uncertainty and thermodynamic uptake conventions are not known enough to call it falsification.",
        "scope": "Measured-temperature comparison. Both temperatures are observed; neither is held out from this descriptive calculation. This is separate from the frozen32-group highest-temperature predictor validation and from heat-demand validation.",
        "interpolation": "Compare linear pressure and log pressure, both using raw retained lowT points and highT measured pressure locations.",
        "inputs_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                PARENT / "data/multitemperature_curves.parquet",
                PARENT / "data/multitemperature_points.parquet",
            )
        },
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": summary,
    }
    (ROOT / "qa/experimental_response_scope.txt").write_text(json.dumps(scope, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
