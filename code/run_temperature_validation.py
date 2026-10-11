# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "polars==2.0.0", "scipy==1.18.1", "pydantic==2.13.5"]
# ///
# Run: uv run --frozen python code/run_temperature_validation.py
"""Hold out the highest measured temperature; never invent regeneration labels."""

import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict
from temperature_fits import Observations, fit_model, predict
from thermal_core import R

ROOT: Final = Path(__file__).resolve().parents[1]


class Group(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    thermal_group_key: str
    doi: str
    material_hash: str
    material_name: str
    reported_convention: str
    selected_curve_ids: list[str]
    group_eligible: bool


class Audit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    groups: list[Group]


def main() -> None:
    audit = Audit.model_validate_json((ROOT / "qa/data_audit.json").read_text(encoding="utf-8-sig"))
    meta = pl.read_parquet(ROOT / "data/multitemperature_curves.parquet")
    points = pl.read_parquet(ROOT / "data/multitemperature_points.parquet")
    cache = {
        str(key[0]): part.unique("pressure_bar", keep="first").sort("pressure_bar")
        for key, part in points.partition_by("curve_id", as_dict=True).items()
    }
    results, predictions, rejected = [], [], []
    for group in audit.groups:
        if not group.group_eligible:
            continue
        curves = meta.filter(pl.col("curve_id").is_in(group.selected_curve_ids)).sort("temperature_K")
        temperatures = curves["temperature_K"].to_numpy()
        low = max(0.001, float(np.max(curves["pmin_bar"].to_numpy())))
        high = min(1.0, float(np.min(curves["pmax_bar"].to_numpy())))
        if high / low < 5:
            rejected.append(
                {"group": group.thermal_group_key, "reason": "common_pressure_span_below_factor5"}
            )
            continue
        pressure, uptake, thermal = [], [], []
        for row in curves.iter_rows(named=True):
            part = cache[row["curve_id"]].filter(pl.col("pressure_bar").is_between(low, high))
            pressure.append(part["pressure_bar"].to_numpy())
            uptake.append(part["uptake_mmol_g"].to_numpy())
            thermal.append(np.full(len(part), row["temperature_K"]))
        if min(len(p) for p in pressure) < 6:
            rejected.append(
                {"group": group.thermal_group_key, "reason": "fewer_than6_original_points_at_any_temperature"}
            )
            continue
        test_supported = (pressure[-1] >= max(p.min() for p in pressure[:-1])) & (
            pressure[-1] <= min(p.max() for p in pressure[:-1])
        )
        pressure[-1], uptake[-1], thermal[-1] = (
            pressure[-1][test_supported],
            uptake[-1][test_supported],
            thermal[-1][test_supported],
        )
        if len(pressure[-1]) < 6:
            rejected.append(
                {
                    "group": group.thermal_group_key,
                    "reason": "fewer_than6_test_points_inside_retained_training_pressure_support",
                }
            )
            continue
        train_p, train_q, train_t = (
            np.concatenate(pressure[:-1]),
            np.concatenate(uptake[:-1]),
            np.concatenate(thermal[:-1]),
        )
        scale = float(train_q.max())
        if scale <= 0:
            rejected.append({"group": group.thermal_group_key, "reason": "zero_training_uptake"})
            continue
        reference = float(1 / np.mean(1 / temperatures[:-1]))
        weights = np.concatenate(
            [np.full(len(p), 1 / np.sqrt(len(p) * (len(pressure) - 1))) for p in pressure[:-1]]
        )
        train = Observations(train_p, train_t, train_q, weights, scale, reference)
        test = Observations(
            pressure[-1], thermal[-1], uptake[-1], np.ones(len(pressure[-1])), scale, reference
        )
        for thermal_model in (False, True):
            fit = fit_model(train, thermal=thermal_model)
            estimated = predict(fit.parameters, test)
            rmse = float(np.sqrt(np.mean((estimated - test.uptake) ** 2)))
            energy = float(fit.parameters[2]) if thermal_model else 0.0
            name = "thermal_langmuir" if thermal_model else "constant_temperature_langmuir"
            results.append(
                {
                    "group_key": group.thermal_group_key,
                    "doi": group.doi,
                    "material_hash": group.material_hash,
                    "material": group.material_name,
                    "convention": group.reported_convention,
                    "model": name,
                    "training_temperatures_K": "|".join(str(t) for t in temperatures[:-1]),
                    "heldout_temperature_K": float(temperatures[-1]),
                    "reference_temperature_K": reference,
                    "n_train_points": len(train_p),
                    "n_test_points": len(test.pressure),
                    "common_low_bar": low,
                    "common_high_bar": high,
                    "train_scale_mmol_g": scale,
                    "train_normalized_rmse": fit.train_error,
                    "test_rmse_mmol_g": rmse,
                    "test_normalized_rmse": rmse / scale,
                    "capacity_mmol_g": float(scale * np.exp(fit.parameters[0])),
                    "logb_reference": float(fit.parameters[1]),
                    "effective_energy_kj_mol": energy,
                    "effective_entropy_loss_kj_mol_K": energy / reference - R * float(fit.parameters[1]),
                    "near_fit_energy_span": fit.near_energy_max - fit.near_energy_min,
                    "near_fit_capacity_span": fit.near_capacity_max - fit.near_capacity_min,
                    "jacobian_condition": fit.jacobian_condition,
                    "converged": fit.converged,
                    "capacity_near_upper_bound": bool(fit.parameters[0] > np.log(30) * 0.999),
                    "energy_near_upper_bound": bool(energy > 119.9),
                }
            )
            predictions.extend(
                {
                    "group_key": group.thermal_group_key,
                    "model": name,
                    "temperature_K": float(temperatures[-1]),
                    "pressure_bar": float(p),
                    "reported_uptake": float(q),
                    "predicted_uptake": float(y),
                }
                for p, q, y in zip(test.pressure, test.uptake, estimated, strict=True)
            )
        sampled = np.array(
            [np.interp(test.pressure, p, q) for p, q in zip(pressure[:-1], uptake[:-1], strict=True)]
        )
        logq = np.log(np.maximum(sampled, scale * 1e-6))
        design = np.column_stack((np.ones(len(temperatures) - 1), 1 / temperatures[:-1] - 1 / reference))
        beta = np.linalg.lstsq(design, logq, rcond=None)[0]
        estimated = np.exp(np.clip(np.array([1.0, 1 / temperatures[-1] - 1 / reference]) @ beta, -700, 700))
        rmse = float(np.sqrt(np.mean((estimated - test.uptake) ** 2)))
        rowspec = results[-1].copy()
        rowspec.update(
            {
                "model": "pressurewise_logq_vanthoff",
                "train_normalized_rmse": float("nan"),
                "test_rmse_mmol_g": rmse,
                "test_normalized_rmse": rmse / scale,
                "capacity_mmol_g": float("nan"),
                "logb_reference": float("nan"),
                "effective_energy_kj_mol": float("nan"),
                "effective_entropy_loss_kj_mol_K": float("nan"),
                "near_fit_energy_span": float("nan"),
                "near_fit_capacity_span": float("nan"),
                "jacobian_condition": float("nan"),
                "converged": True,
                "capacity_near_upper_bound": False,
                "energy_near_upper_bound": False,
            }
        )
        results.append(rowspec)
        predictions.extend(
            {
                "group_key": group.thermal_group_key,
                "model": "pressurewise_logq_vanthoff",
                "temperature_K": float(temperatures[-1]),
                "pressure_bar": float(p),
                "reported_uptake": float(q),
                "predicted_uptake": float(y),
            }
            for p, q, y in zip(test.pressure, test.uptake, estimated, strict=True)
        )
        print("Temperature holdout", len(results) // 3, group.material_name, flush=True)
    table = pl.DataFrame(results).sort("group_key", "model")
    table.write_csv(ROOT / "results/temperature_holdout.csv")
    pl.DataFrame(predictions).sort("group_key", "model", "pressure_bar").write_parquet(
        ROOT / "results/temperature_predictions.parquet"
    )
    (ROOT / "qa/temperature_fit_rejections.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    summary = table.group_by("model").agg(
        pl.len().alias("groups"),
        pl.col("doi").n_unique().alias("dois"),
        pl.col("test_normalized_rmse").median().alias("median_test_nrmse"),
        (pl.col("heldout_temperature_K") >= 353).sum().alias("heldout_atleast353K"),
        pl.col("capacity_near_upper_bound").sum().alias("capacity_boundary_fits"),
        pl.col("converged").sum().alias("converged"),
    )
    summary.write_csv(ROOT / "results/temperature_summary.csv")
    print(summary.write_csv())


if __name__ == "__main__":
    main()
