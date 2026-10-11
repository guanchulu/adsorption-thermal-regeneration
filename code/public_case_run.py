# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "polars==2.0.0", "pydantic==2.13.5"]
# ///
# Run: uv run --project .. --frozen python code/public_case_run.py
"""Export the published-model case without changing manuscript or figures."""

import json
from dataclasses import asdict

import numpy as np
import polars as pl
from public_case_model import ROOT, budget_temperature, case_heat, load_case
from run_endpoint_study import endpoint_case


def main() -> None:
    case = load_case()
    rows = [asdict(case_heat(case, float(t))) for t in np.linspace(298, 423, 251)]
    pl.DataFrame(rows).with_columns(
        pl.col("temperature_K").is_in([303.0, 323.0]).alias("underlying_source_measured_T_marker"),
        (pl.col("temperature_K") > 323).alias("above_underlying_source_max_measured_T"),
    ).write_csv(ROOT / "results/public_case_temperature_path.csv")
    budgets: list[dict[str, float | bool]] = []
    for budget in np.arange(5.0, 251.0, 2.5):
        full_t = budget_temperature(case, float(budget), complete=True)
        reduced_t = budget_temperature(case, float(budget), complete=False)
        if full_t is None or reduced_t is None:
            continue
        full = case_heat(case, full_t)
        reduced = case_heat(case, reduced_t)
        budgets.append(
            {
                "budget_kJ_kg": float(budget),
                "full_temperature_K": full_t,
                "full_release_mol_kg": full.release_mol_kg,
                "full_heat_kJ_kg": full.path_heat_kJ_kg,
                "unused_budget_kJ_kg": max(0.0, float(budget) - full.path_heat_kJ_kg),
                "reduced_temperature_K": reduced_t,
                "reduced_release_mol_kg": reduced.release_mol_kg,
                "actual_heat_at_reduced_target_kJ_kg": reduced.path_heat_kJ_kg,
                "over_budget_at_reduced_target_kJ_kg": max(0.0, reduced.path_heat_kJ_kg - float(budget)),
                "at_temperature_cap": full_t == case.temperature_cap_K,
            }
        )
    pl.DataFrame(budgets).write_csv(ROOT / "results/public_case_budget_release.csv")
    bounds = []
    for budget in (25.0, 50.0, 100.0):
        result = endpoint_case(case.setting(budget), [303.0, 323.0], resolution=0.05)
        bounds.append(
            {
                "budget_kJ_kg": budget,
                **result,
                "endpoint_upper_mol_kg": result["endpoint_upper"] * case.capacity,
                "endpoint_lower_mol_kg": result["endpoint_lower"] * case.capacity,
            }
        )
    pl.DataFrame(bounds).write_csv(ROOT / "results/public_case_resource_bounds.csv")
    summary = {
        "case_id": case.case_id,
        "capacity_mol_kg": case.capacity,
        "dry_cp_kJ_kg_K": case.dry_cp_kJ_kg_K,
        "pressure_leg": asdict(case_heat(case, case.adsorption_K)),
        "temperature_cap": asdict(case_heat(case, case.temperature_cap_K)),
        "budgets": [row for row in budgets if row["budget_kJ_kg"] in (25, 50, 100)],
        "status": case.status,
    }
    (ROOT / "results/public_case_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
