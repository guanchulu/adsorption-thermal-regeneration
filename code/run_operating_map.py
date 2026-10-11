#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "polars==2.0.0"]
# ///
# How to run: uv run --project .. --frozen python code/run_operating_map.py
"""Sample heat budgets; monotonicity, not interpolation, encloses budget cells."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from statistics import median
from time import monotonic
from typing import Final

import polars as pl
from path_design import Design, improve_witness, resource_bracket
from path_model import Setting, Site, heat_path
from run_endpoint_study import endpoint_case

ROOT: Final = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    budget_kJ_kg: float
    dry_cp_kJ_kg_K: float
    capacity_mol_kg: float
    reduced_upper_fraction: float
    endpoint_upper_fraction: float
    refined_upper_fraction: float
    feasible_fraction: float
    reduced_upper_mol_kg: float
    refined_upper_mol_kg: float
    feasible_mol_kg: float
    witness_T_K: float
    witness_E_kJ_mol: float
    witness_s_kJ_mol_K: float
    witness_heat_kJ_kg: float
    witness_dry_heat_kJ_kg: float
    witness_binding_heat_kJ_kg: float
    witness_retained_heat_kJ_kg: float
    reduced_design_T_K: float
    reduced_design_heat_kJ_kg: float
    reduced_budget_excess_pct: float
    gap_pct: float
    interval_count: int
    active_bin_width_K: float
    successful_starts: int
    total_starts: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class BudgetCell:
    B_left_kJ_kg: float
    B_right_kJ_kg: float
    feasible_floor_mol_kg: float
    upper_ceiling_mol_kg: float


def solve_point(setting: Setting) -> OperatingPoint:
    """Retain a directly integrated feasible design and a continuous-T ceiling."""
    start = monotonic()
    bracket = resource_bracket(setting)
    witness = improve_witness(setting, bracket)
    endpoint = endpoint_case(
        setting, [bracket.lower.temperature, bracket.upper.temperature, witness.design.temperature]
    )
    endpoint_site = Site(float(endpoint["endpoint_E"]), float(endpoint["endpoint_s"]), setting.cluster)
    endpoint_design = Design(
        float(endpoint["endpoint_T"]),
        endpoint_site,
        float(endpoint["endpoint_lower"]),
        float(endpoint["endpoint_path_heat"]),
    )
    best = max((bracket.lower, witness.design, endpoint_design), key=lambda design: design.release)
    state = heat_path(setting, best.site, best.temperature)
    upper = min(bracket.upper.release, float(endpoint["endpoint_upper"]))
    if state.total > setting.budget + 1e-8 or state.release > upper + 1e-7:
        raise ArithmeticError("Operating-map witness violates budget or upper bound")
    if abs(state.release - best.release) > 1e-9:
        raise ArithmeticError("Exported witness release disagrees with its explicit path")
    return OperatingPoint(
        budget_kJ_kg=setting.budget,
        dry_cp_kJ_kg_K=setting.dry_cp,
        capacity_mol_kg=setting.capacity,
        reduced_upper_fraction=bracket.upper.release,
        endpoint_upper_fraction=float(endpoint["endpoint_upper"]),
        refined_upper_fraction=upper,
        feasible_fraction=state.release,
        reduced_upper_mol_kg=setting.capacity * bracket.upper.release,
        refined_upper_mol_kg=setting.capacity * upper,
        feasible_mol_kg=setting.capacity * state.release,
        witness_T_K=best.temperature,
        witness_E_kJ_mol=best.site.energy,
        witness_s_kJ_mol_K=best.site.entropy,
        witness_heat_kJ_kg=state.total,
        witness_dry_heat_kJ_kg=state.dry,
        witness_binding_heat_kJ_kg=state.binding,
        witness_retained_heat_kJ_kg=state.loading,
        reduced_design_T_K=bracket.upper.temperature,
        reduced_design_heat_kJ_kg=bracket.upper.path_heat,
        reduced_budget_excess_pct=100 * (bracket.upper.path_heat / setting.budget - 1),
        gap_pct=100 * (upper - state.release) / upper,
        interval_count=int(endpoint["interval_count"]),
        active_bin_width_K=float(endpoint["maximum_bound_bin_width_K"]),
        successful_starts=witness.successful_starts,
        total_starts=witness.total_starts,
        elapsed_seconds=monotonic() - start,
    )


def monotone_cells(points: list[OperatingPoint]) -> list[BudgetCell]:
    """Enclose every intervening budget using smaller-budget witnesses and larger-budget bounds."""
    ordered = sorted(points, key=lambda point: point.budget_kJ_kg)
    cells: list[BudgetCell] = []
    for index, (left, right) in enumerate(pairwise(ordered)):
        floor = max(point.feasible_mol_kg for point in ordered[: index + 1])
        ceiling = min(point.refined_upper_mol_kg for point in ordered[index + 1 :])
        if floor > ceiling + 1e-7:
            raise ArithmeticError("Budget monotonicity contradicts computed release bounds")
        cells.append(BudgetCell(left.budget_kJ_kg, right.budget_kJ_kg, floor, ceiling))
    return cells


def write_methods(points: list[OperatingPoint], elapsed: float) -> None:
    baseline = [point for point in points if point.dry_cp_kJ_kg_K == 1.0]
    gaps = [point.gap_pct for point in baseline]
    excess = [point.reduced_budget_excess_pct for point in baseline]
    text = f"""OPERATING MAP: METHODS AND NUMERICAL SCOPE

Reference system: Q=4 mol/kg, C=1 kJ/(kg K), m=1, Ta=298 K,
fa=0.15 bar, fd=0.10 bar, Tmax=423 K. These are declared theoretical
resource settings, not fitted or measured material properties.

36 reference budgets: B=25,30,...,200 kJ/kg. Six additional calculations:
C=0.5 and 2 kJ/(kg K), each at B=50,100,200 kJ/kg.
Pressure-first isothermal depressurization followed by heating at fixed fd;
heat includes dry-bed sensible, binding and retained-gas sensible terms.
No pressure-work, heat-recovery, kinetics or mixture-performance claim.

At each budget, resource_bracket supplies a reduced-model upper bound and
an analytical feasible construction. improve_witness supplies a multistart
SLSQP feasible candidate. endpoint_case supplies an endpoint-cost ceiling
covering the full temperature domain and a constructive endpoint candidate.
The maximum feasible candidate and minimum valid ceiling are retained.
Every retained witness is reintegrated by heat_path before CSV export.
Witness energies and entropies are mathematical constructions, not measured
material parameters. SLSQP improves feasibility only; it is not a global proof.

The temperature enclosure uses the existing 16-bin initial partition and
adaptive bisection to an active upper-bound-bin width <=0.05 K. All bins
remain represented. Existing supporting-line and 1e-8 relative-plus-absolute
numerical guards are preserved. Standard double precision is used;
this is not formal interval arithmetic or a machine-verified certificate.

At a sampled B, any release target <= feasible_mol_kg is met or exceeded by
an explicit budget-feasible witness. Targets > refined_upper_mol_kg are
excluded within the declared model and numerical enclosure. The intervening
range is unresolved, not necessarily physically infeasible.

For each budget cell [Bi,Bi+1], operating_map_cells.csv uses the largest
feasible witness at budgets <=Bi and the smallest upper bound at budgets
>=Bi+1. Since the admissible set expands with heat budget, these constant
cell limits enclose all budgets in that cell. Linear or spline interpolation
between pointwise limits has no certification claim. Different dry heat
capacities are not interpolated or mixed in these monotonicity cells.

Reference map: median relative upper-to-feasible gap={median(gaps):.6f}%;
range={min(gaps):.6f}% to {max(gaps):.6f}%.
Reduced-optimal design heat-budget excess={min(excess):.6f}% to {max(excess):.6f}%.
Budget excess is Hpath(reduced design)/B - 1; it is not an energy saving.
Total computations={len(points)}; elapsed={elapsed:.3f} seconds.

Exact source calls: path_design.resource_bracket, path_design.improve_witness,
run_endpoint_study.endpoint_case, path_model.heat_path.
Run from latex_v5: uv run --project .. --frozen python code/run_operating_map.py
"""
    (ROOT / "results/operating_map_methods.txt").write_text(text, encoding="utf-8")


def main() -> None:
    settings = [Setting(budget=float(budget)) for budget in range(25, 201, 5)]
    settings.extend(
        Setting(dry_cp=cp, budget=float(budget)) for cp in (0.5, 2.0) for budget in (50, 100, 200)
    )
    start = monotonic()
    points: list[OperatingPoint] = []
    for index, setting in enumerate(settings, 1):
        point = solve_point(setting)
        points.append(point)
        print(
            f"Map {index}/{len(settings)}: C={setting.dry_cp:g}, B={setting.budget:g}, "
            f"release=[{point.feasible_mol_kg:.6f}, {point.refined_upper_mol_kg:.6f}] mol/kg, "
            f"{point.elapsed_seconds:.2f}s",
            flush=True,
        )
    pl.DataFrame([asdict(point) for point in points]).write_csv(ROOT / "results/operating_map_points.csv")
    cells = monotone_cells([point for point in points if point.dry_cp_kJ_kg_K == 1.0])
    pl.DataFrame([asdict(cell) for cell in cells]).write_csv(ROOT / "results/operating_map_cells.csv")
    write_methods(points, monotonic() - start)


if __name__ == "__main__":
    main()
