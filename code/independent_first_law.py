#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "pydantic==2.13.5"]
# ///
# How to run: uv run --project .. --frozen python code/independent_first_law.py
"""Reconstruct first-law heat without using the original heat decomposition."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, astuple, dataclass
from math import log, tanh
from pathlib import Path
from typing import Final

import numpy as np
from path_model import Setting, Site, heat_path
from pydantic import BaseModel, ConfigDict, Field

ROOT: Final = Path(__file__).resolve().parents[1]
GAS_CONSTANT: Final = 0.008314462618


class FirstLawInputError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(f"Invalid first-law path: {detail}")


@dataclass(frozen=True, slots=True)
class FirstLawCase:
    adsorption_K: float = 298.0
    desorption_K: float = 353.0
    fa_bar: float = 0.15
    fd_bar: float = 0.1
    capacity: float = 4.0
    dry_cp: float = 1.0
    energy: float = 30.0
    entropy: float = 0.075
    cluster: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(astuple(self)).all():
            raise FirstLawInputError("finite case parameters")
        if not 298 <= self.adsorption_K < self.desorption_K <= 1200:
            raise FirstLawInputError("298 <= Ta < Td <= 1200 K")
        if not 0 < self.fd_bar <= self.fa_bar:
            raise FirstLawInputError("0 < fd <= fa")
        if self.capacity <= 0 or min(self.dry_cp, self.energy) < 0 or self.cluster < 1:
            raise FirstLawInputError("positive capacity/cluster and nonnegative heat capacity/energy")


class WitnessRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", allow_inf_nan=False)
    adsorption_K: float
    fa_bar: float
    fd_bar: float
    capacity: float
    dry_cp: float
    budget: float = Field(gt=0)
    cluster: int = Field(ge=1)
    witness_T: float
    witness_E: float
    witness_s: float
    witness_path_heat: float

    def case(self) -> FirstLawCase:
        return FirstLawCase(
            adsorption_K=self.adsorption_K,
            desorption_K=self.witness_T,
            fa_bar=self.fa_bar,
            fd_bar=self.fd_bar,
            capacity=self.capacity,
            dry_cp=self.dry_cp,
            energy=self.witness_E,
            entropy=self.witness_s,
            cluster=self.cluster,
        )


@dataclass(frozen=True, slots=True)
class HeatBalance:
    pressure_heat: float
    heating_heat: float
    internal_energy_change: float
    outgoing_enthalpy: float
    released_mol_kg: float

    @property
    def total(self) -> float:
        return self.pressure_heat + self.heating_heat


def first_law(case: FirstLawCase, steps: int = 1000, reference: float = 0.0) -> HeatBalance:
    """Sum stored-energy changes and outgoing enthalpy along the specified path."""
    if steps < 1 or not np.isfinite(reference):
        raise FirstLawInputError("positive interval count and finite enthalpy reference")
    temperatures = np.linspace(case.adsorption_K, case.desorption_K, steps + 1)
    argument = case.cluster * (
        log(case.fd_bar) + case.energy / (GAS_CONSTANT * temperatures) - case.entropy / GAS_CONSTANT
    )
    inventory = case.capacity * 0.5 * (1 + np.tanh(argument / 2))
    initial_argument = case.cluster * (
        log(case.fa_bar) + case.energy / (GAS_CONSTANT * case.adsorption_K) - case.entropy / GAS_CONSTANT
    )
    initial_inventory = case.capacity * 0.5 * (1 + tanh(initial_argument / 2))
    t = temperatures / 1000
    primitive = 24.99735 * t + 55.18696 * t**2 / 2 - 33.69137 * t**3 / 3 + 7.948387 * t**4 / 4 + 0.136638 / t
    gas_enthalpy = primitive - primitive[0] + reference
    stored_energy = case.dry_cp * (temperatures - case.adsorption_K) + inventory * (
        gas_enthalpy - case.energy
    )
    initial_energy = initial_inventory * (reference - case.energy)
    pressure_outflow = reference * (initial_inventory - float(inventory[0]))
    heating_outflow = float(np.dot((gas_enthalpy[1:] + gas_enthalpy[:-1]) / 2, -np.diff(inventory)))
    pressure_heat = float(stored_energy[0]) - initial_energy + pressure_outflow
    heating_heat = float(stored_energy[-1] - stored_energy[0]) + heating_outflow
    return HeatBalance(
        pressure_heat,
        heating_heat,
        float(stored_energy[-1]) - initial_energy,
        pressure_outflow + heating_outflow,
        initial_inventory - float(inventory[-1]),
    )


def load_witnesses(source: Path) -> list[WitnessRecord]:
    with source.open(encoding="utf-8", newline="") as stream:
        return [WitnessRecord.model_validate(row) for row in csv.DictReader(stream)]


def run_check(source: Path, destination: Path) -> None:
    witnesses = load_witnesses(source)
    observations: list[dict[str, float | int]] = []
    reference_differences: list[float] = []
    stored_differences: list[float] = []
    for case_id, witness in enumerate(witnesses, start=1):
        case = witness.case()
        setting = Setting(
            adsorption_K=case.adsorption_K,
            fa_bar=case.fa_bar,
            fd_bar=case.fd_bar,
            capacity=case.capacity,
            dry_cp=case.dry_cp,
            budget=witness.budget,
            temperature_cap=case.desorption_K,
            cluster=case.cluster,
        )
        original = heat_path(setting, Site(case.energy, case.entropy, case.cluster), case.desorption_K)
        stored_differences.append(abs(original.total - witness.witness_path_heat))
        for steps in (250, 1000, 4000):
            unshifted = first_law(case, steps)
            for reference in (-1000.0, 0.0, 1000.0):
                independent = first_law(case, steps, reference)
                reference_differences.append(abs(independent.total - unshifted.total))
                observations.append(
                    {
                        "case_id": case_id,
                        **asdict(case),
                        "budget_kj_kg": witness.budget,
                        "steps": steps,
                        "enthalpy_reference_kj_mol": reference,
                        "original_heat_kj_kg": original.total,
                        "independent_heat_kj_kg": independent.total,
                        "absolute_error_kj_kg": abs(independent.total - original.total),
                        "relative_error": abs(independent.total / original.total - 1),
                        "reference_shift_difference_kj_kg": abs(independent.total - unshifted.total),
                        **asdict(independent),
                    }
                )
    destination.mkdir(exist_ok=True)
    with (destination / "first_law_cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    by_mesh = []
    for steps in (250, 1000, 4000):
        selected = [
            row for row in observations if row["steps"] == steps and row["enthalpy_reference_kj_mol"] == 0
        ]
        by_mesh.append(
            {
                "steps": steps,
                "max_absolute_error_kj_kg": max(row["absolute_error_kj_kg"] for row in selected),
                "median_absolute_error_kj_kg": float(
                    np.median([row["absolute_error_kj_kg"] for row in selected])
                ),
                "max_relative_error": max(row["relative_error"] for row in selected),
            }
        )
    summary = {
        "scope": "Numerical first-law closure within the declared model; not experimental validation",
        "witness_cases": len(witnesses),
        "evaluations": len(observations),
        "enthalpy_reference_shifts_kj_mol": [-1000, 0, 1000],
        "max_reference_shift_difference_kj_kg": max(reference_differences),
        "max_original_vs_stored_heat_difference_kj_kg": max(stored_differences),
        "mesh_convergence": by_mesh,
        "source": str(source.relative_to(ROOT)),
        "independence": "tanh occupations, analytical gas enthalpy, stored-energy endpoints and outgoing enthalpy; original heat_path used only as comparator",
    }
    (destination / "first_law_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    methods = (
        "Independent first-law numerical cross-check\n\n"
        "For each of the 54 retained homogeneous path witnesses, we independently reconstructed "
        "q(T,f) using the hyperbolic-tangent occupation identity and evaluated gas enthalpy from "
        "the analytical Shomate primitive. The bed internal energy was U=C(T-Ta)+q[h_g(T)-E]. "
        "Heat was calculated as Delta U + integral h_g dn_out, with dn_out=-dq. The isothermal "
        "pressure leg used exact endpoint inventory and a constant gas enthalpy. Heating used "
        "250, 1000 and 4000 equal temperature intervals with trapezoidal enthalpy weighting of "
        "the released gas in each interval. This implementation did not call the original "
        "occupation, heat-capacity or binding-plus-sensible-heat integrator. The original "
        "96-point Gauss-Legendre implementation was used only for comparison.\n\n"
        f"At 4000 intervals, the maximum absolute heat difference was {by_mesh[-1]['max_absolute_error_kj_kg']:.8g} "
        f"kJ kg^-1 and the maximum relative difference was {by_mesh[-1]['max_relative_error']:.8g}. "
        f"Shifting the common enthalpy zero by +/-1000 kJ mol^-1 changed total heat by at most "
        f"{max(reference_differences):.8g} kJ kg^-1 across all meshes and cases.\n\n"
        "This verifies numerical closure, sign conventions, molecular cluster normalization and "
        "enthalpy-reference invariance for the tested model paths. It does not independently "
        "validate the constant-E caloric model, the Shomate correlation, material parameters, "
        "transport, kinetics, mixtures, pressure work, or measured process duty.\n"
    )
    (destination / "first_law_methods.txt").write_text(methods, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_check(ROOT / "results/path_brackets.csv", ROOT / "results")
