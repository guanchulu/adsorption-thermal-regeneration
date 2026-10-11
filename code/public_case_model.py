# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "pydantic==2.13.5"]
# ///
# Run: uv run --project .. --frozen python code/public_case_run.py
"""Literature-parameter dual-Langmuir case under the manuscript heat path."""

from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from typing import Final

import numpy as np
from endpoint_bounds import gas_heat_increment
from path_model import Setting, Site, co2_cp, heat_path
from pydantic import BaseModel, ConfigDict, Field
from scipy.integrate import quad
from scipy.optimize import brentq
from thermal_core import R

ROOT: Final = Path(__file__).resolve().parents[1]


class PublicCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    case_id: str
    capacity_1_mol_kg: float = Field(gt=0)
    capacity_2_mol_kg: float = Field(gt=0)
    b01_per_kPa: float = Field(gt=0)
    b02_per_kPa: float = Field(gt=0)
    binding_1_kJ_mol: float = Field(ge=0)
    binding_2_kJ_mol: float = Field(ge=0)
    dry_cp_kJ_kg_K: float = Field(gt=0)
    adsorption_K: float = Field(ge=298)
    fa_bar: float = Field(gt=0)
    fd_bar: float = Field(gt=0)
    temperature_cap_K: float = Field(gt=298)
    source_model_min_K: float
    source_model_max_K: float
    underlying_measurement_temperatures_K: tuple[float, ...]
    parameter_source_doi: str
    underlying_equilibrium_doi: str
    source_locator: str
    status: str

    @property
    def capacity(self) -> float:
        return self.capacity_1_mol_kg + self.capacity_2_mol_kg

    @property
    def sites(self) -> tuple[tuple[float, Site], ...]:
        """Convert b0 from kPa^-1 to bar^-1; standard fugacity is 1 bar."""
        return (
            (self.capacity_1_mol_kg, Site(self.binding_1_kJ_mol, -R * log(100 * self.b01_per_kPa))),
            (self.capacity_2_mol_kg, Site(self.binding_2_kJ_mol, -R * log(100 * self.b02_per_kPa))),
        )

    def setting(self, budget: float) -> Setting:
        return Setting(
            self.adsorption_K,
            self.fa_bar,
            self.fd_bar,
            self.capacity,
            self.dry_cp_kJ_kg_K,
            budget,
            self.temperature_cap_K,
        )


@dataclass(frozen=True, slots=True)
class CaseHeat:
    temperature_K: float
    q_initial_mol_kg: float
    q_after_pressure_mol_kg: float
    q_final_mol_kg: float
    release_mol_kg: float
    dry_heat_kJ_kg: float
    binding_heat_kJ_kg: float
    retained_heat_kJ_kg: float
    path_heat_kJ_kg: float
    reduced_heat_kJ_kg: float


def load_case() -> PublicCase:
    return PublicCase.model_validate_json((ROOT / "data/public_case_13x.json").read_text(encoding="utf-8"))


def occupation(site: Site, state: tuple[float, float]) -> float:
    temperature, pressure_bar = state
    odds = pressure_bar * exp(site.energy / (R * temperature) - site.entropy / R)
    return odds / (1 + odds)


def case_heat(case: PublicCase, temperature: float) -> CaseHeat:
    """Add each site's extensive heats; split dry heat to avoid double-counting."""
    initial = start = final = binding = loading = 0.0
    for capacity, site in case.sites:
        initial += capacity * occupation(site, (case.adsorption_K, case.fa_bar))
        start += capacity * occupation(site, (case.adsorption_K, case.fd_bar))
        final += capacity * occupation(site, (temperature, case.fd_bar))
        if temperature > case.adsorption_K:
            setting = Setting(
                case.adsorption_K,
                case.fa_bar,
                case.fd_bar,
                capacity,
                case.dry_cp_kJ_kg_K * capacity / case.capacity,
                100.0,
                case.temperature_cap_K,
            )
            heat = heat_path(setting, site, temperature)
            binding += heat.binding
            loading += heat.loading
        else:
            binding += (
                capacity
                * site.energy
                * (
                    occupation(site, (case.adsorption_K, case.fa_bar))
                    - occupation(site, (temperature, case.fd_bar))
                )
            )
    dry = case.dry_cp_kJ_kg_K * (temperature - case.adsorption_K)
    return CaseHeat(
        temperature,
        initial,
        start,
        final,
        initial - final,
        dry,
        binding,
        loading,
        dry + binding + loading,
        dry + binding,
    )


def budget_temperature(case: PublicCase, budget: float, *, complete: bool) -> float | None:
    """Invert monotone heat along this fixed material; None means pressure step unaffordable."""

    def heat_error(temperature: float) -> float:
        heat = case_heat(case, temperature)
        return (heat.path_heat_kJ_kg if complete else heat.reduced_heat_kJ_kg) - budget

    if heat_error(case.adsorption_K) > 0:
        return None
    if heat_error(case.temperature_cap_K) <= 0:
        return case.temperature_cap_K
    root, _info = brentq(heat_error, case.adsorption_K, case.temperature_cap_K, xtol=1e-10, full_output=True)
    return float(root)


def firstlaw_heat(case: PublicCase, temperature: float) -> float:
    """Independent bed-state enthalpy plus discharged-gas enthalpy balance."""
    hg = gas_heat_increment(case.adsorption_K, temperature)
    delta_bed = case.dry_cp_kJ_kg_K * (temperature - case.adsorption_K)
    discharged_enthalpy = 0.0
    for capacity, site in case.sites:
        qa = capacity * occupation(site, (case.adsorption_K, case.fa_bar))
        qd = capacity * occupation(site, (temperature, case.fd_bar))
        delta_bed += qd * (hg - site.energy) + qa * site.energy

        def outflow_integrand(t: float, site: Site = site, capacity: float = capacity) -> float:
            theta = occupation(site, (t, case.fd_bar))
            released_per_K = capacity * theta * (1 - theta) * site.energy / (R * t * t)
            return gas_heat_increment(case.adsorption_K, t) * released_per_K

        value, _error = quad(outflow_integrand, case.adsorption_K, temperature, epsabs=1e-10)
        discharged_enthalpy += float(value)
    return delta_bed + discharged_enthalpy


def independent_loading_heat(case: PublicCase, temperature: float) -> float:
    """Adaptive quadrature cross-check, independent of 96-point Gaussian path integration."""

    def integrand(t: float) -> float:
        uptake = sum(capacity * occupation(site, (t, case.fd_bar)) for capacity, site in case.sites)
        return uptake * float(co2_cp(np.array([t]))[0])

    value, _error = quad(integrand, case.adsorption_K, temperature, epsabs=1e-10)
    return float(value)
