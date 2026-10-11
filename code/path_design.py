# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1"]
# ///
# Run via run_path_study.py with the locked project environment.
"""Analytical feasible/optimistic brackets and multistart path-feasible witnesses."""

from dataclasses import dataclass

import numpy as np
from ideal_cluster_probe import cluster_bound
from path_model import Setting, Site, Vector, co2_cp, endpoint_site, heat_path
from scipy.optimize import minimize
from thermal_core import Cycle, Resources, optimal_joint_temperature


@dataclass(frozen=True, slots=True)
class Design:
    temperature: float
    site: Site
    release: float
    path_heat: float


@dataclass(frozen=True, slots=True)
class Bracket:
    lower: Design
    upper: Design
    cp_ceiling: float


@dataclass(frozen=True, slots=True)
class Witness:
    design: Design
    successful_starts: int
    total_starts: int


def reduced_design(setting: Setting, dry_cp: float) -> Design:
    resources = Resources(setting.capacity, dry_cp, setting.budget)
    temperature = min(optimal_joint_temperature(setting.adsorption_K, resources), setting.temperature_cap)
    cycle = Cycle(setting.adsorption_K, temperature, setting.fa_bar, setting.fd_bar)
    bound = cluster_bound(cycle, resources, setting.cluster)
    site = Site(bound.energy_kj_mol, bound.entropy_loss_kj_mol_K, setting.cluster)
    actual = heat_path(setting, site, temperature)
    return Design(temperature, site, bound.utilization, actual.total)


def resource_bracket(setting: Setting) -> Bracket:
    """Construct a path-feasible design using an occupation-independent Cp ceiling."""
    ceiling = float(co2_cp(np.array([setting.temperature_cap]))[0])
    lower = reduced_design(setting, setting.dry_cp + setting.capacity * ceiling)
    upper = reduced_design(setting, setting.dry_cp)
    if lower.path_heat > setting.budget + 1e-7:
        raise ArithmeticError("Constructive lower design exceeds path heat budget")
    return Bracket(lower, upper, ceiling)


def improve_witness(setting: Setting, bracket: Bracket) -> Witness:
    """Search for feasible designs only; no global-optimum claim is made."""

    def unpack(x: Vector) -> tuple[Site, float]:
        return Site(float(50 * x[0]), float(0.1 * x[1]), setting.cluster), float(
            setting.adsorption_K + 100 * x[2]
        )

    def objective(x: Vector) -> float:
        site, temperature = unpack(x)
        return -heat_path(setting, site, temperature).release

    def constraint(x: Vector) -> float:
        site, temperature = unpack(x)
        return (setting.budget * (1 - 1e-9) - heat_path(setting, site, temperature).total) / setting.budget

    starts = []
    for design in (bracket.lower, bracket.upper):
        starts.append(
            np.array(
                [
                    design.site.energy / 50,
                    design.site.entropy / 0.1,
                    (design.temperature - setting.adsorption_K) / 100,
                ]
            )
        )
        state = heat_path(setting, design.site, design.temperature)
        release = state.release
        if 0.05 < release < 0.95:
            v = (1 - release) / 2 * 0.9
            site = endpoint_site(setting, design.temperature, (v + release, v))
            starts.append(
                np.array(
                    [site.energy / 50, site.entropy / 0.1, (design.temperature - setting.adsorption_K) / 100]
                )
            )
    best = bracket.lower
    successful = 0
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(0, 16), (-5, 20), (0.001, (setting.temperature_cap - setting.adsorption_K) / 100)],
            constraints={"type": "ineq", "fun": constraint},
            options={"ftol": 1e-11, "maxiter": 300},
        )
        site, temperature = unpack(np.asarray(result.x, dtype=float))
        state = heat_path(setting, site, temperature)
        if result.success:
            successful += 1
        if state.total <= setting.budget + 1e-8 and state.release > best.release:
            best = Design(temperature, site, state.release, state.total)
    if best.release > bracket.upper.release + 1e-7:
        raise ArithmeticError("A path witness exceeds the optimistic analytical bound")
    return Witness(best, successful, len(starts))
