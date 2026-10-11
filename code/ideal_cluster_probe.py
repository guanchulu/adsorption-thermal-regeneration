# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "polars==2.0.0", "scipy==1.18.1"]
# ///
# Run: uv run --frozen python code/ideal_cluster_probe.py
"""Ideal all-or-none cluster stress test; no assignment of real material mechanisms."""

from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from thermal_core import (
    Cycle,
    ResourceBound,
    Resources,
    ThermalInputError,
    optimal_joint_temperature,
    resource_bound,
)

ROOT: Final = Path(__file__).resolve().parents[1]


def cluster_bound(cycle: Cycle, resources: Resources, size: int) -> ResourceBound:
    """Map molecule-normalized capacity to independent empty/full clusters exactly."""
    if size < 1:
        raise ThermalInputError("positive integer cluster size")
    transformed = Cycle(
        cycle.adsorption_K,
        cycle.desorption_K,
        cycle.adsorption_fugacity_bar**size,
        cycle.desorption_fugacity_bar**size,
    )
    clusters = Resources(
        resources.capacity_mol_kg / size, resources.heat_capacity_kj_kg_K, resources.heat_budget_kj_kg
    )
    result = resource_bound(transformed, clusters)
    return ResourceBound(
        temperature_feasible=result.temperature_feasible,
        utilization=result.utilization,
        energy_kj_mol=result.energy_kj_mol / size,
        entropy_loss_kj_mol_K=result.entropy_loss_kj_mol_K / size,
        sensible_heat_kj_kg=result.sensible_heat_kj_kg,
        binding_heat_kj_kg=result.binding_heat_kj_kg,
    )


def main() -> None:
    rows = []
    resources = Resources(4.0, 1.0, 100.0)
    optimum = optimal_joint_temperature(298.0, resources)
    for size in (1, 2, 4, 8):
        for temperature in np.sort(np.r_[np.linspace(303.0, 397.0, 189), optimum]):
            cycle = Cycle(298.0, float(temperature), 0.15, 0.1)
            result = cluster_bound(cycle, resources, size)
            rows.append(
                {
                    **asdict(cycle),
                    **asdict(resources),
                    **asdict(result),
                    "cluster_size": size,
                    "joint_temperature_K": optimum,
                    "release_bound_mol_kg": 4 * result.utilization,
                    "entropy_in_example_range": 0.04 <= result.entropy_loss_kj_mol_K <= 0.14,
                    "scope": "ideal empty/full clusters, entropy-relaxed co-design; no observed cooperativity claim",
                }
            )
    pl.DataFrame(rows).write_csv(ROOT / "results/ideal_cluster_probe.csv")
    print("Ideal-cluster mechanism cases", len(rows))


if __name__ == "__main__":
    main()
