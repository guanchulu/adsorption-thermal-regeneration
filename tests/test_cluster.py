# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "pytest==9.1.1"]
# ///
# Run: uv run --frozen pytest tests/test_cluster.py
import numpy as np
import pytest
from ideal_cluster_probe import cluster_bound
from thermal_core import Cycle, Resources, difference, optimal_joint_temperature, resource_bound


def test_cluster_one_reduces_to_independent_sites() -> None:
    # Given a one-molecule cluster.
    cycle = Cycle(298.0, 353.0, 0.15, 0.1)
    resources = Resources(4.0, 1.0, 100.0)
    # When the exact cluster mapping is used.
    cluster = cluster_bound(cycle, resources, 1)
    independent = resource_bound(cycle, resources)
    # Then both outputs are identical.
    assert cluster == independent


@pytest.mark.parametrize("size", [2, 4, 8])
def test_cluster_binding_heat_is_per_molecule_not_per_cluster(size: int) -> None:
    # Given fixed molecular capacity and a declared ideal-cluster partition model.
    cycle = Cycle(298.0, 353.0, 0.15, 0.1)
    resources = Resources(4.0, 1.0, 100.0)
    # When the mapped bound is converted back to molecular units.
    result = cluster_bound(cycle, resources, size)
    transformed = Cycle(298.0, 353.0, 0.15**size, 0.1**size)
    response = difference(
        np.array([size * result.energy_kj_mol]), np.array([size * result.entropy_loss_kj_mol_K]), transformed
    )[0]
    heat = 4 * result.energy_kj_mol * response + 55.0
    # Then molecular release and total heat agree without an erroneous factor of size.
    assert response == pytest.approx(result.utilization, abs=1e-11)
    assert heat == pytest.approx(100.0, abs=1e-8)


@pytest.mark.parametrize("size", [2, 4])
def test_joint_temperature_scaling_survives_ideal_cluster_extension(size: int) -> None:
    # Given co-designed ideal clusters under the same thermal resources.
    resources = Resources(4.0, 1.0, 100.0)
    optimum = optimal_joint_temperature(298.0, resources)
    # When release at the closed-form optimum is compared with an independent temperature grid.
    best = cluster_bound(Cycle(298.0, optimum, 0.15, 0.1), resources, size).utilization
    alternatives = [
        cluster_bound(Cycle(298.0, float(t), 0.15, 0.1), resources, size).utilization
        for t in np.linspace(303.0, 397.0, 95)
    ]
    # Then changing cooperativity changes the release bound, not this conditional optimal temperature.
    assert best >= max(alternatives) - 1e-11
