from dataclasses import replace

import pytest
import run_operating_map as operating
from path_model import Setting, Site, heat_path


@pytest.fixture(scope="module")
def reference_point() -> operating.OperatingPoint:
    return operating.solve_point(Setting())


def test_reference_bracket_when_default_resources(reference_point: operating.OperatingPoint) -> None:
    # Given: the independently archived v4 default-resource bracket.
    expected = (0.4497393235, 0.4536924338)
    # When: the operating map solves the same physical problem.
    lower, upper = reference_point.feasible_fraction, reference_point.refined_upper_fraction
    # Then: the recomputed release interval reproduces the archived result.
    assert (lower, upper) == pytest.approx(expected, abs=2e-7)


def test_feasible_witness_when_reintegrated(reference_point: operating.OperatingPoint) -> None:
    # Given: the retained witness parameters in an exported map point.
    site = Site(reference_point.witness_E_kJ_mol, reference_point.witness_s_kJ_mol_K)
    # When: the original path model integrates this design independently of the map wrapper.
    state = heat_path(Setting(), site, reference_point.witness_T_K)
    # Then: the witness attains its declared release within its heat budget.
    assert state.total <= 100.0 + 1e-8
    assert 4.0 * state.release == pytest.approx(reference_point.feasible_mol_kg, abs=1e-9)


def test_cell_bounds_when_budget_samples_are_nonmonotone(reference_point: operating.OperatingPoint) -> None:
    # Given: conservative but nonmonotone numerical bounds at increasing heat budgets.
    points = [
        replace(reference_point, budget_kJ_kg=50.0, feasible_mol_kg=1.0, refined_upper_mol_kg=2.0),
        replace(reference_point, budget_kJ_kg=100.0, feasible_mol_kg=0.9, refined_upper_mol_kg=1.9),
        replace(reference_point, budget_kJ_kg=150.0, feasible_mol_kg=1.2, refined_upper_mol_kg=2.1),
    ]
    # When: monotonicity supplies conservative constant bounds inside each budget cell.
    cells = operating.monotone_cells(points)
    # Then: lower limits come from past witnesses; upper limits come from future ceilings.
    assert [(cell.feasible_floor_mol_kg, cell.upper_ceiling_mol_kg) for cell in cells] == [
        (1.0, 1.9),
        (1.0, 2.1),
    ]
