# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "pytest==9.1.1", "pydantic==2.13.5"]
# ///
# How to run: uv run --project .. --frozen pytest -c pyproject.toml tests/test_independent_first_law.py
from dataclasses import replace

import pytest
from endpoint_bounds import gas_heat_increment
from independent_first_law import FirstLawCase, first_law
from path_model import Setting, Site, heat_path


def test_zero_energy_reduces_to_dry_and_constant_inventory_heat() -> None:
    # Given zero binding energy and entropy, q=Q*f/(1+f) is temperature invariant.
    case = FirstLawCase(energy=0.0, entropy=0.0)
    inventory = case.capacity * case.fd_bar / (1 + case.fd_bar)
    expected = case.dry_cp * 55 + inventory * gas_heat_increment(298.0, 353.0)
    # When the stored-energy and outlet-enthalpy balance is integrated.
    result = first_law(case, steps=20, reference=100.0)
    # Then pressure heat vanishes and heating has the exact constant-inventory limit.
    assert result.pressure_heat == pytest.approx(0.0, abs=1e-12)
    assert result.total == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("cluster", [1, 4])
def test_first_law_converges_to_original_quadrature(cluster: int) -> None:
    # Given independent and ideal-cluster cases with pressure reduction and heating.
    case = FirstLawCase(cluster=cluster)
    original = heat_path(Setting(), Site(case.energy, case.entropy, cluster), case.desorption_K)
    # When an independently reconstructed energy-flow balance uses 4000 intervals.
    result = first_law(case, steps=4000)
    # Then heat and net release agree without an extra molecular cluster factor.
    assert result.total == pytest.approx(original.total, rel=1e-8)
    assert result.released_mol_kg == pytest.approx(case.capacity * original.release, abs=1e-12)


def test_reference_shift_preserves_heat_but_changes_energy_terms() -> None:
    # Given one physical path and a 1000 kJ/mol shift of the gas enthalpy zero.
    case = FirstLawCase()
    base = first_law(case, reference=0.0)
    # When both adsorbed internal energy and outgoing gas share the shifted zero.
    shifted = first_law(case, reference=1000.0)
    # Then the two reference-dependent terms cancel and heat stays invariant.
    expected_shift = 1000 * base.released_mol_kg
    assert shifted.total == pytest.approx(base.total, abs=1e-10)
    assert shifted.outgoing_enthalpy - base.outgoing_enthalpy == pytest.approx(expected_shift)
    assert shifted.internal_energy_change - base.internal_energy_change == pytest.approx(-expected_shift)


def test_mesh_refinement_has_second_order_heat_convergence() -> None:
    # Given a heating path with a nonconstant occupation and gas enthalpy.
    case = FirstLawCase()
    reference = heat_path(Setting(), Site(case.energy, case.entropy), case.desorption_K).total
    # When the heating mesh is refined by factors of four.
    errors = [abs(first_law(case, steps=n).total - reference) for n in (25, 100, 400)]
    # Then the Stieltjes trapezoid error decreases at its expected second order.
    assert 14 < errors[0] / errors[1] < 18
    assert 14 < errors[1] / errors[2] < 18


def test_equal_fugacity_has_no_pressure_step_heat() -> None:
    # Given identical initial and heating fugacities.
    case = replace(FirstLawCase(), fa_bar=0.1)
    # When the full two-leg balance is evaluated.
    result = first_law(case)
    # Then the isothermal pressure leg contributes exactly zero.
    assert result.pressure_heat == 0.0
