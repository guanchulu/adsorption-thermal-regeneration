# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pytest"]
# ///
# Run: PYTHONPATH=code uv run --project .. --no-sync --offline pytest tests/test_gas_fugacity.py
import numpy as np
import pytest
from gas_fugacity import Gas, GasFugacityError, gas_fugacity


@pytest.mark.parametrize(
    ("gas", "expected"),
    [(Gas.CO2, [0.9929114427562412, 0.9945357132660135, 0.9957385348108995]),
     (Gas.N2, [0.9992350878586089, 0.9995466422420981, 0.9997677925041160])],
)
def test_one_bar_coefficients_match_independent_source_parameter_calculation(
    gas: Gas, expected: list[float],
) -> None:
    # Given the source temperatures and a pressure of one bar.
    temperature = np.array([273.0, 298.0, 323.0])
    # When applying the documented pure-gas Peng-Robinson equation.
    result = gas_fugacity(np.ones(3), temperature, gas)
    # Then it matches the independently audited coefficients and bar units.
    np.testing.assert_allclose(result.coefficient, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.fugacity_bar, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("gas", [Gas.CO2, Gas.N2])
def test_zero_and_dilute_pressure_reach_ideal_limit(gas: Gas) -> None:
    # Given zero and nearly zero pressure in the declared domain.
    pressure = np.array([0.0, 1e-10, 1e-8])
    # When calculating reservoir fugacity.
    result = gas_fugacity(pressure, np.full(3, 298.0), gas)
    # Then the analytic ideal limit is recovered without NaN or division by zero.
    np.testing.assert_allclose(result.coefficient, 1.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(result.fugacity_bar, pressure, rtol=1e-9, atol=0)


@pytest.mark.parametrize(("pressure", "temperature"), [(1.1, 298.0), (0.1, 272.0), (0.1, 324.0), (-0.1, 298.0)])
def test_unsupported_thermodynamic_domain_is_rejected(pressure: float, temperature: float) -> None:
    # Given a condition outside the scope justified by the source audit.
    # When attempting the restricted vapor-phase conversion.
    with pytest.raises(GasFugacityError):
        gas_fugacity(np.array([pressure]), np.array([temperature]), Gas.CO2)
    # Then the helper never silently extends its validated domain.
