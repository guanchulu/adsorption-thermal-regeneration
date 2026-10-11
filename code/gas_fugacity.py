# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
# Run through the parent benchmark in its locked environment.
"""Pure-gas PR vapor fugacity, CRAFTED Table 3, limited to 273-323 K and 0-1 bar."""
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, assert_never

import numpy as np
from numpy.typing import NDArray

type Vector = NDArray[np.float64]
R: Final = 8.314462618
CO2_CRITICAL: Final = (304.1282, 7377300.0, 0.22394)
N2_CRITICAL: Final = (126.192, 3395800.0, 0.0372)


class Gas(StrEnum):
    CO2 = "CO2"
    N2 = "N2"


class GasFugacityError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class FugacityResult:
    fugacity_bar: Vector
    coefficient: Vector


def gas_fugacity(pressure_bar: Vector, temperature_k: Vector, gas: Gas) -> FugacityResult:
    """Convert matched pressure/temperature vectors using source Tc[K], Pc[Pa], omega.

    Select the largest real PR root in the restricted vapor domain; zero pressure
    uses the analytic phi=1 limit. Original pressure must remain the reporting axis.
    """
    if (pressure_bar.ndim != 1 or pressure_bar.size == 0
            or pressure_bar.shape != temperature_k.shape
            or not np.isfinite(pressure_bar).all() or not np.isfinite(temperature_k).all()
            or np.any(pressure_bar < 0) or np.any(pressure_bar > 1)
            or np.any(temperature_k < 273) or np.any(temperature_k > 323)):
        raise GasFugacityError("Require matching finite vectors within 0-1 bar and 273-323 K")
    match gas:
        case Gas.CO2:
            tc, pc, omega = CO2_CRITICAL
        case Gas.N2:
            tc, pc, omega = N2_CRITICAL
        case _:
            assert_never(gas)
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + kappa * (1 - np.sqrt(temperature_k / tc)))**2
    attraction = 0.45724 * alpha * (R * tc)**2 / pc
    covolume = 0.07780 * R * tc / pc
    pressure_pa = pressure_bar * 100000
    a = attraction * pressure_pa / (R * temperature_k)**2
    b = covolume * pressure_pa / (R * temperature_k)
    phi = np.ones(pressure_bar.shape, dtype=np.float64)
    sqrt2 = np.sqrt(2.0)
    for index in np.flatnonzero(pressure_bar > 0):
        ai, bi = float(a[index]), float(b[index])
        roots = np.roots([1.0, bi - 1, ai - 3 * bi**2 - 2 * bi, -(ai * bi - bi**2 - bi**3)])
        real_roots = roots.real[(np.abs(roots.imag) < 1e-9) & (roots.real > bi)]
        if real_roots.size == 0:
            raise GasFugacityError("No physical real Peng-Robinson vapor root")
        z = float(real_roots.max())
        log_ratio = np.log1p(2 * sqrt2 * bi / (z + (1 - sqrt2) * bi))
        log_phi = z - 1 - np.log(z - bi) - ai / (2 * sqrt2 * bi) * log_ratio
        phi[index] = np.exp(log_phi)
    return FugacityResult(np.asarray(pressure_bar * phi, dtype=np.float64), phi)
