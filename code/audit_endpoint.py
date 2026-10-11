# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "polars==2.0.0"]
# ///
# Run: uv run --project .. --no-sync --offline python code/audit_endpoint.py
"""Independent scalar, integration and feasible-design checks of endpoint certificates."""

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, localcontext
from math import log
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from endpoint_bounds import endpoint_cost, gas_heat_increment, release_bound, temperature_interval_upper
from path_model import Setting, Site, heat_path
from scipy.integrate import quad
from scipy.optimize import fminbound
from scipy.special import expit
from thermal_core import R

ROOT: Final = Path(__file__).resolve().parents[1]


class CertificateError(ArithmeticError):
    def __init__(self, detail: str) -> None:
        super().__init__(f"Endpoint audit: {detail}")


@dataclass(frozen=True, slots=True)
class Domain:
    release: float
    factor: float
    gas_heat: float
    ratio: float
    cluster: int

    def objective(self, occupation: float, *, upper: bool) -> float:
        loaded = occupation + self.release
        odds = log(loaded) - np.log1p(-loaded) - log(occupation) + np.log1p(-occupation)
        energy = (odds / self.cluster - self.ratio) / self.factor
        after_pressure = loaded / (np.exp(self.cluster * self.ratio) * (1 - loaded) + loaded)
        return float(
            max(0.0, self.release * energy) + self.gas_heat * (after_pressure if upper else occupation)
        )


def precise_minimum(domain: Domain, *, upper: bool) -> float:
    with localcontext() as context:
        context.prec = 70
        d, a, j, ratio = (
            Decimal(value) for value in (domain.release, domain.factor, domain.gas_heat, domain.ratio)
        )
        one, m = Decimal(1), Decimal(domain.cluster)
        k = (m * ratio).exp() - one

        def state(v: Decimal) -> tuple[Decimal, Decimal]:
            p = v + d
            energy = ((p.ln() - (one - p).ln() - v.ln() + (one - v).ln()) / m - ratio) / a
            sensible = j * (one + k) / (one + k * (one - p)) ** 2 if upper else j
            gradient = (
                d / (m * a) * (one / (p * (one - p)) - one / (v * (one - v))) + sensible
                if energy >= 0
                else sensible
            )
            value = max(Decimal(0), d * energy) + j * (p / (one + k * (one - p)) if upper else v)
            return value, gradient

        low, high = (one - d) * Decimal("1e-30"), (one - d) * (one - Decimal("1e-30"))
        for _ in range(230):
            mid = (low + high) / 2
            if state(mid)[1] < 0:
                low = mid
            else:
                high = mid
        return float(state((low + high) / 2)[0])


def scalar_checks() -> pl.DataFrame:
    rng = np.random.default_rng(91731)
    rows = []
    for index in range(600):
        temperature = float(rng.uniform(298.01, 423.0))
        cluster = 1 if index % 2 == 0 else 4
        ratio = 0.0 if index % 5 == 0 else log(1.5)
        release = float(rng.uniform(1e-6, 0.999))
        if index % 10 == 0:
            release = float(10 ** rng.uniform(-10, -4))
        if index % 10 == 1:
            release = float(1 - 10 ** rng.uniform(-10, -4))
        factor = (1 / 298.0 - 1 / temperature) / R
        gas_heat = gas_heat_increment(298.0, temperature)
        domain = Domain(release, factor, gas_heat, ratio, cluster)
        for upper in (False, True):
            result = endpoint_cost(release, factor, gas_heat, ratio, cluster, upper=upper)

            def objective(v: float, *, upper: bool = upper, domain: Domain = domain) -> float:
                return domain.objective(v, upper=upper)

            epsilon = min(release, 1 - release) * 1e-12
            location, minimum, _status, _count = fminbound(
                objective, epsilon, 1 - release - epsilon, xtol=1e-28, full_output=True
            )
            minimum = float(minimum)
            float_minimum = minimum
            gap = result.lower_support - minimum
            if gap > 0:
                minimum = precise_minimum(domain, upper=upper)
                gap = result.lower_support - minimum
            if gap > 1e-11 * (1 + abs(minimum)):
                raise CertificateError("support exceeds independently minimized convex cost")
            direct = domain.objective(result.v, upper=upper)
            if abs(direct - result.value) > 2e-7 * (1 + abs(direct)):
                raise CertificateError("reported endpoint objective mismatch")
            rows.append(
                {
                    "case": index,
                    "upper": upper,
                    "d": release,
                    "Td": temperature,
                    "m": cluster,
                    "independent_minimum": minimum,
                    "float_minimum": float_minimum,
                    "reported_value": result.value,
                    "support": result.lower_support,
                    "support_minus_minimum": gap,
                    "relative_value_error": abs(result.value - minimum) / (1 + abs(minimum)),
                    "independent_v": float(location),
                    "reported_v": result.v,
                }
            )
    return pl.DataFrame(rows)


def adaptive_total(setting: Setting, site: Site, temperature: float) -> float:
    def integrand(value: float) -> float:
        t = value / 1000.0
        cp = (24.99735 + 55.18696 * t - 33.69137 * t * t + 7.948387 * t**3 - 0.136638 / (t * t)) / 1000.0
        theta = expit(site.cluster * (log(setting.fd_bar) + site.energy / (R * value) - site.entropy / R))
        return float(theta * cp)

    loading, _error = quad(integrand, setting.adsorption_K, temperature, epsabs=1e-11, epsrel=1e-11)
    state = heat_path(setting, site, temperature)
    return state.dry + state.binding + setting.capacity * loading


def case_checks() -> pl.DataFrame:
    data = pl.read_csv(ROOT / "results/endpoint_brackets.csv")
    rng = np.random.default_rng(10731)
    rows = []
    for row in data.iter_rows(named=True):
        setting = Setting(**{key: row[key] for key in Setting.__dataclass_fields__})
        site = Site(row["endpoint_E"], row["endpoint_s"], setting.cluster)
        total = adaptive_total(setting, site, row["endpoint_T"])
        if total > setting.budget + 1e-7 or abs(total - row["endpoint_path_heat"]) > 1e-8:
            raise CertificateError("endpoint witness fails adaptive heat check")
        active = temperature_interval_upper(setting, row["active_bin_low"], row["active_bin_high"])
        if abs(active - row["endpoint_upper"]) > 1e-12:
            raise CertificateError("active interval bound mismatch")
        sampled = 0
        largest_margin = -1.0
        temperatures = np.linspace(298.01, 423.0, 33)
        temperatures = temperatures[setting.budget - setting.dry_cp * (temperatures - 298) > 0]
        for temperature in temperatures:
            available = (setting.budget - setting.dry_cp * (temperature - 298)) / setting.capacity
            fixed = release_bound(
                (1 / 298 - 1 / temperature) / R,
                gas_heat_increment(298, float(temperature)),
                log(1.5),
                setting.cluster,
                float(available),
                upper_heat=False,
            )
            if fixed > row["endpoint_upper"] + 2e-7:
                raise CertificateError("sampled fixed-temperature bound exceeds global bin certificate")
            for _ in range(20):
                candidate = Site(float(rng.uniform(0, 100)), float(rng.uniform(-0.05, 0.2)), setting.cluster)
                state = heat_path(setting, candidate, float(temperature))
                if state.total <= setting.budget:
                    sampled += 1
                    largest_margin = max(largest_margin, state.release - row["new_upper"])
                    if state.release > row["new_upper"] + 2e-7:
                        raise CertificateError("sampled feasible path exceeds upper certificate")
        lower, upper = (
            max(row["old_witness"], row["endpoint_lower"]),
            min(row["old_upper"], row["endpoint_upper"]),
        )
        if (
            abs(lower - row["new_lower"]) > 1e-12
            or abs(upper - row["new_upper"]) > 1e-12
            or lower > upper + 1e-8
        ):
            raise CertificateError("combined bracket mismatch")
        rows.append(
            {
                "Q": setting.capacity,
                "C": setting.dry_cp,
                "B": setting.budget,
                "m": setting.cluster,
                "adaptive_heat_error": abs(total - row["endpoint_path_heat"]),
                "feasible_samples": sampled,
                "largest_sample_minus_upper": largest_margin,
                "new_relative_gap": (upper - lower) / upper,
                "active_bin_width": row["active_bin_high"] - row["active_bin_low"],
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    scalars, cases = scalar_checks(), case_checks()
    scalars.write_csv(ROOT / "results/audit_endpoint_scalar.csv")
    cases.write_csv(ROOT / "results/audit_endpoint_cases.csv")
    files = (
        "code/endpoint_bounds.py",
        "code/run_endpoint_study.py",
        "code/audit_endpoint.py",
        "results/endpoint_brackets.csv",
    )
    result = {
        "status": "PASS",
        "scalar_cost_checks": len(scalars),
        "path_cases": len(cases),
        "max_support_excess": float(np.max(scalars["support_minus_minimum"].to_numpy())),
        "max_relative_cost_discrepancy": float(np.max(scalars["relative_value_error"].to_numpy())),
        "feasible_random_path_samples": int(cases["feasible_samples"].sum()),
        "max_adaptive_heat_error": float(np.max(cases["adaptive_heat_error"].to_numpy())),
        "gap_min_median_max": [
            float(f(cases["new_relative_gap"].to_numpy())) for f in (np.min, np.median, np.max)
        ],
        "scope": "Analytical convex/bin inequalities evaluated in ordinary floating point, 1e-8 guard; not formal interval arithmetic",
        "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
    }
    (ROOT / "results/audit_endpoint.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
