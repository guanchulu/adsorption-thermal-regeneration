# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "polars==2.0.0"]
# ///
# Run: uv run --project .. --no-sync --offline python code/audit_path.py
"""Independent adaptive-quadrature audit of saved path designs and local theorem."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from path_model import Setting, Site, endpoint_site
from scipy.integrate import quad
from scipy.special import expit
from thermal_core import R

ROOT: Final = Path(__file__).resolve().parents[1]


class AuditError(ArithmeticError):
    def __init__(self, detail: str) -> None:
        super().__init__(f"Path audit failed: {detail}")


def require(*, condition: bool, detail: str) -> None:
    if not condition:
        raise AuditError(detail)


def gas_cp(temperature: float) -> float:
    scaled = temperature / 1000.0
    return (
        24.99735 + 55.18696 * scaled - 33.69137 * scaled**2 + 7.948387 * scaled**3 - 0.136638 / scaled**2
    ) / 1000.0


def occupancy(site: Site, temperature: float, fugacity: float) -> float:
    return float(
        expit(site.cluster * (np.log(fugacity) + site.energy / (R * temperature) - site.entropy / R))
    )


def adaptive_heat(setting: Setting, site: Site, temperature: float) -> tuple[float, float, float]:
    def integrand(value: float) -> float:
        return occupancy(site, value, setting.fd_bar) * gas_cp(value)

    integral, error = quad(integrand, setting.adsorption_K, temperature, epsabs=1e-11, epsrel=1e-11)
    release = occupancy(site, setting.adsorption_K, setting.fa_bar) - occupancy(
        site, temperature, setting.fd_bar
    )
    loading = setting.capacity * integral
    total = (
        setting.dry_cp * (temperature - setting.adsorption_K)
        + setting.capacity * site.energy * release
        + loading
    )
    return total, loading, setting.capacity * error


def path_checks() -> pl.DataFrame:
    data = pl.read_csv(ROOT / "results/path_brackets.csv")
    rows = []
    for row in data.iter_rows(named=True):
        setting = Setting(**{key: row[key] for key in Setting.__dataclass_fields__})
        differences = []
        quadrature_errors = []
        for prefix in ("lower", "upper", "witness"):
            site = Site(row[f"{prefix}_E"], row[f"{prefix}_s"], setting.cluster)
            temperature = row[f"{prefix}_T"]
            total, loading, error = adaptive_heat(setting, site, temperature)
            saved = row["upper_design_path_heat"] if prefix == "upper" else row[f"{prefix}_path_heat"]
            differences.append(abs(total - saved))
            quadrature_errors.append(error)
            require(condition=abs(total - saved) < 1e-8, detail=f"96-node/adaptive heat mismatch: {prefix}")
            release = occupancy(site, setting.adsorption_K, setting.fa_bar) - occupancy(
                site, temperature, setting.fd_bar
            )
            require(
                condition=abs(release - row[f"{prefix}_release"]) < 1e-12,
                detail=f"release mismatch: {prefix}",
            )
            require(
                condition=loading > 0 and release > 0 and site.energy >= 0,
                detail="nonpositive path component",
            )
            ceiling = setting.capacity * row["cp_g_ceiling"] * (temperature - setting.adsorption_K)
            require(
                condition=loading <= ceiling + 1e-10, detail="loading exceeds occupation-independent ceiling"
            )
            if prefix == "upper":
                require(
                    condition=abs(total - loading - setting.budget) < 1e-8,
                    detail="reduced upper heat is not B",
                )
            else:
                require(condition=total <= setting.budget + 1e-8, detail=f"infeasible path design: {prefix}")
        require(
            condition=row["lower_release"] <= row["witness_release"] + 1e-10,
            detail="witness below constructive lower",
        )
        require(
            condition=row["witness_release"] <= row["upper_release"] + 1e-10,
            detail="witness exceeds reduced upper",
        )
        rows.append(
            {
                **row,
                "adaptive_max_error": max(differences),
                "adaptive_reported_error": max(quadrature_errors),
                "bracket_gap_fraction": 1 - row["lower_release"] / row["upper_release"],
                "witness_gap_fraction": 1 - row["witness_release"] / row["upper_release"],
                "upper_path_overrun_fraction": row["upper_design_path_heat"] / setting.budget - 1,
                "witness_loading_budget_fraction": row["witness_loading"] / setting.budget,
                "lower_slack_fraction": 1 - row["lower_path_heat"] / setting.budget,
                "witness_endpoint_asymmetry": 1 - row["witness_theta_a"] - row["witness_theta_d"],
            }
        )
    require(condition=len(rows) == 54, detail="expected 54 path cases")
    return pl.DataFrame(rows)


def symmetry_checks() -> pl.DataFrame:
    data = pl.read_csv(ROOT / "results/symmetry_breaking.csv")
    rows = []
    for row in data.iter_rows(named=True):
        setting = Setting(fa_bar=0.1, fd_bar=0.1, capacity=row["Q"], cluster=row["m"])
        delta, temperature, center = row["delta"], row["Td"], row["v_symmetric"]
        site = endpoint_site(setting, temperature, (center + delta, center))

        def derivative_integrand(
            value: float, site: Site = site, center: float = center, delta: float = delta
        ) -> float:
            theta = occupancy(site, value, 0.1)
            return gas_cp(value) * theta * (1 - theta) / ((center + delta) * (1 - center - delta))

        integral, _error = quad(
            derivative_integrand, setting.adsorption_K, temperature, epsabs=1e-11, epsrel=1e-11
        )
        derivative = setting.capacity * integral
        step = 1e-6
        values = [
            adaptive_heat(
                setting,
                endpoint_site(setting, temperature, (center + offset + delta, center + offset)),
                temperature,
            )[0]
            for offset in (-step, step)
        ]
        numeric = (values[1] - values[0]) / (2 * step)
        symmetric = adaptive_heat(setting, site, temperature)[0]
        endpoint = row["v_feasible_minimum"]
        improved = adaptive_heat(
            setting, endpoint_site(setting, temperature, (endpoint + delta, endpoint)), temperature
        )[0]
        require(condition=derivative > 0 and numeric > 0, detail="wrong fixed-release derivative sign")
        require(
            condition=abs(numeric - derivative) < 1e-5, detail="analytic fixed-release derivative mismatch"
        )
        require(condition=abs(symmetric - row["heat_symmetric"]) < 1e-8, detail="symmetric heat mismatch")
        require(condition=abs(improved - row["heat_improved"]) < 1e-8, detail="improved heat mismatch")
        require(condition=endpoint < center and improved < symmetric, detail="no lower-inventory improvement")
        rows.append(
            {
                **row,
                "analytic_center_derivative": derivative,
                "finite_difference_derivative": numeric,
                "derivative_error": abs(derivative - numeric),
            }
        )
    require(condition=len(rows) == 18, detail="expected 18 symmetry cases")
    return pl.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class Range:
    minimum: float
    median: float
    maximum: float


def ranges(table: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "metric": column,
                **asdict(
                    Range(
                        float(np.min(table[column].to_numpy())),
                        float(np.median(table[column].to_numpy())),
                        float(np.max(table[column].to_numpy())),
                    )
                ),
            }
            for column in columns
        ]
    )


def main() -> None:
    paths, symmetry = path_checks(), symmetry_checks()
    paths.write_csv(ROOT / "results/audit_path_cases.csv")
    symmetry.write_csv(ROOT / "results/audit_symmetry_cases.csv")
    metrics = ranges(
        paths,
        (
            "thermal_loading_ratio",
            "bracket_gap_fraction",
            "witness_gap_fraction",
            "upper_path_overrun_fraction",
            "witness_loading_budget_fraction",
            "lower_slack_fraction",
            "witness_endpoint_asymmetry",
            "adaptive_max_error",
        ),
    )
    symmetry_metrics = ranges(
        symmetry, ("heat_saved", "fraction_saved", "analytic_center_derivative", "derivative_error")
    )
    representative = paths.filter(
        (pl.col("capacity") == 4) & (pl.col("dry_cp") == 1) & (pl.col("budget") == 100)
    )
    files = (
        "code/path_model.py",
        "code/path_design.py",
        "code/run_path_study.py",
        "code/audit_path.py",
        "results/path_brackets.csv",
        "results/symmetry_breaking.csv",
    )
    result = {
        "status": "PASS",
        "path_cases": len(paths),
        "symmetry_cases": len(symmetry),
        "quadrature": "96-node Gauss Legendre independently checked against adaptive quad",
        "metrics": metrics.to_dicts(),
        "symmetry_metrics": symmetry_metrics.to_dicts(),
        "representative_Q4_C1_B100": representative.to_dicts(),
        "scope": "Reduced-model sharp upper; constructive path lower and feasible search witness, not global path optimum",
        "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
    }
    (ROOT / "results/audit_path.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(metrics.write_csv())
    print(symmetry_metrics.write_csv())
    print(
        representative.select(
            "cluster",
            "lower_release",
            "upper_release",
            "witness_release",
            "lower_T",
            "upper_T",
            "witness_T",
            "witness_dry",
            "witness_binding",
            "witness_loading",
            "witness_E",
            "witness_s",
        ).write_csv()
    )


if __name__ == "__main__":
    main()
