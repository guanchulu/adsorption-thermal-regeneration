# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "polars==2.0.0"]
# ///
# Run: uv run --project .. --frozen python code/run_path_study.py
"""Run predeclared physical-path checks before manuscript writing."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from path_design import improve_witness, resource_bracket
from path_model import Setting, co2_cp, endpoint_site, heat_path
from scipy.optimize import fminbound

ROOT: Final = Path(__file__).resolve().parents[1]


def path_benchmarks() -> None:
    rows = []
    for q in (2.0, 4.0, 8.0):
        for c in (0.5, 1.0, 2.0):
            for b in (50.0, 100.0, 200.0):
                for m in (1, 4):
                    setting = Setting(capacity=q, dry_cp=c, budget=b, cluster=m)
                    bracket = resource_bracket(setting)
                    witness = improve_witness(setting, bracket)
                    state = heat_path(setting, witness.design.site, witness.design.temperature)
                    reduced_state = heat_path(setting, bracket.upper.site, bracket.upper.temperature)
                    rows.append(
                        {
                            **asdict(setting),
                            "cp_g_ceiling": bracket.cp_ceiling,
                            "thermal_loading_ratio": q * bracket.cp_ceiling / c,
                            "lower_release": bracket.lower.release,
                            "upper_release": bracket.upper.release,
                            "lower_T": bracket.lower.temperature,
                            "lower_E": bracket.lower.site.energy,
                            "lower_s": bracket.lower.site.entropy,
                            "lower_path_heat": bracket.lower.path_heat,
                            "upper_T": bracket.upper.temperature,
                            "upper_E": bracket.upper.site.energy,
                            "upper_s": bracket.upper.site.entropy,
                            "upper_design_path_heat": bracket.upper.path_heat,
                            "upper_design_loading": reduced_state.loading,
                            "witness_release": witness.design.release,
                            "witness_T": witness.design.temperature,
                            "witness_E": witness.design.site.energy,
                            "witness_s": witness.design.site.entropy,
                            "witness_path_heat": state.total,
                            "witness_dry": state.dry,
                            "witness_binding": state.binding,
                            "witness_loading": state.loading,
                            "witness_theta_a": state.theta_ads,
                            "witness_theta_d": state.theta_des,
                            "successful_starts": witness.successful_starts,
                            "total_starts": witness.total_starts,
                        }
                    )
        print("Path cases completed for Q", q, flush=True)
    pl.DataFrame(rows).write_csv(ROOT / "results/path_brackets.csv")


def symmetry_breaking() -> None:
    rows = []
    profiles = []
    for q in (2.0, 4.0, 8.0):
        for m in (1, 4):
            for delta in (0.4, 0.6, 0.8):
                setting = Setting(fa_bar=0.1, fd_bar=0.1, capacity=q, cluster=m)
                temperature = 350.0
                v0 = (1 - delta) / 2

                def cost(v: float, setting: Setting = setting, delta: float = delta) -> float:
                    return heat_path(setting, endpoint_site(setting, 350.0, (v + delta, v)), 350.0).total

                optimum, _value, _status, _evaluations = fminbound(
                    cost, max(1e-8, v0 * 0.001), v0, xtol=1e-12, full_output=True
                )
                v = float(optimum)
                symmetric = heat_path(
                    setting, endpoint_site(setting, temperature, (v0 + delta, v0)), temperature
                )
                improved = heat_path(
                    setting, endpoint_site(setting, temperature, (v + delta, v)), temperature
                )
                rows.append(
                    {
                        "Q": q,
                        "m": m,
                        "delta": delta,
                        "Ta": 298.0,
                        "Td": temperature,
                        "v_symmetric": v0,
                        "v_feasible_minimum": v,
                        "heat_symmetric": symmetric.total,
                        "heat_improved": improved.total,
                        "heat_saved": symmetric.total - improved.total,
                        "fraction_saved": 1 - improved.total / symmetric.total,
                    }
                )
                if q == 4.0 and delta == 0.6:
                    for vv in np.linspace(v0 * 0.5, v0 * 1.5, 251):
                        state = heat_path(
                            setting,
                            endpoint_site(setting, temperature, (float(vv + delta), float(vv))),
                            temperature,
                        )
                        profiles.append(
                            {
                                "m": m,
                                "v": float(vv),
                                "heat_total": state.total,
                                "heat_binding": state.binding,
                                "heat_loading": state.loading,
                                "heat_symmetric": symmetric.total,
                                "binding_symmetric": symmetric.binding,
                                "loading_symmetric": symmetric.loading,
                            }
                        )
    pl.DataFrame(rows).write_csv(ROOT / "results/symmetry_breaking.csv")
    pl.DataFrame(profiles).write_csv(ROOT / "results/symmetry_profiles.csv")


if __name__ == "__main__":
    path_benchmarks()
    symmetry_breaking()
    temperatures = np.linspace(298.0, 423.0, 1001)
    cp = co2_cp(temperatures)
    (ROOT / "qa/path_scope.json").write_text(
        json.dumps(
            {
                "path": "isothermal depressurization atTa fromfa tofd, followed by fixed-fd monotone heating; fd<=fa",
                "heat": "dry C deltaT + Q E(theta_a-theta_d) + Q integral theta(T,fd) Cpg(T)dT",
                "CO2_Cp_source": "https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=1",
                "Cp_units": "kJ/mol/K; NIST J values divided by1000",
                "cp_min": float(cp.min()),
                "cp_max": float(cp.max()),
                "monotone_on_checked_grid": bool(np.all(np.diff(cp) > 0)),
                "numerical_witness": "multistart SLSQP feasible point, not asserted global optimum",
                "domain": "free entropy and E>=0 for analytical envelopes; physical synthesizability not guaranteed",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
