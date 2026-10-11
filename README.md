# Thermodynamic bounds on gas release under finite regeneration heat

Data and computational code supporting the manuscript by Guanchu Lu, Peng Zhao, Hao Yu and Xianfeng Fan.

The model bounds gas release from a rigid single-gas adsorbent with a finite regeneration heat budget. It combines dry-solid heating, desorption and the sensible heat of retained adsorbate. Independent Langmuir sites provide the baseline, with an ideal empty/full cluster extension.

## Download the fixed research version

- [Version 1.0.0](https://github.com/guanchulu/adsorption-thermal-regeneration/releases/tag/v1.0.0)
- [Input data and full prediction tables](https://github.com/guanchulu/adsorption-thermal-regeneration/releases/download/v1.0.0/research-data-v1.0.0.zip)

The repository contains code, tests, numerical summaries, figure source tables and six main-figure PDFs. Large Parquet inputs, complete prediction batches and the original CRAFTED archive are supplied in the release data ZIP. Extract that ZIP into the repository root, keeping the `data/` and `results/` directories.

## Reproduce

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are required. Dependency versions are pinned in `uv.lock`.

```sh
uv sync --frozen
uv run --frozen pytest tests code/public_case_tests.py -q
uv run --frozen python reproduce.py
```

The first command prepares the environment; the second runs 82 verification cases. The third regenerates the core numerical results and six main figures using the archived benchmark predictions, then runs the tests. It does not refit all 97,704 public-source isotherms. The full benchmark fitting commands are described in [REPRODUCING.md](REPRODUCING.md).

## Main results and files

| Result | Source files |
| --- | --- |
| 54 heat-budget settings | `results/path_brackets.csv`, `results/endpoint_brackets.csv` |
| Feasible and excluded release targets | `results/operating_map_points.csv`, `results/operating_map_cells.csv` |
| 5,246 temperature–release heat intervals | `results/heat_interval_map.csv` |
| Public uptake and held-out enthalpy predictions | `results/crafted_reference_metrics.parquet`, `results/crafted_full/` in the release data ZIP |
| Normalized enthalpy error | `results/normalized_heat_error.csv` |
| Published 13X parameter example | `data/public_case_13x.json`, `results/public_case_*.csv` |
| First-law checks | `results/first_law_cases.csv` |

The public benchmark covers 32,453 matched condition groups across 1,355 structures. The reference CO2 set contains 1,352 structures; the normalized heat-error statistic uses 1,351 nonzero reference scales. These are different evaluation counts, not additive samples.

## Interpretation

The heat bounds apply to the stated equilibrium model and regeneration path. The public comparisons test uptake and adsorption enthalpy. The 13X case uses a published parameterization; none of these comparisons is a measurement of complete-cycle regeneration duty.

## Data sources and attribution

- CRAFTED 2.0.1: [doi:10.5281/zenodo.10120180](https://doi.org/10.5281/zenodo.10120180).
- NIST/ARPA-E adsorption database: [doi:10.18434/T43882](https://doi.org/10.18434/T43882); the frozen upstream commit and checksum are in `data/snapshot_manifest.json`.
- NIST gas heat-capacity data and the published 13X parameters are documented in [DATA_SOURCES.md](DATA_SOURCES.md).

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source licenses and transformation notices. Source observations, normalized input tables and calculated results are distinguished there.

## Citation and version

Use the versioned release URL above when referring to the data underlying this study. This GitHub release has no DOI. Machine-readable author and version information is in [CITATION.cff](CITATION.cff).

OpenAI Codex assisted with code development, scientific drafting, numerical analysis and figure preparation. The scripts, input records, numerical outputs and tests are available for inspection. Repository publication does not indicate journal acceptance.

