# Reproducing the calculations

## 1. Obtain code and data

Clone or download the repository at tag `v1.0.0`. Download `research-data-v1.0.0.zip` and `SHA256SUMS.txt` from the same GitHub release.

Check the ZIP against SHA256SUMS.txt, then extract it into the repository root. The ZIP contains only relative paths under data/ and results/, plus its inventory. Keep its directory layout.

Windows:
```powershell
Get-FileHash research-data-v1.0.0.zip -Algorithm SHA256
Expand-Archive -LiteralPath research-data-v1.0.0.zip -DestinationPath . -Force
```

Linux/macOS:
```sh
sha256sum -c SHA256SUMS.txt
unzip -o research-data-v1.0.0.zip
```

SHA256SUMS.txt covers both release ZIPs. If only one has been downloaded, compare its individual hash or download the complete set before using the check-all command.

## 2. Install the pinned environment

```sh
uv sync --frozen
uv run --frozen pytest tests code/public_case_tests.py -q
```

The test suite includes thermodynamic identities, endpoint bounds, heat integration, gas fugacity, temperature fitting, operating maps, 13X unit conversions and three text-conversion checks.

## 3. Regenerate core calculations and main figures

```sh
uv run --frozen python reproduce.py
```

The runner creates output directories, recalculates the normalized enthalpy statistics, heat-budget map, independent first-law check, 13X example and heat-interval map, then recreates the six main figures.

Generated timing fields, PDF metadata and font rendering can vary by platform. The numerical model and comparison data are unchanged. Plotting uses Arial when available and Matplotlib fallback fonts otherwise.

## 4. Refit the public benchmark

This is a separate, longer calculation. The fixed release includes all saved predictions so it is not required just to regenerate the main figures.

```sh
uv run --frozen python code/parse_crafted.py
uv run --frozen python code/run_crafted_benchmark.py --workers 4
uv run --frozen python code/summarize_crafted.py
uv run --frozen python code/normalized_heat_error.py
```

The full archive has 97,704 isotherm–enthalpy pairs. Fitting uses 273/298 K uptake; 323 K uptake and adsorption enthalpy are held out. See `docs/benchmark_protocol.txt` and the original-source metadata.

For additional checks:
```sh
uv run --frozen python code/audit_crafted_raw.py
uv run --frozen python code/audit_path.py
uv run --frozen python code/audit_endpoint.py
```

Scripts may write diagnostic records to qa/. These are generated outputs and are ignored by Git.

