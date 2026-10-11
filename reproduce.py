# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# Run: uv run --frozen python reproduce.py
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
for name in ("results", "figures", "qa"):
    (root / name).mkdir(exist_ok=True)
for script in (
    "normalized_heat_error.py",
    "run_operating_map.py",
    "independent_first_law.py",
    "public_case_run.py",
    "heat_interval_map.py",
    "fig_release_final.py",
    "fig_mechanism_v5.py",
    "fig_framework_v5.py",
    "fig_decisions_v5.py",
    "fig_external_v5.py",
    "fig_public_case_v5.py",
):
    print(f"Running {script}", flush=True)
    subprocess.run([sys.executable, str(root / "code" / script)], cwd=root, check=True)  # noqa: S603 - Fixed local script names only.
subprocess.run(
    [sys.executable, "-m", "pytest", "tests", "code/public_case_tests.py", "-q"],
    cwd=root,
    check=True,
)

