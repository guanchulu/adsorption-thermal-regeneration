# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "polars"]
# ///
from __future__ import annotations

import hashlib
import io
import json
import math
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Final

import numpy as np
import polars as pl

ROOT: Final = Path(__file__).resolve().parents[1]
ARCHIVE: Final = ROOT / "data/crafted_raw/CRAFTED-2.0.1.tar.xz"
GRID: Final = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000,
                       50000, 100000, 200000, 500000, 1000000], dtype=float)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        raise ArithmeticError(message)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    started = time.monotonic()
    files = [ROOT / "data/crafted_curves.parquet", ROOT / "data/crafted_points.parquet",
             ROOT / "data/crafted_temperature_groups.parquet", ARCHIVE]
    before = {str(f.relative_to(ROOT)): sha(f) for f in files}
    require(before[str(ARCHIVE.relative_to(ROOT))] ==
            "f29f7b3d6b7d6c51fd113138bc4a811c448d4dcce66ed37fddbfa4141a238dde", "Archive hash drift")
    curves = pl.read_parquet(files[0])
    points = pl.read_parquet(files[1])
    groups = pl.read_parquet(files[2])
    inventory = json.loads((ROOT / "results/crafted_inventory.json").read_text(encoding="utf8"))
    ranks = [hashlib.sha256(cid.encode()).hexdigest() for cid in curves["curve_id"]]
    ranked = curves.with_columns(pl.Series("audit_selection_sha256", ranks)).sort("audit_selection_sha256")
    selected: set[str] = set()
    for stratum in ranked.partition_by(["gas", "force_field", "temperature_K"]):
        selected.add(str(stratum["curve_id"][0]))
    nonfinite_ids = points.filter(~pl.col("enthalpy_kj_mol").is_finite())["curve_id"].unique().to_list()
    selected.update(ranked.filter(pl.col("curve_id").is_in(nonfinite_ids)).head(2)["curve_id"].to_list())
    for cid in ranked["curve_id"]:
        if len(selected) >= 20:
            break
        selected.add(str(cid))
    sample_meta = ranked.filter(pl.col("curve_id").is_in(list(selected)))
    require(sample_meta.height == 20, "Sample size mismatch")
    require(sample_meta.select(pl.struct("gas", "force_field", "temperature_K").n_unique()).item() == 12,
            "Missing gas/forcefield/temperature stratum")
    normalized = {str(key[0]): part.sort("pressure_Pa") for key, part in
                  points.filter(pl.col("curve_id").is_in(list(selected))).partition_by("curve_id", as_dict=True).items()}
    metadata = {row["curve_id"]: row for row in sample_meta.iter_rows(named=True)}
    require(curves.height == curves["curve_id"].n_unique() == 97704, "Curve key/count mismatch")
    require(points.height == 1270152, "Point count mismatch")
    require(points.select(pl.struct("curve_id", "pressure_Pa").n_unique()).item() == points.height,
            "Non-unique curve-pressure join")
    require(points["enthalpy_kj_mol"].null_count() == 0, "Missing enthalpy join")
    np.testing.assert_array_equal(points["pressure_bar"].to_numpy(), points["pressure_Pa"].to_numpy()/100000)
    np.testing.assert_array_equal(points["reported_minus_enthalpy_kj_mol"].to_numpy(),
                                  -points["enthalpy_kj_mol"].to_numpy())
    require(curves["structure_id"].n_unique() == 1357, "Structure count mismatch")
    require(groups.height == 32568 and bool((groups["n_temperatures"] == 3).all()), "Three-temperature count mismatch")
    require(points.filter(pl.col("pressure_bar") <= 1).height == 977040, "Low-pressure count mismatch")
    require(not any(PurePosixPath(p).name.startswith("._") for p in curves["source_member"]),
            "AppleDouble member retained")
    source_q: set[str] = set()
    source_h: set[str] = set()
    nonfinite_h = nonfinite_h_error = skipped = checked_pairs = 0
    sample_records: dict[str, dict[str, str | int]] = {cid: {"curve_id": cid} for cid in selected}
    with tarfile.open(ARCHIVE, mode="r|xz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if path.name.startswith("._"):
                skipped += 1
                continue
            if path.suffix != ".csv":
                continue
            is_q, is_h = "ISOTHERM_FILES" in path.parts, "ENTHALPY_FILES" in path.parts
            if not (is_q or is_h):
                continue
            cid = path.stem
            target_set = source_q if is_q else source_h
            require(cid not in target_set, f"Duplicate raw curve member: {cid}")
            target_set.add(cid)
            if not is_h and cid not in selected:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ArithmeticError(f"Cannot read {member.name}")
            raw = stream.read()
            text = raw.decode("utf-8-sig")
            if is_h:
                lines = [line.split(",") for line in text.splitlines() if line.strip() and not line.startswith("#")]
                require(len(lines) == 13, f"Unexpected enthalpy point count: {cid}")
                np.testing.assert_array_equal(np.array([float(line[0]) for line in lines]), GRID)
                nonfinite_h += sum(not math.isfinite(float(line[1])) for line in lines)
                nonfinite_h_error += sum(not math.isfinite(float(line[2])) for line in lines)
            if cid not in selected:
                continue
            values = np.loadtxt(io.StringIO(text), delimiter=",", ndmin=2)
            dest = normalized[cid]
            np.testing.assert_array_equal(values[:, 0], dest["pressure_Pa"].to_numpy())
            kind = "uptake" if is_q else "enthalpy"
            value_field = "uptake_mol_kg" if is_q else "enthalpy_kj_mol"
            error_field = "uptake_reported_error" if is_q else "enthalpy_reported_error"
            np.testing.assert_array_equal(values[:, 1], dest[value_field].to_numpy())
            np.testing.assert_array_equal(values[:, 2], dest[error_field].to_numpy())
            record = sample_records[cid]
            record[kind+"_source_member"] = member.name
            record[kind+"_sha256"] = hashlib.sha256(raw).hexdigest()
            record[kind+"_points_checked"] = len(values)
            if is_q:
                require(record["uptake_sha256"] == metadata[cid]["sha256"], f"Uptake hash mismatch: {cid}")
                require(metadata[cid]["source_member"] == member.name, f"Source-path mismatch: {cid}")
                require(len(values) == metadata[cid]["n_points"], f"Metadata point count mismatch: {cid}")
            checked_pairs += 1
            if checked_pairs % 10 == 0:
                print("Validated raw samples", checked_pairs, "/40", flush=True)
    require(source_q == source_h == set(curves["curve_id"]), "Raw uptake/enthalpy curve sets differ")
    require(len(source_q) == len(source_h) == 97704 and checked_pairs == 40, "Raw source/sample count mismatch")
    normalized_nonfinite = points.filter(~pl.col("enthalpy_kj_mol").is_finite()).height
    normalized_error_nonfinite = points.filter(~pl.col("enthalpy_reported_error").is_finite()).height
    require(nonfinite_h == normalized_nonfinite == 1249, "Nonfinite enthalpy count mismatch")
    require(nonfinite_h_error == normalized_error_nonfinite, "Nonfinite enthalpy uncertainty count mismatch")
    require(skipped == inventory["sidecars_skipped"], "Sidecar count mismatch")
    after = {str(f.relative_to(ROOT)): sha(f) for f in files}
    require(after == before, "Data changed during audit; re-audit frozen files")
    report = ["INDEPENDENT CRAFTED INGESTION AUDIT — 2026-10-07", "VERDICT: PASS",
        "Read-only validation of normalized data against the fixed raw source. Only this report is written.",
        "Selected20 curve pairs (40real CSVmembers), deterministically ranked bySHA256(curve_id): oneper gas×forcefield×temperature stratum, twohash-selected nonfinite-enthalpy curves, then lowestremaininghashes. All3temperatures, bothgases and bothforcefields covered.",
        "All20uptake fileSHA256 andsourcepaths agree withcurve metadata. Every sampledpressure, uptake, enthalpy andbothreporteduncertaintycolumns matches raw CSV floats exactly, including NaN preservation;260points perproperty. Same-basename q/H pairing verified for samples.",
        "Entire rawtar streamed once: exactreal uptake/enthalpy IDsets both97704 andequalnormalizedcurve IDs. Everyenthalpyfile has the13declaredPa nodes. Normalized(curve_id,pressure_Pa)keys unique andenthalpy joins0null. Dataset1270152pairedrows;1357sourceIDs;32568three-temperature structure/forcefield/charge/gas groups;977040rows<=1bar.",
        "Allnormalized pressure conversions Pa/100000=bar verified exactly. Source uptake mol/kg is numerically equal to mmol/g; no1000factor applied orrequired. Stored enthalpy retains raw sign; reported_minus_enthalpy is exactly its negative, with noextraRT orothercaloric adjustment.",
        f"Raw nonfinite enthalpyvalues={nonfinite_h}; normalized={normalized_nonfinite}. Raw nonfiniteenthalpyerrorvalues={nonfinite_h_error}; normalized={normalized_error_nonfinite}. Values werepreserved, not silently filled ordiscarded. Sidecars skipped={skipped}.",
        "LIMITS:20pairedcurves are a deterministic stratified provenance sample, not a full elementwise raw-value audit ofall97704curves. Whole-source pressure grids, keysets andnonfinite counts werechecked separately. PASS certifies ingestion consistency, notGCMC accuracy, experimental validation or suitability ofeverycurve forfitting.",
        "The official v1.0.0workflow exports total simulatedmolecule counts perframework mass withoutbulk-density subtraction. Itsuncertainty selector defaults touSD andpermitsSD/SE/uSD/uSE; the releasedmean_error should remain reporteduncertainty unlessper-run settings areestablished. Itis not automaticallySEM or95%CI. Pressure-to-fugacity handling remains aseparate sourceissue; noassumptionf=P isvalidated here.",
        "DREIDING+DDEC may be a predeclared computational reference, not a best-validated forcefield. Forcefield/charge variants andtemperatures do not createindependentmaterial identities. Retainexperimental vsGCMC strata.",
        "SAMPLE SELECTION", sample_meta.select("curve_id", "gas", "force_field", "charge", "temperature_K", "audit_selection_sha256").write_csv(),
        "RAW SAMPLE EVIDENCE", json.dumps([sample_records[cid] for cid in sorted(selected)], indent=2),
        "FROZEN DATA SHA256", json.dumps(before, indent=2),
        "AUDIT SCRIPT SHA256 "+sha(Path(__file__)), f"Elapsed seconds {time.monotonic()-started:.3f}"]
    (ROOT / "qa/crafted_ingestion_audit.txt").write_text("\n\n".join(report)+"\n", encoding="utf8")
    print("PASS", checked_pairs, "rawfiles", nonfinite_h, "nonfiniteH", flush=True)


if __name__ == "__main__":
    main()
