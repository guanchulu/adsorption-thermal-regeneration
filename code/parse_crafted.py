# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "polars"]
# ///
# Run: uv run --project .. --no-sync --offline python code/parse_crafted.py
import hashlib
import io
import json
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, TypedDict

import numpy as np
import polars as pl
from numpy.typing import NDArray

ROOT: Final = Path(__file__).resolve().parents[1]
ARCHIVE: Final = ROOT/"data/crafted_raw/CRAFTED-2.0.1.tar.xz"


class CraftedParseError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class CurveMetadata(TypedDict):
    curve_id: str
    structure_id: str
    force_field: str
    charge: str
    gas: str
    temperature_K: int


class CurveInventory(CurveMetadata):
    n_points: int
    pmin_Pa: float
    pmax_Pa: float
    finite_values: int
    negative_uptake_points: int
    max_uptake: float
    max_downstep_relative_peak: float
    source_member: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ParsedMember:
    rows: list[dict[str, str | int | float]]
    inventory: CurveInventory | None


def metadata(path: str) -> CurveMetadata:
    file = PurePosixPath(path).name
    prefix, force_field, gas, temperature = file.removesuffix(".csv").rsplit("_", 3)
    charge, structure = prefix.split("_", 1)
    if force_field not in {"UFF", "DREIDING"} or gas not in {"CO2", "N2"} or int(temperature) not in {273,298,323}:
        raise CraftedParseError(f"Unexpected CRAFTED file metadata: {file}")
    return {"curve_id": file.removesuffix(".csv"), "structure_id": structure,
            "force_field": force_field, "charge": charge, "gas": gas, "temperature_K": int(temperature)}


def select_member(member: tarfile.TarInfo) -> tuple[bool, int]:
    if not member.isfile():
        return False, 0
    path = PurePosixPath(member.name)
    if path.name.startswith("._"):
        return False, 1
    return path.suffix == ".csv", 0


def member_columns(path: PurePosixPath) -> tuple[str, str, str] | None:
    if "ISOTHERM_FILES" in path.parts:
        return "isotherm", "uptake_mol_kg", "uptake_reported_error"
    if "ENTHALPY_FILES" in path.parts:
        return "enthalpy", "enthalpy_kj_mol", "enthalpy_reported_error"
    return None


def inventory_record(meta: CurveMetadata, values: NDArray[np.float64], member_name: str, raw: bytes) -> CurveInventory:
    peak = float(np.nanmax(values[:,1]))
    return {**meta, "n_points": len(values), "pmin_Pa": float(np.nanmin(values[:,0])),
            "pmax_Pa": float(np.nanmax(values[:,0])), "finite_values": int(np.isfinite(values).all()),
            "negative_uptake_points": int(np.sum(values[:,1]<0)), "max_uptake": peak,
            "max_downstep_relative_peak": float(max(0., -np.min(np.diff(values[:,1])))/peak) if peak>0 else float("nan"),
            "source_member": member_name, "sha256": hashlib.sha256(raw).hexdigest()}


def read_member(archive: tarfile.TarFile, member: tarfile.TarInfo, columns: tuple[str, str, str]) -> ParsedMember:
    kind, value_name, sigma_name = columns
    stream = archive.extractfile(member)
    if stream is None:
        raise CraftedParseError(f"Cannot read {member.name}")
    with stream:
        raw = stream.read()
    text = raw.decode("utf-8-sig")
    header = text.splitlines()[0].lower()
    required = "mean_volume[mol/kg]" if kind == "isotherm" else "enthalpy_kj/mol"
    if required not in header:
        raise CraftedParseError(f"Unexpected units: {member.name}: {header}")
    values = np.loadtxt(io.StringIO(text), delimiter=",", ndmin=2)
    if values.shape[1] != 3:
        raise CraftedParseError(f"Unexpected column count: {member.name}")
    meta = metadata(member.name)
    point_metadata: dict[str, str | int | float] = {
        "curve_id": meta["curve_id"], "structure_id": meta["structure_id"],
        "force_field": meta["force_field"], "charge": meta["charge"],
        "gas": meta["gas"], "temperature_K": meta["temperature_K"]}
    rows: list[dict[str, str | int | float]] = [
        {**point_metadata, "pressure_Pa": float(pressure), value_name: float(value), sigma_name: float(error)}
        for pressure, value, error in values]
    record = inventory_record(meta, values, member.name, raw) if kind == "isotherm" else None
    return ParsedMember(rows, record)


def main() -> None:
    (ROOT / "data/crafted_parts").mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    rows: dict[str, list[dict[str, str | int | float]]] = {"isotherm": [], "enthalpy": []}
    parts = {"isotherm": 0, "enthalpy": 0}
    counts = {"isotherm": 0, "enthalpy": 0, "sidecars_skipped": 0}
    inventory: list[CurveInventory] = []

    def flush(kind: str) -> None:
        if rows[kind]:
            pl.DataFrame(rows[kind]).write_parquet(ROOT/f"data/crafted_parts/{kind}_{parts[kind]:03}.parquet")
            rows[kind].clear()
            parts[kind] += 1

    with tarfile.open(ARCHIVE, mode="r|xz") as archive:
        for member in archive:
            selected, sidecars_skipped = select_member(member)
            counts["sidecars_skipped"] += sidecars_skipped
            if not selected:
                continue
            columns = member_columns(PurePosixPath(member.name))
            if columns is None:
                continue
            kind = columns[0]
            parsed = read_member(archive, member, columns)
            rows[kind].extend(parsed.rows)
            if parsed.inventory is not None:
                inventory.append(parsed.inventory)
            counts[kind] += 1
            if len(rows[kind]) >= 50000:
                flush(kind)
            if counts[kind] % 20000 == 0:
                print(kind, counts[kind], "curves parsed", flush=True)
    for kind in rows:
        flush(kind)
    curves = pl.DataFrame(inventory)
    curves.write_parquet(ROOT/"data/crafted_curves.parquet")
    q = pl.scan_parquet(str(ROOT/"data/crafted_parts/isotherm_*.parquet"))
    h = pl.scan_parquet(str(ROOT/"data/crafted_parts/enthalpy_*.parquet")).select("curve_id", "pressure_Pa", "enthalpy_kj_mol", "enthalpy_reported_error")
    points = q.join(h, on=["curve_id", "pressure_Pa"], how="left", validate="1:1").with_columns(
        (pl.col("pressure_Pa")/100000).alias("pressure_bar"), (-pl.col("enthalpy_kj_mol")).alias("reported_minus_enthalpy_kj_mol"))
    joined = points.collect()
    joined.write_parquet(ROOT/"data/crafted_points.parquet")
    groups = curves.group_by("structure_id", "force_field", "charge", "gas").agg(
        pl.col("temperature_K").n_unique().alias("n_temperatures"), pl.col("n_points").sum().alias("n_points"))
    groups.write_parquet(ROOT/"data/crafted_temperature_groups.parquet")
    summary = {**counts, "point_records_with_uptake_and_enthalpy": joined.height,
               "source_structure_ids": curves["structure_id"].n_unique(), "temperature_groups": groups.height,
               "three_temperature_groups": groups.filter(pl.col("n_temperatures")==3).height,
               "missing_enthalpy_values": joined["enthalpy_kj_mol"].null_count(),
               "nonfinite_enthalpy_values": joined.filter(~pl.col("enthalpy_kj_mol").is_finite()).height,
               "low_pressure_records_le_1bar": joined.filter(pl.col("pressure_bar")<=1).height,
               "gas_counts": curves.group_by("gas").len().to_dicts(),
               "charge_counts": curves.group_by("charge").len().to_dicts(),
               "source":"https://doi.org/10.5281/zenodo.10120180", "version":"2.0.1",
               "evidence_type":"GCMC simulation; modeling variants are repeated conditions within each structure",
               "elapsed_seconds": time.monotonic()-start}
    (ROOT/"results/crafted_inventory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
