"""Government pipe-survey ingestion pipeline.

Accepts a CSV of field-surveyed pipe dimensions, validates the rows,
shows a human-readable diff against the current pune_base.inp, writes a
versioned backup of the old .inp, patches the [XSECTIONS] and [CONDUITS]
roughness columns in-place, and optionally triggers the recalibration
pipeline so the GNN training data stays consistent with real asset data.

CSV format expected from survey teams (one row per pipe):
    pipe_id, diameter_mm, shape, material, condition, survey_date, surveyor

Only `pipe_id` and `diameter_mm` are mandatory. All other columns are
optional but are stored in the audit log so the change is traceable.

Material → Manning's n mapping follows IS:7916 / standard drainage
practice.  Condition downgrades n to model sediment/biofilm build-up.

Run as a script:
    python dimension_ingest.py survey_2026_09.csv [--dry-run] [--no-recal]

Or import and call `ingest_survey(path)` from the FastAPI admin endpoint.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
INP_PATH = BASE_DIR / "pune_base.inp"
VERSIONS_DIR = BASE_DIR / "data" / "inp_versions"
AUDIT_LOG = BASE_DIR / "data" / "dimension_audit.log"

# ---------------------------------------------------------------------------
# Material -> Manning's n  (IS:7916 Table 4 / standard concrete-drain values)
# ---------------------------------------------------------------------------
MATERIAL_MANNINGS: dict[str, float] = {
    "concrete":  0.013,
    "rcc":       0.013,   # reinforced cement concrete
    "pvc":       0.009,
    "hdpe":      0.010,
    "brick":     0.015,
    "stone":     0.017,
    "earth":     0.025,
    "unknown":   0.013,   # default: smooth concrete
}

# Condition multipliers applied to Manning's n (worse condition = more roughness)
CONDITION_MULTIPLIERS: dict[str, float] = {
    "good":      1.00,
    "fair":      1.15,    # ~15% rougher — typical sediment build-up
    "poor":      1.30,    # ~30% rougher — significant blockage / deformation
    "critical":  1.50,    # severely degraded; flag for maintenance
    "unknown":   1.10,    # conservative default
}

MIN_DIAMETER_MM = 150    # smallest realistic urban sewer
MAX_DIAMETER_MM = 3000   # largest trunk sewer in Indian cities


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

class SurveyRow:
    """One validated row from the survey CSV."""
    __slots__ = ("pipe_id", "diameter_m", "shape", "material", "condition",
                 "survey_date", "surveyor", "mannings_n")

    def __init__(
        self,
        pipe_id: str,
        diameter_mm: float,
        shape: str = "CIRCULAR",
        material: str = "unknown",
        condition: str = "unknown",
        survey_date: str = "",
        surveyor: str = "",
    ):
        self.pipe_id     = pipe_id.strip().upper()
        self.diameter_m  = diameter_mm / 1000.0
        self.shape       = shape.strip().upper() or "CIRCULAR"
        self.material    = material.strip().lower() or "unknown"
        self.condition   = condition.strip().lower() or "unknown"
        self.survey_date = survey_date.strip()
        self.surveyor    = surveyor.strip()

        mat_n      = MATERIAL_MANNINGS.get(self.material, MATERIAL_MANNINGS["unknown"])
        cond_mult  = CONDITION_MULTIPLIERS.get(self.condition, CONDITION_MULTIPLIERS["unknown"])
        self.mannings_n = round(mat_n * cond_mult, 4)


def parse_csv(path: str | Path) -> tuple[list[SurveyRow], list[str]]:
    """Return (valid_rows, error_messages).

    Rows that fail validation are collected as errors rather than crashing,
    so a partially-valid upload does not discard the good rows.
    """
    rows: list[SurveyRow] = []
    errors: list[str] = []

    with open(path, newline="", encoding="utf-8-sig") as f:   # utf-8-sig strips Excel BOM
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            errors.append("CSV appears to be empty or has no header row.")
            return rows, errors

        for line_num, raw in enumerate(reader, start=2):      # 1 = header row
            row = {k.strip().lower(): v.strip() for k, v in raw.items()}

            pipe_id = row.get("pipe_id", "").strip()
            if not pipe_id:
                errors.append(f"Line {line_num}: missing pipe_id — row skipped.")
                continue

            raw_diam = row.get("diameter_mm", "").replace(",", ".")
            if not raw_diam:
                errors.append(f"Line {line_num} ({pipe_id}): missing diameter_mm — row skipped.")
                continue
            try:
                diameter_mm = float(raw_diam)
            except ValueError:
                errors.append(
                    f"Line {line_num} ({pipe_id}): diameter_mm '{raw_diam}' is not a number — skipped."
                )
                continue

            if not (MIN_DIAMETER_MM <= diameter_mm <= MAX_DIAMETER_MM):
                errors.append(
                    f"Line {line_num} ({pipe_id}): diameter_mm={diameter_mm} outside "
                    f"plausible range [{MIN_DIAMETER_MM}, {MAX_DIAMETER_MM}] — skipped."
                )
                continue

            rows.append(SurveyRow(
                pipe_id=pipe_id,
                diameter_mm=diameter_mm,
                shape=row.get("shape", "CIRCULAR"),
                material=row.get("material", "unknown"),
                condition=row.get("condition", "unknown"),
                survey_date=row.get("survey_date", ""),
                surveyor=row.get("surveyor", ""),
            ))

    return rows, errors


# ---------------------------------------------------------------------------
# Diff against current .inp
# ---------------------------------------------------------------------------

def _current_xsections(inp_path: Path) -> dict[str, dict]:
    """Parse [XSECTIONS] from .inp -> {pipe_id: {shape, diameter_m}}."""
    result: dict[str, dict] = {}
    in_section = False
    with open(inp_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("["):
                in_section = stripped.strip("[]") == "XSECTIONS"
                continue
            if not in_section or not stripped or stripped.startswith(";"):
                continue
            parts = stripped.split()
            if len(parts) >= 3:
                result[parts[0]] = {"shape": parts[1], "diameter_m": float(parts[2])}
    return result


def _current_roughness(inp_path: Path) -> dict[str, float]:
    """Parse [CONDUITS] roughness column -> {pipe_id: mannings_n}."""
    result: dict[str, float] = {}
    in_section = False
    with open(inp_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("["):
                in_section = stripped.strip("[]") == "CONDUITS"
                continue
            if not in_section or not stripped or stripped.startswith(";"):
                continue
            parts = stripped.split()
            # CONDUITS: Name  Node1  Node2  Length  Roughness ...
            if len(parts) >= 5:
                result[parts[0]] = float(parts[4])
    return result


def build_diff(
    rows: list[SurveyRow],
    inp_path: Path,
) -> tuple[list[dict], list[str]]:
    """Return (changes, unknown_pipe_warnings).

    changes: list of {pipe_id, field, old, new} dicts describing what will change.
    unknown_pipe_warnings: pipe_ids in the CSV not found in the current .inp.
    """
    current_xs  = _current_xsections(inp_path)
    current_rgh = _current_roughness(inp_path)
    changes: list[dict] = []
    warnings: list[str] = []

    for row in rows:
        if row.pipe_id not in current_xs:
            warnings.append(
                f"  WARNING  {row.pipe_id} not found in {inp_path.name}; "
                "skipped (new pipes must be added via generate_network.py)."
            )
            continue

        old_diam = current_xs[row.pipe_id]["diameter_m"]
        if abs(old_diam - row.diameter_m) > 1e-4:
            changes.append({
                "pipe_id":  row.pipe_id,
                "field":    "diameter_m",
                "old":      f"{old_diam:.3f}",
                "new":      f"{row.diameter_m:.3f}",
                "surveyor": row.surveyor,
                "date":     row.survey_date,
            })

        old_n = current_rgh.get(row.pipe_id, 0.013)
        if abs(old_n - row.mannings_n) > 1e-5:
            changes.append({
                "pipe_id":  row.pipe_id,
                "field":    "mannings_n",
                "old":      f"{old_n:.4f}",
                "new":      f"{row.mannings_n:.4f}",
                "surveyor": row.surveyor,
                "date":     row.survey_date,
            })

    return changes, warnings


def print_diff(changes: list[dict], warnings: list[str]) -> None:
    if warnings:
        print("\nWarnings (pipes not in current network — skipped):")
        for w in warnings:
            print(w)

    if not changes:
        print("\n[OK] No differences — current .inp already matches the survey data.")
        return

    print(f"\nProposed changes ({len(changes)}):")
    print(f"  {'Pipe':<20} {'Field':<14} {'Old':>10}  ->  {'New':<10}  Surveyor")
    print("  " + "-" * 72)
    for c in changes:
        print(
            f"  {c['pipe_id']:<20} {c['field']:<14} "
            f"{c['old']:>10}  ->  {c['new']:<10}  {c['surveyor']} ({c['date']})"
        )


# ---------------------------------------------------------------------------
# Write patched .inp  (versioned backup first)
# ---------------------------------------------------------------------------

def _backup_inp(inp_path: Path) -> Path:
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = VERSIONS_DIR / f"pune_base_{stamp}.inp"
    shutil.copy2(inp_path, dest)
    return dest


def _patch_inp(rows: list[SurveyRow], inp_path: Path) -> None:
    """Rewrite [XSECTIONS] diameters and [CONDUITS] roughness in-place."""
    survey_map  = {r.pipe_id: r for r in rows}
    current_xs  = _current_xsections(inp_path)

    with open(inp_path) as f:
        lines = f.readlines()

    out: list[str] = []
    section: Optional[str] = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped.strip("[]")
            out.append(line)
            continue

        if section == "XSECTIONS" and stripped and not stripped.startswith(";"):
            parts = stripped.split()
            pipe_id = parts[0]
            if pipe_id in survey_map and pipe_id in current_xs:
                r = survey_map[pipe_id]
                out.append(
                    f"{pipe_id:<16} {r.shape:<12} {r.diameter_m:<10.3f}"
                    f" 0          0          0          1\n"
                )
                continue

        if section == "CONDUITS" and stripped and not stripped.startswith(";"):
            parts = stripped.split()
            pipe_id = parts[0]
            if pipe_id in survey_map and len(parts) >= 5:
                r = survey_map[pipe_id]
                # CONDUITS: Name Node1 Node2 Length Roughness InOffset OutOffset InitFlow MaxFlow
                parts[4] = f"{r.mannings_n:.4f}"
                out.append("  ".join(parts) + "\n")
                continue

        out.append(line)

    with open(inp_path, "w") as f:
        f.writelines(out)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def _write_audit(rows: list[SurveyRow], changes: list[dict], csv_path: str | Path) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    with open(AUDIT_LOG, "a") as f:
        f.write(f"\n[{stamp}] Ingest from: {csv_path}\n")
        f.write(f"  Rows parsed: {len(rows)}, Changes applied: {len(changes)}\n")
        for c in changes:
            f.write(
                f"  {c['pipe_id']}  {c['field']}  {c['old']} -> {c['new']}"
                f"  surveyor={c['surveyor']}  date={c['date']}\n"
            )


# ---------------------------------------------------------------------------
# Recalibration trigger
# ---------------------------------------------------------------------------

def _trigger_recalibration() -> None:
    """Re-run recalibrate_pipes.py so Manning's equation diameters stay
    consistent with any pipe-dimension updates.  Called after patching."""
    import importlib
    try:
        recal = importlib.import_module("recalibrate_pipes")
        diameters = recal.recalibrate()
        recal.rewrite_xsections(diameters)
        logger.info("recalibrate_pipes: updated %d pipe diameters.", len(diameters))
    except Exception:
        logger.exception(
            "recalibrate_pipes failed after dimension update — manual review needed."
        )


# ---------------------------------------------------------------------------
# Public API  (called by FastAPI admin endpoint or directly)
# ---------------------------------------------------------------------------

def ingest_survey(
    csv_path: str | Path,
    inp_path: Path = INP_PATH,
    dry_run: bool = False,
    run_recalibration: bool = True,
) -> dict:
    """Parse a survey CSV, diff against the current .inp, optionally patch it.

    Returns:
        {
            "rows_parsed": int,
            "changes": list[dict],
            "warnings": list[str],
            "errors": list[str],
            "backup_path": str | None,
            "dry_run": bool,
        }
    """
    rows, errors = parse_csv(csv_path)
    changes, warnings = build_diff(rows, inp_path)

    result: dict = {
        "rows_parsed": len(rows),
        "changes": changes,
        "warnings": warnings,
        "errors": errors,
        "backup_path": None,
        "dry_run": dry_run,
    }

    if not dry_run and changes:
        backup = _backup_inp(inp_path)
        result["backup_path"] = str(backup)
        _patch_inp(rows, inp_path)
        _write_audit(rows, changes, csv_path)
        logger.info(
            "Dimension update applied: %d changes, backup at %s", len(changes), backup
        )
        if run_recalibration:
            _trigger_recalibration()

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Ingest a pipe survey CSV and update pune_base.inp."
    )
    parser.add_argument("csv", help="Path to the survey CSV file.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the diff without writing any changes.",
    )
    parser.add_argument(
        "--no-recal", action="store_true",
        help="Skip recalibrate_pipes.py after patching.",
    )
    args = parser.parse_args()

    result = ingest_survey(
        args.csv,
        dry_run=args.dry_run,
        run_recalibration=not args.no_recal,
    )

    if result["errors"]:
        print(f"\nValidation errors ({len(result['errors'])}):")
        for e in result["errors"]:
            print(f"  x  {e}")

    print_diff(result["changes"], result["warnings"])

    if args.dry_run:
        print("\n[dry-run] No changes written.")
    elif result["changes"]:
        print(f"\n[OK] {len(result['changes'])} change(s) applied.")
        print(f"     Backup: {result['backup_path']}")
        print(f"     Audit:  {AUDIT_LOG}")
    else:
        print("\n[OK] Nothing to update.")
