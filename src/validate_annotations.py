"""Phase 2: schema validation for an annotator's script-tag file.

Enforces the domains and rules fixed in data/annotations/protocol.md.

Run: python src/validate_annotations.py <path-to-annotator-csv>
Exit 0 = valid, 1 = invalid (STOP and report).
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UNITS = REPO / "data" / "annotation_units.csv"

SCRIPTS = {"urdu", "english", "digit"}
GLYPHS = {"western", "eastern_arabic_indic", "na", "unclear"}
COLS = ["unit_id", "script", "digit_glyph", "ambiguous", "note"]


def main(path: Path) -> int:
    path = path.resolve()
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    unit_ids = {int(u["unit_id"])
                for u in csv.DictReader(UNITS.open(encoding="utf-8"))}
    errs: list[str] = []

    if rows and list(rows[0].keys()) != COLS:
        errs.append(f"columns are {list(rows[0].keys())}, expected {COLS}")

    seen: set[int] = set()
    for i, r in enumerate(rows, 2):
        uid = int(r["unit_id"])
        if uid in seen:
            errs.append(f"line {i}: duplicate unit_id {uid}")
        seen.add(uid)
        if uid not in unit_ids:
            errs.append(f"line {i}: unit_id {uid} not in annotation_units.csv")
        if r["script"] not in SCRIPTS:
            errs.append(f"line {i}: script '{r['script']}' not in {SCRIPTS}")
        if r["digit_glyph"] not in GLYPHS:
            errs.append(f"line {i}: digit_glyph '{r['digit_glyph']}' invalid")
        if r["ambiguous"] not in {"0", "1"}:
            errs.append(f"line {i}: ambiguous must be 0 or 1")
        # protocol rule: digit_glyph mandatory when script=digit
        if r["script"] == "digit" and r["digit_glyph"] in {"na", ""}:
            errs.append(f"line {i}: script=digit requires a real digit_glyph")
        if r["script"] != "digit" and r["digit_glyph"] != "na":
            errs.append(f"line {i}: digit_glyph must be 'na' when script!=digit")
        # protocol rule: ambiguous=1 requires a note
        if r["ambiguous"] == "1" and not r["note"].strip():
            errs.append(f"line {i}: ambiguous=1 requires a note")

    missing = unit_ids - seen
    if missing:
        errs.append(f"{len(missing)} units unannotated: "
                    f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")

    print(f"file      : {path.relative_to(REPO)}")
    print(f"rows      : {len(rows)}  (expected {len(unit_ids)})")
    print(f"script    : {dict(Counter(r['script'] for r in rows))}")
    print(f"glyph     : {dict(Counter(r['digit_glyph'] for r in rows))}")
    print(f"ambiguous : {sum(int(r['ambiguous']) for r in rows)}")
    if errs:
        print(f"\n{len(errs)} SCHEMA ERROR(S):")
        for e in errs[:40]:
            print(f"  - {e}")
        print("\nRESULT: INVALID")
        return 1
    print("\nRESULT: VALID")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python src/validate_annotations.py <csv>")
    raise SystemExit(main(Path(sys.argv[1])))
