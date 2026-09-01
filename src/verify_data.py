"""Phase 1: dataset acquisition verification.

Generates data/raw/manifest.txt and data/raw/checksums.csv, then runs
the guide's Phase 1 validation gate (852 images, 426 YES / 426 NO).

Run:  python src/verify_data.py
Exit code 0 = gate passed, 1 = gate failed (STOP and report).
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
CLASS_DIRS = ["Yes", "No"]          # as released; Yes -> dyslexic (y=1)
EXPECTED_TOTAL = 852
EXPECTED_PER_CLASS = 426


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def collect() -> list[dict]:
    rows: list[dict] = []
    for cls in CLASS_DIRS:
        d = RAW / cls
        if not d.is_dir():
            sys.exit(f"FAIL: missing class directory {d}")
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            with Image.open(p) as im:
                w, h = im.size
                mode, fmt = im.mode, im.format
            rows.append(
                {
                    "uid": f"{cls}/{p.name}",
                    "filename": p.name,
                    "class_folder": cls,
                    "label": 1 if cls == "Yes" else 0,
                    "sha256": sha256_of(p),
                    "bytes": p.stat().st_size,
                    "width": w,
                    "height": h,
                    "mode": mode,
                    "format": fmt,
                }
            )
    return rows


def write_manifest(rows: list[dict]) -> None:
    lines = ["data/raw:"] + [f"  {c}/" for c in CLASS_DIRS] + [""]
    for cls in CLASS_DIRS:
        names = [r["filename"] for r in rows if r["class_folder"] == cls]
        lines.append(f"data/raw/{cls}:  ({len(names)} files)")
        lines.extend(f"  {n}" for n in names)
        lines.append("")
    (RAW / "manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def write_checksums(rows: list[dict]) -> None:
    cols = ["uid", "filename", "class_folder", "label", "sha256",
            "bytes", "width", "height", "mode", "format"]
    with (RAW / "checksums.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)


def gate(rows: list[dict]) -> bool:
    ok = True
    per_class = Counter(r["class_folder"] for r in rows)
    checks: list[tuple[str, bool, str]] = []

    checks.append(("total == 852", len(rows) == EXPECTED_TOTAL,
                   f"got {len(rows)}"))
    for cls in CLASS_DIRS:
        checks.append((f"{cls} == 426", per_class[cls] == EXPECTED_PER_CLASS,
                       f"got {per_class[cls]}"))

    uids = [r["uid"] for r in rows]
    checks.append(("uid unique", len(set(uids)) == len(uids),
                   f"{len(uids) - len(set(uids))} duplicate uids"))

    names = [r["filename"] for r in rows]
    dup_names = [n for n, c in Counter(names).items() if c > 1]
    checks.append(("filename unique across classes", not dup_names,
                   f"{len(dup_names)} names reused across Yes/No"))

    digests = [r["sha256"] for r in rows]
    dup_hash = [h for h, c in Counter(digests).items() if c > 1]
    checks.append(("no byte-identical duplicate images", not dup_hash,
                   f"{len(dup_hash)} sha256 collisions"))

    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:38s} {detail}")
        ok &= passed
    return ok


def main() -> int:
    print("Phase 1 verification")
    rows = collect()
    write_manifest(rows)
    write_checksums(rows)
    print(f"  wrote {RAW/'manifest.txt'}")
    print(f"  wrote {RAW/'checksums.csv'}")

    print("\nSummary")
    print(f"  images            : {len(rows)}")
    for cls in CLASS_DIRS:
        n = sum(1 for r in rows if r["class_folder"] == cls)
        print(f"  {cls:<18}: {n}")
    print(f"  formats           : {dict(Counter(r['format'] for r in rows))}")
    print(f"  modes             : {dict(Counter(r['mode'] for r in rows))}")
    sizes = Counter((r["width"], r["height"]) for r in rows)
    print(f"  distinct WxH      : {len(sizes)}  most common: {sizes.most_common(3)}")
    mb = sum(r["bytes"] for r in rows) / 1e6
    print(f"  total size        : {mb:.1f} MB")

    print("\nPhase 1 gate")
    passed = gate(rows)
    print(f"\nRESULT: {'PASS' if passed else 'FAIL - STOP AND REPORT'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
