"""Phase 2 -> Phase 3 bridge: attach final script tags to every file.

Joins data/corpus_v1.csv (852 released files, with corpus membership
flags) to data/annotations/scripts_final.csv (638 adjudicated units).

Because the annotation unit is the distinct image, every duplicate file
inherits its unit's tag, so byte-identical files are guaranteed the same
script tag.

Output: data/corpus_v1_scripts.csv  -- the single table every downstream
stage reads (split, data loading, per-script evaluation).

Run: python src/propagate_tags.py
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def main() -> int:
    corpus = list(csv.DictReader((DATA / "corpus_v1.csv")
                                 .open(encoding="utf-8")))
    tags = {int(r["unit_id"]): r
            for r in csv.DictReader((DATA / "annotations" /
                                     "scripts_final.csv")
                                    .open(encoding="utf-8"))}

    missing = {int(c["unit_id"]) for c in corpus} - set(tags)
    if missing:
        raise SystemExit(f"{len(missing)} units have no final tag: "
                         f"{sorted(missing)[:10]}")

    out = []
    for c in corpus:
        t = tags[int(c["unit_id"])]
        out.append({**c,
                    "script": t["script"],
                    "digit_glyph": t["digit_glyph"],
                    "script_ambiguous": t["ambiguous"],
                    "script_source": t["source"]})

    # every duplicate group must carry one and only one script tag
    by_unit: dict[str, set[str]] = {}
    for r in out:
        by_unit.setdefault(r["unit_id"], set()).add(r["script"])
    bad = {u: s for u, s in by_unit.items() if len(s) > 1}
    if bad:
        raise SystemExit(f"inconsistent tags within a unit: {bad}")

    path = DATA / "corpus_v1_scripts.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    prim = [r for r in out if r["in_primary"] == "1"]
    assert len(prim) == 618, f"primary corpus is {len(prim)}, expected 618"

    print(f"wrote {path}  ({len(out)} file rows)")
    print(f"\nPRIMARY corpus ({len(prim)} images)")
    print(f"  by class : {dict(Counter(r['class_folder'] for r in prim))}")
    print(f"  by script: {dict(Counter(r['script'] for r in prim))}")
    print("\n  joint (class x script) strata:")
    joint = Counter((r["class_folder"], r["script"]) for r in prim)
    for (cls, scr), n in sorted(joint.items()):
        print(f"    {cls:4s} x {scr:8s}: {n:4d}   "
              f"(80/10/10 -> {round(n*0.8)}/{round(n*0.1)}/{round(n*0.1)})")
    print(f"\n  smallest stratum: {min(joint.values())}")
    print(f"\nSENSITIVITY corpus ({len(out)} files)")
    print(f"  by class : {dict(Counter(r['class_folder'] for r in out))}")
    print(f"  by script: {dict(Counter(r['script'] for r in out))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
