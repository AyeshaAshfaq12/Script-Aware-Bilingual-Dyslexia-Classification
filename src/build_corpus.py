"""Phase 1 addendum: define the experiment corpora (resolves D-004).

Authors' decision, 2026-09-01:
  PRIMARY     = deduplicated corpus, one representative per distinct
                image, with the 20 cross-class (contradictory) images
                removed entirely.            -> expected 618 images
  SENSITIVITY = the released corpus as-is.   -> 852 files

Annotation unit is the distinct image (sha256), not the file: identical
bytes get one script tag, propagated to every file carrying them. That
makes Phase 2 a 638-image job instead of 852 and guarantees duplicate
files can never receive inconsistent tags.

Representative selection is deterministic: lexicographically smallest
uid within the sha256 group. No randomness, no seed consumed.

Outputs:
  data/corpus_v1.csv        852 rows, one per released file, with flags
  data/annotation_units.csv 638 rows, one per distinct image (Phase 2)
  data/corpus_v1_summary.json

Run: python src/build_corpus.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

EXPECTED_FILES = 852
EXPECTED_DISTINCT = 638
EXPECTED_PRIMARY = 618
EXPECTED_PRIMARY_YES = 406
EXPECTED_PRIMARY_NO = 212


def main() -> int:
    rows = list(csv.DictReader((DATA / "raw" / "checksums.csv")
                               .open(encoding="utf-8")))
    assert len(rows) == EXPECTED_FILES, f"expected 852 files, got {len(rows)}"

    by_hash: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_hash[r["sha256"]].append(r)
    assert len(by_hash) == EXPECTED_DISTINCT, \
        f"expected 638 distinct, got {len(by_hash)}"

    # stable group ids, ordered by the group's smallest uid
    order = sorted(by_hash, key=lambda h: min(r["uid"] for r in by_hash[h]))
    gid_of = {h: i for i, h in enumerate(order, 1)}

    units: list[dict] = []
    for h in order:
        grp = by_hash[h]
        folders = {r["class_folder"] for r in grp}
        cross = len(folders) > 1
        rep = min(grp, key=lambda r: r["uid"])
        units.append({
            "unit_id": gid_of[h],
            "sha256": h,
            "representative_uid": rep["uid"],
            "n_files": len(grp),
            "class_folders": "|".join(sorted(folders)),
            "cross_class": int(cross),
            # label is undefined for cross-class units
            "label": "" if cross else ("1" if rep["class_folder"] == "Yes" else "0"),
            "in_primary": int(not cross),
            "width": rep["width"],
            "height": rep["height"],
        })

    unit_by_hash = {u["sha256"]: u for u in units}

    corpus: list[dict] = []
    for r in sorted(rows, key=lambda r: r["uid"]):
        u = unit_by_hash[r["sha256"]]
        corpus.append({
            "uid": r["uid"],
            "filename": r["filename"],
            "class_folder": r["class_folder"],
            "label": r["label"],
            "sha256": r["sha256"],
            "unit_id": u["unit_id"],
            "dup_group_size": u["n_files"],
            "is_duplicate_file": int(u["n_files"] > 1),
            "cross_class_unit": u["cross_class"],
            "representative_uid": u["representative_uid"],
            "is_representative": int(r["uid"] == u["representative_uid"]),
            "in_primary": int(u["in_primary"] and
                              r["uid"] == u["representative_uid"]),
            "in_sensitivity": 1,
            "width": r["width"],
            "height": r["height"],
        })

    primary = [c for c in corpus if c["in_primary"]]
    pc = Counter(c["class_folder"] for c in primary)

    # --- integrity assertions (fail loudly, never silently continue) ---
    assert len(primary) == EXPECTED_PRIMARY, \
        f"primary corpus expected 618, got {len(primary)}"
    assert pc["Yes"] == EXPECTED_PRIMARY_YES, f"primary Yes {pc['Yes']}"
    assert pc["No"] == EXPECTED_PRIMARY_NO, f"primary No {pc['No']}"
    assert len({c["sha256"] for c in primary}) == len(primary), \
        "primary corpus contains duplicate content"
    assert not any(c["cross_class_unit"] for c in primary), \
        "primary corpus contains a contradictory image"
    assert sum(c["in_sensitivity"] for c in corpus) == EXPECTED_FILES
    assert len(units) == EXPECTED_DISTINCT

    def dump(path: Path, recs: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
            wr.writeheader()
            wr.writerows(recs)

    dump(DATA / "corpus_v1.csv", corpus)
    dump(DATA / "annotation_units.csv", units)

    summary = {
        "decision": "D-004 resolved: primary=deduplicated 618, "
                    "sensitivity=released 852",
        "released_files": len(corpus),
        "distinct_images": len(units),
        "annotation_units": len(units),
        "cross_class_units_excluded": sum(u["cross_class"] for u in units),
        "primary": {
            "n": len(primary),
            "yes_dyslexic": pc["Yes"],
            "no_nondyslexic": pc["No"],
            "class_ratio_yes_to_no": round(pc["Yes"] / pc["No"], 4),
        },
        "sensitivity": {
            "n": len(corpus),
            "yes": sum(1 for c in corpus if c["class_folder"] == "Yes"),
            "no": sum(1 for c in corpus if c["class_folder"] == "No"),
        },
        "representative_rule": "lexicographically smallest uid in sha256 group",
    }
    (DATA / "corpus_v1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nall integrity assertions passed")
    for p in ("corpus_v1.csv", "annotation_units.csv", "corpus_v1_summary.json"):
        print(f"wrote data/{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
