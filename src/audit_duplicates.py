"""Phase 1 addendum: exact-duplicate audit of the released corpus.

Reads data/raw/checksums.csv and reports byte-identical duplicate
groups, distinguishing within-class duplication from cross-class
duplication (the same image bytes filed under both Yes and No, which
is a label contradiction).

Outputs:
  results/duplicate_groups.csv   one row per file in a duplicate group
  results/duplicate_audit.json   summary counts

Run: python src/audit_duplicates.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKSUMS = REPO / "data" / "raw" / "checksums.csv"
RESULTS = REPO / "results"


def main() -> int:
    rows = list(csv.DictReader(CHECKSUMS.open(encoding="utf-8")))
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_hash[r["sha256"]].append(r)

    groups = {h: g for h, g in by_hash.items() if len(g) > 1}
    cross = {h: g for h, g in groups.items()
             if len({r["class_folder"] for r in g}) > 1}
    within = {h: g for h, g in groups.items() if h not in cross}

    yes = {r["sha256"] for r in rows if r["class_folder"] == "Yes"}
    no = {r["sha256"] for r in rows if r["class_folder"] == "No"}

    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "duplicate_groups.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["group_id", "kind", "sha256", "uid", "class_folder",
                     "filename", "width", "height", "bytes"])
        for gid, (h, g) in enumerate(sorted(groups.items()), 1):
            kind = "cross_class" if h in cross else "within_class"
            for r in sorted(g, key=lambda r: r["uid"]):
                wr.writerow([gid, kind, h, r["uid"], r["class_folder"],
                             r["filename"], r["width"], r["height"],
                             r["bytes"]])

    summary = {
        "files_total": len(rows),
        "files_per_class": dict(Counter(r["class_folder"] for r in rows)),
        "distinct_images_by_sha256": len(by_hash),
        "redundant_copies": len(rows) - len(by_hash),
        "duplicate_groups": len(groups),
        "duplicate_group_size_hist": dict(
            sorted(Counter(len(g) for g in groups.values()).items())),
        "cross_class_groups": len(cross),
        "within_class_groups": len(within),
        "within_class_groups_by_folder": dict(Counter(
            g[0]["class_folder"] for g in within.values())),
        "distinct_content_yes_only": len(yes - no),
        "distinct_content_no_only": len(no - yes),
        "distinct_content_both_classes": len(yes & no),
        "distinct_per_class_folder": {
            cls: len({r["sha256"] for r in rows
                      if r["class_folder"] == cls})
            for cls in ("Yes", "No")},
        "deduplicated_corpus_excluding_contradictions": {
            "yes": len(yes - no),
            "no": len(no - yes),
            "total": len(yes - no) + len(no - yes),
        },
    }
    (RESULTS / "duplicate_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {RESULTS/'duplicate_groups.csv'}")
    print(f"wrote {RESULTS/'duplicate_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
