"""Phase 3: the fixed train/val/test split (guide section 4).

Generated ONCE with global seed 1337 and never regenerated. Every arm
loads data/splits/split_v1.json.

Design decision that follows from DEVIATIONS.md D-004: the split is
drawn at the level of the **annotation unit** (the distinct image),
not the file. Assigning a unit assigns every byte-identical copy of it
to the same partition, which is what makes the sensitivity corpus
(852 files, 214 of them duplicates) leakage-free. A file-level draw
would put pixel-identical images on both sides of train/test.

  PRIMARY split      618 deduplicated images, stratified jointly on
                     (class x script), 80/10/10. Carries the co-primary
                     endpoints.
  SENSITIVITY split  the same unit -> partition assignment expanded to
                     all 852 released files, plus the 20 cross-class
                     (contradictory-label) units. Those 20 get a plain
                     seeded 80/10/10 shuffle: their class is undefined
                     and their script mix is 19 digit / 1 urdu, so no
                     stratified draw is possible. All copies of a
                     cross-class unit land in one partition, so the
                     contradiction is contained and never straddles a
                     boundary.

Run: python src/make_split.py          (refuses to overwrite)
     python src/make_split.py --force  (only with an explicit deviation)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = DATA / "splits" / "split_v1.json"

SEED = 1337
TEST_FRAC = 0.10
VAL_FRAC = 0.10
PARTS = ["train", "val", "test"]


def stratified_80_10_10(keys: list, strata: list[str], seed: int):
    """80/10/10 stratified split. Two calls, both seeded with `seed`."""
    train, rest, _, rest_str = train_test_split(
        keys, strata, test_size=VAL_FRAC + TEST_FRAC,
        random_state=seed, stratify=strata, shuffle=True)
    val, test = train_test_split(
        rest, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        random_state=seed, stratify=rest_str, shuffle=True)
    return sorted(train), sorted(val), sorted(test)


def counts_block(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "by_class": dict(sorted(Counter(r["class_folder"]
                                        for r in rows).items())),
        "by_script": dict(sorted(Counter(r["script"] for r in rows).items())),
        "by_class_script": {f"{c}|{s}": n for (c, s), n in
                            sorted(Counter((r["class_folder"], r["script"])
                                           for r in rows).items())},
    }


def frac(rows: list[dict], field: str) -> dict[str, float]:
    c = Counter(r[field] for r in rows)
    return {k: v / len(rows) for k, v in c.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing split (logged deviation only)")
    a = ap.parse_args()

    if OUT.exists() and not a.force:
        raise SystemExit(
            f"{OUT} already exists. The split is generated ONCE and never "
            f"regenerated (guide section 4). Use --force only with an "
            f"entry in DEVIATIONS.md.")

    rows = list(csv.DictReader((DATA / "corpus_v1_scripts.csv")
                               .open(encoding="utf-8")))

    # ---------- primary: draw over units ----------
    prim_files = [r for r in rows if r["in_primary"] == "1"]
    assert len(prim_files) == 618, f"primary is {len(prim_files)}"
    unit_of = {r["unit_id"]: r for r in prim_files}
    units = sorted(unit_of, key=int)
    strata = [f"{unit_of[u]['class_folder']}|{unit_of[u]['script']}"
              for u in units]

    tr_u, va_u, te_u = stratified_80_10_10(units, strata, SEED)
    part_of_unit = {u: p for p, us in
                    zip(PARTS, [tr_u, va_u, te_u]) for u in us}

    primary = {p: sorted(unit_of[u]["uid"] for u in us)
               for p, us in zip(PARTS, [tr_u, va_u, te_u])}
    prim_rows = {p: [unit_of[u] for u in us]
                 for p, us in zip(PARTS, [tr_u, va_u, te_u])}

    # ---------- sensitivity: expand units to all files ----------
    files_by_unit: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        files_by_unit[r["unit_id"]].append(r)

    cross_units = sorted({r["unit_id"] for r in rows
                          if r["cross_class_unit"] == "1"}, key=int)
    assert len(cross_units) == 20, f"{len(cross_units)} cross-class units"

    # Cross-class units have no defined class label, and their script mix
    # is 19 digit / 1 urdu, so a stratified draw is impossible (the urdu
    # stratum is a singleton). They get a plain seeded 80/10/10 shuffle.
    # They are excluded from the primary corpus and therefore never touch
    # the co-primary endpoints; this affects the sensitivity arm only.
    cx_shuffled = list(cross_units)
    random.Random(SEED).shuffle(cx_shuffled)
    n_cx = len(cx_shuffled)
    n_te = round(n_cx * TEST_FRAC)
    n_va = round(n_cx * VAL_FRAC)
    cx_te = sorted(cx_shuffled[:n_te])
    cx_va = sorted(cx_shuffled[n_te:n_te + n_va])
    cx_tr = sorted(cx_shuffled[n_te + n_va:])
    for p, us in zip(PARTS, [cx_tr, cx_va, cx_te]):
        for u in us:
            part_of_unit[u] = p

    sens_rows: dict[str, list[dict]] = {p: [] for p in PARTS}
    for u, fs in files_by_unit.items():
        sens_rows[part_of_unit[u]].extend(fs)
    sensitivity = {p: sorted(r["uid"] for r in sens_rows[p]) for p in PARTS}

    # ---------- assertions (guide section 4) ----------
    def check(name: str, parts: dict[str, list[str]], total: int,
              rowsets: dict[str, list[dict]], all_rows: list[dict],
              tol: float = 0.05) -> None:
        sets = {p: set(v) for p, v in parts.items()}
        for x in PARTS:
            for y in PARTS:
                if x < y:
                    assert not (sets[x] & sets[y]), \
                        f"{name}: {x} and {y} overlap"
        union = set().union(*sets.values())
        assert len(union) == total, \
            f"{name}: union is {len(union)}, expected {total}"
        assert sum(len(v) for v in parts.values()) == total, \
            f"{name}: partition sizes do not sum to {total}"
        # class balance and script mix preserved within rounding
        for field in ("class_folder", "script"):
            overall = frac(all_rows, field)
            for p in PARTS:
                got = frac(rowsets[p], field)
                for k, v in overall.items():
                    assert abs(got.get(k, 0.0) - v) <= tol, (
                        f"{name}/{p}: {field}={k} is {got.get(k,0):.3f}, "
                        f"corpus {v:.3f}, tol {tol}")
        # no duplicate group straddles a partition
        seen: dict[str, str] = {}
        for p in PARTS:
            for r in rowsets[p]:
                prev = seen.setdefault(r["unit_id"], p)
                assert prev == p, \
                    f"{name}: unit {r['unit_id']} spans {prev} and {p}"

    check("primary", primary, 618, prim_rows, prim_files)
    # The sensitivity corpus carries 48 files from 20 unstratified
    # cross-class units on top of the stratified draw, so its per-partition
    # mix drifts slightly further from the corpus proportions.
    check("sensitivity", sensitivity, 852, sens_rows, rows, tol=0.08)

    # ---------- write ----------
    payload = {
        "seed": SEED,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "src/make_split.py",
        "corpus_table": "data/corpus_v1_scripts.csv",
        "record_key": "uid = '<class_folder>/<filename>'",
        "draw_level": "annotation unit (distinct image, sha256)",
        "note": ("Drawn at unit level so that every byte-identical copy of "
                 "an image lands in one partition. See DEVIATIONS.md D-004."),
        "primary": {
            "description": "618 deduplicated images, 406 dyslexic / 212 "
                           "non-dyslexic; carries the co-primary endpoints",
            "stratified_on": "class x script",
            "counts": {p: counts_block(prim_rows[p]) for p in PARTS},
        },
        "sensitivity": {
            "description": "all 852 released files, incl. 214 duplicate "
                           "copies and 20 cross-class units; secondary "
                           "analysis only",
            "stratified_on": "class x script for the 618 primary units; "
                             "the 20 cross-class units get a plain seeded "
                             "shuffle (their class is undefined and their "
                             "script mix is 19 digit / 1 urdu, so no "
                             "stratified draw is possible)",
            "counts": {p: counts_block(sens_rows[p]) for p in PARTS},
        },
        # guide section 4 top-level shape: the primary split
        "train": primary["train"],
        "val": primary["val"],
        "test": primary["test"],
        "train_units": [int(u) for u in tr_u],
        "val_units": [int(u) for u in va_u],
        "test_units": [int(u) for u in te_u],
        "sensitivity_files": sensitivity,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote {OUT}  (seed {SEED})")
    for name, blocks, rs in (("PRIMARY", payload["primary"]["counts"],
                              prim_rows),
                             ("SENSITIVITY", payload["sensitivity"]["counts"],
                              sens_rows)):
        print(f"\n{name}")
        for p in PARTS:
            b = blocks[p]
            print(f"  {p:5s} n={b['n']:4d}  class={b['by_class']}  "
                  f"script={b['by_script']}")
    print("\nall split assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
