# Phase Record — Phase 3: Fixed split

- **Date:** 2026-09-02
- **Phase:** Phase 3 (Guide §4)
- **Status:** **PASS**

## Objective

One fixed 80/10/10 train/val/test partition, global seed **1337**,
stratified jointly on (class × script), generated once and never
regenerated. Every arm loads it unchanged.

## Actions performed

1. `src/propagate_tags.py` joined `corpus_v1.csv` (852 files) to
   `scripts_final.csv` (638 adjudicated units) into
   `data/corpus_v1_scripts.csv`, asserting that every file carries a
   tag and that no duplicate group holds two different tags.
2. `src/make_split.py` drew the split and wrote
   `data/splits/split_v1.json`.

## Key design decision: the draw is at unit level, not file level

The split is drawn over **annotation units** (distinct images), then
expanded to files. Assigning a unit assigns every byte-identical copy
of it to the same partition. A file-level draw would place
pixel-identical images on both sides of train/test — the leakage
identified in D-004. This is what makes the sensitivity corpus (852
files, 214 of them duplicates) usable at all.

Two splits are stored in one file, so neither is ever regenerated:

- **PRIMARY** — 618 deduplicated images, stratified jointly on
  (class × script). Carries the co-primary endpoints.
- **SENSITIVITY** — the same unit→partition assignment expanded to all
  852 files, plus the 20 cross-class units. Secondary analysis only.

**Cross-class units are not stratified.** Their class label is
undefined (they appear in both folders) and their script mix is 19
digit / 1 urdu, so the urdu stratum is a singleton and `sklearn`
cannot stratify it. They receive a plain seeded 80/10/10 shuffle
(seed 1337). They are excluded from the primary corpus and therefore
never touch the co-primary endpoints; this is a sensitivity-arm-only
departure from "stratify jointly", recorded here and in the docstring
and payload of `make_split.py` / `split_v1.json`.

## Configuration used

- Seed **1337** for the stratified draws and the cross-class shuffle.
- `sklearn.model_selection.train_test_split`, two calls: 80/20, then
  the 20% halved into val/test, both `random_state=1337`.
- Record key `uid = "<class_folder>/<filename>"`.

## Files created / updated

| Path | Description |
|---|---|
| `src/propagate_tags.py` | joins final tags to all 852 files |
| `data/corpus_v1_scripts.csv` | 852 rows, the table every stage reads |
| `src/make_split.py` | the split generator (refuses to overwrite) |
| `data/splits/split_v1.json` | **the fixed split — never regenerate** |

## Results

### PRIMARY (618)

| partition | n | No | Yes | urdu | english | digit |
|---|---|---|---|---|---|---|
| train | 494 | 170 | 324 | 335 | 65 | 94 |
| val | 62 | 21 | 41 | 41 | 9 | 12 |
| test | 62 | 21 | 41 | 43 | 8 | 11 |

Joint strata, train: `No|digit 41, No|english 35, No|urdu 94,
Yes|digit 53, Yes|english 30, Yes|urdu 241`.
Joint strata, test: `No|digit 5, No|english 4, No|urdu 12,
Yes|digit 6, Yes|english 4, Yes|urdu 31`.

### SENSITIVITY (852 files)

| partition | n | No | Yes | urdu | english | digit |
|---|---|---|---|---|---|---|
| train | 684 | 344 | 340 | 443 | 65 | 176 |
| val | 83 | 40 | 43 | 54 | 9 | 20 |
| test | 85 | 42 | 43 | 55 | 8 | 22 |

## Validation results

| Check | Primary | Sensitivity |
|---|---|---|
| partitions pairwise disjoint | **PASS** | **PASS** |
| union == corpus size | **PASS** (618) | **PASS** (852) |
| sizes sum to corpus size | **PASS** | **PASS** |
| class balance preserved | **PASS** (tol 0.05) | **PASS** (tol 0.08) |
| script mix preserved | **PASS** (tol 0.05) | **PASS** (tol 0.08) |
| **no duplicate group straddles partitions** | **PASS** | **PASS** |
| regeneration guarded | **PASS** (`--force` required) | — |

The wider tolerance on the sensitivity corpus is because 48 files from
20 unstratified cross-class units ride on top of the stratified draw.

## Issues encountered

1. **Cross-class stratification impossible** (19 digit / 1 urdu). Fixed
   with a seeded shuffle, scoped to the sensitivity arm, documented in
   three places. Caught by an assertion, not silently worked around.
2. No other issues. `make_split.py` refuses to overwrite an existing
   `split_v1.json` without `--force`, which the guide permits only with
   a logged deviation.

## Deviations

None beyond the cross-class shuffle described above, which is a
consequence of the already-accepted D-004 and affects no primary
endpoint.

## Status

**PASS.** `split_v1.json` is committed and must never be regenerated.
Ready for Phase 4 (models and smoke tests).
