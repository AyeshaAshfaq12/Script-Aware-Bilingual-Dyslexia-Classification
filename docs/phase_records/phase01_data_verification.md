# Phase Record — Phase 1: Data acquisition and verification

- **Date:** 2026-09-01
- **Phase:** Phase 1 (Guide §2), preceded by Phase 0 repo scaffold (§1)
- **Status:** **BLOCKED** — guide-mandated gate passed; supplementary
  integrity audit failed. See DEVIATIONS.md D-004.

## Actions performed

1. Created the §1 repository layout at the workspace root: `configs/`,
   `data/{raw,annotations,splits}/`, `src/`, `runs/`, `results/`,
   `figures/`, `paper/`, plus `docs/phase_records/` for these records.
2. Wrote `.gitignore` excluding `data/raw/**`, `Related_Research_Work/`,
   `runs/**`, and model weights, per hard rule 7 (no image
   redistribution) and §1.
3. Wrote `DEVIATIONS.md` with entries D-001..D-004.
4. Moved the dataset: `Reference_Dataset/Dataset/{Yes,No}` ->
   `data/raw/{Yes,No}`. `Reference_Dataset/` removed (now empty).
5. Implemented `src/verify_data.py`; generated `data/raw/manifest.txt`
   and `data/raw/checksums.csv` (uid, filename, class_folder, label,
   sha256, bytes, width, height, mode, format).
6. Implemented `src/audit_duplicates.py` after the checksum pass
   surfaced hash collisions; generated `results/duplicate_groups.csv`
   and `results/duplicate_audit.json`.
7. Started TensorFlow 2.20 (CPU) install into Python 3.13.7.

## Decisions made

- Annotator A = Claude Code, annotator B = human author (D-001),
  authorised by the authors 2026-09-01.
- As-released class-folder casing `Yes`/`No` preserved (D-002); label
  mapping `Yes -> y=1 (dyslexic)`, `No -> y=0 (non-dyslexic)`.
- Layout created at workspace root, not in a `script-aware-dyslexia/`
  subfolder; `Documents_and_Guides/` and `Related_Research_Work/`
  retained.
- A stable record key `uid = "<class_folder>/<filename>"` is adopted in
  place of bare filenames, because 20 filenames are reused across the
  two class folders and the guide's `split_v1.json` filename lists
  would otherwise be ambiguous.

## Configuration used

- Python 3.13.7, Windows 10 Pro 19045, CPU only.
- numpy 2.2.6, pandas 2.3.2, pillow 11.3.0, scikit-learn 1.7.2,
  scipy 1.16.2, PyYAML 6.0.2. TensorFlow install in progress.
- No seeds consumed in this phase; nothing stochastic was run.

## Files created / updated

| Path | Description |
|---|---|
| `.gitignore` | excludes images, source archive, run artifacts |
| `DEVIATIONS.md` | D-001 .. D-004 |
| `src/verify_data.py` | Phase 1 gate + manifest + checksums |
| `src/audit_duplicates.py` | exact-duplicate audit |
| `data/raw/Yes/`, `data/raw/No/` | 852 images (gitignored) |
| `data/raw/manifest.txt` | folder listing as accessed |
| `data/raw/checksums.csv` | 852 rows, sha256 + image properties |
| `results/duplicate_groups.csv` | 361 rows, 147 duplicate groups |
| `results/duplicate_audit.json` | audit summary counts |
| `docs/phase_records/phase01_data_verification.md` | this record |

## Validation results

Guide-mandated gate (§2 step 3):

| Check | Result |
|---|---|
| total images == 852 | **PASS** (852) |
| Yes == 426 | **PASS** (426) |
| No == 426 | **PASS** (426) |
| all files decode as images | **PASS** (852/852 JPEG, RGB) |
| uid unique | **PASS** |

Supplementary integrity checks:

| Check | Result |
|---|---|
| filename unique across class folders | **FAIL** — 20 names reused |
| no byte-identical duplicate images | **FAIL** — 147 groups, 214 redundant copies |

Corpus properties: 852 files, 638 distinct by sha256; all JPEG/RGB;
538 distinct resolutions (most common 491x491, n=21); 57.3 MB total.

## Issues encountered

1. **Duplicate content (blocking).** 638 distinct images among 852
   files. `No/` holds only 232 distinct images across 426 files. 20
   images are byte-identical across `Yes/` and `No/`, carrying
   contradictory labels. Deduplicated corpus: 406 vs 212. Full detail
   in DEVIATIONS.md D-004.
2. **Filename collisions.** 20 filenames occur in both class folders,
   so the guide's filename-keyed `split_v1.json` is ambiguous as
   specified. Mitigation adopted: `uid = class_folder/filename`.
3. **Perceptual near-duplicate scan was inconclusive and is not
   relied upon.** A dHash pass over the 638 distinct images produced
   chained clusters and collisions between images of different
   dimensions and different content, which is expected for single
   characters on plain paper (very low image entropy). No
   near-duplicate claim is made; only the exact-hash result, which is
   definitive, is reported.
4. **Provenance not yet recorded.** The guide (§2 step 2) requires the
   access URL and access date in README.md. These are author-supplied
   facts and have not been provided; they are NOT invented. README's
   access block remains an explicit TODO.

## Deviations

- D-001 annotator A is Claude Code (ACCEPTED)
- D-002 class-folder casing (RESOLVED)
- D-003 CPU-only local environment (RESOLVED, informational)
- D-004 duplicate/contradictory images (**OPEN, BLOCKING**)

## Status

**BLOCKED** — awaiting author decision on D-004 and the dataset
provenance facts. Phase 2 annotation tooling can proceed in parallel;
Phase 3 (split) cannot start until D-004 is resolved, because the
duplicate handling determines what the split is built over.
