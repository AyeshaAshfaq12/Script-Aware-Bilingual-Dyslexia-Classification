# Script-Aware Bilingual Dyslexia Classification

A controlled ablation testing whether **script-conditioned attention**
improves binary dyslexia classification on Urdu–English children's
handwriting, against a **matched-capacity script-agnostic control**.

Companion code for `paper/ScriptAwareDyslexia_TBD_v1_2026-09.tex`.
Protocol: `Documents_and_Guides/ScriptAwareDyslexia_ExperimentGuide_v1_2026-09.md`.

> **Status:** in progress. Phase 1 complete, Phase 2 (script
> annotation) underway. No test-set evaluation has occurred; the
> `freeze-v1` tag does not yet exist.

## Dataset

The corpus is the bilingual children's handwriting dataset of
**Kashif et al. (2026)**, used as released. We do **not** redistribute
the images (see *Licence and redistribution*); `data/raw/` is
gitignored.

### Access record

| Field | Value |
|---|---|
| Source | Kashif et al. (2026), *Engineering Reports*, Section 3.8 / endnote 1 |
| Dataset URL | <https://drive.google.com/drive/folders/1L9Dc3xQ7dqlL10CD4JaRfG_H9JzuP-Tw?usp=drive_link> |
| Source-study results + code | <https://drive.google.com/drive/folders/1kiig7QTYF90vqAvvJR6mz-tO5yOOsSkY?usp=sharing> |
| Access route | The public link published by the source study |
| Access date | **2026-08-31** |
| Folder listing as accessed | `data/raw/manifest.txt` |
| Per-file checksums | `data/raw/checksums.csv` (852 rows, SHA-256) |

The access date is evidenced by the download artifact itself: Google
Drive's bulk export named the archive
`Dataset-20260831T212010Z-1-001.zip`, encoding **2026-08-31T21:20:10Z**.
The archive is retained under `Related_Research_Work/` (gitignored, not
redistributed).

Per the source study's Section 3.8 the data is reachable at a public
link, while its formal Data Availability Statement describes
availability on request; we treat the published link above as the
operative access route, as stated in the paper's Data section.

### Released contents and the integrity audit

The release contains 852 JPEG files, 426 under `Yes/` (dyslexic) and
426 under `No/` (non-dyslexic). A SHA-256 audit found that these 852
files hold only **638 distinct images**:

| | Files | Distinct images |
|---|---|---|
| `Yes/` | 426 | 426 |
| `No/` | 426 | **232** |
| **Total** | **852** | **638** |

20 images are **byte-identical across both class folders**, carrying
contradictory labels. See `DEVIATIONS.md` D-004,
`results/duplicate_audit.json`, and `results/duplicate_groups.csv`.

Two corpora are therefore defined (`src/build_corpus.py`):

| Corpus | Definition | n | Dyslexic | Non-dyslexic |
|---|---|---|---|---|
| **Primary** | deduplicated, contradictions removed | **618** | 406 | 212 |
| **Sensitivity** | released files as-is | 852 | 426 | 426 |

The co-primary endpoints are evaluated on the **primary** corpus. The
sensitivity corpus is analysed secondarily to retain comparability with
the source study.

Label mapping: `Yes` → `y=1` (dyslexic), `No` → `y=0` (non-dyslexic).
Record key: `uid = "<class_folder>/<filename>"`, because 20 filenames
are reused across the two class folders.

## Repository layout

```
configs/      grid.yaml, endpoints.yaml   (frozen at Phase 7)
data/
  raw/        dataset as accessed (GITIGNORED) + manifest, checksums
  annotations/ script tags: annotA, annotB, final, agreement report
  splits/     split_v1.json  (generated once, never regenerated)
  corpus_v1.csv, annotation_units.csv
src/          data, models, train, evaluate, stats, figures
runs/         one folder per run (GITIGNORED, archived separately)
results/      all_runs.csv, stats summaries, audit artifacts
figures/      generated vector PDFs
paper/        .tex and .bib
docs/phase_records/  one record per completed phase
DEVIATIONS.md every departure from the guide, with impact
```

## Environment

Python 3.13.7, TensorFlow/Keras (see `requirements.txt` for pinned
versions). Local development and smoke tests run on CPU; GPU runs
(tuning, pilot, final) execute on Colab/Kaggle. TensorFlow >2.10 has no
native Windows GPU support (`DEVIATIONS.md` D-003).

```bash
pip install -r requirements.txt
```

## Reproducing

Every table and figure is generated from artifacts on disk. No number is
hand-entered.

| Stage | Command |
|---|---|
| Phase 1 gate, manifest, checksums | `python src/verify_data.py` |
| Duplicate audit | `python src/audit_duplicates.py` |
| Corpus definitions (618 / 852) | `python src/build_corpus.py` |
| Script-tag agreement + adjudication | `python src/agreement.py --stage final --policy annotator_b` |
| Fixed split (generated once) | `python src/make_split.py` |
| Smoke tests | `python -m pytest tests/ -q` |
| Phase 5 tuning sweep | `python src/run_tuning.py --arms A1 A2` |
| | `python src/run_tuning.py --arms A3 A0 A5` |
| | `python src/run_tuning.py --arms A0_others` |
| Phase 5 config selection | `python src/select_configs.py` |
| Phase 6 pilot + power | `python src/run_pilot.py` |
| Phase 8 final runs (post-freeze) | `python src/run_final.py --arms A1 A2 A0 A3` |
| | `python src/run_final.py --arms A5 A2p A4` |
| Phase 9 statistics | `python src/stats.py` |
| Phase 10 figures and tables | `python src/figures.py` |

All runners are **resumable**: each finished run is appended to
`results/all_runs.csv` and skipped on re-invocation, so a sweep can be
interrupted and continued.

## Key design decisions

| Decision | Where recorded |
|---|---|
| Deduplicated 618-image primary corpus | `DEVIATIONS.md` D-004 |
| Accuracy endpoints kept despite 406/212 imbalance | `DEVIATIONS.md` D-005 |
| Primary A2-vs-A1 pair at a **matched insertion depth** | `DEVIATIONS.md` D-006 |
| CNN-from-scratch trains end to end (no cacheable prefix) | `DEVIATIONS.md` D-007 |
| Tuning grid trimmed before the freeze (depth sweep kept) | `configs/grid.yaml` |
| S = 30 with the achieved power recorded honestly | `configs/endpoints.yaml` |

### Compute

All experiments run on a local CPU (Intel i7-6600U, 2 cores). This costs
no result quality — the computation is identical, so hardware changes
wall-clock time only — and CPU is preferable here because
`enable_op_determinism()` is fully reliable on CPU and unavailable for
several GPU kernels.

Because the backbone is frozen and no augmentation is used, each arm's
frozen prefix is cached and only the trainable suffix is fitted. That is
**exactly equivalent** to end-to-end training and is asserted
bit-for-bit in `tests/test_smoke.py::TestSplitEquivalence`.

## Reproducibility commitments

- One fixed split (`data/splits/split_v1.json`, seed 1337), generated
  once, reused by every arm.
- Fixed seed lists: tuning 101–103, pilot 201–205, final 301..(300+S).
- Identical tuning grid and budget for every arm.
- No test-set evaluation before the `freeze-v1` tag.
- No augmentation, no synthetic data, no external data.
- Every reported number traces to a run artifact on disk.

## Licence and redistribution

Code: **MIT** (chosen by the authors 2026-09-01, per guide §1).
The `LICENSE` file is **not yet written**: it requires the exact
copyright holder name(s), which are author-supplied and are not
inferred. **TODO — authors to supply the copyright holder line.**

The handwriting images are **not** redistributed here. They remain
available from Kashif et al. (2026) via the access route published in
that paper. Our released artifacts (script annotations, split indices,
code, results) are keyed to the original filenames.

## AI assistance

Development used Claude (planning, drafting) and Claude Code
(implementation). Claude Code additionally serves as annotator A in the
script-annotation pass; see `DEVIATIONS.md` D-001. Venue policy
compliance is checked at venue selection.
