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
| Access URL | **TODO — authors to supply the exact link used** |
| Access date | **TODO — authors to confirm (archive stamped 2026-08-31)** |
| Access route | **TODO — published link, or other** |
| Folder listing as accessed | `data/raw/manifest.txt` |
| Per-file checksums | `data/raw/checksums.csv` |

*(These three fields resolve a TODO in the paper's Data section and are
recorded verbatim from the authors. They are not inferred.)*

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

| Artifact | Command |
|---|---|
| Phase 1 gate, manifest, checksums | `python src/verify_data.py` |
| Duplicate audit | `python src/audit_duplicates.py` |
| Corpus definitions | `python src/build_corpus.py` |

*(Remaining commands are added as each phase completes; per guide §13
every table and figure must be reproducible with one command.)*

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
