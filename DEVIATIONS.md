# DEVIATIONS

Every departure from `ScriptAwareDyslexia_ExperimentGuide_v1_2026-09.md`
is recorded here, per hard rule 4. Format: date, guide clause, reason,
impact on the claim, resolution. Deviations are reviewed by both
authors before the results handoff (guide §14).

Status legend: `OPEN` (unresolved), `ACCEPTED` (authors approved),
`RESOLVED` (no longer applies).

---

## D-001 — Script annotation: Claude Code as annotator A

- **Date:** 2026-09-01
- **Guide clause:** §3 (Phase 2) — "Both authors annotate all 852
  images independently."
- **Deviation:** Annotator A is Claude Code (vision-based, image-only,
  no filename or folder cues). Annotator B is a human author.
  Adjudication of all disagreements is performed by the human author.
- **Reason:** Authorised by the authors on 2026-09-01 in preference to
  a two-human pass, on turnaround grounds.
- **Impact on the claim:** Cohen's kappa in `agreement_report.md`
  measures **human-vs-AI** agreement, not inter-human agreement, and
  MUST be described that way in the paper's Data section. The final
  tag for every image is decided by a human author, so
  `scripts_final.csv` remains human-adjudicated. Script tags are a
  covariate (conditioning input and per-script reporting axis), not
  the outcome label, so annotation noise affects both arm A1 and arm
  A2 identically and does not bias the co-primary A2-A1 contrast.
- **Resolution:** ACCEPTED. Reporting language fixed at Phase 2; the
  annotation protocol given to both annotators is archived verbatim in
  `data/annotations/protocol.md`.

---

## D-002 — Dataset class-folder casing

- **Date:** 2026-09-01
- **Guide clause:** §2 (Phase 1) — "426 in YES, 426 in NO".
- **Deviation:** The release ships the folders as `Yes` and `No`
  (mixed case), not `YES`/`NO`.
- **Reason:** As-released naming; renaming would break traceability to
  the original archive.
- **Impact on the claim:** None. Cosmetic. Counts are 426/426 as
  specified.
- **Resolution:** RESOLVED. As-released casing preserved throughout;
  the mapping `Yes -> dyslexic (y=1)`, `No -> non-dyslexic (y=0)` is
  recorded in `README.md` and applied in `src/data.py`.

---

## D-003 — Local development environment is CPU-only

- **Date:** 2026-09-01
- **Guide clause:** §1 — "local VS Code for development and smoke
  tests (CPU is fine)".
- **Deviation:** None in substance; recorded for reproducibility.
  TensorFlow >2.10 has no native Windows GPU support, so the local
  Windows environment is CPU-only. All GPU work (Phases 5, 6, 8) runs
  on Colab/Kaggle.
- **Reason:** Platform constraint.
- **Impact on the claim:** None. The guide already designates local as
  smoke-test-only. Full op determinism is verified on CPU (§5.4 check
  4); if it cannot be attained on the GPU runtime, that is logged
  separately and the repeated-seeds design carries the inference.
- **Resolution:** RESOLVED (informational).

---

## D-004 — BLOCKING: released corpus contains 214 byte-identical duplicate files, 20 of them cross-class

- **Date:** 2026-09-01
- **Guide clause:** §2 (Phase 1) step 3 — "Expected: 852 images total,
  426 in YES, 426 in NO. If counts differ, STOP and report."
- **Status:** ACCEPTED / RESOLVED 2026-09-01 by author decision.
- **Finding:** The *file* counts match the guide exactly (852 / 426 /
  426). The *content* does not. SHA-256 over all 852 files yields only
  **638 distinct images**; 214 files are redundant copies.
  - `Yes/`: 426 files, 426 distinct — no internal duplication.
  - `No/` : 426 files, **232 distinct** — 194 redundant copies.
  - **20 images are byte-identical across classes**, i.e. the same
    file appears in both `Yes/` and `No/` (verified pixel-identical,
    e.g. `Yes/IMG_0185.jpeg` == `No/IMG_0185.jpeg`). These carry
    contradictory labels.
  - Deduplicated and dropping the contradictory images, the corpus is
    **406 dyslexic vs 212 non-dyslexic (618 total)** — a 2:1 class
    imbalance, not the 1:1 balance the paper's Data section asserts.
- **Evidence artifacts:** `data/raw/checksums.csv`,
  `results/duplicate_groups.csv`, `results/duplicate_audit.json`,
  produced by `src/verify_data.py` and `src/audit_duplicates.py`.
- **Impact on the claim:**
  1. *Leakage.* With an image-level split, a duplicate group straddling
     train and test puts a pixel-identical image on both sides. 147
     groups are affected. Every absolute accuracy figure is inflated by
     memorisation, ours and the source study's alike.
  2. *Label contradiction.* 20 images are simultaneously labelled
     dyslexic and non-dyslexic. No model can be right on both copies;
     they impose an irreducible error floor and corrupt any partition
     they straddle.
  3. *Stated class balance is an artefact.* The 426/426 balance holds
     only at the file level and is produced by duplication within
     `No/`. The paper's Data section, written against 426/426, cannot
     stand as drafted.
  4. *Co-primary endpoint.* The A2-A1 paired contrast is comparatively
     robust: both arms train and test on the identical corrupted split
     and both benefit from the same leakage. The contrast is not
     destroyed, but its interpretation weakens, and the per-script Urdu
     endpoint depends on how duplicates distribute across scripts,
     which cannot be assessed until Phase 2 annotation exists.
- **Resolution:** ACCEPTED 2026-09-01. The authors chose option (a):
  - **PRIMARY corpus** = deduplicated, 618 images
    (406 dyslexic / 212 non-dyslexic). One representative per distinct
    sha256, chosen deterministically as the lexicographically smallest
    uid in the group; all 20 cross-class contradictory images removed
    entirely. This corpus carries the co-primary endpoints.
  - **SENSITIVITY corpus** = the released 852 files as-is, analysed
    secondarily so comparability with Kashif et al. (2026) is retained.
  - Built by `src/build_corpus.py` into `data/corpus_v1.csv` (852 rows
    with membership flags), `data/annotation_units.csv` (638 distinct
    images), `data/corpus_v1_summary.json`. All integrity assertions
    pass.
  - Consequence for Phase 2: the annotation unit is the distinct image
    (sha256), not the file. 638 units are annotated; every duplicate
    file inherits its unit's tag, so duplicate files cannot receive
    inconsistent script tags.
  - Consequence for the paper: the Data section must be rewritten. The
    released 426/426 balance is a file-level artefact of duplication
    inside `No/`; the deduplicated corpus is 406/212.

---

## D-005 — Class imbalance on the primary corpus vs. accuracy-based endpoints

- **Date:** 2026-09-01
- **Guide clause:** §10 / endpoints — co-primary endpoints are defined
  as *accuracy* differences (pooled and Urdu-subset).
- **Status:** CLOSED 2026-09-02 — authors confirmed the analysis
  plan is unchanged.
- **Finding:** Following D-004, the primary corpus is 406 dyslexic vs
  212 non-dyslexic (1.92:1). The majority-class baseline is therefore
  **65.7% accuracy**, not 50%. On the released 852 it was 50%.
- **Impact on the claim:** The co-primary endpoints are *paired
  differences* between A2 and A1 on an identical split, so the
  imbalance affects both arms identically and does not bias the
  contrast. It does, however, make absolute accuracy a weak descriptive
  statistic, and it raises the question of whether class weighting or a
  balanced metric should enter the protocol.
- **Resolution:** CLOSED. The authors confirmed on 2026-09-02 that the
  plan stands as written: accuracy deltas remain the co-primary
  endpoints, no class weighting is added, and the deduplicated corpus is
  used as-is. The 65.7% majority-class baseline is reported alongside
  absolute accuracy so the numbers are read correctly. The paired A2-A1
  contrast is unaffected either way, since both arms see the identical
  split.
