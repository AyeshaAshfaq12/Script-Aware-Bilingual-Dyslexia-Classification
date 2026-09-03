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

---

## D-006 — Primary A2-vs-A1 comparison fixed at a matched insertion depth

- **Date:** 2026-09-02
- **Guide clause:** §6 — "A1 and A2 sweep depth_config identically; each
  reports its own best fair configuration, per the paper."
- **Status:** ACCEPTED (decided before the Phase 7 freeze; no test data
  had been touched).
- **Deviation:** The co-primary comparison is run at a single common
  insertion depth (`mid`) for both arms, rather than at each arm's own
  best depth. Each arm still selects its own optimizer and learning rate
  inside that depth, so the tuning budget and search space remain equal.
- **Reason:** Selection on validation chose different depths for the two
  arms — A1 (the control) chose `all`, A2 chose `mid`. That leaves the
  compared models with different capacities:

  | arm | depth | n_params |
  |---|---|---|
  | A1 (control) | all | 9,821,161 |
  | A2 (treatment) | mid | 9,685,537 |

  a 135,624-parameter gap (1.40%) **in the control's favour**. The
  paper's central design claim is that A1 and A2 have identical capacity
  and differ only in whether the attention gates see the script. Running
  the primary endpoint on models of unequal size would contradict that
  claim in the one comparison the paper rests on.
- **Why `mid`:** it is A2's best depth *and* tied-best for A1 (A1 scores
  0.8065 at both `mid` and `all`), so neither arm is handicapped by the
  choice. At `mid` both arms have exactly 9,685,537 parameters.
- **Impact on the claim:** This does not change the observed effect.
  The validation A2-A1 delta is **-0.0054 either way**, because A1's
  best score is identical at `mid` and `all`. The choice therefore
  affects the integrity of the capacity-matching claim, not the number,
  and it was made before any test-set evaluation. The guide's
  each-arm-own-best comparison is retained and reported as an
  exploratory sensitivity analysis in
  `results/selected_configs.json -> secondary_pair_each_arm_own_best`.
- **Resolution:** ACCEPTED. `results/selected_configs.json` records both
  the matched-depth primary pair and the each-arm-own-best secondary
  pair. The paper must state that the primary comparison is at a fixed
  common depth, with the full depth sweep reported separately.

---

## D-007 — A0 CNN-from-scratch trains end to end and is far slower

- **Date:** 2026-09-02
- **Guide clause:** §5.3 — A0 re-runs include CNN-from-scratch.
- **Status:** RESOLVED (informational; no change to the protocol).
- **Finding:** Every other arm uses a frozen backbone, so its frozen
  prefix is cached and only the trainable suffix is fitted. The
  CNN-from-scratch baseline has no frozen part, so it trains end to end
  on raw images and cannot use that path. Measured: **23.8 min for one
  run** (50 epochs, no early stop) versus ~30 s for a cached A0 run.
  Nine cnn_scratch runs cost ~3.6 h, which is most of the A0_others
  budget.
- **Impact on the claim:** None. It is a supporting baseline, not part
  of the co-primary comparison, and it runs under the same split, seeds
  and protocol as every other arm.
- **Resolution:** Run as specified. A bug that assumed every arm has a
  cacheable prefix was fixed in `src/train.py` (it now fits the full
  model on images when there is no frozen prefix).

---

## D-008 — Duplicate rows from two concurrent tuning runners

- **Date:** 2026-09-02
- **Guide clause:** §9 — `results/all_runs.csv` is the append-only
  master log, one row per run.
- **Status:** RESOLVED.
- **What happened:** An A5 tuning sweep was launched twice. The first
  launch was believed dead but was still alive, so two runner processes
  overlapped. Each checks `already_done()` before starting a run, and
  neither had logged yet, so both trained and logged the same
  `(config, seed)` pairs. **9 duplicate rows** resulted, across 8
  distinct keys.
- **Integrity check before removing anything:** every duplicated
  `(phase, arm, config_hash, seed)` key was compared. All 8 pairs are
  **bit-identical** — same accuracy to six decimals and the same epoch
  count. This is an unplanned but genuine confirmation of the
  determinism requirement in guide §5.4 check 4: the same seed run twice
  in separate processes gives exactly the same result.
- **Impact on the claim:** None. No run's result changed; only the row
  counts were inflated. The affected arm (A5) is a context arm, not part
  of the co-primary comparison.
- **Resolution:**
  1. The raw log was preserved verbatim as
     `results/all_runs_raw_with_duplicates.csv`.
  2. `results/all_runs.csv` was deduplicated on
     `(phase, arm, config_hash, seed)`, keeping the first occurrence:
     222 rows -> 213 unique.
  3. `src/run_tuning.py` now takes an exclusive lock
     (`runs/.runner.lock`) for the duration of a sweep, so a second
     runner exits immediately with an explanatory message instead of
     silently duplicating work. The lock is released through a `with`
     block, so it survives exceptions and early exits.

---

## D-009 — "Attention maps" rendered as channel-gate profiles, not spatial heat maps

- **Date:** 2026-09-03
- **Phase:** 10 (figures), guide §11 deliverable 5
- **Status:** ACCEPTED

**Deviation.** The guide lists `fig_attention_maps.pdf` among the
required figures. The delivered figure shows SCA **channel-gate
profiles** and a script-counterfactual, not spatial attention heat maps
over the handwriting images.

**Reason.** The SCA block is squeeze-and-excitation style *channel*
attention. Its gate is a vector `a` in (0,1)^C applied as `f + f*a`,
broadcast identically across every spatial position
(`src/models.py::SCA.call`). There is no spatial term anywhere in the
mechanism, so there is no spatial attention to visualise. Producing a
heat map would have required substituting a different method (Grad-CAM
or similar) and presenting it as if it showed the SCA gates — which
would misrepresent what the model does. Overlaying gates on dataset
images would also have conflicted with hard rule 7.

**Impact on the claim.** None negative; the substitute is strictly more
informative for the question the paper asks. The counterfactual probe
(`src/attention_probe.py`) holds each image fixed and varies only the
script id, which isolates the causal contribution of the conditioning
signal — the exact quantity the A2-vs-A1 contrast tests.

**Result it produced.** Script-induced gate SD = 0.000282 against
image-induced gate SD = 0.008609, a ratio of **0.0327**: changing the
script moves the attention gates about 3% as much as changing the image
does. This is a direct mechanistic account of the null co-primary
result — the conditioning input barely perturbs the attention it is
supposed to steer.

**Resolution.** The filename `fig_attention_maps.pdf` is kept so the
deliverable list maps 1:1, but the figure title, caption and this entry
all state plainly that these are channel gates and that SCA has no
spatial component. The probe carries a null check: A1's gates cannot
depend on the script by construction, and its measured script-induced
variation is 4.87e-09 — float32 round-off, 58,000x below A2's — which
confirms the probe reads the right tensor.
