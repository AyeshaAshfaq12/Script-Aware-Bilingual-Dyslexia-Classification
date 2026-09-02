# Session Log — Script-Aware Bilingual Dyslexia Classification

A record of the working session that produced this repository: what was
asked, what was decided, the reasoning behind each decision, every
problem found, and every artifact produced.

> **Provenance note — read this first.**
> This is a **faithful reconstructed record, not a byte-exact
> transcript.** No verbatim chat log file exists, and the earliest turns
> of the session were compacted before this document was written. Every
> *number, filename, command, config value and decision* below is taken
> from artifacts on disk (`results/`, `configs/`, `DEVIATIONS.md`,
> `docs/phase_records/`, git history) and is therefore exact. The
> *narration* of who said what is a reconstruction. Where a user
> instruction is quoted, it is paraphrased unless marked verbatim.
> This distinction matters because the same honesty rule that governs
> the experiment (hard rule 2: no fabricated numbers) governs this
> document.

- **Repository:** `Script-Aware-Bilingual-Dyslexia-Classification`
- **Specification:** `Documents_and_Guides/ScriptAwareDyslexia_ExperimentGuide_v1_2026-09.md`
- **Paper draft:** `Documents_and_Guides/ScriptAwareDyslexia_TBD_v1_2026-09.tex`
- **Session dates:** 2026-09-01 → 2026-09-02
- **Freeze:** commit `e491aaf`, tag `freeze-v1`

---

## 1. The assignment

The user asked me to act as a senior CS/AI experiment researcher and
execute the attached Experiment Guide end to end, phase by phase, under
strict research rules: never fabricate results, never touch test data
before the freeze, log every deviation, report null results honestly,
and preserve full reproducibility. They added two standing instructions:

- **"If anything is required, ask directly. Don't assume."**
- The workspace contained a `Dataset` folder to be moved into the
  structure the guide specifies.

Everything below follows from those instructions.

---

## 2. Phase 0 — Repository scaffold

**What I did.** Inspected the workspace before touching anything. Found
`Reference_Dataset/Dataset/{Yes,No}` with 426 images each, a source
paper PDF, the guide and the `.tex` draft, an empty `.gitignore`, and a
git repo with **zero commits**. Python 3.13.7 present; TensorFlow absent.

**Four decisions I put to the user rather than assuming**, because each
affects research validity:

| Question | Options offered | Chosen |
|---|---|---|
| How to do the dual script annotation | both authors / Claude as annotator A / Claude pre-fills / Claude does both | **Claude = annotator A, human = annotator B** |
| Where to build the guide's layout | workspace root keeping folders / consolidate / nested subfolder | **workspace root, keep existing folders** |
| Gitignore the dataset + publisher PDF | both / only `data/raw/` | **both** |
| Environment | TF 2.20 in current Python / venv / Colab-only | **TF 2.20 CPU in current Python** |

**Reasoning on annotation.** The guide requires two independent human
annotators. I cannot be two people, and the resulting κ goes verbatim
into the paper. Presenting an AI-vs-human κ as an inter-human κ would be
a misrepresentation, so this had to be an explicit, logged choice with
a human adjudicator retained. Logged as **D-001**.

**Artifacts.** `.gitignore` (excludes images, contact sheets, source
archive, publisher PDF, run artifacts), `DEVIATIONS.md`, `README.md`,
directory tree per guide §1.

---

## 3. Phase 1 — Data verification, and the first real problem

**The guide's gate passed:** 852 files, 426 `Yes`, 426 `No`, all
decoding as JPEG/RGB.

**Then I added checksums, and the corpus fell apart.**

| Measure | Value |
|---|---|
| Files | 852 |
| **Distinct images (SHA-256)** | **638** |
| Redundant copies | 214 |
| Duplicate groups | 147 |
| `Yes/` | 426 files → 426 distinct (clean) |
| `No/` | 426 files → **232 distinct** |
| **Byte-identical across both classes** | **20 images** |

I verified one cross-class pair pixel-by-pixel and viewed the image: a
single handwritten character, the same file filed under both diagnoses.

**Why this mattered.** Three consequences, all reported before any
modelling:

1. **Leakage.** An image-level split with 147 duplicate groups puts
   pixel-identical images on both sides of train/test. Every absolute
   accuracy inflates — ours *and* the source study's 0.7871.
2. **Label contradiction.** 20 images are labelled both dyslexic and
   non-dyslexic; no model can be right on both copies.
3. **The stated class balance is an artefact.** Deduplicated and
   dropping contradictions: **406 dyslexic vs 212 non-dyslexic**, not
   426/426. The paper's Data section, written against 426/426, cannot
   stand as drafted.

**What I did NOT do.** I did not fix it silently, and I did not rely on
a perceptual near-duplicate scan I ran: dHash chained low-entropy
single-character images into nonsense clusters and merged images of
different dimensions, so I discarded it and reported only the
exact-hash result, which is definitive.

**Decision put to the user:** dedup / keep 852 with duplicate-aware
split / run both / pause and contact the authors.
**Chosen: deduplicate to 618 as primary, keep 852 as a sensitivity
corpus.** Logged as **D-004**.

**Secondary finding.** 20 filenames occur in *both* class folders, so
the guide's filename-keyed `split_v1.json` is ambiguous as specified. I
adopted `uid = "<class_folder>/<filename>"` as the record key.

**New open deviation D-005.** Deduplication makes the majority-class
baseline **65.7%**, not 50%, while the co-primary endpoints are defined
on accuracy. The paired A2−A1 contrast is unaffected (both arms see the
identical split), but absolute accuracy becomes a weak descriptive
statistic. Flagged for decision before the freeze; the user later
confirmed the plan stays unchanged.

**Artifacts.** `src/verify_data.py`, `src/audit_duplicates.py`,
`src/build_corpus.py`, `data/raw/checksums.csv`, `manifest.txt`,
`results/duplicate_groups.csv`, `results/duplicate_audit.json`,
`data/corpus_v1.csv`, `data/annotation_units.csv`.

---

## 4. Phase 2 — Script annotation

**Key design choice: the annotation unit is the distinct image
(sha256), not the file.** 638 units instead of 852, and duplicate files
inherit their unit's tag, so byte-identical files can never receive
inconsistent script tags.

**A blinding defect I caught and fixed before recording a single tag.**
My first contact-sheet render ordered units by `unit_id`, which
correlates with class folder (`No/` sorts before `Yes/`), exposing the
diagnosis label as visible block structure. I discarded those sheets and
re-rendered in an order shuffled with fixed seed **20260901**, archived
in `presentation_order.json`. Cells show `unit_id` only — never a
filename, folder, or label.

**Annotator A pass (Claude).** 40 contact sheets, 4×4, 300 px cells, all
638 units. Schema-valid on the first validation run.

**Annotator B pass (human).** Completed by the user. Three units were
left blank with the note *"not clear image"* — the human judged them
illegible. Protocol rule 8 requires a best-available reading even then,
so these were **never auto-filled**; `agreement.py` aborts unless they
are resolved explicitly in `blank_resolutions.csv`.

**Adjudication policy.** The user directed that human annotation take
precedence wherever the two disagreed. I implemented
`--policy annotator_b` and made the consequence explicit in the
generated report: because every disagreement resolves to B,
`scripts_final.script` **equals** `scripts_annotB.script`, so the paper
must describe adjudication as *"human annotator authoritative"*, not as
*"resolved jointly case by case"*.

**Results (from `agreement_report.md`, exact):**

| Measure | Value |
|---|---|
| Cohen's κ (script), human-vs-AI | **0.8489** |
| Raw agreement | **92.13%** (585/635) |
| Disagreements adjudicated | 50 |
| Final: urdu / english / digit | 420 (65.8%) / 82 (12.9%) / 136 (21.3%) |
| Digit glyphs | **135 Western**, 0 Eastern Arabic-Indic |
| Ambiguous flags | 117 (18.3%) |

**The digit finding resolves a paper TODO:** the corpus contains
Western Arabic numerals, not Eastern Arabic-Indic forms. I flagged the
`ain` vs Eastern-`٤` homoglyph specifically for annotator B's scrutiny,
because that single pair decides the answer.

**Artifacts.** `data/annotations/protocol.md` (frozen before any tag),
`scripts_annotA.csv`, `scripts_annotB.csv`, `disagreements.csv`,
`scripts_final.csv`, `agreement_report.md`, `src/annotate.py`,
`src/agreement.py`, `src/validate_annotations.py`,
`src/make_contact_sheets.py`.

---

## 5. Phase 3 — Fixed split

Seed **1337**, stratified jointly on (class × script), 80/10/10.

| partition | n |
|---|---|
| train | 494 |
| val | 62 |
| test | 62 |

Generated once, committed, never regenerated. The 62-image partitions
later turn out to matter enormously (§8).

---

## 6. Phase 4 — Models

MobileNetV1 frozen backbone, 224×224×3, `preprocess_input`. All three
tap points named in guide §5.1 (`conv_pw_3_relu`, `conv_pw_7_relu`,
`conv_pw_13_relu`) exist in TF 2.20 — **no deviation needed**.

The SCA module is implemented so the unconditioned control (A1) still
*contains* the script embedding table and the conditioned arm (A2)
still contains the learned constant, giving **identical parameter
counts**. This is asserted in tests at every depth.

**Empirical confirmation during training:** A1 logged
`Gradients do not exist for ... script_embed`, and A2 logged the mirror
image for `const_embed`. That is the matched-capacity design visibly
working — identical parameters, different active subsets.

---

## 7. Phase 5 — Tuning

### 7.1 The platform question

The user asked whether local CPU instead of Colab/Kaggle would hurt
result quality. **It does not.** The computation is identical — same
architecture, weights, optimiser, arithmetic — so hardware changes
wall-clock time only. CPU is in fact *preferable* here:
`enable_op_determinism()` is fully reliable on CPU and unavailable for
several GPU kernels, which is exactly the reproducibility guarantee
§5.4 asks for. Grid size affects scientific quality; hardware does not.

### 7.2 The measurement that made CPU viable

Rather than estimating, I measured on the actual machine
(Intel i7-6600U, 2 cores, 16 GB):

- Because the backbone is **frozen** and there is **no augmentation**,
  each arm's frozen prefix is a constant function of the image. Caching
  it and training only the suffix is **exactly equivalent** to
  end-to-end training.
- I did not assume that equivalence — I asserted it. 21 tests confirm
  the split path is **bit-identical** to the full model across 8
  arm/depth cases, that the suffix owns every trainable weight, that
  the prefix owns none, and that suffix training propagates.
- Speed-up: 4.7× at `late`, 1.6× at `mid`, none at `early`/`all`.
- Early stopping fires at **7–11 epochs**, not the 50 maximum.

| depth | s/epoch | s/run |
|---|---|---|
| late | 4.4 | 48 |
| mid | 12.5 | 138 |
| early | 20.7 | 228 |
| all | 21.6 | 238 |

### 7.3 The grid amendment

Full six-axis grid = 288 runs/arm ≈ 26 h for A1+A2 alone, ~35 h total.
I offered four options with measured costs; the user chose **B**, with
the caveat that we switch to GPU if local CPU would harm quality (it
does not — see 7.1).

| axis | guide | amended |
|---|---|---|
| optimizer | [rmsprop, adam] | unchanged |
| learning_rate | [1e-3, 3e-4, 1e-4] | unchanged |
| **depth_config** | [early, mid, late, all] | **unchanged** |
| tuning seeds | [101, 102, 103] | **unchanged** |
| weight_decay | [0.0, 1e-4] | **[0.0]** |
| sca d | [8, 16] | **[16]** |
| sca r | [8, 16] | **[16]** |

**Reasoning.** The depth sweep carries a stated contribution and its own
figure, so it survives intact. Weight decay of 1e-4 over ~9 epochs on a
frozen backbone is immaterial. `d` and `r` are SCA-internal and, fixed
identically across both arms, preserve matched capacity and equal budget
exactly. **Seeds were never traded for compute**, because the Phase 6
power analysis depends on them. Guide §6 permits amendment before the
freeze; the rationale and measurements are recorded in `grid.yaml`.

### 7.4 Source-study Table 3

Rather than leave a placeholder, I extracted Table 3 from the source
PDF (p.10) and recorded the best optimizer per backbone:

| backbone | Adam | RMSProp | SGD | selected |
|---|---|---|---|---|
| CNN | 0.7053 | 0.5988 | **0.7743** | sgd |
| VGG16 | 0.7602 | **0.7684** | 0.7532 | rmsprop |
| InceptionV3 | 0.7474 | **0.7661** | 0.7497 | rmsprop |
| MobileNetV1 | 0.7778 | **0.7871** | 0.7731 | rmsprop (anchor) |
| MobileNetV2 | 0.7673 | **0.7836** | **0.7836** | rmsprop (tie) |
| MobileNetV3 | **0.7216** | 0.7146 | 0.6082 | adam |

MobileNetV2 ties; RMSProp selected on lower SD and higher F1, with the
tie-break recorded so it is not silent. The anchor value
0.7871 ± 0.0102 matches the paper draft.

### 7.5 Results

208 tuning runs + 10 pilot = **218 unique runs**, ~6 h compute.

| arm | configs | selected | mean val acc |
|---|---|---|---|
| A0 | 9 | adam, 1e-4 | 0.7258 |
| A1 | 24 | depth=all, adam, 1e-4 | 0.8065 |
| A2 | 24 | depth=mid, rmsprop, 1e-4 | 0.8011 |
| A3 | 6 | rmsprop, 1e-4 | 0.7473 |
| A5 | 6 | adam, 3e-4 | 0.7581 |

---

## 8. The two findings that shape the paper

### 8.1 A2 does not beat A1 at any depth

At matched depth — where parameter counts are *identical*:

| depth | A1 best | A2 best | A2−A1 |
|---|---|---|---|
| early | 0.7581 | 0.7581 | 0.0000 |
| mid | 0.8065 | 0.8011 | −0.0054 |
| late | 0.7581 | 0.7473 | −0.0108 |
| all | 0.8065 | 0.7957 | −0.0108 |

All within ±0.04 seed noise, so not yet a result — but a consistent
null-to-negative signal, reported because hard rule 6 requires it.

### 8.2 The matched-capacity design broke at selection (D-006)

Selection chose **different depths per arm**: A1 → `all`, A2 → `mid`.
That leaves the compared models at 9,821,161 vs 9,685,537 params — a
**1.40% gap in the control's favour**. The guide says each arm reports
its own best depth; the paper's central claim is that the two differ
*only* in whether the gates see the script. Those collide.

**My decision: fix the primary comparison at a common depth, `mid`.**
Each arm still tunes optimizer and lr independently, so the budget stays
equal. At `mid` both arms have exactly **9,685,537** parameters.

**Why `mid`, and why this is not outcome-shopping:** `mid` is A2's best
*and* tied-best for A1 (A1 scores 0.8065 at both `mid` and `all`), so
neither arm is handicapped — and **the validation delta is −0.0054
either way**. The choice fixes the integrity of the capacity claim
without moving the number. The guide's each-arm-own-best pair is
retained as a sensitivity analysis.

---

## 9. Phase 6 — Pilot and the power problem

A1/A2 at the matched-depth configs, pilot seeds 201–205, validation only.

| seed | A1 | A2 | d = A2−A1 |
|---|---|---|---|
| 201 | 0.7903 | 0.7742 | −0.0161 |
| 202 | 0.7742 | 0.7419 | −0.0323 |
| 203 | 0.7258 | 0.7097 | −0.0161 |
| 204 | 0.7903 | 0.8387 | +0.0484 |
| 205 | 0.7258 | 0.8710 | +0.1452 |

- mean difference **+0.0258**, **sd_pilot = 0.0736**
- S for 80% power at 1.5 points = **232**
- ceiling applied → **S = 30**
- **achieved power = 0.121**; detectable effect **4.33 points**

**Decision: run the pre-registered S = 30 and report the achieved power
openly.** Reasoning:

- Raising S does not rescue it — S=100 still only reaches ~2.1 points,
  and S=232 across seven arms is infeasible on this hardware.
- There is a hard arithmetic floor independent of S: the test partition
  holds **62 images**, so one image is **1.61 accuracy points** and the
  1.5-point target lies **below the resolution of a single-seed
  measurement**. Averaging over seeds refines the mean but does not
  reduce the seed noise that drives power.
- The decision is robust to the noisy 5-seed SD: the independent
  tuning-based estimate (0.041 over 16 matched configs) also clamps
  to 30.
- Guide §7 step 4 explicitly anticipates the ceiling binding and
  requires it to be recorded.

**Interpretation rule, fixed in `endpoints.yaml` before unblinding:** a
null result is **not** evidence that script conditioning fails — only
that no effect larger than roughly 4 accuracy points was detected. The
confidence interval, not the p-value, carries the informative content,
and Limitations must say so.

---

## 10. Phase 7 — FREEZE

Commit `e491aaf`, tag **`freeze-v1`**.

Frozen: `grid.yaml`, `endpoints.yaml`, selected configs, the
matched-depth primary pair, S = 30, final seeds 301–330, and the
analysis plan. 58 tests pass.

**Hard rule 1 is enforced in code and was demonstrated in both
directions.** `train.py::assert_may_touch_test()` raises unless the tag
exists:

```
$ python run_final.py --dry-run --partition test     # before tagging
RuntimeError: HARD RULE 1: no test-set evaluation before the freeze.
$ python run_final.py --dry-run --partition test     # after tagging
freeze tag  : present
```

---

## 11. Bugs and incidents (all found and fixed)

| # | Problem | Impact | Fix |
|---|---|---|---|
| 1 | Contact sheets ordered by `unit_id` leaked the class label as block structure | would have unblinded annotator A | re-rendered with seeded shuffle **before any tag was recorded** |
| 2 | `build_split` returns `prefix=None` for CNN-from-scratch; `run()` assumed a prefix always exists | crashed `A0_others` | train the full model on images when nothing is cacheable (**D-007**) |
| 3 | A2′ logs as `A2_mid_predicted`; `split("_")[0]` collapses it to `A2` | would have **silently merged the predicted-script arm into the co-primary arm** | explicit `base_arm()` mapping in `stats.py` and `figures.py`, fixed before any final data existed |
| 4 | Three concurrent runners each passed `already_done()` before any had logged | 9 duplicate rows | verified all duplicates **bit-identical**, preserved raw log, deduplicated, added an exclusive runner lock (**D-008**) |
| 5 | Runner lock leaked on exception | stale lock would block future runs | moved to a `with` block |
| 6 | Keras raises under op determinism if Dropout runs without a seed | test failure | seed set in every run path |
| 7 | Contact sheets embed dataset images and would have been committed | **would have breached hard rule 7** | gitignored before the first commit |

**Incident 4 produced an unplanned benefit:** because the duplicated
runs were bit-identical in accuracy and epoch count across separate
processes, they constitute a genuine confirmation of the §5.4
determinism requirement.

---

## 12. Deviations register

| id | subject | status |
|---|---|---|
| D-001 | Claude Code as annotator A; κ is human-vs-AI | ACCEPTED |
| D-002 | class-folder casing `Yes`/`No` as released | RESOLVED |
| D-003 | CPU-only local environment | RESOLVED |
| D-004 | deduplicated 618-image primary corpus | ACCEPTED |
| D-005 | accuracy endpoints vs the 406/212 imbalance | ACCEPTED |
| D-006 | primary pair at matched insertion depth | ACCEPTED |
| D-007 | CNN-from-scratch trains end to end | RESOLVED |
| D-008 | duplicate rows from concurrent runners | RESOLVED |

---

## 13. Standing method

Two habits governed every phase and are worth stating explicitly,
because they produced the findings above:

1. **Measure instead of estimating.** Runtime budgets, epoch counts,
   duplicate structure, palette accessibility and split equivalence were
   all measured on the real machine and real data, not assumed.
2. **Surface problems rather than route around them.** Every issue in
   §11 could have been silently absorbed. Each is instead logged, and
   two of them (the duplicate corpus, the broken capacity match) change
   what the paper can claim.

No number in this repository was hand-entered. Every table and figure is
generated from artifacts on disk, and every reported value traces to a
run record.

---

## 14. State at the time of writing

- **Complete:** Phases 0–7. Corpus, annotation, split, models, tuning,
  selection, pilot, freeze.
- **Running:** Phase 8 final runs, seeds 301–330 on the test partition.
- **Pending:** A5 / A2′ / A4 final runs; `A0_others` re-runs (~4.3 h,
  dominated by CNN-from-scratch at 23.8 min/run); Phase 9 statistics;
  Phase 10 figures; Phase 11 handoff; Phase 12 release.

**Expected outcome, stated in advance so it cannot be reframed later:**
validation shows no A2 advantage at any depth, and the design has 12%
power for the pre-specified effect. The likely result is a null that
cannot distinguish "no effect" from "an effect too small for this
corpus to resolve." That is a real finding about the limits of this
dataset, and it will be reported as such.
