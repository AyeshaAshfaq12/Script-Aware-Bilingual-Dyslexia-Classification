# Experiment Guide: Script-Aware Bilingual Dyslexia Classification
Companion to draft `ScriptAwareDyslexia_TBD_v1_2026-09.tex`.
Audience: the authors and Claude Code (VS Code). Version 1, Sept 2026.

---

## 0. Scope and ground rules (read first, applies to every phase)

**What we are building.** A controlled ablation on the Kashif et al.
(2026) bilingual dataset (852 character images, binary labels)
testing whether script-conditioned attention (arm A2) beats a
matched-capacity script-agnostic control (arm A1), plus context arms
A0, A2', A3, A4, A5. Two co-primary endpoints: pooled A2-A1 accuracy
delta and Urdu-subset A2-A1 accuracy delta.

**What Claude Code is expected to do.** Implement, smoke-test, and
run exactly what this guide specifies; log everything; surface
problems instead of silently working around them.

**Hard rules (violations invalidate the paper):**
1. NO test-set evaluation before the freeze tag (Phase 7). Tuning
   and model selection use the validation partition only.
2. NO fabricated, estimated, or "representative" numbers anywhere.
   Every reported number must trace to a run artifact on disk.
3. NO data augmentation, no synthetic data, no external data. The
   852 released images, as-is, are the entire corpus.
4. NO silent deviations from this spec. Any deviation gets logged in
   `DEVIATIONS.md` with date and reason, and reported to the authors.
5. The same fixed split, the same seed list, and the same tuning
   budget for every arm. If an arm fails a run, rerun that seed, do
   not swap seeds.
6. Null or negative results are recorded and reported exactly like
   positive ones.
7. Do not re-host or redistribute the dataset images. We release
   only our own artifacts (annotations, split indices, code,
   results) keyed to the original filenames.

---

## 1. Environments and repository layout

Two environments: local VS Code for development and smoke tests
(CPU is fine), Colab or Kaggle for GPU runs. Same repo both places.

**Framework decision (resolves a paper TODO):** TensorFlow/Keras,
to stay close to the source study's stack. Record the exact version.
If any SCA component proves impractical in Keras, switching to
PyTorch is allowed ONLY as a logged deviation, and then A0 must be
re-run in PyTorch too so the internal anchor stays in one framework.

**requirements.txt (pin after first successful install):**
```
tensorflow==2.*        # pin exact version once installed
numpy
pandas
scikit-learn
scipy
statsmodels
matplotlib
pillow
```

**Repository layout (create exactly this):**
```
script-aware-dyslexia/
  README.md
  DEVIATIONS.md
  LICENSE                  # authors choose; MIT suggested for code
  requirements.txt
  configs/
    grid.yaml              # tuning grid (frozen at Phase 7)
    endpoints.yaml         # co-primary endpoints + analysis plan
  data/
    raw/                   # dataset as downloaded (gitignored)
    annotations/
      scripts_annotA.csv   # annotator 1
      scripts_annotB.csv   # annotator 2
      scripts_final.csv    # adjudicated
      agreement_report.md
    splits/
      split_v1.json        # fixed train/val/test filename lists
  src/
    data.py                # loading, resize, normalize
    annotate.py            # annotation helper
    make_split.py
    models.py              # backbone, SCA, all arms
    train.py               # one run = one (arm, config, seed)
    evaluate.py            # metrics incl. per-script
    stats.py               # paired tests, dz, CI, power
    figures.py
  runs/                    # one folder per run (gitignored, archived)
  results/
    all_runs.csv           # master log, append-only
  figures/
  paper/                   # the .tex and .bib live here
```

---

## 2. Phase 1: Data acquisition and verification

1. Download the dataset from the Google Drive link published in
   Kashif et al. (2026), Section 3.8 / endnote 1, into `data/raw/`.
2. **Record in README.md:** access date, the link used, and save the
   folder listing (`ls -R > data/raw/manifest.txt`). This resolves
   the access-date TODO in the paper's Data section.
3. Verify counts programmatically and write `data/raw/checksums.csv`
   (filename, sha256, class folder). Expected: 852 images total,
   426 in YES, 426 in NO. **If counts differ, STOP and report;**
   the paper's Data section is written against 426/426.
4. Recommended (authors, not Claude Code): email the corresponding
   author confirming use is within intended terms; file the reply.

---

## 3. Phase 2: Script annotation (dual annotator)

The release has no script metadata. Both authors annotate all 852
images independently.

**`src/annotate.py`:** minimal loop that displays each image and
records one row per image. CSV schema (both annotator files):
```
filename, class_folder, script, digit_glyph, ambiguous, note
# script      in {urdu, english, digit}
# digit_glyph in {western, eastern_arabic_indic, na, unclear}
# ambiguous   in {0, 1}   (glyph form ambiguous between scripts)
```
Rules: annotate from the image alone; `digit_glyph` is mandatory
when script=digit; when unsure set ambiguous=1 and describe in note.

**Adjudication and agreement (`agreement_report.md`):**
- Cohen's kappa on `script` between annotators (sklearn
  `cohen_kappa_score`), plus raw agreement percentage.
- Disagreements resolved jointly; every resolution logged
  (filename, A's tag, B's tag, final, reason).
- Report the digit-glyph composition (count western vs eastern) and
  the count of ambiguous flags. These numbers go into the paper's
  Data section TODOs verbatim.
Output: `scripts_final.csv`, the single source of truth for script
tags. Release this file (it contains no images).

---

## 4. Phase 3: Fixed split

`src/make_split.py`:
- Input: `scripts_final.csv`.
- Stratify jointly on (class, script): 80/10/10 train/val/test.
- Global split seed: **1337** (recorded in split_v1.json).
- Output `data/splits/split_v1.json`:
  `{"seed": 1337, "train": [...], "val": [...], "test": [...]}`
  as filename lists, plus per-partition class x script counts.
- Sanity assertions: partitions disjoint, union = 852, each
  partition preserves class balance and script mix within rounding.
This file is generated ONCE and never regenerated. Every arm loads
it. Commit it.

---

## 5. Phase 4: Models

All code below is a **reference implementation**: correct in intent,
must be smoke-tested (Section 5.4) before any real run.

### 5.1 Backbone and preprocessing
- `tf.keras.applications.MobileNet` (V1), ImageNet weights,
  `include_top=False`, all conv layers frozen.
- **Input resolution: 224x224x3** (resolves a paper TODO; matches
  the source study's MobileNet input). Normalize with the
  MobileNet `preprocess_input`. Identical for every arm.
- Expose three tap points for insertion depths. Using layer names of
  Keras MobileNetV1: early = after `conv_pw_3_relu`, mid = after
  `conv_pw_7_relu`, late = after `conv_pw_13_relu`. If layer names
  differ in the pinned TF version, pick the nearest equivalents,
  record the mapping in DEVIATIONS.md.

### 5.2 SCA module (the core of the paper)
```python
import tensorflow as tf
from tensorflow.keras import layers

class SCA(layers.Layer):
    """Script-conditioned channel attention (SE-style).
    conditioned=True  -> gates see the script embedding (arm A2)
    conditioned=False -> gates see a learned constant (arm A1)
    Parameter counts are IDENTICAL across the two settings."""
    def __init__(self, channels, d=16, r=16, conditioned=True, **kw):
        super().__init__(**kw)
        self.conditioned = conditioned
        self.embed = layers.Embedding(3, d)          # urdu/english/digit
        self.const = self.add_weight(                # used when not conditioned
            "const_embed", shape=(d,), trainable=True)
        self.fc1 = layers.Dense(max(channels // r, 4), activation="relu")
        self.fc2 = layers.Dense(channels, activation="sigmoid")

    def call(self, f, script_id):
        z = tf.reduce_mean(f, axis=[1, 2])           # squeeze: (B, C)
        if self.conditioned:
            e = self.embed(script_id)                # (B, d)
        else:
            _ = self.embed(script_id)                # keep graph identical
            e = tf.tile(self.const[None, :], [tf.shape(f)[0], 1])
        a = self.fc2(self.fc1(tf.concat([z, e], -1)))
        return f + f * a[:, None, None, :]           # residual gate
```
Note: the unconditioned branch still *contains* the embedding table
so trainable-parameter counts match exactly; it just never uses the
script input. This is the matched-capacity design from the paper.

### 5.3 Arms
- **A0 re-run:** MobileNetV1 frozen + Flatten + Dense(128, relu) +
  Dropout(0.5) + Dense(1, sigmoid), per the source study's head; run
  with Adam, SGD, RMSProp. Also re-run CNN-from-scratch, VGG16,
  InceptionV3, MobileNetV2, MobileNetV3Small each with its best
  optimizer from the source study's Table 3. (Full 6x3 matrix is
  optional if budget allows; log which was done.)
- **A1:** backbone + SCA(conditioned=False) at swept depths + A0
  head.
- **A2:** same with conditioned=True, oracle script id from
  `scripts_final.csv`.
- **A2':** A2 at its selected config, script id from a small script
  classifier (frozen-backbone features + Dense(3, softmax)) trained
  on the same train partition; report its val/test script accuracy.
- **A3:** backbone (no SCA) + concat one-hot script onto the
  128-dim penultimate features + Dense(1, sigmoid).
- **A4:** the A0 MobileNet head trained separately on Urdu-only and
  English-only subsets; inference routed by oracle script; digits
  evaluated by whichever expert the adjudicated glyph form assigns,
  and reported separately.
- **A5:** shared frozen features; script classifier routes to one of
  three script-specific Dense(128)+Dense(1) heads.

**Capacity assertion (mandatory):**
```python
assert count_params(build_A1(cfg)) == count_params(build_A2(cfg))
```
Print both counts into every run log.

### 5.4 Smoke tests (before anything else)
`pytest`-style checks, CPU, tiny subset (32 images):
1. Data pipeline yields (image, script_id, label) with correct
   shapes and value ranges.
2. Each arm builds, trains 2 epochs without NaN, saves and reloads.
3. A1/A2 parameter equality assertion passes at every depth config.
4. Deterministic rerun: same seed twice -> identical first-epoch
   loss (set `tf.keras.utils.set_random_seed(seed)` and
   `tf.config.experimental.enable_op_determinism()`; if full
   determinism is impossible on GPU, log it and rely on the
   repeated-seeds design).

---

## 6. Phase 5: Tuning protocol

**Proposed grid (goes into `configs/grid.yaml`; authors may amend
BEFORE the freeze, never after):**
```
optimizer:        [rmsprop, adam]
learning_rate:    [1e-3, 3e-4, 1e-4]
weight_decay:     [0.0, 1e-4]
sca_embed_dim d:  [8, 16]          # SCA arms only
sca_ratio r:      [8, 16]          # SCA arms only
depth_config:     [early, mid, late, all]   # SCA arms only
batch_size:       16
max_epochs:       50
early_stopping:   patience 5 on val loss, restore best weights
loss:             binary cross-entropy
```
**Procedure (identical for every arm, this is the fairness
guarantee):**
1. For each config in the arm's grid, train with 3 fixed tuning
   seeds (101, 102, 103); score = mean val accuracy.
2. Select the arm's best config by that score. Ties: higher val F1,
   then fewer parameters.
3. The selected config is what runs in the pilot and final phase.
Log every tuning run to `results/all_runs.csv` with
`phase=tuning`. A1 and A2 sweep depth_config identically; each
reports its own best fair configuration, per the paper.

---

## 7. Phase 6: Pilot and power (resolves the "S" TODO)

1. Run A1 and A2 at their selected configs with pilot seeds
   201..205 (5 seeds), shared between arms.
2. Compute the five seed-paired val-accuracy differences
   (A2 - A1) and their standard deviation `sd_pilot`.
3. Choose S:
```python
from statsmodels.stats.power import TTestPower
delta_min = 0.015          # pre-specified minimal effect: 1.5 pts
S = TTestPower().solve_power(effect_size=delta_min/sd_pilot,
                             power=0.80, alpha=0.025)  # Holm-adjusted
S = int(min(max(np.ceil(S), 10), 30))   # floor 10, ceiling 30
```
4. Record `sd_pilot`, the computed S, and the applied floor/ceiling
   in `configs/endpoints.yaml` and in the run log. These numbers go
   into the paper's Setup TODOs verbatim.

---

## 8. Phase 7: FREEZE (the integrity gate)

Before ANY test-set evaluation:
1. `configs/grid.yaml` (as actually used), `configs/endpoints.yaml`
   (co-primary endpoints, Holm across the two, delta_min, S, seed
   list, analysis plan) are final.
2. Commit and tag: `git tag freeze-v1`.
3. From this point the test partition may be touched, and nothing in
   configs may change. Any post-freeze change = new tag + logged
   deviation + authors decide whether results survive.

---

## 9. Phase 8: Final runs

- Final seed list: 301..(300+S), identical across all arms.
- Run every arm at its selected config for all S seeds. Each run
  writes `runs/<arm>_<config-hash>_<seed>/` containing: resolved
  config, param counts, training history CSV, final weights, and
  per-image test predictions (`filename, y_true, y_prob, script`).
- Append one row per run to `results/all_runs.csv`:
```
phase, arm, config_hash, seed, split_file, n_params,
acc, prec, rec, f1, auc,
acc_urdu, acc_english, acc_digit, f1_urdu, f1_english, f1_digit,
runtime_s, env, timestamp
```
- Compute note: frozen backbone, 852 images, <=50 epochs means
  minutes per run on a Colab/Kaggle GPU; the full matrix (7 arms x
  up to 30 seeds + tuning) fits a free-tier budget across a few
  sessions. Checkpoint `all_runs.csv` to Drive after every run.

---

## 10. Phase 9: Statistical analysis (`src/stats.py`)

Per co-primary endpoint (pooled delta; Urdu-subset delta), on the S
seed-paired differences d_i = A2_i - A1_i:
```python
from scipy import stats
t, p_t   = stats.ttest_rel(a2, a1)
w, p_w   = stats.wilcoxon(a2 - a1)          # robustness check
dz       = (a2 - a1).mean() / (a2 - a1).std(ddof=1)
ci       = stats.t.interval(0.95, S-1,
             loc=(a2-a1).mean(),
             scale=stats.sem(a2 - a1))
```
- Apply Holm across the two co-primary p-values (t-test is primary;
  Wilcoxon reported alongside).
- Everything else (A2 vs A3, depth sweep, A2', A4, A5, non-Urdu
  breakdowns) is EXPLORATORY: report means, SDs, and CIs, no
  significance claims.
- Output `results/stats_summary.json` + a human-readable
  `results/stats_summary.md`. Wide CIs are expected at n=852;
  report them as they are.

---

## 11. Phase 10: Figures and tables (`src/figures.py`)

All figures: matplotlib, vector PDF into `figures/`, colorblind-safe
palette, error bars = SD over seeds unless stated, no chartjunk.
1. `tab_main.tex`: all arms x metrics, mean +/- SD, pooled; plus
   n_params column. Generated, not hand-typed.
2. `fig_primary_delta.pdf`: the two co-primary paired deltas with
   95% CI (point + interval), one panel pooled, one Urdu-subset.
3. `fig_per_script.pdf`: grouped bars, accuracy per script per arm
   (A0-best, A1, A2, A3), SD error bars.
4. `fig_depth_sweep.pdf`: val accuracy vs depth_config for A1 and
   A2 (tuning-phase data, labeled as such).
5. `fig_attention_maps.pdf`: qualitative panel; for a few test
   images per script, channel-gate profiles or Grad-CAM overlays of
   A2 vs A1. Labeled qualitative support in the caption; images
   shown must come from the released dataset (consent for open
   publication is documented in the source paper, cite it in the
   caption).
6. Appendix: training curves per arm (one seed, stated).

---

## 12. Phase 11: Results handoff package (what comes back to me)

To draft the Results section I need, verbatim files, no summaries:
1. `results/all_runs.csv`
2. `results/stats_summary.json` and `.md`
3. `data/annotations/agreement_report.md` and the digit-glyph counts
4. `data/splits/split_v1.json` counts block
5. The generated `tab_main.tex` and all `fig_*.pdf`
6. `DEVIATIONS.md` (even if empty)
7. The freeze tag name and repo commit hash
I will write Results/Limitations text strictly from these artifacts;
any number I cannot find in them, I will not write.

---

## 13. Phase 12: Release and the paper's links section

1. Public GitHub repo (code, configs, split_v1.json,
   scripts_final.csv, all_runs.csv, stats summaries, figure code).
   NOT the images (see rule 7).
2. Archive the release: Zenodo GitHub integration -> versioned DOI.
3. The paper's Declarations section will list, as links: (a) code +
   configs repository, (b) Zenodo DOI of the archived release,
   (c) script-annotation file, (d) split indices, (e) per-seed raw
   results, (f) the original dataset via its published source link
   (credited to Kashif et al., not re-hosted). A stub for this now
   exists in the .tex; links get filled at submission time.
4. README must state: how to reproduce every table and figure with
   one command each, the pinned environment, and the license.
5. AI-assistance note for the paper's statement: development used
   Claude (planning/drafting) and Claude Code (implementation);
   final venue policy check happens at venue selection.

---

## 14. Final checklist before results come back
- [ ] 852/426/426 verified, access date recorded
- [ ] Dual annotation done, kappa reported, digit glyphs counted
- [ ] split_v1.json committed, never regenerated
- [ ] Smoke tests pass, incl. A1==A2 param assertion
- [ ] Tuning done with identical budget per arm, all runs logged
- [ ] Pilot run, sd_pilot and S recorded
- [ ] freeze-v1 tagged BEFORE any test evaluation
- [ ] Final S seeds run for all arms, per-image predictions saved
- [ ] Stats + figures generated from artifacts only
- [ ] DEVIATIONS.md reviewed by both authors
