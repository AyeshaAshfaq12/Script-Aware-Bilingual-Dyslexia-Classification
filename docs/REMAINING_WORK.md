# Remaining Work — runbook to completion

What is left, in execution order, with commands, measured time
estimates, blockers, and who owns each item.

- **Generated:** 2026-09-02
- **Freeze:** `freeze-v1` at commit `e491aaf` — configs are final and
  must not change
- **Spec:** `Documents_and_Guides/ScriptAwareDyslexia_ExperimentGuide_v1_2026-09.md`

---

## 1. Status at a glance

| Phase | State |
|---|---|
| 0 Scaffold | ✅ done |
| 1 Data verification + corpus definition | ✅ done |
| 2 Script annotation (κ = 0.8489) | ✅ done |
| 3 Fixed split (seed 1337) | ✅ done |
| 4 Models + smoke tests (58 pass) | ✅ done |
| 5 Tuning + selection (218 runs) | ✅ done |
| 6 Pilot + power (S = 30) | ✅ done |
| 7 **FREEZE** (`freeze-v1`) | ✅ done |
| 8 Final runs | 🔄 **in progress** — 11/210 |
| 9 Statistics | ⏳ blocked on 8 |
| 10 Figures + tables | ⏳ partly blocked on 8 and 9 |
| 11 Handoff package | ⏳ blocked on 9, 10 |
| 12 Release (Zenodo, links) | ⏳ authors |

**Total remaining compute: ≈ 9 h** on the local CPU
(Intel i7-6600U, 2 cores), before thermal throttling.

---

## 2. Phase 8 — Final runs (IN PROGRESS)

Seeds **301–330** (S = 30), identical across all arms, evaluated on the
**test** partition. Permitted only because `freeze-v1` exists.

### 2.1 Core arms — running now

```bash
cd src
python run_final.py --arms A1 A2 A0 A3 --partition test
```

| arm | done | per run | remaining |
|---|---|---|---|
| A1 | 11/30 | 138 s | ~45 min |
| A2 | 0/30 | 138 s | ~69 min |
| A0 | 0/30 | ~30 s | ~15 min |
| A3 | 0/30 | ~30 s | ~15 min |

**≈ 2.4 h remaining.** Resumable — interrupt and re-run the same command.

### 2.2 Context arms — not started

```bash
python run_final.py --arms A5 A2p A4 --partition test
```

| arm | what it does | estimate |
|---|---|---|
| A5 | script classifier routes to three script-specific heads | ~20 min |
| A2p | A2 with **predicted** script ids (classifier trained per seed) | ~80 min |
| A4 | per-script experts (Urdu / English), routed at inference | ~35 min |

**≈ 2.3 h.**

> ⚠️ **Risk to check first.** The A5 / A2′ / A4 code paths are written
> and parse, but have **never been executed**. Run one seed before
> committing to all 30:
> ```bash
> python -c "
> import run_final, sys
> sys.argv = ['x','--arms','A5','--partition','test']
> run_final.main()" 2>&1 | head -20
> ```
> Specifically worth verifying: A4's digit routing (all digits are
> Western numerals, so they route to the English expert, and must be
> reported separately per guide §5.3), and that A2′ logs under
> `A2_mid_predicted` so `base_arm()` maps it to `A2p` rather than
> merging it into the co-primary arm.

### 2.3 A0 supporting backbones — 44 runs outstanding

```bash
python run_tuning.py --arms A0_others
```

Re-runs of the source study's other backbones under our fixed split, at
each one's best optimizer from Kashif et al. Table 3.

| backbone | runs | note |
|---|---|---|
| cnn_scratch | 1/9 done | **23.8 min each** — trains end to end, no cacheable prefix (D-007) |
| vgg16 | 0/9 | slow feature extraction, then fast heads |
| inceptionv3 | 0/9 | as above |
| mobilenetv2 | 0/9 | fast |
| mobilenetv3small | 0/9 | fast |

**≈ 3.9 h, dominated by CNN-from-scratch.** These are validation-phase
supporting re-runs (3 seeds), not the S-seed test protocol, and must be
reported in a **separate supporting table** labelled as such — not in
`tab_main.tex`.

---

## 3. Phase 9 — Statistics

```bash
python src/stats.py
```

Blocked until A1 and A2 finals are complete. Produces
`results/stats_summary.json` and `results/stats_summary.md`.

Runs: paired *t*-test (primary), Wilcoxon (robustness), Cohen's *d_z*,
95% CI, and **Holm across the two co-primary endpoints** (pooled delta,
Urdu-subset delta). Everything else is emitted as exploratory with no
significance claim.

**Reporting obligation fixed before unblinding** (`endpoints.yaml`):
achieved power is **0.121** for the pre-specified 1.5-point effect, and
the design detects ~4.33 points at 80% power. A null must be reported
as *"no effect larger than ~4 points was detected"* — not as evidence
that script conditioning fails. **The confidence interval carries the
content, not the p-value.**

---

## 4. Phase 10 — Figures and tables

```bash
python src/figures.py
```

| Guide §11 item | Status |
|---|---|
| 1. `tab_main.tex` | implemented, needs final runs |
| 2. `fig_primary_delta.pdf` | implemented, needs `stats_summary.json` |
| 3. `fig_per_script.pdf` | implemented, needs final runs |
| 4. `fig_depth_sweep.pdf` | ✅ **generated** |
| 5. `fig_attention_maps.pdf` | ❌ **NOT IMPLEMENTED** |
| 6. Appendix training curves | implemented, needs final runs |

### The one real gap: `fig_attention_maps.pdf`

Guide §11 item 5 asks for a qualitative panel — channel-gate profiles or
Grad-CAM overlays comparing A2 against A1 on a few test images per
script. It is **not written yet**, because it needs the saved suffix
weights that Phase 8 is producing now.

Recommended approach (simpler and more honest than Grad-CAM here):
plot the **SCA channel-gate activation profiles** per script. A2's gates
are conditioned on script, so if the mechanism does anything the three
script curves should separate; A1's gates see a learned constant, so its
curves must be identical by construction. That contrast is the figure's
actual scientific content, and it doubles as a mechanism check.

Caption must state it is qualitative support, and cite the source paper
for consent to publish the images.

---

## 5. Phase 11 — Handoff package

Guide §12 asks for these verbatim, no summaries:

| # | Artifact | Status |
|---|---|---|
| 1 | `results/all_runs.csv` | 🔄 filling |
| 2 | `results/stats_summary.json` + `.md` | ⏳ Phase 9 |
| 3 | `data/annotations/agreement_report.md` + digit-glyph counts | ✅ ready |
| 4 | `data/splits/split_v1.json` counts block | ✅ ready |
| 5 | `figures/tab_main.tex` + all `fig_*.pdf` | ⏳ Phase 10 |
| 6 | `DEVIATIONS.md` | ✅ ready (D-001…D-008) |
| 7 | Freeze tag + commit hash | ✅ `freeze-v1` / `e491aaf` |

---

## 6. Phase 12 — Release (authors)

1. Repo is public at
   `github.com/AyeshaAshfaq12/Script-Aware-Bilingual-Dyslexia-Classification`
   — code, configs, `split_v1.json`, `scripts_final.csv`, `all_runs.csv`,
   stats, figure code. **Never the images** (hard rule 7, verified: no
   image or archive exists anywhere in git history).
2. Archive via Zenodo → versioned DOI.
3. Fill the paper's Declarations links: repo, DOI, annotation file,
   split indices, per-seed raw results, original dataset source.
4. README must state one-command reproduction, the pinned environment,
   and the licence.

---

## 7. Blocked on author input

| Item | Why it is blocked |
|---|---|
| **`LICENSE` file** | MIT chosen, but the copyright holder line needs exact author names. Not inferred. `README.md` line 170. |
| **Author list / affiliations** | paper `\todonote{Author order to confirm}` |
| **Corresponding-author confirmation** | guide §2 step 4 recommends written confirmation that this use is within intended terms |
| **Both authors to review `DEVIATIONS.md`** | guide §14 final checklist |

---

## 8. Paper TODOs

**19 `\todonote` markers remain** in
`Documents_and_Guides/ScriptAwareDyslexia_TBD_v1_2026-09.tex`.

Several are already answerable from artifacts produced here:

| Paper TODO | Answer from this repo |
|---|---|
| input resolution / normalization | 224×224×3, MobileNet `preprocess_input` |
| framework choice | TensorFlow 2.20.0 / Keras 3.15.1, CPU |
| digit glyph forms | **135 Western, 0 Eastern Arabic-Indic** |
| annotation agreement statistic | κ = **0.8489**, raw 92.13% (human-vs-AI, D-001) |
| ambiguous-glyph count | **117** (18.3%) |
| access date + link | 2026-08-31, public link (README access record) |
| concrete tuning grid | `configs/grid.yaml` (frozen) |
| pilot SD and resulting S | sd_pilot **0.0736**, S = **30** (ceiling applied) |

**Sections needing rewriting, not just filling:**

- **Data.** The released 426/426 balance is a file-level artefact of
  duplication inside `No/`. The deduplicated corpus is **406/212** from
  **618** unique images. This must be stated plainly.
- **Setup.** Must report the achieved power (0.121), not merely the
  target (0.80).
- **Method.** Must say the primary comparison is at a **fixed common
  insertion depth** (D-006), with the full depth sweep reported
  separately.
- **Limitations.** Must add: duplicate/contradictory images in the
  release; the 62-image partition meaning one image = 1.61 accuracy
  points; and that the study is powered only for ~4-point effects.

---

## 9. Ordered runbook

```bash
cd "d:/Github Reporsitories/Script-Aware-Bilingual-Dyslexia-Classification/src"

# 1. finish core final runs               (~2.4 h, running now)
python run_final.py --arms A1 A2 A0 A3 --partition test

# 2. smoke one context arm BEFORE all 30  (~2 min)
python run_final.py --arms A5 --partition test   # watch the first seed

# 3. context arms                         (~2.3 h)
python run_final.py --arms A5 A2p A4 --partition test

# 4. statistics                           (seconds)
python stats.py

# 5. figures and tables                   (seconds)
python figures.py

# 6. supporting backbone re-runs          (~3.9 h, can run last/overnight)
python run_tuning.py --arms A0_others

# 7. regenerate figures to include them
python figures.py
```

Then: implement `fig_attention_maps.pdf`, update
`docs/SESSION_LOG.md` §14, write the Phase 8–11 phase records, and
commit + push.

**Every runner is resumable and holds an exclusive lock** — interrupt
freely, re-run the same command, and never start two at once (D-008).

---

## 10. Expected result, stated in advance

Validation showed **no A2 advantage at any insertion depth**
(0.0000, −0.0054, −0.0108, −0.0108), and the design has **12% power**
for the pre-specified effect.

The likely outcome is a **null that cannot distinguish "no effect" from
"an effect too small for this corpus to resolve."** That is recorded
here before the test results exist so it cannot be reframed afterwards.
It is a genuine finding about the limits of this dataset, and hard rule
6 requires reporting it with the same prominence as a positive result.

---

## 11. Moving to another machine

**Cloning the repo is not sufficient.** The code is fully portable —
`corpus_v1.csv` stores relative uids (`No/1 (1).jpeg`), `data.py`
resolves paths from the repo root at runtime, and a normal `git clone`
brings the `freeze-v1` tag so the hard-rule-1 gate keeps working — but
four directories are deliberately gitignored.

| Directory | Size | Needed? | How to restore |
|---|---|---|---|
| `data/raw/` | 57 MB | **MANDATORY** | re-download from the Drive link in README; **never** committed (hard rule 7) |
| `runs/` | **837 MB** | **YES — see the trap below** | copy across, or archive to Drive/Zenodo |
| `data/annotations/contact_sheets/` | 40 MB | no | regenerate: `python src/make_contact_sheets.py` |
| `Related_Research_Work/` | 64 MB | no | source PDF + archive; reference only |

### Setup on the new machine

```bash
git clone https://github.com/AyeshaAshfaq12/Script-Aware-Bilingual-Dyslexia-Classification.git
cd Script-Aware-Bilingual-Dyslexia-Classification
pip install -r requirements.txt          # TF 2.20.0, Python 3.9-3.13

# restore the images into data/raw/{Yes,No}, then VERIFY them:
python src/verify_data.py
```

`verify_data.py` re-checks all 852 files against the **committed**
`data/raw/checksums.csv` (SHA-256 per file), so a corrupted or wrong
download is caught immediately rather than silently changing results.
Confirm the freeze tag survived the clone:

```bash
git tag --list freeze-v1     # must print freeze-v1
```

If it does not (e.g. a shallow clone), `train.py` will refuse every
test-partition run — which is the gate behaving correctly, not a bug.

### ⚠️ The `runs/` trap

`results/all_runs.csv` **is** tracked, so every resumable runner will
**skip** all completed runs on the new machine. But their per-run
artifacts live in `runs/` and are **not** in the repo. Skipped runs are
never regenerated, so you would end up with results rows whose
underlying artifacts are missing. That breaks:

- guide §9's requirement that per-image test predictions be saved per run,
- `fig_training_curves.pdf`, which reads `runs/*/history.csv`,
- `fig_attention_maps.pdf`, which needs the saved suffix weights,
- the Phase 11 handoff package.

Three ways out, best first:

1. **Copy `runs/` across** (837 MB, USB or cloud). Cleanest, loses
   nothing. This is what the guide intends — its layout comment reads
   `runs/ # one folder per run (gitignored, archived)`.
2. **Archive `runs/` to Drive/Zenodo** and pull it on the new machine.
   Same effect, and it is needed for release anyway.
3. **Re-run from scratch.** Delete the affected rows from
   `results/all_runs.csv` and re-run. Runs are deterministic — verified
   bit-identical across separate processes (D-008) — so results
   reproduce exactly, but you pay the full recompute (~6 h of tuning
   plus whatever finals are done).

**If you are mid-Phase-8, copy `runs/` — do not restart.** Rerunning
would reproduce the same numbers, but it would waste hours for nothing.
