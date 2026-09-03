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
| 5 Tuning + selection (252 runs incl. A0_others 45/45) | ✅ done |
| 6 Pilot + power (S = 30) | ✅ done |
| 7 **FREEZE** (`freeze-v1`) | ✅ done |
| 8 Final runs | ✅ done — 210/210, all 7 arms × 30 seeds |
| 9 Statistics | ✅ done — both co-primary endpoints null |
| 10 Figures + tables | ✅ done — 5 figures, 2 tables |
| 11 Handoff package | ⏳ remaining |
| 12 Release (Zenodo, links) | ⏳ authors (needs LICENSE) |

**Total remaining compute: none.** All 472 runs are logged. Phases 11
and 12 are documentation and release steps only.

---

## 2-4. Phases 8, 9 and 10 — COMPLETE (2026-09-03)

All compute is finished. 472 rows in `results/all_runs.csv`
(252 tuning, 10 pilot, 210 final). Nothing here remains to run; the
commands are kept so the work can be reproduced.

```bash
python src/run_final.py --arms A1 A2 A0 A3 --partition test   # done
python src/run_final.py --arms A5 A2p A4 --partition test     # done
python src/run_tuning.py --arms A0_others                     # done, 45/45
python src/stats.py                                           # done
python src/attention_probe.py                                 # done
python src/a0_table.py                                        # done
python src/figures.py                                         # done
```

**Phase 8** — all 7 arms at 30/30 on the test partition. Details and
the exploratory observations are in
`docs/phase_records/phase08_final_runs.md`.

**Phase 9** — **both co-primary endpoints are null.** Pooled A2−A1
+0.0075, 95% CI [−0.0174, +0.0325], p = 0.542. Urdu-subset +0.0008,
95% CI [−0.0216, +0.0232], p = 0.944. Neither survives Holm; Wilcoxon
agrees with the t-test on both. Reported under the interpretation rule
fixed in `endpoints.yaml` before unblinding: the CI carries the content,
not the p-value, and a null is evidence only that no effect above
~4 points was detected. See `docs/phase_records/phase09_statistics.md`.

**Phase 10** — five figures and two LaTeX tables. The
`fig_attention_maps.pdf` gap flagged in the original runbook is closed:
SCA is channel attention with no spatial term, so it is rendered as gate
profiles plus a script counterfactual rather than invented heat maps
(D-009). Holding the image fixed and varying only the script id moves
the gates **3.3%** as much as varying the image does — a mechanistic
account of the null. See `docs/phase_records/phase10_figures.md`.

Three defects were found and fixed along the way: the A2′ arm-tag
collapse (caught before any final data existed), a `tab_main` caption
that reported one arm's seed count for every row, and a leaked tuning
lock (D-011). Two A0 backbones did not train at all (D-010).

---

## 5. Phase 11 — Handoff package

Guide §12 asks for these verbatim, no summaries:

| # | Artifact | Status |
|---|---|---|
| 1 | `results/all_runs.csv` | ✅ complete — 472 rows |
| 2 | `results/stats_summary.json` + `.md` | ✅ ready |
| 3 | `data/annotations/agreement_report.md` + digit-glyph counts | ✅ ready |
| 4 | `data/splits/split_v1.json` counts block | ✅ ready |
| 5 | `figures/tab_main.tex` + all `fig_*.pdf` | ✅ ready — 5 figures, 2 tables |
| 6 | `DEVIATIONS.md` | ✅ ready (D-001…D-011), authors still to review |
| 7 | Freeze tag + commit hash | ✅ `freeze-v1` / `e491aaf` |
| 8 | `results/attention_gates.json` + `.npz` | ✅ ready (mechanism probe) |
| 9 | `results/a0_backbones.md` + `.json` | ✅ ready (backbone reproduction) |

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

| Item | Status |
|---|---|
| **Author list** | **RESOLVED 2026-09-03.** First author Ayesha Ashfaq; corresponding author Abdullah Butt. Set in the paper's `\author{}` block. |
| **Affiliations** | still open — `\todonote{affiliation}` in the paper. Not inferred. |
| **`LICENSE` file** | **DEFERRED** at the authors' instruction (2026-09-03). MIT was chosen but no `LICENSE` file exists yet and the copyright holder line is unwritten. Must be settled before Phase 12 release. `README.md` line 170. |
| **Corresponding-author confirmation** | still open — guide §2 step 4 recommends written confirmation from the dataset authors that this use is within intended terms. |
| **Authors to review `DEVIATIONS.md`** | still open — deferred to a later review (2026-09-03). Guide §14 final checklist. D-001 (Claude Code as annotator A), D-004 (618-image deduplication) and D-010 (two backbones that did not train) are the entries a reviewer is most likely to question. |

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
