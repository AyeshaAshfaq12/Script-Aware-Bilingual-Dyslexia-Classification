# Phase Record — Phase 10: Figures, tables and the mechanism probe

- **Date:** 2026-09-03
- **Phase:** Phase 10 (Guide §11)
- **Status:** **COMPLETE.** All five figures and both tables generated.

## Objective

Produce the guide's §11 deliverables, every number traced to an artifact
on disk and none hand-entered.

## Deliverables

| # | deliverable | file | status |
|---|---|---|---|
| 1 | main results table | `figures/tab_main.tex` | done |
| 2 | co-primary deltas | `figures/fig_primary_delta.pdf` | done |
| 3 | per-script breakdown | `figures/fig_per_script.pdf` | done |
| 4 | depth sweep | `figures/fig_depth_sweep.pdf` | done |
| 5 | attention visualisation | `figures/fig_attention_maps.pdf` | done, see D-009 |
| 6 | training curves | `figures/fig_training_curves.pdf` | done |
| 7 | A0 backbone comparison | `figures/tab_a0_backbones.tex` | done |

Style: vector PDF, Okabe-Ito colourblind-safe palette assigned per arm
and never cycled, error bars are SD over seeds, recessive grid.

## Deliverable 5: what was actually built, and why

The guide names `fig_attention_maps.pdf`. SCA is squeeze-and-excitation
style **channel** attention: the gate is a vector `a` in (0,1)^C applied
as `f + f*a`, broadcast identically across every spatial position. There
is no spatial term anywhere in the mechanism, so there is no spatial
attention to visualise. Producing a heat map would have meant
substituting a different method (Grad-CAM or similar) and presenting it
as though it showed the SCA gates. Recorded as **D-009**.

`src/attention_probe.py` measures the gates directly and answers the
question the ablation actually asks:

> holding the image fixed, how far does the gate move when only the
> script id changes?

**Result.** Over 30 seeds on the test partition, 512 channels at the
`conv_pw_7_relu` tap:

| quantity | value |
|---|---|
| script-induced gate SD (image fixed) | **0.000282** |
| image-induced gate SD (script fixed) | **0.008609** |
| ratio | **0.0327** |

The conditioning signal perturbs the attention it is meant to steer by
about **3% of the image-driven scale**. This is a direct mechanistic
account of the null co-primary result: A2 barely differs from A1 in
behaviour because the script input barely differs from a constant
*inside the model*. The behavioural and mechanistic findings were
obtained independently and agree.

**Probe null check.** A1's gates cannot depend on the script by
construction — the embedding output is discarded for a learned constant.
Its measured script-induced variation is 4.87e-09, which is float32
round-off (eps ≈ 1.2e-7 against gates ≈ 0.5), **58,000× below A2's**.
The check threshold is set at that round-off scale rather than at exact
zero, because three separate float32 matmuls over identical inputs are
not bit-identical under oneDNN threading. This confirms the probe reads
the correct tensor.

No dataset image is written to disk or embedded in any figure, so hard
rule 7 is respected.

## Deliverable 7: A0 backbone reproduction

45 runs, validation partition only, five alternative backbones against
Kashif et al. Table 3. VGG16, InceptionV3 and MobileNetV2 land within
2–4 accuracy points.

`cnn_scratch` and `mobilenetv3small` **collapsed to constant
majority-class predictors** — balanced accuracy exactly 0.5000, recall
1.000, zero variance across all seeds and learning rates, AUC 0.5029 and
0.5482. Their 0.6613 is the validation base rate (41/62), not
performance. Both rows are retained and flagged rather than dropped, per
hard rule 6. Recorded as **D-010**; `src/a0_table.py` detects the
condition automatically so it cannot be missed on a re-run.

## Defects found and fixed during this phase

**`tab_main` misreported the seed count.** It took `S` from an arbitrary
arm via `next(iter(arms.values()))` and printed it in the caption for
every row, so any incomplete arm would have produced a wrong `S` for the
whole table. Now reports per-arm `n` and warns when arms are uneven.

**`run_tuning.py` leaked its lock file.** It called
`RunnerLock().__enter__()` by hand, so `__exit__` never ran and the lock
survived clean exits, blocking the next runner against a dead PID
(**D-011**). Now a `with` block, matching `run_final.py`. No results
were affected.

## Artifacts

- `results/attention_gates.json`, `results/attention_gates.npz`
- `results/a0_backbones.md`, `results/a0_backbones.json`
- `results/stats_summary.json`, `results/stats_summary.md`
- `figures/` — 5 PDFs, 2 `.tex` tables

## Reproduce

```
python src/attention_probe.py
python src/a0_table.py
python src/stats.py
python src/figures.py
```

## Status

**COMPLETE.** 58 tests pass. Proceeds to Phase 11 (handoff package).
