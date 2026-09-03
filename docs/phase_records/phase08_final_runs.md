# Phase Record — Phase 8: Final runs

- **Date:** 2026-09-02 to 2026-09-03
- **Phase:** Phase 8 (Guide §9)
- **Status:** **COMPLETE.** All 7 arms at 30/30 on the test partition.

## Objective

Run every arm at the frozen configuration across the 30 pre-registered
final seeds (301–330), identical seeds for every arm, evaluating on the
test partition now that `freeze-v1` permits it.

## What ran

| arm | description | n | test accuracy |
|---|---|---|---|
| A0 | frozen backbone, no attention | 30 | 0.7554 ± 0.0347 |
| A1 | matched-capacity control (script-agnostic) | 30 | 0.7419 ± 0.0536 |
| A2 | script-conditioned attention (oracle script) | 30 | 0.7495 ± 0.0497 |
| A2′ | as A2, predicted script ids | 30 | 0.7360 ± 0.0414 |
| A3 | — | 30 | 0.7253 ± 0.0560 |
| A4 | per-script experts, routed at inference | 30 | 0.7199 ± 0.0429 |
| A5 | identify-then-branch | 30 | **0.7720 ± 0.0390** |

210 final rows. A1 and A2 are the co-primary pair at matched depth
(D-006), identical capacity 9,685,537 parameters. Only they carry
inferential claims; everything else is exploratory and descriptive, per
guide §10 and the frozen `endpoints.yaml`.

## Execution notes

Run in two batches under the D-008 runner lock: A0/A1/A2/A3 first, then
A5, then A2′/A4. The lock serialised them deliberately — this machine
has 2 physical cores, so overlapping runners would not have been faster
and would have risked the duplicate-row corruption of D-008.

A5, A2′ and A4 were code paths that had never executed before this
phase. Each was smoke-checked on its first seed before the remaining 29
were allowed to proceed. All three worked on the first attempt.

The A2′ arm logs under the tag `A2_mid_predicted`. `base_arm()` in both
`stats.py` and `figures.py` maps that to `A2p` explicitly; a naive
`split("_")[0]` would have silently merged the predicted-script arm into
the co-primary oracle arm.

## Exploratory observations

No significance is claimed for any of the following.

**A5 is the strongest arm on every aggregate** — accuracy 0.7720,
balanced accuracy 0.7045, F1 0.8413, and digit accuracy 0.7788. Hard
routing to script-specific heads outperforms modulating attention gates
with a script embedding. This is consistent with the Phase 10 gate probe:
the SCA gates move only 3.3% as much for a script change as for an image
change, whereas A5's routing is a discrete architectural switch that
cannot be ignored.

**A0 beats both co-primary arms.** The plain frozen backbone with no
attention block at all (0.7554) scores above A2 (0.7495) and A1
(0.7419). The 9.7M-parameter SCA apparatus buys nothing over the
simplest baseline on this test set.

**A2′ falls below the A1 control** (0.7360 vs 0.7419). The script
classifier is competent — 0.8559 ± 0.0239 test accuracy over 30 seeds —
yet replacing oracle script ids with predicted ones costs more than
conditioning gains. Under realistic deployment, where no oracle script
label exists, script conditioning is worse than not doing it. This is
the most practically consequential result in the set.

**A4's digit accuracy is 0.5061, at chance.** The design has no digit
expert. Every digit in this corpus is a Western Arabic numeral, the form
used in English writing, so digits route to the English expert (guide
§5.3), which never saw a digit in training. This is a property of the
arm as specified, not a defect, but A4's pooled 0.7199 is propped up by
non-digit images and the digit figure must be reported separately rather
than averaged in.

## Caveats

Seed SDs are 3.5–5.6 accuracy points and the test partition holds 62
images, so one image is worth 1.61 points. Every arm-to-arm gap above,
including A5's 3.0-point lead over A1, sits inside that noise. A5 was
never a pre-registered endpoint and cannot be promoted to one after
seeing the data.

Balanced accuracy runs far below raw accuracy for every arm — the
406/212 class imbalance of D-005 surfacing as anticipated. Raw accuracy
flatters these models and must not be quoted alone.

Absolute values are within-dataset and are not generalisation estimates.

## Artifacts

- `results/all_runs.csv` — 210 final rows
- `runs/<arm>_<config_hash>_<seed>/` — config, history, per-image
  predictions, weights, for every run
- `results/final_a2p_a4.log`

## Status

**COMPLETE.** Proceeds to Phase 9 (statistics) and Phase 10 (figures).
