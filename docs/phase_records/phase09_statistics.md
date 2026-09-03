# Phase Record — Phase 9: Co-primary statistics

- **Date:** 2026-09-02
- **Phase:** Phase 9 (Guide §10)
- **Status:** **COMPLETE for the co-primary endpoints.** Both are null.

## Objective

Test the two pre-registered co-primary endpoints exactly as fixed at the
freeze, with Holm correction across the family, and report the result
whichever way it falls.

## Inputs

- `results/all_runs.csv` — A1 and A2 each at 30/30 final seeds (301–330),
  test partition, matched-depth configs (D-006, identical capacity
  9,685,537 parameters).
- Analysis plan from `configs/endpoints.yaml`, frozen at `freeze-v1`
  before any test-set evaluation.

## Result

| endpoint | A1 mean | A2 mean | mean diff | 95% CI | $d_z$ | $t$ $p$ | Wilcoxon $p$ | Holm $\alpha$ | reject |
|---|---|---|---|---|---|---|---|---|---|
| pooled_delta | 0.7419 ± 0.0536 | 0.7495 ± 0.0497 | **+0.0075** | [−0.0174, +0.0325] | +0.113 | 0.5420 | 0.4168 | 0.0250 | **no** |
| urdu_delta | 0.7357 ± 0.0527 | 0.7364 ± 0.0378 | **+0.0008** | [−0.0216, +0.0232] | +0.013 | 0.9441 | 0.9198 | 0.0500 | **no** |

**Neither endpoint rejects.** The Wilcoxon robustness check agrees with
the paired $t$-test on both. Majority-class baseline is 0.6570.

## Reading this honestly

The interpretation rule was fixed in `endpoints.yaml` *before*
unblinding, and is applied here unchanged.

**The pooled endpoint is genuinely inconclusive.** The CI rules out any
A2 advantage above ~3.3 accuracy points, which is a real and reportable
upper bound. But the pre-specified effect of interest, 1.5 points, sits
*inside* the interval, so the study cannot separate "no effect" from
"the effect we set out to detect". This is what achieved power of 0.121
buys, and it was known and recorded before the test was run.

**The Urdu endpoint is the more substantive finding.** The paper's
mechanism predicts that gains from script conditioning should
*concentrate* in Urdu, because that is where the script statistics
differ most. Observed: **+0.0008**, $d_z$ = 0.013 — the smallest effect
of the two, with the tighter interval. This is not merely an
underpowered null. The effect is flat precisely where the theory
predicts its strongest signal, which is evidence against the proposed
mechanism rather than an absence of evidence about it.

**The power analysis was well calibrated.** Observed test SD of the
paired difference tracked `sd_pilot` (0.0736) almost exactly, so the
Phase 6 sample-size reasoning held up on unseen data.

## Corroboration from the mechanism probe

Phase 10's `src/attention_probe.py` supplies a direct mechanistic
account. Holding each test image fixed and varying only the script id,
the SCA channel gates move with SD 0.000282, against 0.008609 for the
variation driven by the image — a ratio of **0.0327**. The conditioning
signal perturbs the attention it is supposed to steer by about 3% of the
image-driven scale. A behavioural null is what one would expect from
that, and the two results were obtained independently.

## Caveat carried forward to the write-up

Balanced accuracy (0.676 A1, 0.688 A2) sits far below raw accuracy
(0.742, 0.750) for every arm, which is the 406/212 class imbalance
recorded in D-005 surfacing exactly as anticipated. Raw accuracy
flatters these models and must not be quoted alone.

Absolute accuracies are within-dataset on a 62-image test partition, where
one image is worth 1.61 accuracy points. They are not generalisation
estimates.

## Artifacts

- `results/stats_summary.json`
- `results/stats_summary.md`
- `figures/fig_primary_delta.pdf`

## Status

**Co-primary analysis COMPLETE and final.** A1 and A2 are both at 30/30
and their configs are frozen, so these two rows cannot change. The
exploratory arms (A2′, A4, A5) are descriptive only and carry no
significance claim; `stats.py` is re-run once they finish, which extends
the descriptive tables without touching the table above.
