# Phase Record — Phase 7: FREEZE (the integrity gate)

- **Date:** 2026-09-02
- **Phase:** Phase 7 (Guide §8)
- **Status:** **PASS** — configs frozen, `freeze-v1` tagged.

## Objective

Fix the tuning grid, the endpoints, and the analysis plan **before any
test-set evaluation**. After this tag the test partition may be touched
and nothing in `configs/` may change.

## Pre-freeze checklist

| Requirement (guide §8) | Status |
|---|---|
| `configs/grid.yaml` final, as actually used | **frozen: true** |
| `configs/endpoints.yaml` final (endpoints, Holm, delta_min, S, seeds, analysis plan) | **frozen: true** |
| Every arm's config selected on validation only | **PASS** |
| No test-set evaluation has occurred | **PASS** |
| Smoke tests pass | **PASS** (58 tests) |
| Deviations logged | **PASS** (D-001 … D-008) |
| Commit and tag `freeze-v1` | **done** |

## What is frozen

**Grid** (`configs/grid.yaml`) — amended before the freeze as guide §6
permits, with the rationale and the measurements recorded in the file:
optimizer × learning rate × the full four-point depth sweep; weight
decay fixed at 0.0, SCA `d` and `r` fixed at 16; tuning seeds 101–103.

**Selected configurations** (`results/selected_configs.json`), all
chosen on validation by mean accuracy, ties by val F1 then fewer
parameters:

| arm | configs evaluated | selected | mean val acc |
|---|---|---|---|
| A0 | 9 | adam, 1e-4 | 0.7258 |
| A1 | 24 | depth=all, adam, 1e-4 | 0.8065 |
| A2 | 24 | depth=mid, rmsprop, 1e-4 | 0.8011 |
| A3 | 6 | rmsprop, 1e-4 | 0.7473 |
| A5 | 6 | adam, 3e-4 | 0.7581 |

**Primary pair** (D-006), at matched depth `mid` — identical capacity:

| arm | optimizer | lr | n_params |
|---|---|---|---|
| A1 (control) | rmsprop | 3e-4 | **9,685,537** |
| A2 (treatment) | rmsprop | 1e-4 | **9,685,537** |

**Endpoints** (`configs/endpoints.yaml`): co-primary pooled and
Urdu-subset A2−A1 accuracy deltas; Holm across the two; delta_min =
0.015; **S = 30**; final seeds **301–330**, identical across all arms.

**Power, recorded honestly:** sd_pilot = 0.0736; 80% power for a
1.5-point effect would require S = 232; the ceiling of 30 binds, giving
**achieved power 0.121** and a detectable effect of **4.33 points**. The
interpretation rule — that a null is not evidence of no effect, only of
no effect above ~4 points, and that the CI carries the content — is
fixed in `endpoints.yaml` *before* unblinding.

## Tuning totals at the freeze

218 unique runs logged (208 tuning + 10 pilot), all on validation:
A1 72, A2 72, A0 27, A3 18, A5 18, cnn_scratch 1.

## Hard rule 1, enforced and demonstrated

`src/train.py::assert_may_touch_test()` raises unless the `freeze-v1`
tag exists. Demonstrated before tagging:

```
$ python run_final.py --dry-run --partition test
RuntimeError: HARD RULE 1: no test-set evaluation before the freeze.
Tag `freeze-v1` does not exist.
```

The same command with `--partition val` ran normally. The gate is
checked before any test data is read, not after.

## Issues encountered

**Concurrent runners produced duplicate rows** (`DEVIATIONS.md` D-008).
Three runner processes overlapped, each passing its own `already_done`
check before any had logged. All duplicated `(phase, arm, config, seed)`
keys were verified **bit-identical** in accuracy and epoch count before
anything was removed — an unplanned confirmation of the §5.4
determinism requirement. The raw log is preserved as
`results/all_runs_raw_with_duplicates.csv`; the working log was
deduplicated; and `run_tuning.py` now holds an exclusive lock for the
duration of a sweep.

## Deviations at the freeze

| id | subject | status |
|---|---|---|
| D-001 | Claude Code as annotator A | ACCEPTED |
| D-002 | class-folder casing | RESOLVED |
| D-003 | CPU-only environment | RESOLVED |
| D-004 | deduplicated 618-image primary corpus | ACCEPTED |
| D-005 | accuracy endpoints vs 406/212 imbalance | ACCEPTED |
| D-006 | primary pair at matched depth | ACCEPTED |
| D-007 | CNN-from-scratch trains end to end | RESOLVED |
| D-008 | duplicate rows from concurrent runners | RESOLVED |

## Status

**PASS.** `freeze-v1` tagged. Test-set evaluation is now permitted.
