# Phase Record — Phase 6: Pilot and power

- **Date:** 2026-09-02
- **Phase:** Phase 6 (Guide §7)
- **Status:** **PASS** — with a material finding about statistical power
  that must reach the paper's Limitations section.

## Objective

Run the primary pair at their selected configs under the pilot seeds,
estimate the between-seed SD of the paired A2−A1 accuracy difference,
and solve for the repeat count S. Validation partition only.

## Actions performed

1. Resolved the Phase 5 design tension (`DEVIATIONS.md` D-006): the
   primary comparison is fixed at a **matched insertion depth** so that
   A1 and A2 have identical parameter counts.
2. Implemented `src/run_pilot.py`; ran A1 and A2 at the matched-depth
   configs under pilot seeds 201–205 (10 runs, validation only).
3. Computed `sd_pilot`, solved for S with the guide's formula, applied
   the floor/ceiling, and recorded the achieved power.
4. Wrote `results/pilot_power.json` and filled the corresponding fields
   in `configs/endpoints.yaml`.

## Configuration used

Primary pair, matched depth = `mid`, identical parameter counts
(**9,685,537** each):

| arm | depth | optimizer | lr | n_params |
|---|---|---|---|---|
| A1 (control) | mid | rmsprop | 3e-4 | 9,685,537 |
| A2 (treatment) | mid | rmsprop | 1e-4 | 9,685,537 |

Pilot seeds 201–205, shared between arms. Corpus: primary (618).

## Results

| seed | A1 | A2 | d = A2−A1 |
|---|---|---|---|
| 201 | 0.7903 | 0.7742 | −0.0161 |
| 202 | 0.7742 | 0.7419 | −0.0323 |
| 203 | 0.7258 | 0.7097 | −0.0161 |
| 204 | 0.7903 | 0.8387 | +0.0484 |
| 205 | 0.7258 | 0.8710 | +0.1452 |

- mean difference: **+0.0258**
- **sd_pilot = 0.0736**
- effect size (delta_min / sd_pilot) = 0.204
- **S unclamped = 232**; ceiling applied → **S = 30**
- **achieved power at S=30 = 0.121** (target 0.80)
- detectable effect at S=30, 80% power = **4.33 accuracy points**

Final seed list: **301–330**, identical across all arms.

## The power finding (must reach the paper)

Detecting the pre-specified 1.5-point effect at 80% power would need
**S = 232 seeds**. The guide's ceiling of 30 binds, leaving the
co-primary tests with about **12% power** for that effect.

There is also a hard arithmetic limit independent of S. The test
partition holds **62 images**, so a single image is worth **1.61
accuracy points** and the 1.5-point target lies *below the resolution of
a single-seed measurement*. Averaging over seeds refines the estimated
mean but does not reduce the seed-to-seed variance that drives power.

Raising S does not rescue this: even S=100 would only reach roughly 2.1
detectable points, and S=232 across seven arms is not feasible on the
available compute. **Decision: run the pre-registered S = 30, and report
the achieved power openly rather than presenting an underpowered null as
if it were an informative negative.** Guide §7 step 4 anticipates this
by requiring the applied ceiling to be recorded; hard rule 6 requires
null results to be reported with the same prominence as positive ones.

The decision is robust to the noisiness of a 5-seed SD estimate: the
independent estimate from the tuning sweep (0.041, pooled over 16
matched configs) also clamps to S = 30.

**Interpretation rule, fixed before any test-set evaluation:** the
co-primary tests run exactly as pre-registered, but a null result is
*not* evidence that script conditioning fails. It is only evidence that
no effect larger than roughly 4 accuracy points was detected. The
confidence interval, not the p-value, carries the informative content.

## Files created / updated

| Path | Description |
|---|---|
| `src/run_pilot.py` | pilot runner + power solver |
| `results/pilot_power.json` | sd_pilot, S, clamp, achieved power |
| `configs/endpoints.yaml` | S, final seeds, power record |
| `results/selected_configs.json` | matched-depth primary pair (D-006) |
| `src/run_final.py` | Phase 8 runner (all arms, incl. A2', A4, A5) |
| `src/stats.py` | Phase 9 analysis |
| `src/figures.py` | Phase 10 figures and tables |
| `docs/phase_records/phase06_pilot_power.md` | this record |

## Validation results

| Check | Result |
|---|---|
| primary pair has identical parameter counts | **PASS** (9,685,537 both) |
| pilot used shared seeds across arms | **PASS** (201–205) |
| pilot evaluated on validation only | **PASS** |
| S within [floor 10, ceiling 30] | **PASS** (30, ceiling applied) |
| sd_pilot, S and clamp recorded | **PASS** |
| achieved power recorded | **PASS** (0.121) |
| no test data touched | **PASS** (`freeze-v1` absent) |

## Issues encountered

1. **Underpowered design** — documented above; the central finding of
   this phase.
2. `sd_pilot` is inflated by one seed (205, d = +0.145). With five seeds
   the SD estimate is itself noisy; the tuning-based estimate is 0.041.
   Both clamp to S = 30, so the conclusion does not depend on which is
   used. Reported as measured, per hard rule 2.
3. A background A5 tuning launch exited after one run and was re-run;
   no artifact was affected (the runner is resumable and skips logged
   runs).

## Deviations

- D-006 (matched-depth primary pair) governs this phase. No new
  deviations. The S ceiling is applied as the guide specifies, not as a
  departure from it.

## Status

**PASS** — ready for the Phase 7 freeze.
