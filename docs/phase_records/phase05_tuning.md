# Phase Record — Phase 5: Tuning protocol

- **Date:** 2026-09-02
- **Phase:** Phase 5 (Guide §6)
- **Status:** **INFRASTRUCTURE COMPLETE — sweep pending.** The grid is
  amended and approved, the runner is verified on real runs, and the
  sweep is ready for the authors to launch.

## Execution platform decision

All experiments run on the **local CPU** (Intel i7-6600U, 2 physical
cores, 16 GB RAM), not Colab/Kaggle. The authors raised whether this
costs result quality; it does not. The computation is identical —
same architecture, weights, optimiser and arithmetic — so hardware
changes wall-clock time, not results. CPU is in fact preferable here:
`tf.config.experimental.enable_op_determinism()` is fully reliable on
CPU and unavailable for several GPU kernels, which is exactly the
reproducibility guarantee §5.4 asks for. Staying on one platform also
keeps every arm internally consistent.

What affects scientific quality is grid size, not hardware, which is
why the amendment below preserves the depth sweep.

## The cached-prefix optimisation

The backbone is frozen and no augmentation is used (hard rule 3), so
the output of the frozen prefix is a constant function of the input
image. Computing it once per (backbone, tap) and training only the
suffix is **exactly equivalent** to end-to-end training, and is what
makes a local-CPU budget feasible.

Equivalence is not assumed. `tests/test_smoke.py::TestSplitEquivalence`
asserts, for all eight arm/depth combinations, that
`suffix(prefix(x))` is **bit-identical** to the full model, that the
suffix owns every trainable weight of the full model, that the prefix
holds none, and that training the suffix propagates to the full model.

Measured speed-up: 4.7× at `late`, 1.6× at `mid`, none at
`early`/`all` (the SCA sits early, so most conv blocks still re-run).

## Grid amendment (guide §6 permits this BEFORE the freeze)

| axis | guide | amended | kept? |
|---|---|---|---|
| optimizer | [rmsprop, adam] | unchanged | ✔ |
| learning_rate | [1e-3, 3e-4, 1e-4] | unchanged | ✔ |
| **depth_config** | [early, mid, late, all] | **unchanged** | ✔ |
| tuning seeds | [101, 102, 103] | **unchanged** | ✔ |
| weight_decay | [0.0, 1e-4] | **[0.0]** | trimmed |
| sca d | [8, 16] | **[16]** | trimmed |
| sca r | [8, 16] | **[16]** | trimmed |

96 configs → **24 configs** per SCA arm; 288 → **72 runs** per arm.

Rationale, recorded in `configs/grid.yaml`: the depth sweep carries a
stated contribution and its own figure, so it was kept in full;
weight decay of 1e-4 over the ~9 epochs early stopping actually uses,
on a frozen backbone, is immaterial; `d` and `r` are SCA-internal and,
fixed identically across A1 and A2, preserve matched capacity and equal
tuning budget exactly. **Seeds were not traded for compute**, since the
Phase 6 power analysis depends on them.

## Measured costs on this machine

| depth | s/epoch | s/run |
|---|---|---|
| late | 4.4 | 48 |
| mid | 12.5 | 138 |
| early | 20.7 | 228 |
| all | 21.6 | 238 |

Early stopping fires at **7–11 epochs** (observed), not the 50 maximum.

| stage | runs | estimate |
|---|---|---|
| A1 + A2 tuning | 144 | ~6.5 h |
| A3 tuning | 18 | ~0.2 h |
| A0 anchor (MobileNetV1 × 3 optimizers) | 27 | ~0.3 h |
| A0 others (5 backbones) | 45 | ~0.5 h + extraction |
| **total tuning** | **234** | **~7.5 h** |

Estimates exclude thermal throttling, which on a 2-core U-series CPU
under sustained load will realistically add 20–50%. `A0_others`
additionally pays a one-time feature-extraction cost per backbone;
VGG16 and InceptionV3 are markedly slower than MobileNet here.

## Source-study Table 3 transcribed

Guide §5.3 requires each A0 backbone to run at its best optimizer from
Kashif et al. (2026) Table 3. Extracted from the PDF (p.10) rather than
left as a placeholder:

| backbone | Adam | RMSProp | SGD | selected |
|---|---|---|---|---|
| CNN | 0.7053 | 0.5988 | **0.7743** | sgd |
| VGG16 | 0.7602 | **0.7684** | 0.7532 | rmsprop |
| InceptionV3 | 0.7474 | **0.7661** | 0.7497 | rmsprop |
| MobileNetV1 | 0.7778 | **0.7871** | 0.7731 | rmsprop (anchor) |
| MobileNetV2 | 0.7673 | **0.7836** | **0.7836** | rmsprop (tie) |
| MobileNetV3 | **0.7216** | 0.7146 | 0.6082 | adam |

MobileNetV2 ties on mean accuracy under RMSProp and SGD; RMSProp is
selected for its lower SD (0.0128 vs 0.0161) and higher F1 (0.7832 vs
0.7756). The tie-break is recorded in `grid.yaml` so it is not silent.

The anchor value **0.7871 ± 0.0102, CI [0.7730, 0.8013]** matches the
figure quoted in the paper draft's Introduction.

## Hard rule 1 enforced in code

`train.py::assert_may_touch_test()` raises unless the `freeze-v1` git
tag exists. Tuning and pilot runs can only ever evaluate on the
validation partition. The gate is checked before any test data is read,
not after.

## Files created / updated

| Path | Description |
|---|---|
| `configs/grid.yaml` | amended grid, rationale, measurements, Table 3 |
| `configs/endpoints.yaml` | co-primary endpoints, Holm, delta_min, S slot |
| `src/models.py` | `build_split()` — cacheable prefix + trainable suffix |
| `src/evaluate.py` | pooled and per-script metrics from predictions |
| `src/train.py` | one run = (arm, config, seed); freeze gate; artifacts |
| `src/run_tuning.py` | resumable grid driver, grouped by depth |
| `tests/test_smoke.py` | +21 split-equivalence tests (58 total) |
| `results/all_runs.csv` | append-only master log (2 verification rows) |

## Validation results

| Check | Result |
|---|---|
| split path bit-identical to full model (8 arm/depth cases) | **PASS** |
| suffix owns all trainable weights; prefix owns none | **PASS** |
| A1/A2 capacity match via split builder, all depths | **PASS** |
| full smoke suite | **PASS** (58 tests) |
| end-to-end training run writes all artifacts | **PASS** |
| `all_runs.csv` schema matches guide §9 | **PASS** |
| resume skips already-logged runs | **PASS** |
| test partition unreachable without freeze tag | **PASS** |
| grid.yaml has no unresolved placeholders | **PASS** |

Two real verification runs are logged (`A3`, adam, 1e-4, seeds 101 and
102): val accuracy 0.7097 at 14 and 17 epochs. They are genuine tuning
runs and will be skipped, not repeated, when the sweep resumes.

## Issues encountered

1. Keras raises `RuntimeError: Random ops require a seed` under op
   determinism if Dropout runs without a seed set. Every run path calls
   `keras.utils.set_random_seed(seed)` first; one test needed the same.
2. `A0_others` per-run estimates do not include the one-time
   feature-extraction cost of VGG16 / InceptionV3, which is
   substantially higher than MobileNet on this CPU.

## Deviations

None. The grid amendment is expressly permitted by guide §6 before the
freeze, is approved by the authors, and is recorded in `grid.yaml` with
its rationale and the measurements behind it.

## Status

**READY TO RUN.** Nothing has touched the test partition;
`freeze-v1` does not exist.
