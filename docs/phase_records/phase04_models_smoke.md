# Phase Record — Phase 4: Models and smoke tests

- **Date:** 2026-09-02
- **Phase:** Phase 4 (Guide §5)
- **Status:** **PASS** — all 37 smoke tests green on CPU.

## Actions performed

1. `src/data.py` — the shared pipeline. 224×224×3, MobileNet
   `preprocess_input` (→ [-1, 1]), yields `((image, script_id), label)`.
   Reads `split_v1.json`; supports both the primary and sensitivity
   corpora. Script encoding fixed: **urdu=0, english=1, digit=2**.
2. `src/models.py` — the `SCA` layer and every arm. All arms take the
   same two inputs and emit one sigmoid, so one training loop serves
   all of them; arms that ignore script simply do not read input 2.
3. `tests/` — the four §5.4 checks plus additional integrity guards.
4. `results/capacity_check.csv` — parameter counts for every config,
   generated, not hand-typed. Feeds the `n_params` column of
   `tab_main.tex`.

## Architecture notes

- **SCA insertion.** MobileNetV1 is a linear chain (asserted at build
  time), so the arm builders re-host its layers one by one onto a fresh
  input and inject an `SCA` after each tap. Taps confirmed present in
  TF 2.20: `conv_pw_3_relu` (early, 128 ch), `conv_pw_7_relu` (mid, 512
  ch), `conv_pw_13_relu` (late, 1024 ch). **No §5.1 deviation needed.**
- **Matched capacity.** A1 and A2 are one builder with `conditioned`
  flipped. Both instantiate and build the embedding table *and* the
  learned constant, so counts match exactly at all 16 configs.
- **A3/A5 use registered custom layers** (`ScriptOneHot`,
  `RouteSelect`) rather than `Lambda`, because Keras 3 refuses to
  deserialise `Lambda`-wrapped Python lambdas without unsafe mode. Arms
  must round-trip through `.keras` for the per-run artifacts in Phase 8.
- **A2-prime is not a new architecture:** it is A2 fed predicted script
  ids from `build_script_classifier` (frozen features + GAP +
  Dense(3, softmax); 3,075 trainable params).
- **A4** is deliberately not a single Keras model: it is two
  independently trained A0 models plus a routing rule, assembled in
  Phase 8.

## Parameter counts (from `results/capacity_check.csv`)

| arm | total | trainable |
|---|---|---|
| A0 MobileNetV1 | 9,651,649 | 6,422,785 |
| A0 CNN-from-scratch | 12,938,561 | 12,938,561 |
| A1 = A2 (mid, d=16, r=16) | 9,685,537 | 6,456,673 |
| A1 = A2 (all, d=16, r=16) | 9,821,161 | 6,592,297 |
| A3 | 9,651,652 | 6,422,788 |
| A5 | 22,497,219 | 19,268,355 |
| script classifier | 3,231,939 | 3,075 |

A1 and A2 match exactly at **all 16 configs** (4 depths × d in {8,16} ×
r in {8,16}).

## Validation results (guide §5.4)

| # | Required check | Result |
|---|---|---|
| 1 | pipeline yields correct shapes and ranges | **PASS** |
| 2 | every arm builds, trains 2 epochs without NaN, saves and reloads | **PASS** (9 arms) |
| 3 | A1/A2 parameter equality at every depth config | **PASS** (16/16) |
| 4 | deterministic rerun: same seed gives identical first-epoch loss | **PASS** |

Additional guards added:

| Check | Result |
|---|---|
| partitions disjoint and complete, both corpora | **PASS** |
| **no duplicate image spans partitions** (D-004 leakage guard) | **PASS** |
| script ids match adjudicated tags | **PASS** |
| A1 output invariant to script id; A2 output is not | **PASS** |
| both arms contain `script_embed` **and** `const_embed` | **PASS** |
| backbone taps are numerically alive | **PASS** |
| different seeds give different losses | **PASS** |

`tf.config.experimental.enable_op_determinism()` is enabled for the
suite and full determinism holds on CPU, so §5.4's GPU fallback clause
is not invoked locally. It must be re-checked on the Colab/Kaggle GPU
before Phase 5.

## Issues encountered

1. **A randomly-initialised MobileNetV1 is numerically dead.** Measured
   at `conv_pw_7_relu` on real images: with `weights=None`, mean and
   max activation are both exactly **0.000000**; with
   `weights="imagenet"`, mean 0.841 / max 6.0. Because SCA is a
   *residual* gate (`f + f*a`), a zero feature map makes the gate
   inert and A2 becomes bit-identical to A1 — the script-sensitivity
   test failed with delta = 0.000e+00. This was a defect in the
   **test**, not the model: the smoke tests were building untrained
   backbones. Fixed by running the smoke tests on ImageNet weights, as
   the real runs do, and by adding `TestBackboneHealth` so the failure
   mode cannot return silently. With pretrained weights the script-id
   sensitivity is delta = 1.45e-03 at initialisation, as expected for a
   near-identity residual gate with a randomly initialised embedding.
2. **Keras 3 refuses to reload `Lambda` layers.** A3 and A5 failed
   save/reload. Replaced with registered serialisable layers.
3. **Expected optimiser warnings, not defects.** Keras reports "no
   gradients for `const_embed`" in A2 and "no gradients for
   `script_embed`" in A1. This is precisely the matched-capacity
   design: identical parameter counts, different active subsets. Worth
   stating in the paper that the *gradient-receiving* parameter counts
   differ by 2*d (32 parameters at d=16, i.e. 5e-4 % of trainable
   capacity) even though total capacity is identical.

## Deviations

None. The §5.1 tap-layer mapping matched the pinned TF version, so the
DEVIATIONS.md entry that guide §5.1 anticipates is not required.

## Files created / updated

| Path | Description |
|---|---|
| `src/data.py` | shared data pipeline |
| `src/models.py` | SCA, ScriptOneHot, RouteSelect, arms A0–A5 |
| `tests/conftest.py` | determinism + 32-image fixture |
| `tests/test_smoke.py` | 37 tests |
| `pytest.ini` | test config |
| `results/capacity_check.csv` | generated parameter table |

## Status

**PASS.** Ready for Phase 5 (tuning), which is the first GPU phase.
Nothing has touched the test partition.
