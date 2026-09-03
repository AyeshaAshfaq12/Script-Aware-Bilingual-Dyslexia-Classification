"""Phase 10: probe what the SCA gates actually do (guide section 11, item 5).

The guide asks for "attention maps". The SCA block is squeeze-and-excitation
style CHANNEL attention: the gate is a vector a in (0,1)^C applied as
f + f*a, with no spatial term at all. There is therefore no spatial
heat map to draw, and painting one would be an invention. What can be
measured honestly is the gate vector itself, so this probe reports gate
profiles and, more importantly, a counterfactual:

    holding the image fixed, how far does the gate move when the script
    id is changed?

That isolates the causal contribution of the conditioning signal, which
is exactly the mechanism the A2-vs-A1 contrast is testing. It is
compared against the gate variation driven by the image, which sets the
natural scale.

A1 is included as a correctness check on the probe, not as a result: its
gates cannot depend on the script by construction (the embedding output
is discarded for a learned constant), so its script-induced variation
must come out at exactly 0.

Writes results/attention_gates.json (summary) and
results/attention_gates.npz (per-seed arrays).

No dataset image is written to disk or embedded in any figure, so this
respects hard rule 7.

Run:
  python src/attention_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

import models
from train import (FeatureCache, RunSpec, assert_may_touch_test,
                   git_commit)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
CONFIGS = REPO / "configs"
RUNS = REPO / "runs"

SCRIPTS = ["urdu", "english", "digit"]


def spec_for(arm: str) -> RunSpec:
    grid = yaml.safe_load((CONFIGS / "grid.yaml").read_text(encoding="utf-8"))
    sel = json.loads((RESULTS / "selected_configs.json").read_text(
        encoding="utf-8"))
    cfg = sel["primary_pair"]["arms"][arm]
    return RunSpec(
        arm=arm, backbone=cfg["backbone"], depth_config=cfg["depth_config"],
        d=cfg["sca_d"], r=cfg["sca_r"], optimizer=cfg["optimizer"],
        lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
        batch_size=grid["common"]["batch_size"],
        max_epochs=grid["common"]["max_epochs"],
        patience=grid["common"]["early_stopping"]["patience"],
        corpus=grid["common"]["corpus"])


def gates(sca, f: np.ndarray, script_id: np.ndarray) -> np.ndarray:
    """Recompute the SCA gate vector a for features f and script ids.

    Mirrors SCA.call up to the residual application, so it stays correct
    if the layer's internals are edited.
    """
    import tensorflow as tf
    z = tf.reduce_mean(tf.constant(f), axis=[1, 2])
    if sca.conditioned:
        e = sca.embed(tf.constant(script_id, dtype="int32"))
    else:
        e = tf.tile(sca.const[None, :], [z.shape[0], 1])
    a = sca.fc2(sca.fc1(tf.concat([z, e], axis=-1)))
    return np.asarray(a)


def probe_arm(arm: str, seeds: list[int], cache: FeatureCache) -> dict:
    import keras
    spec = spec_for(arm)
    taps = models.taps_for(spec.depth_config)
    if len(taps) != 1:
        raise SystemExit(
            f"probe assumes a single tap (depth_config='{spec.depth_config}' "
            f"has {len(taps)}); the cached prefix output is only the input "
            f"to the FIRST SCA block.")

    keras.utils.set_random_seed(seeds[0])
    prefix, suffix, _ = models.build_split(
        arm, backbone=spec.backbone, depth_config=spec.depth_config,
        d=spec.d, r=spec.r)
    recs = cache.records(spec, "test")
    F = cache.features(spec, "test", prefix)
    true_ids = np.array([r["script_id"] for r in recs], dtype="int32")
    n = len(recs)

    per_seed = []
    used = []
    for s in seeds:
        rd = RUNS / f"{spec.tag()}_{spec.hash()}_{s}" / "suffix.weights.h5"
        if not rd.exists():
            continue
        suffix.load_weights(rd)
        sca = [l for l in suffix.layers if isinstance(l, models.SCA)][0]
        # a[script, image, channel] under each forced script id
        A = np.stack([gates(sca, F, np.full(n, k, dtype="int32"))
                      for k in range(len(SCRIPTS))])
        per_seed.append(A)
        used.append(s)

    if not per_seed:
        raise SystemExit(f"no saved weights found for {arm}")
    A = np.stack(per_seed)                    # (S, 3, N, C)

    # Script-induced: fix the image, vary the script id.
    script_sd = A.std(axis=1, ddof=0).mean(axis=(1, 2))      # (S,)
    # Image-induced: fix the script (each image's TRUE script), vary image.
    idx = true_ids[None, :, None]
    A_true = np.take_along_axis(A, np.broadcast_to(
        idx[None], (A.shape[0], 1, A.shape[2], A.shape[3])), axis=1)[:, 0]
    image_sd = A_true.std(axis=1, ddof=0).mean(axis=1)       # (S,)

    return {
        "arm": arm,
        "seeds": used,
        "n_images": n,
        "n_channels": int(A.shape[-1]),
        "tap": taps[0],
        "gate_mean": float(A_true.mean()),
        "gate_min": float(A.min()),
        "gate_max": float(A.max()),
        "script_induced_sd_mean": float(script_sd.mean()),
        "script_induced_sd_sd": float(script_sd.std(ddof=1)),
        "image_induced_sd_mean": float(image_sd.mean()),
        "image_induced_sd_sd": float(image_sd.std(ddof=1)),
        "ratio_script_over_image": float(
            script_sd.mean() / image_sd.mean()) if image_sd.mean() else 0.0,
        "_arrays": {"A": A, "true_ids": true_ids,
                    "script_sd": script_sd, "image_sd": image_sd},
    }


def main() -> int:
    assert_may_touch_test()
    ep = yaml.safe_load((CONFIGS / "endpoints.yaml").read_text(
        encoding="utf-8"))
    seeds = [300 + i for i in range(1, int(ep["repeats"]["S"]) + 1)]
    cache = FeatureCache()

    out, arrays = {}, {}
    for arm in ("A2", "A1"):
        print(f"[{arm}] probing gates ...")
        r = probe_arm(arm, seeds, cache)
        arr = r.pop("_arrays")
        arrays[f"{arm}_A"] = arr["A"].astype("float32")
        arrays[f"{arm}_script_sd"] = arr["script_sd"]
        arrays[f"{arm}_image_sd"] = arr["image_sd"]
        arrays["true_ids"] = arr["true_ids"]
        out[arm] = r
        print(f"  seeds={len(r['seeds'])} channels={r['n_channels']} "
              f"gate mean={r['gate_mean']:.4f}")
        print(f"  script-induced gate SD = {r['script_induced_sd_mean']:.6f}")
        print(f"  image-induced  gate SD = {r['image_induced_sd_mean']:.6f}")
        print(f"  ratio = {r['ratio_script_over_image']:.4f}")

    # A1 is the probe's correctness check: its gates cannot see the script.
    # It does not come out at exactly 0. The three forced-script passes are
    # separate float32 matmuls over identical inputs, and oneDNN does not
    # guarantee bit-identical results across calls, so the floor is
    # round-off, not zero: float32 eps is ~1.2e-7 and the gates are ~0.5,
    # giving ~6e-8. The threshold is set at that scale rather than at 0.
    a1_leak = out["A1"]["script_induced_sd_mean"]
    a2_signal = out["A2"]["script_induced_sd_mean"]
    out["probe_check"] = {
        "a1_script_induced_sd": a1_leak,
        "expected": "float32 round-off (~1e-8), not exactly 0",
        "threshold": 1e-6,
        "passed": bool(a1_leak < 1e-6),
        "a2_over_a1": float(a2_signal / a1_leak) if a1_leak else float("inf"),
        "note": "A1 discards the embedding for a learned constant, so its "
                "gates cannot depend on the script. Any script-induced "
                "variation above float32 round-off would mean the probe is "
                "reading the wrong tensor.",
    }
    out["git_commit"] = git_commit()
    (RESULTS / "attention_gates.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    np.savez_compressed(RESULTS / "attention_gates.npz", **arrays)
    print(f"\nprobe check (A1 script-induced SD == 0): "
          f"{'PASS' if out['probe_check']['passed'] else 'FAIL'}")
    print("wrote results/attention_gates.json, attention_gates.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
