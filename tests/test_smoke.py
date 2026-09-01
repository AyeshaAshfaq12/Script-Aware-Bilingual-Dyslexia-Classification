"""Phase 4 smoke tests — guide section 5.4.

Four required checks, CPU, 32-image subset:
  1. data pipeline shapes and value ranges
  2. every arm builds, trains 2 epochs without NaN, saves and reloads
  3. A1/A2 parameter equality at EVERY depth config
  4. deterministic rerun: same seed twice -> identical first-epoch loss

These must all pass before any real run (guide section 5).
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import SMOKE_N, SMOKE_SEED

import keras
import tensorflow as tf

import data
import models


# ------------------------------------------------------------------ 1
class TestDataPipeline:
    def test_batch_shapes_and_ranges(self):
        ds = data.make_dataset("val", "primary", batch_size=8)
        (x, s), y = next(iter(ds))
        assert x.shape[1:] == (data.IMG_SIZE, data.IMG_SIZE, 3)
        assert x.dtype == tf.float32
        # MobileNet preprocess_input maps to [-1, 1]
        assert float(tf.reduce_min(x)) >= -1.0
        assert float(tf.reduce_max(x)) <= 1.0
        assert s.dtype == tf.int32
        assert int(tf.reduce_min(s)) >= 0
        assert int(tf.reduce_max(s)) < data.N_SCRIPTS
        assert set(np.unique(y.numpy())).issubset({0.0, 1.0})

    def test_partitions_disjoint_and_complete(self):
        for corpus, total in (("primary", 618), ("sensitivity", 852)):
            uids = {p: {r["uid"] for r in data.records(p, corpus)}
                    for p in data.PARTITIONS}
            assert not uids["train"] & uids["val"]
            assert not uids["train"] & uids["test"]
            assert not uids["val"] & uids["test"]
            assert sum(len(v) for v in uids.values()) == total

    def test_no_duplicate_image_across_partitions(self):
        """The leakage guard from D-004, checked at the data layer."""
        for corpus in ("primary", "sensitivity"):
            seen: dict[int, str] = {}
            for p in data.PARTITIONS:
                for r in data.records(p, corpus):
                    prev = seen.setdefault(r["unit_id"], p)
                    assert prev == p, (
                        f"{corpus}: unit {r['unit_id']} in {prev} and {p}")

    def test_script_ids_match_tags(self):
        for r in data.records("train", "primary")[:200]:
            assert data.SCRIPT_ID[r["script"]] == r["script_id"]


# ------------------------------------------------------------------ 3
class TestCapacityMatch:
    """Mandatory assertion from guide section 5.3."""

    @pytest.mark.parametrize("depth", models.DEPTH_CONFIGS)
    @pytest.mark.parametrize("d,r", [(8, 8), (8, 16), (16, 8), (16, 16)])
    def test_a1_a2_param_equality(self, depth, d, r):
        cfg = {"depth_config": depth, "d": d, "r": r, "weights": None}
        p = models.assert_capacity_match(cfg)
        assert p["trainable"] > 0

    def test_embedding_present_in_both(self):
        """A1 must still CONTAIN the embedding table it never reads."""
        cfg = {"depth_config": "mid", "d": 16, "r": 16, "weights": None}
        for build in (models.build_a1, models.build_a2):
            m = build(**cfg)
            names = [w.path for w in m.trainable_weights]
            assert any("script_embed" in n for n in names), names
            assert any("const_embed" in n for n in names), names


# ------------------------------------------------------------------ 2
def _compile(m):
    m.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m


# Real runs use ImageNet weights, so the smoke tests do too. A
# randomly-initialised MobileNet is numerically dead at every tap point
# (see TestBackboneHealth), which would make these tests vacuous.
W = "imagenet"

ARMS = {
    "A0_mobilenet": lambda: models.build_a0("mobilenet", weights=W),
    "A0_cnn_scratch": lambda: models.build_a0("cnn_scratch", weights=None),
    "A1_mid": lambda: models.build_a1(depth_config="mid", weights=W),
    "A2_mid": lambda: models.build_a2(depth_config="mid", weights=W),
    "A1_all": lambda: models.build_a1(depth_config="all", weights=W),
    "A2_all": lambda: models.build_a2(depth_config="all", weights=W),
    "A3": lambda: models.build_a3(weights=W),
    "A5": lambda: models.build_a5(weights=W),
    "A4_expert": lambda: models.build_a4_expert(weights=W, tag="urdu"),
}


class TestBackboneHealth:
    """Guards the failure mode found during Phase 4 development.

    With weights=None the frozen MobileNetV1 is numerically dead by
    conv_pw_7_relu (every activation exactly 0), so the SCA residual
    gate f + f*a has nothing to modulate and A2 becomes indistinguishable
    from A1. Any run must therefore use pretrained weights.
    """

    @pytest.mark.parametrize("depth", ["early", "mid", "late"])
    def test_tap_activations_alive_with_imagenet(self, tiny, depth):
        x, s, _ = tiny
        m = models.build_a2(depth_config=depth, weights=W)
        sca = [l for l in m.layers if isinstance(l, models.SCA)][0]
        tap = sca._inbound_nodes[0].arguments.args[0]
        f = keras.Model(m.inputs, tap).predict([x, s], verbose=0)
        assert np.isfinite(f).all()
        assert f.max() > 0.0, f"{depth}: backbone tap is dead"
        assert (f != 0).mean() > 0.01, f"{depth}: tap almost entirely zero"


class TestArms:
    @pytest.mark.parametrize("name", list(ARMS))
    def test_builds_trains_saves_reloads(self, name, tiny, tmp_path):
        x, s, y = tiny
        keras.utils.set_random_seed(SMOKE_SEED)
        m = _compile(ARMS[name]())

        h = m.fit([x, s], y, epochs=2, batch_size=8, verbose=0)
        losses = h.history["loss"]
        assert len(losses) == 2
        assert all(np.isfinite(losses)), f"{name}: non-finite loss {losses}"

        p0 = m.predict([x, s], verbose=0)
        assert p0.shape == (SMOKE_N, 1)
        assert np.all(np.isfinite(p0))
        assert p0.min() >= 0.0 and p0.max() <= 1.0

        path = tmp_path / f"{name}.keras"
        m.save(path)
        reloaded = keras.models.load_model(path)
        p1 = reloaded.predict([x, s], verbose=0)
        np.testing.assert_allclose(p0, p1, rtol=1e-5, atol=1e-6)

    def test_script_classifier(self, tiny):
        x, s, _ = tiny
        keras.utils.set_random_seed(SMOKE_SEED)
        m = models.build_script_classifier(weights=W)
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        h = m.fit(x, s, epochs=2, batch_size=8, verbose=0)
        assert all(np.isfinite(h.history["loss"]))
        p = m.predict(x, verbose=0)
        assert p.shape == (SMOKE_N, data.N_SCRIPTS)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)

    def test_a1_ignores_script_input_a2_does_not(self, tiny):
        """The factor under test: only A2's output may move with script."""
        x, s, _ = tiny
        alt = (s + 1) % data.N_SCRIPTS

        keras.utils.set_random_seed(SMOKE_SEED)
        a1 = models.build_a1(depth_config="mid", weights=W)
        d1 = np.abs(a1.predict([x, s], verbose=0)
                    - a1.predict([x, alt], verbose=0)).max()
        assert d1 == 0.0, "A1 output changed with script id"

        keras.utils.set_random_seed(SMOKE_SEED)
        a2 = models.build_a2(depth_config="mid", weights=W)
        # untrained embeddings are random, so the gate must already differ
        d2 = np.abs(a2.predict([x, s], verbose=0)
                    - a2.predict([x, alt], verbose=0)).max()
        assert d2 > 0.0, "A2 output did not change with script id"


# ------------------------------------------------------------------ 4
class TestDeterminism:
    def _first_epoch_loss(self, seed: int) -> float:
        x, s, y = data.as_arrays("train", "primary", limit=SMOKE_N)
        keras.utils.set_random_seed(seed)
        m = _compile(models.build_a2(depth_config="mid", weights=W))
        return m.fit([x, s], y, epochs=1, batch_size=8,
                     shuffle=False, verbose=0).history["loss"][0]

    def test_same_seed_same_first_epoch_loss(self):
        a = self._first_epoch_loss(SMOKE_SEED)
        b = self._first_epoch_loss(SMOKE_SEED)
        assert a == b, f"non-deterministic: {a} != {b}"

    def test_different_seed_differs(self):
        a = self._first_epoch_loss(SMOKE_SEED)
        c = self._first_epoch_loss(SMOKE_SEED + 1)
        assert a != c, "different seeds produced identical loss"
