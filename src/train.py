"""Phase 5+: one run = one (arm, config, seed). Guide sections 6, 8, 9.

Training uses the cached-prefix path: the frozen backbone prefix is
computed once per (backbone, tap) and reused by every run that shares
it. This is exactly equivalent to end-to-end training because the
backbone is frozen and no augmentation is used; the equivalence is
asserted in tests/test_smoke.py::TestSplitEquivalence.

HARD RULE 1 is enforced in code: evaluating on the test partition
raises unless the `freeze-v1` git tag exists. Tuning and pilot runs can
only ever see the validation partition.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

import data
import models
from evaluate import compute_metrics

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
RESULTS = REPO / "results"
ALL_RUNS = RESULTS / "all_runs.csv"
FREEZE_TAG = "freeze-v1"

ROW_FIELDS = [
    "phase", "arm", "config_hash", "seed", "corpus", "split_file",
    "eval_partition", "n_params", "n_params_trainable",
    "epochs_run", "best_epoch",
    "acc", "bal_acc", "prec", "rec", "f1", "auc",
    "acc_urdu", "acc_english", "acc_digit",
    "f1_urdu", "f1_english", "f1_digit",
    "n_urdu", "n_english", "n_digit",
    "optimizer", "learning_rate", "weight_decay",
    "depth_config", "sca_d", "sca_r", "backbone",
    "runtime_s", "env", "timestamp",
]


# --------------------------------------------------------------- spec
@dataclass(frozen=True)
class RunSpec:
    arm: str                      # A0 | A1 | A2 | A3 | A5 | script_clf
    backbone: str = "mobilenet"
    depth_config: str | None = None
    d: int = 16
    r: int = 16
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.5
    batch_size: int = 16
    max_epochs: int = 50
    patience: int = 5
    corpus: str = "primary"
    script_source: str = "oracle"   # oracle | predicted (A2')
    train_subset: str | None = None  # A4 experts: 'urdu' | 'english'

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:10]

    def tag(self) -> str:
        bits = [self.arm]
        if self.depth_config:
            bits.append(self.depth_config)
        if self.train_subset:
            bits.append(self.train_subset)
        if self.script_source != "oracle":
            bits.append(self.script_source)
        return "_".join(bits)


# ------------------------------------------------------- freeze guard
def freeze_tag_exists() -> bool:
    try:
        out = subprocess.run(["git", "tag", "--list", FREEZE_TAG],
                             cwd=REPO, capture_output=True, text=True,
                             check=True)
        return FREEZE_TAG in out.stdout.split()
    except Exception:
        return False


def assert_may_touch_test() -> None:
    if not freeze_tag_exists():
        raise RuntimeError(
            "HARD RULE 1: no test-set evaluation before the freeze. "
            f"Tag `{FREEZE_TAG}` does not exist. Tuning and model "
            "selection use the validation partition only.")


# ------------------------------------------------------ feature cache
class FeatureCache:
    """Frozen-prefix features, keyed by (backbone, tap, partition, corpus).

    Held in memory for the life of a runner process. Recomputing a
    prefix costs 7-15 s, so grouping runs by depth keeps this cheap and
    avoids multi-GB on-disk caches.
    """

    def __init__(self) -> None:
        self._feats: dict[tuple, np.ndarray] = {}
        self._meta: dict[tuple, list[dict]] = {}
        self._images: dict[tuple, np.ndarray] = {}

    def _key(self, spec: RunSpec, partition: str) -> tuple:
        first_tap = (models.taps_for(spec.depth_config)[0]
                     if spec.arm in ("A1", "A2") else "backbone_out")
        return (spec.backbone, first_tap, partition, spec.corpus,
                spec.train_subset)

    def records(self, spec: RunSpec, partition: str) -> list[dict]:
        key = self._key(spec, partition)
        if key not in self._meta:
            recs = data.records(partition, spec.corpus)
            if spec.train_subset and partition == "train":
                recs = [r for r in recs if r["script"] == spec.train_subset]
            self._meta[key] = recs
        return self._meta[key]

    def images(self, spec: RunSpec, partition: str) -> np.ndarray:
        key = self._key(spec, partition)
        if key not in self._images:
            recs = self.records(spec, partition)
            self._images[key] = np.stack(
                [data._decode(tf.constant(r["path"])).numpy() for r in recs])
        return self._images[key]

    def features(self, spec: RunSpec, partition: str,
                 prefix: keras.Model) -> np.ndarray:
        key = self._key(spec, partition)
        if key not in self._feats:
            x = self.images(spec, partition)
            self._feats[key] = prefix.predict(x, batch_size=spec.batch_size,
                                              verbose=0)
        return self._feats[key]

    def clear(self) -> None:
        self._feats.clear()
        self._images.clear()


# ------------------------------------------------------------ helpers
def make_optimizer(spec: RunSpec):
    kw = {"learning_rate": spec.lr}
    if spec.weight_decay:
        kw["weight_decay"] = spec.weight_decay
    return {"adam": keras.optimizers.Adam,
            "rmsprop": keras.optimizers.RMSprop,
            "sgd": keras.optimizers.SGD}[spec.optimizer](**kw)


def env_string() -> str:
    return (f"tf{tf.__version__}|keras{keras.__version__}|"
            f"py{platform.python_version()}|{platform.system()}|cpu")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def already_done(phase: str, arm_tag: str, config_hash: str,
                 seed: int) -> bool:
    if not ALL_RUNS.exists():
        return False
    with ALL_RUNS.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row["phase"] == phase and row["arm"] == arm_tag
                    and row["config_hash"] == config_hash
                    and int(row["seed"]) == seed):
                return True
    return False


def append_row(row: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    new = not ALL_RUNS.exists()
    with ALL_RUNS.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------- run
def run(spec: RunSpec, seed: int, phase: str,
        eval_partition: str = "val",
        cache: FeatureCache | None = None,
        save_weights: bool = False,
        script_ids_override: dict[str, np.ndarray] | None = None) -> dict:
    """Train one (arm, config, seed) and return its results row."""
    if eval_partition == "test":
        assert_may_touch_test()
    arm_tag = spec.tag()
    chash = spec.hash()
    if already_done(phase, arm_tag, chash, seed):
        print(f"  skip (already logged): {arm_tag} {chash} seed={seed}")
        return {}

    cache = cache or FeatureCache()
    t0 = time.time()
    keras.utils.set_random_seed(seed)

    prefix, suffix, full = models.build_split(
        arm=spec.arm, backbone=spec.backbone,
        depth_config=spec.depth_config or "mid",
        d=spec.d, r=spec.r, dropout=spec.dropout, weights="imagenet")

    tr = cache.records(spec, "train")
    ev = cache.records(spec, eval_partition)
    Ftr = cache.features(spec, "train", prefix)
    Fev = cache.features(spec, eval_partition, prefix)
    ytr = np.array([r["label"] for r in tr], dtype="float32")
    yev = np.array([r["label"] for r in ev], dtype="float32")
    str_ = np.array([r["script_id"] for r in tr], dtype="int32")
    sev = np.array([r["script_id"] for r in ev], dtype="int32")

    # A2': oracle ids replaced by a script classifier's predictions
    if script_ids_override:
        str_ = script_ids_override["train"]
        sev = script_ids_override[eval_partition]

    suffix.compile(optimizer=make_optimizer(spec),
                   loss="binary_crossentropy", metrics=["accuracy"])
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=spec.patience,
        restore_best_weights=True)
    hist = suffix.fit([Ftr, str_], ytr,
                      validation_data=([Fev, sev], yev),
                      epochs=spec.max_epochs, batch_size=spec.batch_size,
                      callbacks=[es], verbose=0, shuffle=True)

    y_prob = suffix.predict([Fev, sev], batch_size=spec.batch_size,
                            verbose=0).ravel()
    m = compute_metrics(yev, y_prob, [r["script"] for r in ev])
    counts = models.param_counts(full)
    runtime = time.time() - t0
    epochs_run = len(hist.history["loss"])
    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1

    # ---- artifacts ----
    rd = RUNS / f"{arm_tag}_{chash}_{seed}"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(json.dumps(
        {**asdict(spec), "seed": seed, "phase": phase,
         "eval_partition": eval_partition, "split_file": str(data.SPLIT.name),
         "git_commit": git_commit(), "env": env_string(),
         "param_counts": counts,
         "n_train": len(tr), "n_eval": len(ev)}, indent=2), encoding="utf-8")
    with (rd / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch"] + list(hist.history))
        for i in range(epochs_run):
            w.writerow([i + 1] + [hist.history[k][i] for k in hist.history])
    with (rd / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "y_true", "y_prob", "script", "partition"])
        for r, yt, yp in zip(ev, yev, y_prob):
            w.writerow([r["uid"], int(yt), f"{yp:.6f}", r["script"],
                        eval_partition])
    if save_weights:
        suffix.save_weights(rd / "suffix.weights.h5")

    row = {
        "phase": phase, "arm": arm_tag, "config_hash": chash, "seed": seed,
        "corpus": spec.corpus, "split_file": data.SPLIT.name,
        "eval_partition": eval_partition,
        "n_params": counts["total"],
        "n_params_trainable": counts["trainable"],
        "epochs_run": epochs_run, "best_epoch": best_epoch,
        "optimizer": spec.optimizer, "learning_rate": spec.lr,
        "weight_decay": spec.weight_decay,
        "depth_config": spec.depth_config or "", "sca_d": spec.d,
        "sca_r": spec.r, "backbone": spec.backbone,
        "runtime_s": round(runtime, 2), "env": env_string(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{k: (round(v, 6) if isinstance(v, float) else v)
           for k, v in m.items()},
    }
    append_row(row)
    print(f"  {arm_tag:14s} {spec.optimizer:7s} lr={spec.lr:<8.0e} "
          f"seed={seed} ep={epochs_run:2d} "
          f"{eval_partition}_acc={m['acc']:.4f} ({runtime:.0f}s)")
    return row
