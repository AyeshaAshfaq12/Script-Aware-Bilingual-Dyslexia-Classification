"""Phase 4: data loading (guide section 5.1).

Yields (image, script_id), label for every arm, identically.

- Input resolution 224x224x3, MobileNet `preprocess_input` normalisation
  (guide section 5.1; resolves the paper's resolution TODO).
- No augmentation, no synthetic data, no external data (hard rule 3).
- Partitions come from data/splits/split_v1.json and are never redrawn.

Script id encoding is fixed here and must not change:
    urdu = 0, english = 1, digit = 2
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SPLIT = DATA / "splits" / "split_v1.json"
CORPUS = DATA / "corpus_v1_scripts.csv"

IMG_SIZE = 224
N_SCRIPTS = 3
SCRIPT_ID = {"urdu": 0, "english": 1, "digit": 2}
ID_SCRIPT = {v: k for k, v in SCRIPT_ID.items()}
PARTITIONS = ("train", "val", "test")


def load_split() -> dict:
    return json.loads(SPLIT.read_text(encoding="utf-8"))


def load_corpus() -> dict[str, dict]:
    """uid -> record, for all 852 released files."""
    return {r["uid"]: r
            for r in csv.DictReader(CORPUS.open(encoding="utf-8"))}


def records(partition: str, corpus: str = "primary") -> list[dict]:
    """Records for one partition of one corpus, in a fixed order.

    corpus="primary"     618 deduplicated images (co-primary endpoints)
    corpus="sensitivity" all 852 released files (secondary analysis)
    """
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}")
    if corpus not in ("primary", "sensitivity"):
        raise ValueError("corpus must be 'primary' or 'sensitivity'")

    split = load_split()
    uids = (split[partition] if corpus == "primary"
            else split["sensitivity_files"][partition])
    table = load_corpus()
    out = []
    for uid in sorted(uids):          # sorted => deterministic order
        r = table[uid]
        out.append({
            "uid": uid,
            "path": str(DATA / "raw" / uid),
            "label": int(r["label"]),
            "script": r["script"],
            "script_id": SCRIPT_ID[r["script"]],
            "unit_id": int(r["unit_id"]),
            "class_folder": r["class_folder"],
        })
    return out


def _decode(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img = tf.cast(img, tf.float32)
    return preprocess_input(img)      # scales to [-1, 1]


def make_dataset(partition: str, corpus: str = "primary",
                 batch_size: int = 16, shuffle: bool = False,
                 seed: int | None = None,
                 limit: int | None = None) -> tf.data.Dataset:
    """((image, script_id), label) batches.

    shuffle=True is for training only; it consumes `seed` so that data
    ordering is reproducible per run seed.
    """
    recs = records(partition, corpus)
    if limit is not None:
        recs = recs[:limit]

    paths = tf.constant([r["path"] for r in recs])
    sids = tf.constant([r["script_id"] for r in recs], dtype=tf.int32)
    labels = tf.constant([r["label"] for r in recs], dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, sids, labels))
    if shuffle:
        ds = ds.shuffle(len(recs), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, s, y: ((_decode(p), s), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def as_arrays(partition: str, corpus: str = "primary",
              limit: int | None = None):
    """Eager (images, script_ids, labels) — for small evaluation sets."""
    recs = records(partition, corpus)
    if limit is not None:
        recs = recs[:limit]
    imgs = np.stack([_decode(tf.constant(r["path"])).numpy() for r in recs])
    sids = np.array([r["script_id"] for r in recs], dtype=np.int32)
    ys = np.array([r["label"] for r in recs], dtype=np.float32)
    return imgs, sids, ys


def class_counts(partition: str, corpus: str = "primary") -> dict[str, int]:
    recs = records(partition, corpus)
    return {"n": len(recs),
            "pos": sum(r["label"] for r in recs),
            "neg": sum(1 - r["label"] for r in recs)}


if __name__ == "__main__":
    for c in ("primary", "sensitivity"):
        print(f"\n{c}")
        for p in PARTITIONS:
            cc = class_counts(p, c)
            recs = records(p, c)
            scr = {k: sum(1 for r in recs if r["script"] == k)
                   for k in SCRIPT_ID}
            print(f"  {p:5s} n={cc['n']:4d} pos={cc['pos']:4d} "
                  f"neg={cc['neg']:4d}  {scr}")
