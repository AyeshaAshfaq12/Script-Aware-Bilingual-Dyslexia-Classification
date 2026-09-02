"""Phase 8: final runs for every arm (guide section 9).

Final seeds are 301..(300+S), identical across all arms. Each run writes
its own folder under runs/ with the resolved config, parameter counts,
training history, per-image predictions and weights, and appends one row
to results/all_runs.csv.

Resumable: anything already logged is skipped.

HARD RULE 1: evaluating on the test partition requires the `freeze-v1`
git tag. train.py raises without it, so this script cannot touch test
data before the freeze.

Arms
  A1, A2  co-primary pair, at the MATCHED-DEPTH configs (D-006)
  A0, A3  context, at their selected configs
  A5      script classifier routes to three script-specific heads
  A2p     A2 with predicted (not oracle) script ids
  A4      per-script experts (urdu / english), routed at inference

Run:
  python src/run_final.py --arms A1 A2 A0 A3
  python src/run_final.py --arms A5 A2p A4
  python src/run_final.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

import models
from run_tuning import RunnerLock
from evaluate import compute_metrics, script_classifier_metrics
from train import (FeatureCache, RunSpec, already_done, append_row,
                   assert_may_touch_test, env_string, freeze_tag_exists,
                   git_commit, make_optimizer, run,
                   train_script_classifier)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
CONFIGS = REPO / "configs"
RUNS = REPO / "runs"


def load_all():
    ep = yaml.safe_load((CONFIGS / "endpoints.yaml").read_text(
        encoding="utf-8"))
    grid = yaml.safe_load((CONFIGS / "grid.yaml").read_text(encoding="utf-8"))
    sel = json.loads((RESULTS / "selected_configs.json").read_text(
        encoding="utf-8"))
    return ep, grid, sel


def spec_from(cfg: dict, grid: dict, **over) -> RunSpec:
    base = dict(
        arm=cfg["arm_tag"].split("_")[0],
        backbone=cfg["backbone"],
        depth_config=cfg["depth_config"],
        d=cfg["sca_d"], r=cfg["sca_r"],
        optimizer=cfg["optimizer"],
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        batch_size=grid["common"]["batch_size"],
        max_epochs=grid["common"]["max_epochs"],
        patience=grid["common"]["early_stopping"]["patience"],
        corpus=grid["common"]["corpus"],
    )
    base.update(over)
    return RunSpec(**base)


def final_seeds(ep: dict) -> list[int]:
    S = ep["repeats"]["S"]
    if not S:
        raise SystemExit("endpoints.yaml has no S. Run Phase 6 first "
                         "(python src/run_pilot.py).")
    return [300 + i for i in range(1, int(S) + 1)]


# ------------------------------------------------------------- A4
def run_a4(seed: int, grid: dict, sel: dict, cache: FeatureCache,
           eval_partition: str) -> dict:
    """Per-script experts: train Urdu-only and English-only A0 heads,
    route at inference by the adjudicated script tag.

    Digits carry no expert of their own. Guide 5.3: they are evaluated by
    whichever expert the adjudicated glyph form assigns. Every digit in
    this corpus is a Western Arabic numeral (agreement_report.md), which
    is the form used in English writing, so digits route to the English
    expert and are reported separately.
    """
    import time
    import keras
    tag = "A4"
    a0 = sel["selected"]["A0"]
    chash = f"a4-{a0['config_hash']}"
    if already_done("final", tag, chash, seed):
        print(f"  skip (already logged): {tag} {chash} seed={seed}")
        return {}
    t0 = time.time()

    preds: dict[str, float] = {}
    experts = {}
    for subset in ("urdu", "english"):
        spec = spec_from(a0, grid, train_subset=subset)
        keras.utils.set_random_seed(seed)
        prefix, suffix, full = models.build_split(
            "A0", backbone=spec.backbone, weights="imagenet")
        tr = cache.records(spec, "train")
        Ftr = cache.features(spec, "train", prefix)
        ytr = np.array([r["label"] for r in tr], dtype="float32")
        str_ = np.array([r["script_id"] for r in tr], dtype="int32")
        # validation signal for early stopping: same-script val images
        probe = spec_from(a0, grid)
        va_all = cache.records(probe, "val")
        Fva_all = cache.features(probe, "val", prefix)
        m = np.array([r["script"] == subset for r in va_all])
        suffix.compile(optimizer=make_optimizer(spec),
                       loss="binary_crossentropy", metrics=["accuracy"])
        yva = np.array([r["label"] for r in va_all], dtype="float32")
        sva = np.array([r["script_id"] for r in va_all], dtype="int32")
        suffix.fit([Ftr, str_], ytr,
                   validation_data=([Fva_all[m], sva[m]], yva[m]),
                   epochs=spec.max_epochs, batch_size=spec.batch_size,
                   verbose=0, shuffle=True,
                   callbacks=[keras.callbacks.EarlyStopping(
                       monitor="val_loss", patience=spec.patience,
                       restore_best_weights=True)])
        experts[subset] = (prefix, suffix, full)

    probe = spec_from(a0, grid)
    ev = cache.records(probe, eval_partition)
    Fev = cache.features(probe, eval_partition, experts["urdu"][0])
    sev = np.array([r["script_id"] for r in ev], dtype="int32")
    yev = np.array([r["label"] for r in ev], dtype="float32")

    probs = {k: experts[k][1].predict([Fev, sev], verbose=0).ravel()
             for k in experts}
    # route: urdu -> urdu expert; english and digit -> english expert
    y_prob = np.where([r["script"] == "urdu" for r in ev],
                      probs["urdu"], probs["english"])

    mets = compute_metrics(yev, y_prob, [r["script"] for r in ev])
    counts = models.param_counts(experts["urdu"][2])
    rd = RUNS / f"{tag}_{chash}_{seed}"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(json.dumps({
        "arm": "A4", "experts": ["urdu", "english"],
        "base_config": a0, "seed": seed,
        "digit_routing": "english expert (all digits are Western numerals)",
        "eval_partition": eval_partition, "git_commit": git_commit(),
    }, indent=2), encoding="utf-8")
    with (rd / "predictions.csv").open("w", newline="",
                                       encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "y_true", "y_prob", "script", "routed_to"])
        for r, yt, yp in zip(ev, yev, y_prob):
            w.writerow([r["uid"], int(yt), f"{yp:.6f}", r["script"],
                        "urdu" if r["script"] == "urdu" else "english"])

    row = {"phase": "final", "arm": tag, "config_hash": chash, "seed": seed,
           "corpus": probe.corpus, "split_file": "split_v1.json",
           "eval_partition": eval_partition,
           "n_params": counts["total"] * 2,
           "n_params_trainable": counts["trainable"] * 2,
           "epochs_run": "", "best_epoch": "",
           "optimizer": a0["optimizer"], "learning_rate": a0["learning_rate"],
           "weight_decay": a0["weight_decay"], "depth_config": "",
           "sca_d": a0["sca_d"], "sca_r": a0["sca_r"],
           "backbone": a0["backbone"],
           "runtime_s": round(time.time() - t0, 2), "env": env_string(),
           "timestamp": __import__("datetime").datetime.now(
               __import__("datetime").timezone.utc).isoformat(
                   timespec="seconds"),
           **{k: (round(v, 6) if isinstance(v, float) else v)
              for k, v in mets.items()}}
    append_row(row)
    print(f"  {tag:14s} seed={seed} {eval_partition}_acc={mets['acc']:.4f} "
          f"({row['runtime_s']:.0f}s)")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["A1", "A2", "A0", "A3"],
                    help="A1 A2 A0 A3 A5 A2p A4")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--partition", default="test",
                    choices=["val", "test"])
    a = ap.parse_args()

    ep, grid, sel = load_all()
    seeds = final_seeds(ep)
    if a.partition == "test":
        assert_may_touch_test()

    print(f"final seeds : {seeds[0]}..{seeds[-1]}  (S={len(seeds)})")
    print(f"partition   : {a.partition}")
    print(f"freeze tag  : {'present' if freeze_tag_exists() else 'ABSENT'}")
    if a.dry_run:
        for arm in a.arms:
            print(f"  {arm}: {len(seeds)} runs")
        return 0

    cache = FeatureCache()
    pair = sel["primary_pair"]["arms"]

    with RunnerLock():
      for arm in a.arms:
        print(f"\n[{arm}]")
        if arm in ("A1", "A2"):
            spec = spec_from(pair[arm], grid)
            for s in seeds:
                run(spec, s, phase="final", eval_partition=a.partition,
                    cache=cache, save_weights=True)
        elif arm in ("A0", "A3", "A5"):
            spec = spec_from(sel["selected"][arm], grid)
            for s in seeds:
                run(spec, s, phase="final", eval_partition=a.partition,
                    cache=cache, save_weights=True)
        elif arm == "A2p":
            spec = spec_from(pair["A2"], grid, script_source="predicted")
            for s in seeds:
                sc = train_script_classifier(
                    s, cache, corpus=spec.corpus,
                    partitions=("train", a.partition))
                run(spec, s, phase="final", eval_partition=a.partition,
                    cache=cache, save_weights=True,
                    script_ids_override={
                        "train": sc["predicted"]["train"],
                        a.partition: sc["predicted"][a.partition]})
                print(f"    script classifier acc "
                      f"({a.partition}): {sc['accuracy'][a.partition]:.4f}")
        elif arm == "A4":
            for s in seeds:
                run_a4(s, grid, sel, cache, a.partition)
        else:
            raise SystemExit(f"unknown arm '{arm}'")
      print(f"\nlog: results/all_runs.csv")
      return 0


if __name__ == "__main__":
      raise SystemExit(main())
