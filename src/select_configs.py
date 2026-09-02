"""Phase 5 step 2-3: select each arm's best config (guide section 6).

Selection rule, identical for every arm (this is the fairness
guarantee):
  1. score = mean validation accuracy over the three tuning seeds
  2. ties broken by higher mean val F1, then by fewer parameters
Selection uses the VALIDATION partition only (hard rule 1).

A1 and A2 sweep depth_config identically and each reports its own best
fair configuration, per the paper's design.

Output: results/selected_configs.json  (+ a readable summary on stdout)

Run: python src/select_configs.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
TUNING_SEEDS = {101, 102, 103}
MATCHED_DEPTH = "mid"   # see DEVIATIONS.md D-006


def base_arm(tag: str) -> str:
    return tag.split("_")[0]


def main() -> int:
    rows = [r for r in csv.DictReader(
        (RESULTS / "all_runs.csv").open(encoding="utf-8"))
        if r["phase"] == "tuning"]

    # group by (base arm, config_hash)
    g: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        g[(base_arm(r["arm"]), r["config_hash"])].append(r)

    per_arm: dict[str, list[dict]] = defaultdict(list)
    for (arm, chash), rs in g.items():
        seeds = {int(r["seed"]) for r in rs}
        if seeds != TUNING_SEEDS:
            print(f"  WARNING: {arm} {chash} has seeds {sorted(seeds)}, "
                  f"expected {sorted(TUNING_SEEDS)} - excluded")
            continue
        accs = [float(r["acc"]) for r in rs]
        f1s = [float(r["f1"]) for r in rs]
        r0 = rs[0]
        per_arm[arm].append({
            "config_hash": chash,
            "arm_tag": r0["arm"],
            "backbone": r0["backbone"],
            "depth_config": r0["depth_config"] or None,
            "optimizer": r0["optimizer"],
            "learning_rate": float(r0["learning_rate"]),
            "weight_decay": float(r0["weight_decay"]),
            "sca_d": int(r0["sca_d"]),
            "sca_r": int(r0["sca_r"]),
            "n_params": int(r0["n_params"]),
            "mean_val_acc": round(st.mean(accs), 6),
            "sd_val_acc": round(st.stdev(accs), 6),
            "mean_val_f1": round(st.mean(f1s), 6),
            "n_seeds": len(rs),
        })

    selected = {}
    for arm in sorted(per_arm):
        cands = sorted(
            per_arm[arm],
            key=lambda c: (-c["mean_val_acc"], -c["mean_val_f1"],
                           c["n_params"]))
        best = cands[0]
        tied = [c for c in cands
                if c["mean_val_acc"] == best["mean_val_acc"]]
        best = dict(best)
        best["n_configs_evaluated"] = len(cands)
        best["n_tied_on_accuracy"] = len(tied)
        best["tie_broken_by"] = (
            "none" if len(tied) == 1 else
            "val_f1" if len({c["mean_val_f1"] for c in tied}) > 1 else
            "fewer_params")
        selected[arm] = best

        print(f"\n=== {arm} "
              f"({len(cands)} configs x 3 seeds = {len(cands)*3} runs) ===")
        print(f"{'rank':>4s} {'depth':6s} {'opt':8s} {'lr':>7s} "
              f"{'mean_acc':>9s} {'sd':>7s} {'mean_f1':>8s}")
        for i, c in enumerate(cands[:5], 1):
            mark = " <-- selected" if i == 1 else ""
            print(f"{i:>4d} {str(c['depth_config'] or '-'):6s} "
                  f"{c['optimizer']:8s} {c['learning_rate']:7.0e} "
                  f"{c['mean_val_acc']:9.4f} {c['sd_val_acc']:7.4f} "
                  f"{c['mean_val_f1']:8.4f}{mark}")
        if len(tied) > 1:
            print(f"     ({len(tied)} configs tied on accuracy; "
                  f"broken by {best['tie_broken_by']})")

    # ---- primary pair at MATCHED depth (DEVIATIONS.md D-006) ----
    # Each arm's own best depth differs (A1 -> all, A2 -> mid), which
    # breaks the paper's matched-capacity claim. The primary comparison
    # therefore fixes one common depth; each arm still tunes optimizer
    # and learning rate freely inside it, so the tuning budget stays
    # equal. `mid` is chosen because it is A2's best AND tied-best for
    # A1, so neither arm is handicapped.
    matched = {}
    for arm in ("A1", "A2"):
        cands = [c for c in per_arm.get(arm, [])
                 if c["depth_config"] == MATCHED_DEPTH]
        if cands:
            matched[arm] = sorted(
                cands, key=lambda c: (-c["mean_val_acc"], -c["mean_val_f1"],
                                      c["n_params"]))[0]
    if len(matched) == 2:
        assert matched["A1"]["n_params"] == matched["A2"]["n_params"], \
            "matched-depth pair must have identical parameter counts"
        print(f"\n--- PRIMARY PAIR (matched depth = {MATCHED_DEPTH}) ---")
        for arm in ("A1", "A2"):
            m = matched[arm]
            print(f"  {arm}: {m['optimizer']:8s} lr={m['learning_rate']:.0e} "
                  f"val_acc={m['mean_val_acc']:.4f} "
                  f"n_params={m['n_params']:,}")
        print(f"  parameter counts identical: "
              f"{matched['A1']['n_params']:,}")

    out = {
        "selection_rule": ("mean val accuracy over seeds 101-103; ties by "
                           "mean val F1, then fewer parameters"),
        "partition_used": "val",
        "note": ("Hard rule 1: model selection never touches the test "
                 "partition."),
        "selected": selected,
        "primary_pair": {
            "policy": "matched_depth",
            "depth_config": MATCHED_DEPTH,
            "rationale": ("The paper's central claim is matched capacity. "
                          "Each arm's own best depth differs (A1 all, A2 "
                          "mid), giving the CONTROL 1.40% more parameters. "
                          "Fixing a common depth restores exact capacity "
                          "matching; optimizer and lr are still tuned "
                          "independently per arm, so the budget stays "
                          "equal. `mid` is A2's best and tied-best for A1 "
                          "(0.8065 at both mid and all), so neither arm is "
                          "handicapped, and the observed val delta is "
                          "-0.0054 either way."),
            "deviation": "DEVIATIONS.md D-006",
            "arms": matched,
        },
        "secondary_pair_each_arm_own_best": {
            "policy": "each_arm_own_best_depth",
            "note": ("Guide section 6 / paper as written. Reported as an "
                     "exploratory sensitivity analysis because parameter "
                     "counts differ."),
            "arms": {k: selected[k] for k in ("A1", "A2") if k in selected},
        },
    }
    (RESULTS / "selected_configs.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS/'selected_configs.json'}")

    if "A1" in selected and "A2" in selected:
        a1, a2 = selected["A1"], selected["A2"]
        print("\n--- primary pair ---")
        print(f"  A1: depth={a1['depth_config']} {a1['optimizer']} "
              f"lr={a1['learning_rate']:.0e}  val_acc={a1['mean_val_acc']:.4f}")
        print(f"  A2: depth={a2['depth_config']} {a2['optimizer']} "
              f"lr={a2['learning_rate']:.0e}  val_acc={a2['mean_val_acc']:.4f}")
        assert a1["n_params"] == a2["n_params"] or \
            a1["depth_config"] != a2["depth_config"], \
            "same depth must imply identical parameter counts"
        print(f"  n_params: A1={a1['n_params']:,}  A2={a2['n_params']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
