"""Phase 6: pilot and power analysis (guide section 7).

Runs A1 and A2 at their selected configs under the pilot seeds, then
estimates the between-seed SD of the paired A2-A1 validation accuracy
difference and solves for the repeat count S.

The primary pair is the MATCHED-DEPTH pair (DEVIATIONS.md D-006), so
both arms have identical parameter counts.

Validation partition only — hard rule 1. No test data is touched.

Run:
  python src/run_pilot.py            # run the pilot, then report
  python src/run_pilot.py --report   # recompute from existing rows
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import numpy as np
import yaml
from statsmodels.stats.power import TTestPower

from train import FeatureCache, RunSpec, run

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
CONFIGS = REPO / "configs"


def load_endpoints() -> dict:
    return yaml.safe_load((CONFIGS / "endpoints.yaml").read_text(
        encoding="utf-8"))


def spec_from(cfg: dict, grid: dict) -> RunSpec:
    return RunSpec(
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


def pilot_rows() -> dict[str, dict[int, float]]:
    """{arm: {seed: val_accuracy}} for phase=pilot."""
    out: dict[str, dict[int, float]] = {}
    with (RESULTS / "all_runs.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["phase"] != "pilot":
                continue
            arm = r["arm"].split("_")[0]
            out.setdefault(arm, {})[int(r["seed"])] = float(r["acc"])
    return out


def report(ep: dict) -> int:
    got = pilot_rows()
    if "A1" not in got or "A2" not in got:
        print("pilot rows incomplete; nothing to report yet.")
        return 1
    seeds = sorted(set(got["A1"]) & set(got["A2"]))
    a1 = np.array([got["A1"][s] for s in seeds])
    a2 = np.array([got["A2"][s] for s in seeds])
    d = a2 - a1
    sd_pilot = float(st.stdev(d.tolist())) if len(d) > 1 else float("nan")
    delta_min = float(ep["effect"]["delta_min"])
    alpha = float(ep["multiplicity"]["alpha_per_endpoint_for_power"])
    power = float(ep["effect"]["target_power"])
    floor = int(ep["repeats"]["S_floor"])
    ceiling = int(ep["repeats"]["S_ceiling"])

    print(f"pilot seeds      : {seeds}")
    print(f"A1 val acc       : {np.round(a1, 4).tolist()}")
    print(f"A2 val acc       : {np.round(a2, 4).tolist()}")
    print(f"paired d (A2-A1) : {np.round(d, 4).tolist()}")
    print(f"mean d           : {d.mean():+.4f}")
    print(f"sd_pilot         : {sd_pilot:.4f}")

    if sd_pilot <= 0:
        print("\nsd_pilot is zero: every seed gave an identical difference. "
              "The power formula is undefined; S falls back to the floor.")
        s_raw, s_applied, why = float("nan"), floor, "sd_pilot == 0"
    else:
        s_raw = float(TTestPower().solve_power(
            effect_size=delta_min / sd_pilot, power=power, alpha=alpha,
            alternative="two-sided"))
        s_applied = int(min(max(np.ceil(s_raw), floor), ceiling))
        why = ("ceiling" if np.ceil(s_raw) > ceiling
               else "floor" if np.ceil(s_raw) < floor else "none")
        print(f"effect size      : {delta_min/sd_pilot:.4f} "
              f"(delta_min={delta_min} / sd_pilot)")
        print(f"S (unclamped)    : {np.ceil(s_raw):.0f}")
    print(f"S (applied)      : {s_applied}   clamp={why} "
          f"[floor {floor}, ceiling {ceiling}]")

    achieved = float("nan")
    if sd_pilot > 0:
        achieved = float(TTestPower().power(
            effect_size=delta_min / sd_pilot, nobs=s_applied, alpha=alpha,
            alternative="two-sided"))
        detectable = float(TTestPower().solve_power(
            nobs=s_applied, power=power, alpha=alpha,
            alternative="two-sided")) * sd_pilot
        print(f"achieved power   : {achieved:.3f} "
              f"(target {power}) at S={s_applied}")
        print(f"detectable delta : {detectable*100:.2f} points at "
              f"{power:.0%} power")

    out = {
        "pilot_seeds": seeds,
        "a1_val_acc": a1.tolist(),
        "a2_val_acc": a2.tolist(),
        "paired_differences": d.tolist(),
        "mean_difference": float(d.mean()),
        "sd_pilot": sd_pilot,
        "delta_min": delta_min,
        "alpha_per_endpoint": alpha,
        "target_power": power,
        "S_unclamped": None if np.isnan(s_raw) else float(np.ceil(s_raw)),
        "S_applied": s_applied,
        "clamp_applied": why,
        "achieved_power_at_S": None if np.isnan(achieved) else achieved,
        "partition": "val",
        "note": ("Pilot uses the validation partition only. S is applied "
                 "identically to every arm in Phase 8."),
    }
    (RESULTS / "pilot_power.json").write_text(json.dumps(out, indent=2),
                                              encoding="utf-8")
    print(f"\nwrote {RESULTS/'pilot_power.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="skip training, recompute from logged rows")
    a = ap.parse_args()

    ep = load_endpoints()
    if a.report:
        return report(ep)

    grid = yaml.safe_load((CONFIGS / "grid.yaml").read_text(encoding="utf-8"))
    sel = json.loads((RESULTS / "selected_configs.json").read_text(
        encoding="utf-8"))
    pair = sel["primary_pair"]["arms"]
    seeds = ep["repeats"]["pilot_seeds"]

    print(f"pilot: matched depth = {sel['primary_pair']['depth_config']}, "
          f"seeds {seeds}")
    for arm in ("A1", "A2"):
        print(f"  {arm}: {pair[arm]['optimizer']} "
              f"lr={pair[arm]['learning_rate']:.0e} "
              f"n_params={pair[arm]['n_params']:,}")
    assert pair["A1"]["n_params"] == pair["A2"]["n_params"], \
        "primary pair must have identical parameter counts"

    cache = FeatureCache()
    for arm in ("A1", "A2"):
        spec = spec_from(pair[arm], grid)
        for seed in seeds:
            run(spec, seed, phase="pilot", eval_partition="val", cache=cache)
    return report(ep)


if __name__ == "__main__":
    raise SystemExit(main())
