"""Phase 5 driver: the tuning sweep (guide section 6).

Resumable. Every finished run is appended to results/all_runs.csv and
skipped on a later invocation, so the sweep can be stopped, the laptop
closed or rebooted, and the command re-run to continue exactly where it
left off.

Runs are grouped by (arm-family, depth_config) so the frozen-prefix
feature cache is computed once per group instead of once per run.

Grid: configs/grid.yaml (option B, amended by the authors 2026-09-02
before the freeze). Tuning and model selection use the VALIDATION
partition only — hard rule 1.

Examples
  python src/run_tuning.py --dry-run
  python src/run_tuning.py --arms A1 A2
  python src/run_tuning.py --arms A3 A0
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import yaml

from train import FeatureCache, RunSpec, already_done, run

REPO = Path(__file__).resolve().parents[1]
GRID = REPO / "configs" / "grid.yaml"

# measured on this machine (see grid.yaml -> amendment.measurements)
SEC_PER_RUN = {"early": 228, "mid": 138, "late": 48, "all": 238,
               None: 40, "": 40}


def load_grid() -> dict:
    return yaml.safe_load(GRID.read_text(encoding="utf-8"))


def plan(arms: list[str], g: dict) -> list[tuple[RunSpec, int]]:
    seeds = g["tuning"]["seeds"]
    lrs = g["learning_rate"]
    wds = g["weight_decay"]
    out: list[tuple[RunSpec, int]] = []

    for arm in arms:
        if arm in ("A1", "A2"):
            for depth, opt, lr, wd, d, r in itertools.product(
                    g["sca"]["depth_config"], g["optimizer"], lrs, wds,
                    g["sca"]["d"], g["sca"]["r"]):
                spec = RunSpec(arm=arm, depth_config=depth, optimizer=opt,
                               lr=float(lr), weight_decay=float(wd),
                               d=int(d), r=int(r),
                               batch_size=g["common"]["batch_size"],
                               max_epochs=g["common"]["max_epochs"],
                               patience=g["common"]["early_stopping"]
                               ["patience"],
                               corpus=g["common"]["corpus"])
                out += [(spec, s) for s in seeds]

        elif arm in ("A3", "A5"):
            for opt, lr, wd in itertools.product(g["optimizer"], lrs, wds):
                spec = RunSpec(arm=arm, optimizer=opt, lr=float(lr),
                               weight_decay=float(wd),
                               batch_size=g["common"]["batch_size"],
                               max_epochs=g["common"]["max_epochs"],
                               patience=g["common"]["early_stopping"]
                               ["patience"],
                               corpus=g["common"]["corpus"])
                out += [(spec, s) for s in seeds]

        elif arm == "A0":
            anchor = g["a0"]["anchor"]
            for opt, lr in itertools.product(anchor["optimizers"], lrs):
                spec = RunSpec(arm="A0", backbone=anchor["backbone"],
                               optimizer=opt, lr=float(lr),
                               batch_size=g["common"]["batch_size"],
                               max_epochs=g["common"]["max_epochs"],
                               patience=g["common"]["early_stopping"]
                               ["patience"],
                               corpus=g["common"]["corpus"])
                out += [(spec, s) for s in seeds]

        elif arm == "A0_others":
            for entry in g["a0"]["others"]:
                for lr in lrs:
                    spec = RunSpec(arm="A0", backbone=entry["backbone"],
                                   optimizer=entry["optimizer"],
                                   lr=float(lr),
                                   batch_size=g["common"]["batch_size"],
                                   max_epochs=g["common"]["max_epochs"],
                                   patience=g["common"]["early_stopping"]
                                   ["patience"],
                                   corpus=g["common"]["corpus"])
                    out += [(spec, s) for s in seeds]
        else:
            raise ValueError(f"unknown arm group '{arm}'")
    return out


def group_key(spec: RunSpec):
    """Runs sharing this key share a frozen-prefix cache."""
    return (spec.arm if spec.arm in ("A1", "A2") else "frozen",
            spec.backbone, spec.depth_config, spec.corpus,
            spec.train_subset)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A1", "A2"],
                    help="A1 A2 A3 A5 A0 A0_others")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N runs (for a smoke check)")
    a = ap.parse_args()

    g = load_grid()
    if g.get("frozen"):
        print("NOTE: grid.yaml is marked frozen; running as specified.")
    jobs = plan(a.arms, g)

    todo = [(s, sd) for s, sd in jobs
            if not already_done("tuning", s.tag(), s.hash(), sd)]
    est = sum(SEC_PER_RUN.get(s.depth_config, 40) for s, _ in todo)
    print(f"planned runs : {len(jobs)}")
    print(f"already done : {len(jobs) - len(todo)}")
    print(f"to run       : {len(todo)}")
    print(f"estimated    : {est/3600:.1f} h "
          f"(measured per-run times, excludes thermal throttling)")
    if a.dry_run:
        for s, sd in todo[:15]:
            print(f"    {s.tag():12s} {s.optimizer:7s} lr={s.lr:<8.0e} "
                  f"seed={sd}")
        if len(todo) > 15:
            print(f"    ... and {len(todo)-15} more")
        return 0
    if not todo:
        print("nothing to do.")
        return 0

    todo.sort(key=lambda t: (group_key(t[0]), t[0].optimizer, t[0].lr, t[1]))
    cache, current, t_start = FeatureCache(), None, time.time()
    done = 0
    for spec, seed in todo:
        k = group_key(spec)
        if k != current:
            cache.clear()
            current = k
            print(f"\n[group] arm={spec.arm} backbone={spec.backbone} "
                  f"depth={spec.depth_config}  (prefix cache rebuilt)")
        run(spec, seed, phase="tuning", eval_partition="val", cache=cache)
        done += 1
        if a.limit and done >= a.limit:
            print(f"\nstopping after {done} runs (--limit)")
            break
    print(f"\ncompleted {done} runs in {(time.time()-t_start)/60:.1f} min")
    print(f"log: results/all_runs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
