"""Phase 9: statistical analysis (guide section 10).

Co-primary endpoints, on the S seed-paired differences d_i = A2_i - A1_i:
  1. pooled accuracy delta
  2. Urdu-subset accuracy delta
Each gets a paired t-test (primary), a Wilcoxon signed-rank test
(distribution-free robustness check), Cohen's dz, and a 95% CI. Holm is
applied across the two co-primary p-values.

Everything else -- A2 vs A3, the depth sweep, A2', A4, A5, non-Urdu
breakdowns -- is EXPLORATORY: means, SDs and CIs only, no significance
claims (guide section 10).

Outputs: results/stats_summary.json and results/stats_summary.md

Run: python src/stats.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
CONFIGS = REPO / "configs"


def base_arm(tag: str) -> str:
    """Map a run's arm tag to its reporting arm.

    A2' is logged as "A2_<depth>_predicted"; a naive split on "_" would
    collapse it into A2 and silently merge the predicted-script arm with
    the oracle one.
    """
    if tag.endswith("_predicted"):
        return "A2p"
    return tag.split("_")[0]


def load_final() -> dict[str, dict[int, dict]]:
    """{arm: {seed: row}} for phase=final."""
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    with (RESULTS / "all_runs.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["phase"] == "final":
                out[base_arm(r["arm"])][int(r["seed"])] = r
    return out


def paired(arm_a: dict[int, dict], arm_b: dict[int, dict],
           field: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    seeds = sorted(set(arm_a) & set(arm_b))
    a = np.array([float(arm_a[s][field]) for s in seeds])
    b = np.array([float(arm_b[s][field]) for s in seeds])
    return a, b, seeds


def endpoint(a1: np.ndarray, a2: np.ndarray, name: str) -> dict:
    """Paired analysis of d = A2 - A1."""
    d = a2 - a1
    S = len(d)
    res: dict = {
        "endpoint": name, "S": S,
        "a1_mean": float(a1.mean()), "a1_sd": float(a1.std(ddof=1)),
        "a2_mean": float(a2.mean()), "a2_sd": float(a2.std(ddof=1)),
        "mean_difference": float(d.mean()),
        "sd_difference": float(d.std(ddof=1)),
        "differences": d.tolist(),
    }
    if np.allclose(d, 0):
        res.update({"t": 0.0, "p_t": 1.0, "wilcoxon_w": None, "p_w": 1.0,
                    "dz": 0.0, "ci95": [0.0, 0.0],
                    "note": "all paired differences are exactly zero"})
        return res

    t, p_t = stats.ttest_rel(a2, a1)
    res["t"], res["p_t"] = float(t), float(p_t)
    try:
        w, p_w = stats.wilcoxon(d)
        res["wilcoxon_w"], res["p_w"] = float(w), float(p_w)
    except ValueError as e:                      # all-zero differences
        res["wilcoxon_w"], res["p_w"] = None, None
        res["wilcoxon_note"] = str(e)
    sd = d.std(ddof=1)
    res["dz"] = float(d.mean() / sd) if sd > 0 else float("nan")
    lo, hi = stats.t.interval(0.95, S - 1, loc=d.mean(),
                              scale=stats.sem(d))
    res["ci95"] = [float(lo), float(hi)]
    return res


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm step-down across the co-primary family."""
    items = sorted(((k, v) for k, v in pvals.items() if v is not None),
                   key=lambda kv: kv[1])
    m = len(items)
    out, prev_reject = {}, True
    for i, (k, p) in enumerate(items):
        thresh = alpha / (m - i)
        reject = bool(prev_reject and p <= thresh)
        out[k] = {"p_raw": p, "holm_threshold": thresh,
                  "reject_at_0.05": reject, "rank": i + 1}
        prev_reject = reject
    return out


def descriptive(arms: dict[str, dict[int, dict]], field: str) -> dict:
    out = {}
    for arm, rows in sorted(arms.items()):
        v = np.array([float(r[field]) for r in rows.values()
                      if r[field] not in ("", None)])
        if len(v) == 0:
            continue
        S = len(v)
        entry = {"n_seeds": S, "mean": float(v.mean()),
                 "sd": float(v.std(ddof=1)) if S > 1 else 0.0}
        if S > 1 and v.std(ddof=1) > 0:
            lo, hi = stats.t.interval(0.95, S - 1, loc=v.mean(),
                                      scale=stats.sem(v))
            entry["ci95"] = [float(lo), float(hi)]
        else:
            entry["ci95"] = [entry["mean"], entry["mean"]]
        out[arm] = entry
    return out


def main() -> int:
    ep = yaml.safe_load((CONFIGS / "endpoints.yaml").read_text(
        encoding="utf-8"))
    arms = load_final()
    if "A1" not in arms or "A2" not in arms:
        raise SystemExit("no final A1/A2 rows in all_runs.csv. Run Phase 8.")

    alpha = float(ep["multiplicity"]["alpha_family"])
    results: dict = {
        "corpus": ep["corpus"],
        "primary_pair": {"treatment": "A2", "control": "A1"},
        "delta_min": float(ep["effect"]["delta_min"]),
        "S_planned": ep["repeats"]["S"],
        "co_primary": {},
        "multiplicity": {"method": "holm", "alpha_family": alpha},
        "exploratory": {},
    }

    # ---- co-primary endpoints ----
    a1, a2, seeds = paired(arms["A1"], arms["A2"], "acc")
    results["co_primary"]["pooled_delta"] = endpoint(a1, a2, "pooled_delta")
    results["co_primary"]["pooled_delta"]["seeds"] = seeds

    u1, u2, useeds = paired(arms["A1"], arms["A2"], "acc_urdu")
    results["co_primary"]["urdu_delta"] = endpoint(u1, u2, "urdu_delta")
    results["co_primary"]["urdu_delta"]["seeds"] = useeds

    results["multiplicity"]["holm"] = holm(
        {k: v["p_t"] for k, v in results["co_primary"].items()}, alpha)

    # ---- exploratory ----
    for field, label in (("acc", "accuracy"), ("bal_acc", "balanced_accuracy"),
                         ("f1", "f1"), ("auc", "auc"),
                         ("acc_urdu", "accuracy_urdu"),
                         ("acc_english", "accuracy_english"),
                         ("acc_digit", "accuracy_digit")):
        results["exploratory"][label] = descriptive(arms, field)

    if "A3" in arms:
        b1, b2, _ = paired(arms["A3"], arms["A2"], "acc")
        e = endpoint(b1, b2, "A2_minus_A3_pooled")
        e["label"] = "EXPLORATORY - no significance claim"
        results["exploratory"]["A2_vs_A3"] = e

    results["note"] = (
        "Co-primary endpoints are the pooled and Urdu-subset A2-A1 "
        "accuracy differences, tested with paired t-tests and Holm-"
        "corrected across the two. Wilcoxon is reported alongside as a "
        "distribution-free check. Everything under `exploratory` is "
        "descriptive only and carries no significance claim.")

    (RESULTS / "stats_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # ---- markdown ----
    md = ["# Statistical summary (Phase 9)", "",
          "Generated by `src/stats.py` from `results/all_runs.csv`. "
          "Every number here traces to a run artifact on disk.", "",
          f"- Corpus: primary, {ep['corpus']['primary']}",
          f"- Majority-class baseline: "
          f"**{ep['corpus']['majority_class_baseline']:.4f}**",
          f"- Pre-specified minimal effect: "
          f"**{results['delta_min']}** ({results['delta_min']*100:.1f} pts)",
          f"- S (seeds per arm): **{len(seeds)}**", "",
          "## Co-primary endpoints", "",
          "| endpoint | A1 mean | A2 mean | mean diff | 95% CI | dz | "
          "t p | Wilcoxon p | Holm | reject |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in results["co_primary"].items():
        h = results["multiplicity"]["holm"].get(key, {})
        pw = "n/a" if r.get("p_w") is None else f"{r['p_w']:.4f}"
        md.append(
            f"| {key} | {r['a1_mean']:.4f} ± {r['a1_sd']:.4f} "
            f"| {r['a2_mean']:.4f} ± {r['a2_sd']:.4f} "
            f"| {r['mean_difference']:+.4f} "
            f"| [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] "
            f"| {r['dz']:+.3f} | {r['p_t']:.4f} | {pw} "
            f"| {h.get('holm_threshold', float('nan')):.4f} "
            f"| {'YES' if h.get('reject_at_0.05') else 'no'} |")

    md += ["", "## Per-arm descriptives (mean ± SD over seeds)", "",
           "| arm | accuracy | balanced acc | F1 | AUC | Urdu acc | "
           "English acc | Digit acc |", "|---|---|---|---|---|---|---|---|"]
    ex = results["exploratory"]
    for arm in sorted(ex["accuracy"]):
        def cell(metric):
            e = ex[metric].get(arm)
            return (f"{e['mean']:.4f} ± {e['sd']:.4f}" if e else "-")
        md.append(f"| {arm} | {cell('accuracy')} | "
                  f"{cell('balanced_accuracy')} | {cell('f1')} | "
                  f"{cell('auc')} | {cell('accuracy_urdu')} | "
                  f"{cell('accuracy_english')} | {cell('accuracy_digit')} |")

    md += ["", "## Interpretation guard", "",
           "- Only the two co-primary endpoints above carry inferential "
           "claims, and only after the Holm correction.",
           "- All other rows are descriptive. No significance is claimed "
           "for them (guide section 10).",
           "- Absolute accuracies are within-dataset and are not "
           "generalisation estimates.",
           "- A null or negative result is reported with the same "
           "prominence as a positive one (hard rule 6).", ""]
    (RESULTS / "stats_summary.md").write_text("\n".join(md),
                                              encoding="utf-8")

    print(f"S = {len(seeds)} seeds")
    for key, r in results["co_primary"].items():
        h = results["multiplicity"]["holm"].get(key, {})
        print(f"\n{key}:")
        print(f"  A1 {r['a1_mean']:.4f} ± {r['a1_sd']:.4f}   "
              f"A2 {r['a2_mean']:.4f} ± {r['a2_sd']:.4f}")
        print(f"  mean diff {r['mean_difference']:+.4f}  "
              f"95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
              f"dz={r['dz']:+.3f}")
        print(f"  t p={r['p_t']:.4f}  holm thresh="
              f"{h.get('holm_threshold', float('nan')):.4f}  "
              f"reject={'YES' if h.get('reject_at_0.05') else 'no'}")
    print(f"\nwrote {RESULTS/'stats_summary.json'}")
    print(f"wrote {RESULTS/'stats_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
