"""Phase 2: inter-annotator agreement and adjudication.

Reads scripts_annotA.csv and scripts_annotB.csv, computes Cohen's kappa
and raw agreement on `script`, writes the disagreement worklist for
human adjudication, and (once adjudicated) emits scripts_final.csv plus
agreement_report.md.

Per DEVIATIONS.md D-001 annotator A is Claude Code and annotator B is a
human author, so the kappa reported here is HUMAN-vs-AI agreement and
must be described that way in the paper.

Usage
  python src/agreement.py --stage disagreements
      -> writes data/annotations/disagreements.csv for the human to fill
         in the `final` and `reason` columns.
  python src/agreement.py --stage final
      -> reads the completed disagreements.csv, writes scripts_final.csv
         and agreement_report.md.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, confusion_matrix

REPO = Path(__file__).resolve().parents[1]
ANN = REPO / "data" / "annotations"
LABELS = ["urdu", "english", "digit"]


def load(p: Path) -> dict[int, dict]:
    if not p.exists():
        raise SystemExit(f"missing {p}. Annotator pass not complete.")
    return {int(r["unit_id"]): r
            for r in csv.DictReader(p.open(encoding="utf-8"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["disagreements", "final"],
                    required=True)
    a = ap.parse_args()

    A = load(ANN / "scripts_annotA.csv")
    B = load(ANN / "scripts_annotB.csv")
    ids = sorted(set(A) & set(B))
    if set(A) != set(B):
        raise SystemExit(
            f"unit sets differ: A has {len(A)}, B has {len(B)}, "
            f"overlap {len(ids)}. Both annotators must cover every unit.")

    a_s = [A[i]["script"] for i in ids]
    b_s = [B[i]["script"] for i in ids]
    kappa = cohen_kappa_score(a_s, b_s, labels=LABELS)
    n_agree = sum(x == y for x, y in zip(a_s, b_s))
    raw = n_agree / len(ids)
    disagree = [i for i in ids if A[i]["script"] != B[i]["script"]]

    if a.stage == "disagreements":
        out = ANN / "disagreements.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["unit_id", "annotA_script", "annotA_note",
                        "annotB_script", "annotB_note", "final", "reason"])
            for i in disagree:
                w.writerow([i, A[i]["script"], A[i].get("note", ""),
                            B[i]["script"], B[i].get("note", ""), "", ""])
        print(f"kappa (script) : {kappa:.4f}")
        print(f"raw agreement  : {raw:.4f}  ({n_agree}/{len(ids)})")
        print(f"disagreements  : {len(disagree)}")
        print(f"\nwrote {out}")
        print("Human author: fill in `final` and `reason` for every row, "
              "then rerun with --stage final.")
        return 0

    # ---- stage: final ----
    adj: dict[int, dict] = {}
    dpath = ANN / "disagreements.csv"
    if disagree:
        if not dpath.exists():
            raise SystemExit(f"{len(disagree)} disagreements but {dpath} "
                             f"is missing. Run --stage disagreements first.")
        for r in csv.DictReader(dpath.open(encoding="utf-8")):
            if not r["final"].strip():
                raise SystemExit(
                    f"unit {r['unit_id']} has no adjudicated `final` value. "
                    f"Every disagreement must be resolved by a human.")
            if r["final"] not in LABELS:
                raise SystemExit(f"unit {r['unit_id']}: final "
                                 f"'{r['final']}' not in {LABELS}")
            if not r["reason"].strip():
                raise SystemExit(f"unit {r['unit_id']}: `reason` is required "
                                 f"(guide section 3: log every resolution)")
            adj[int(r["unit_id"])] = r

    final_rows = []
    for i in ids:
        if i in adj:
            script, src = adj[i]["final"], "adjudicated"
        else:
            script, src = A[i]["script"], "agreed"
        ga, gb = A[i]["digit_glyph"], B[i]["digit_glyph"]
        if ga == gb:
            glyph = ga
        else:
            glyph = gb if script == B[i]["script"] else ga
        if script != "digit":
            glyph = "na"
        elif glyph == "na":
            glyph = "unclear"
        final_rows.append({
            "unit_id": i,
            "script": script,
            "digit_glyph": glyph,
            "ambiguous": str(int(A[i]["ambiguous"] == "1"
                                 or B[i]["ambiguous"] == "1")),
            "source": src,
            "annotA_script": A[i]["script"],
            "annotB_script": B[i]["script"],
        })

    fpath = ANN / "scripts_final.csv"
    with fpath.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(final_rows[0].keys()))
        w.writeheader()
        w.writerows(final_rows)

    cm = confusion_matrix(a_s, b_s, labels=LABELS)
    fc = Counter(r["script"] for r in final_rows)
    gc = Counter(r["digit_glyph"] for r in final_rows if r["script"] == "digit")
    amb = sum(int(r["ambiguous"]) for r in final_rows)
    n = len(final_rows)

    md: list[str] = []
    md.append("# Inter-annotator agreement report (Phase 2)")
    md.append("")
    md.append("Generated by `src/agreement.py`. Protocol: "
              "`data/annotations/protocol.md`.")
    md.append("")
    md.append("> **Annotator A is Claude Code; annotator B is a human "
              "author (`DEVIATIONS.md` D-001).** The kappa below is "
              "therefore a **human-vs-AI** agreement statistic, not an "
              "inter-human one, and must be described as such in the "
              "paper. Every disagreement was resolved by the human "
              "author.")
    md.append("")
    md.append("## Agreement on `script`")
    md.append("")
    md.append(f"- Annotation units: **{len(ids)}** (distinct images)")
    md.append(f"- Cohen's kappa: **{kappa:.4f}**")
    md.append(f"- Raw agreement: **{raw*100:.2f}%** ({n_agree}/{len(ids)})")
    md.append(f"- Disagreements adjudicated: **{len(disagree)}**")
    md.append("")
    md.append("### Confusion matrix (rows = annotator A, cols = annotator B)")
    md.append("")
    md.append("| A vs B | " + " | ".join(LABELS) + " |")
    md.append("|---" * (len(LABELS) + 1) + "|")
    for lab, row in zip(LABELS, cm):
        md.append(f"| **{lab}** | " + " | ".join(str(v) for v in row) + " |")
    md.append("")
    md.append("## Adjudicated composition (`scripts_final.csv`)")
    md.append("")
    md.append("| script | n | % |")
    md.append("|---|---|---|")
    for lab in LABELS:
        md.append(f"| {lab} | {fc[lab]} | {fc[lab]/n*100:.1f}% |")
    md.append(f"| **total** | **{n}** | 100% |")
    md.append("")
    md.append("## Digit-glyph composition")
    md.append("")
    md.append("These counts go into the paper's Data section verbatim and "
              "determine whether digit results are interpretable.")
    md.append("")
    md.append("| digit_glyph | n |")
    md.append("|---|---|")
    for g in ["western", "eastern_arabic_indic", "unclear"]:
        md.append(f"| {g} | {gc.get(g, 0)} |")
    md.append("")
    md.append("## Ambiguity")
    md.append("")
    md.append(f"- Units flagged `ambiguous=1` by at least one annotator: "
              f"**{amb}** ({amb/n*100:.1f}%)")
    md.append("- A flag marks a glyph form that is ambiguous *between "
              "scripts*; the annotator still committed to a single "
              "most-likely script.")
    md.append("")
    md.append("## Provenance")
    md.append("")
    md.append(f"- `scripts_annotA.csv` - annotator A (Claude Code), "
              f"{len(A)} rows")
    md.append(f"- `scripts_annotB.csv` - annotator B (human), {len(B)} rows")
    md.append(f"- `disagreements.csv` - {len(disagree)} adjudicated rows "
              f"with reasons")
    md.append("- `scripts_final.csv` - single source of truth for script "
              "tags")
    md.append("")

    (ANN / "agreement_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"kappa {kappa:.4f}  raw {raw:.4f}  disagreements {len(disagree)}")
    print(f"wrote {fpath}")
    print(f"wrote {ANN / 'agreement_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
