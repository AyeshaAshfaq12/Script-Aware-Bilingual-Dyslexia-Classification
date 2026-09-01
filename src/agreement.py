"""Phase 2: inter-annotator agreement and adjudication.

Reads scripts_annotA.csv and scripts_annotB.csv, computes Cohen's kappa
and raw agreement on `script`, and emits scripts_final.csv plus
agreement_report.md.

Per DEVIATIONS.md D-001 annotator A is Claude Code and annotator B is a
human author, so the kappa reported here is HUMAN-vs-AI agreement and
must be described that way in the paper.

Adjudication policy (--policy), fixed by the authors before the final
stage was run:
  annotator_b  every A/B disagreement resolves to annotator B, the
               human author. Consequently scripts_final.script equals
               scripts_annotB.script for every unit, and the paper must
               describe adjudication as "human annotator authoritative",
               NOT as "resolved jointly case by case".
  manual       every disagreement must carry a hand-entered `final` and
               `reason` in disagreements.csv (the guide's default).

Units the human left blank are never auto-filled. They must be resolved
explicitly in data/annotations/blank_resolutions.csv
(unit_id, final, digit_glyph, reason), or the run aborts.

Usage
  python src/agreement.py --stage disagreements
  python src/agreement.py --stage final --policy annotator_b
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
GLYPHS = {"western", "eastern_arabic_indic", "na", "unclear"}


def load(p: Path) -> dict[int, dict]:
    if not p.exists():
        raise SystemExit(f"missing {p}. Annotator pass not complete.")
    return {int(r["unit_id"]): r
            for r in csv.DictReader(p.open(encoding="utf-8"))}


def load_blank_resolutions() -> dict[int, dict]:
    p = ANN / "blank_resolutions.csv"
    if not p.exists():
        return {}
    out = {}
    for r in csv.DictReader(p.open(encoding="utf-8")):
        uid = int(r["unit_id"])
        if r["final"] not in LABELS:
            raise SystemExit(f"blank_resolutions unit {uid}: final "
                             f"'{r['final']}' not in {LABELS}")
        if r.get("digit_glyph", "na") not in GLYPHS:
            raise SystemExit(f"blank_resolutions unit {uid}: bad digit_glyph")
        if not r.get("reason", "").strip():
            raise SystemExit(f"blank_resolutions unit {uid}: reason required")
        out[uid] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["disagreements", "final"],
                    required=True)
    ap.add_argument("--policy", choices=["annotator_b", "manual"],
                    default="manual")
    a = ap.parse_args()

    A = load(ANN / "scripts_annotA.csv")
    B = load(ANN / "scripts_annotB.csv")
    if set(A) != set(B):
        raise SystemExit(
            f"unit sets differ: A has {len(A)}, B has {len(B)}. "
            f"Both annotators must cover every unit.")

    blanks = load_blank_resolutions()
    unresolved = [i for i in sorted(B)
                  if not B[i]["script"].strip() and i not in blanks]
    if unresolved:
        raise SystemExit(
            f"{len(unresolved)} unit(s) have no annotator B script and no "
            f"entry in blank_resolutions.csv: {unresolved}\n"
            f"Protocol rule 8 requires a best-available reading for every "
            f"unit. Resolve them explicitly; they are never auto-filled.")

    # Effective B tag: the human's tag, or the human-signed blank resolution.
    def b_script(i: int) -> str:
        return B[i]["script"].strip() or blanks[i]["final"]

    def b_glyph(i: int) -> str:
        if B[i]["script"].strip():
            return B[i]["digit_glyph"]
        return blanks[i].get("digit_glyph", "na")

    ids = sorted(A)
    # Agreement is computed on the annotators' own tags. Units the human
    # left blank carry no B judgement, so they are excluded from kappa
    # and reported separately rather than counted as agreement.
    kappa_ids = [i for i in ids if B[i]["script"].strip()]
    a_s = [A[i]["script"] for i in kappa_ids]
    b_s = [B[i]["script"] for i in kappa_ids]
    kappa = cohen_kappa_score(a_s, b_s, labels=LABELS)
    n_agree = sum(x == y for x, y in zip(a_s, b_s))
    raw = n_agree / len(kappa_ids)
    disagree = [i for i in kappa_ids if A[i]["script"] != B[i]["script"]]

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
        print(f"raw agreement  : {raw:.4f}  ({n_agree}/{len(kappa_ids)})")
        print(f"disagreements  : {len(disagree)}")
        print(f"blank in B     : {len(ids) - len(kappa_ids)}")
        print(f"\nwrote {out}")
        return 0

    # ---- stage: final ----
    manual: dict[int, dict] = {}
    if a.policy == "manual" and disagree:
        dpath = ANN / "disagreements.csv"
        if not dpath.exists():
            raise SystemExit(f"{len(disagree)} disagreements but {dpath} "
                             f"is missing. Run --stage disagreements first.")
        for r in csv.DictReader(dpath.open(encoding="utf-8")):
            if not r["final"].strip():
                raise SystemExit(
                    f"unit {r['unit_id']} has no adjudicated `final` value.")
            if r["final"] not in LABELS:
                raise SystemExit(f"unit {r['unit_id']}: final "
                                 f"'{r['final']}' not in {LABELS}")
            if not r["reason"].strip():
                raise SystemExit(f"unit {r['unit_id']}: `reason` is required "
                                 f"(guide section 3: log every resolution)")
            manual[int(r["unit_id"])] = r

    final_rows = []
    for i in ids:
        if not B[i]["script"].strip():
            script, glyph = blanks[i]["final"], blanks[i].get("digit_glyph",
                                                              "na")
            src = "blank_resolved"
        elif A[i]["script"] == B[i]["script"]:
            script, glyph, src = A[i]["script"], b_glyph(i), "agreed"
        elif a.policy == "annotator_b":
            script, glyph, src = b_script(i), b_glyph(i), "adjudicated_human"
        else:
            script = manual[i]["final"]
            glyph = b_glyph(i) if script == B[i]["script"] \
                else A[i]["digit_glyph"]
            src = "adjudicated_manual"

        if script != "digit":
            glyph = "na"
        elif glyph in ("na", ""):
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

    # Guide section 3 requires every resolution to be logged per row.
    # Under the annotator_b policy the reason is the standing policy,
    # written back into disagreements.csv so the log is complete.
    if a.policy == "annotator_b" and disagree:
        reason = ("Adjudication policy fixed by the authors 2026-09-02: "
                  "annotator B (human) is authoritative on every A/B "
                  "disagreement; annotator A is Claude Code (D-001).")
        with (ANN / "disagreements.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["unit_id", "annotA_script", "annotA_note",
                        "annotB_script", "annotB_note", "final", "reason"])
            for i in disagree:
                w.writerow([i, A[i]["script"], A[i].get("note", ""),
                            B[i]["script"], B[i].get("note", ""),
                            B[i]["script"], reason])

    fpath = ANN / "scripts_final.csv"
    with fpath.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(final_rows[0].keys()))
        w.writeheader()
        w.writerows(final_rows)

    # Integrity: under the annotator_b policy the final tags must equal
    # the human's tags wherever the human gave one.
    if a.policy == "annotator_b":
        for r in final_rows:
            i = r["unit_id"]
            if B[i]["script"].strip():
                assert r["script"] == B[i]["script"], \
                    f"unit {i}: final != annotator B under annotator_b policy"

    cm = confusion_matrix(a_s, b_s, labels=LABELS)
    fc = Counter(r["script"] for r in final_rows)
    gc = Counter(r["digit_glyph"] for r in final_rows if r["script"] == "digit")
    sc = Counter(r["source"] for r in final_rows)
    amb = sum(int(r["ambiguous"]) for r in final_rows)
    n = len(final_rows)

    policy_text = {
        "annotator_b":
            "Every A/B disagreement was resolved in favour of **annotator "
            "B, the human author**, by a policy the authors fixed before "
            "this stage was run. `scripts_final.csv` therefore reproduces "
            "annotator B's tags exactly wherever annotator B gave one. The "
            "paper must describe adjudication as *human annotator "
            "authoritative*, not as case-by-case joint resolution.",
        "manual":
            "Every A/B disagreement was resolved by hand, with a recorded "
            "reason, in `disagreements.csv`.",
    }[a.policy]

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
              "paper.")
    md.append("")
    md.append("## Adjudication policy")
    md.append("")
    md.append(policy_text)
    md.append("")
    md.append("## Agreement on `script`")
    md.append("")
    md.append(f"- Units with a judgement from both annotators: "
              f"**{len(kappa_ids)}** of {len(ids)}")
    md.append(f"- Cohen's kappa: **{kappa:.4f}**")
    md.append(f"- Raw agreement: **{raw*100:.2f}%** ({n_agree}/{len(kappa_ids)})")
    md.append(f"- Disagreements: **{len(disagree)}**")
    md.append(f"- Units annotator B left blank (illegible), resolved "
              f"separately: **{len(ids) - len(kappa_ids)}**")
    md.append("")
    md.append("Units the human declined to read carry no B judgement and "
              "are excluded from kappa rather than scored as agreement.")
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
    md.append("### Provenance of each final tag")
    md.append("")
    md.append("| source | n |")
    md.append("|---|---|")
    for k in ["agreed", "adjudicated_human", "adjudicated_manual",
              "blank_resolved"]:
        if sc.get(k):
            md.append(f"| {k} | {sc[k]} |")
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
    md.append(f"- `disagreements.csv` - {len(disagree)} A/B disagreements")
    if blanks:
        md.append(f"- `blank_resolutions.csv` - {len(blanks)} units the "
                  f"human left blank, resolved explicitly")
    md.append("- `scripts_final.csv` - single source of truth for script "
              "tags")
    md.append("")

    (ANN / "agreement_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"policy         : {a.policy}")
    print(f"kappa          : {kappa:.4f}")
    print(f"raw agreement  : {raw:.4f} ({n_agree}/{len(kappa_ids)})")
    print(f"disagreements  : {len(disagree)}")
    print(f"final scripts  : {dict(fc)}")
    print(f"digit glyphs   : {dict(gc)}")
    print(f"sources        : {dict(sc)}")
    print(f"\nwrote {fpath}")
    print(f"wrote {ANN / 'agreement_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
