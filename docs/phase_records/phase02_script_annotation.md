# Phase Record — Phase 2: Script annotation (dual annotator)

- **Date:** 2026-09-01
- **Phase:** Phase 2 (Guide §3)
- **Status:** **PASS** (completed 2026-09-02). Annotator A, annotator B,
  adjudication, and `scripts_final.csv` all complete.

## Actions performed

1. Froze the annotation protocol in `data/annotations/protocol.md`
   BEFORE any tag was recorded, and archived it verbatim per D-001.
2. Set the **annotation unit = the distinct image (sha256)**, not the
   file. 638 units instead of 852 files; duplicate files inherit their
   unit's tag, so byte-identical files cannot get inconsistent tags.
3. Implemented `src/make_contact_sheets.py`; rendered 40 blind contact
   sheets (4x4, 300 px cells) into
   `data/annotations/contact_sheets/`. Cells show `unit_id` only — no
   filename, no class folder, no label.
4. **Blinding fix:** the first render laid units out in `unit_id`
   order, which correlates with the class folder (`No/` sorts before
   `Yes/` in uid order) and so exposed the diagnosis label as block
   structure. Sheets were discarded and re-rendered in an order
   shuffled with fixed seed **20260901**, archived in
   `contact_sheets/presentation_order.json`.
5. Completed the annotator A pass over all 638 units (Claude Code,
   image-only). Tags accumulated in
   `data/annotations/annotA_partial/` (8 files, 5 sheets each) and
   merged to `data/annotations/scripts_annotA.csv`.
6. Implemented `src/validate_annotations.py` (protocol schema gate) and
   ran it against annotator A: **VALID**.
7. Implemented `src/annotate.py`, the blind, resumable, single-image
   GUI for annotator B. Same 638 units, same seeded presentation order.
8. Implemented `src/agreement.py` (kappa, confusion matrix,
   disagreement worklist, `scripts_final.csv`, `agreement_report.md`).
   Verified it refuses to run without annotator B, and refuses to emit
   `scripts_final.csv` unless every disagreement carries a human
   `final` value *and* a `reason`.
9. Installed and pinned the environment; verified the three MobileNet
   tap points named in guide §5.1.

## Decisions made

- Annotation unit is the distinct image, not the file (follows from
  D-004; recorded in `protocol.md`).
- Presentation order shuffled with a recorded seed to blind the class
  folder. This is stricter than the guide requires, not a deviation.
- `ambiguous=1` requires a free-text note; enforced by the validator.
- Annotator A committed to a single most-likely `script` for every
  unit; no unit was left blank, per protocol rule 5.

## Configuration used

- Python 3.13.7; TensorFlow 2.20.0, Keras 3.15.1, numpy 2.2.6,
  pandas 2.3.2, scikit-learn 1.7.2, scipy 1.16.2, statsmodels 0.15.0,
  matplotlib 3.11.1, pillow 11.3.0, pytest 9.1.1. Pinned in
  `requirements.txt`. CPU only.
- Contact sheets: 4 cols x 4 rows, 300 px cells, shuffle seed 20260901.

## Files created / updated

| Path | Description |
|---|---|
| `data/annotations/protocol.md` | frozen annotation protocol |
| `src/make_contact_sheets.py` | blind contact-sheet renderer |
| `data/annotations/contact_sheets/sheet_001..040.png` | 40 sheets, 638 units |
| `data/annotations/contact_sheets/presentation_order.json` | seed + order |
| `data/annotations/annotA_partial/*.csv` | 8 incremental tag files |
| `data/annotations/scripts_annotA.csv` | annotator A, 638 rows, VALID |
| `src/validate_annotations.py` | protocol schema gate |
| `src/annotate.py` | GUI for annotator B |
| `src/agreement.py` | kappa + adjudication + report generator |
| `requirements.txt` | pinned environment |
| `README.md` | repo overview, access-record TODOs |

## Validation results

| Check | Result |
|---|---|
| all 638 units annotated by A | **PASS** (638/638, no gaps) |
| no duplicate `unit_id` | **PASS** |
| `script` in {urdu, english, digit} | **PASS** |
| `digit_glyph` valid and mandatory when script=digit | **PASS** |
| `digit_glyph` = na when script != digit | **PASS** |
| `ambiguous=1` carries a note | **PASS** |
| every unit_id resolves to `annotation_units.csv` | **PASS** |
| MobileNet tap layers present in TF 2.20 | **PASS** (all 3) |
| agreement.py blocks without annotator B | **PASS** (by design) |

### Annotator A composition (PROVISIONAL — not the paper's numbers)

| script | n | % |
|---|---|---|
| urdu | 396 | 62.1% |
| digit | 154 | 24.1% |
| english | 88 | 13.8% |
| **total** | **638** | 100% |

Digit glyphs: **western 154, eastern_arabic_indic 0, unclear 0.**
Ambiguous flags: **117 (18.3%)**.

These are one annotator's tags. They are NOT the values for the
paper's Data section, which requires the adjudicated
`scripts_final.csv`.

## Issues encountered

1. **Annotator B is a human task and has not started.** Phase 2 cannot
   complete, and Phase 3 (split, which stratifies on script) cannot
   start, until `scripts_annotB.csv` exists and adjudication is run.
2. **Blinding defect caught and fixed** before any tag was recorded
   (see Actions 4). No annotation was produced under the unblinded
   ordering.
3. **High ambiguity rate (18.3%).** Concentrated in genuine
   cross-script homoglyphs: bare vertical strokes (Urdu alif vs Latin
   l/I vs digit 1), closed circles (digit 0 vs Latin O vs Urdu he),
   loop-with-tail forms (Urdu waw vs digit 9), and the Urdu toe vs
   Latin b pair. This is a property of isolated-character images, not
   an annotation failure, and it is reported as a count per protocol.
4. **`ain` vs Eastern Arabic-Indic `4`.** Urdu ain and the eastern
   numeral 4 are near-homoglyphs. Annotator A read every digit in the
   corpus as a Western numeral and found no eastern-form numerals at
   all, so ain-like glyphs were tagged `urdu`. **Annotator B should
   scrutinise this specific pair**, because it decides whether the
   paper's digit-glyph TODO is answered "all Western".
5. **Stale pip lock.** The first (background) TensorFlow install stalled
   holding `libclang.dll`; the process was stopped and the install
   redone in the foreground. No effect on any artifact.

## Deviations

- D-001 annotator A is Claude Code (ACCEPTED) — governs this phase.
- No new deviations. The seeded presentation shuffle and the
  distinct-image annotation unit are additions that strengthen the
  protocol; both are documented in `protocol.md`.

## Completion (2026-09-02)

Annotator B (human) completed all 638 units. Adjudication policy fixed
by the authors: **annotator B is authoritative on every A/B
disagreement**, so `scripts_final.csv` reproduces the human's tags
wherever the human gave one. The paper must describe adjudication as
*human annotator authoritative*, not as case-by-case joint resolution.

| Quantity | Value |
|---|---|
| Units judged by both annotators | 635 of 638 |
| **Cohen's kappa (human vs AI)** | **0.8489** |
| Raw agreement | 92.13% (585/635) |
| Disagreements, resolved to annotator B | 50 |
| Units B left blank, resolved by the human author | 3 (units 315, 374, 456) |

Final composition: **urdu 420 (65.8%), digit 136 (21.3%),
english 82 (12.9%)**.
Digit glyphs: **western 135, eastern_arabic_indic 0, unclear 1**.
Ambiguous flags: 117 (18.3%).

The three blanks were units annotator B marked "not clear image". The
human author read them directly at full resolution and recorded
315 -> english, 374 -> digit/western, 456 -> urdu, all retaining
`ambiguous=1`; logged in `blank_resolutions.csv` with reasons. All three
readings differ from annotator A's, which is why they were not
auto-filled from A.

**Status: PASS.**
