# Script annotation protocol (Phase 2)

Archived verbatim per `DEVIATIONS.md` D-001. Both annotators work from
this document. Frozen 2026-09-01, before any annotation was recorded.

## Unit of annotation

The **distinct image** (`sha256`), not the file. `data/annotation_units.csv`
lists 638 units; each carries a `unit_id` and a `representative_uid`.
Duplicate files inherit their unit's tag (`src/propagate_tags.py`), so
byte-identical files can never receive inconsistent tags.

## What is recorded

One row per unit, schema per guide §3:

```
unit_id, script, digit_glyph, ambiguous, note
```

| Field | Domain | Rule |
|---|---|---|
| `script` | `urdu` / `english` / `digit` | mandatory |
| `digit_glyph` | `western` / `eastern_arabic_indic` / `na` / `unclear` | mandatory when `script=digit`; otherwise `na` |
| `ambiguous` | `0` / `1` | `1` when the glyph form is ambiguous **between scripts** |
| `note` | free text | mandatory when `ambiguous=1`; describe the ambiguity |

## Decision rules

1. **Annotate from the image alone.** Do not consult the filename, the
   class folder, the diagnosis label, or the other annotator's file.
2. `script=digit` applies to any numeral, in either glyph family. The
   glyph family goes in `digit_glyph`, not in `script`.
   - `western` = 0 1 2 3 4 5 6 7 8 9
   - `eastern_arabic_indic` = ٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩ (Urdu forms ۰۱۲۳۴۵۶۷۸۹)
3. `script=urdu` = a letter of the Urdu alphabet, including
   dot-bearing (nuqta) forms and Nastaliq-style curves and loops.
4. `script=english` = a letter of the Latin alphabet, upper or lower
   case.
5. **Cross-script homoglyphs.** Some forms are genuinely shared, e.g.
   Urdu `ا` (alif) vs Latin `l`/`I`/`1`; Urdu `و` (waw) vs Latin `9`;
   Urdu `ہ` vs Latin `o`; Urdu `ر` vs a comma. Where the form alone
   does not decide, set `ambiguous=1`, record the competing readings in
   `note`, and still commit to the single most likely `script`. Never
   leave `script` blank.
6. **Do not use stroke quality as evidence.** Malformed, reversed, or
   shaky letters are exactly what the dataset is about; a poorly formed
   glyph is still tagged by its intended script.
7. If an image contains more than one character, tag the dominant
   (largest/centred) one and record the fact in `note`.
8. If no character is legible at all, set `ambiguous=1`, `note` it, and
   give the best available reading.

## Blinding

Annotator A works from contact sheets rendered by
`src/make_contact_sheets.py`: 4x4 grids, 300 px cells, labelled with
`unit_id` only. Class folder and filename are never rendered.
Presentation order is shuffled with fixed seed **20260901** and archived
in `contact_sheets/presentation_order.json`, so that the class blocks
(`No/` sorts before `Yes/` in uid order) are not visible as structure.

Annotator B works from `src/annotate.py`, which displays one image at a
time and likewise renders no filename, folder, or label.

## Independence

The two annotators do not see each other's files until both are
complete. `scripts_annotA.csv` is committed before annotator B starts,
so independence is verifiable from git history.

## Agreement and adjudication

- Cohen's kappa on `script` (`sklearn.metrics.cohen_kappa_score`) plus
  raw agreement percentage, computed by `src/agreement.py`.
- Because annotator A is Claude Code (D-001), this is a **human-vs-AI**
  agreement statistic and must be described as such in the paper.
- Every disagreement is resolved by the **human author**, and each
  resolution is logged with: `unit_id`, A's tag, B's tag, final tag,
  reason.
- Output: `scripts_final.csv` (the single source of truth) and
  `agreement_report.md`.

## Reported quantities (these go into the paper's Data section)

1. Cohen's kappa on `script`, and raw agreement %.
2. Final counts per script: urdu / english / digit.
3. Digit-glyph composition: western vs eastern_arabic_indic vs unclear.
4. Count of `ambiguous=1` flags, and how they were resolved.
