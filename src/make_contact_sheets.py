"""Phase 2 helper: render annotation contact sheets.

Lays out the 638 distinct images (data/annotation_units.csv) onto
labelled grids so annotator A can tag them image-only, in unit_id
order, with no filename or class-folder cue visible.

Each cell shows ONLY the unit_id and the image. Class folder is never
rendered, so the annotator cannot see the diagnosis label.

Run: python src/make_contact_sheets.py [--cols 4] [--rows 4] [--cell 300]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "data" / "annotations" / "contact_sheets"

PAD, LABEL_H, MARGIN = 8, 26, 14
BG, FG, CELL_BG = (255, 255, 255), (0, 0, 0), (245, 245, 245)


def load_font(size: int):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit(im: Image.Image, box: int) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    s = box / max(w, h)
    return im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cell", type=int, default=300)
    ap.add_argument("--only", type=int, default=None,
                    help="render only this sheet number")
    ap.add_argument("--shuffle-seed", type=int, default=20260901,
                    help="fixed seed for presentation order (blinding)")
    a = ap.parse_args()

    units = list(csv.DictReader((DATA / "annotation_units.csv")
                                .open(encoding="utf-8")))
    # Presentation order is shuffled with a fixed, recorded seed so that
    # the class folder (which correlates with unit_id order, since
    # "No/..." sorts before "Yes/...") is not visible as block structure
    # to the annotator. unit_id remains the join key.
    units.sort(key=lambda u: int(u["unit_id"]))
    random.Random(a.shuffle_seed).shuffle(units)
    per = a.cols * a.rows
    OUT.mkdir(parents=True, exist_ok=True)
    font, title_font = load_font(17), load_font(20)

    cw = a.cell + 2 * PAD
    ch = a.cell + 2 * PAD + LABEL_H
    sheet_w = MARGIN * 2 + a.cols * cw
    sheet_h = MARGIN * 2 + 34 + a.rows * ch

    n_sheets = (len(units) + per - 1) // per

    # archive the presentation order so the pass is reproducible
    (OUT / "presentation_order.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "presentation_order.json").write_text(json.dumps({
        "shuffle_seed": a.shuffle_seed,
        "cols": a.cols, "rows": a.rows, "cell_px": a.cell,
        "n_units": len(units), "n_sheets": n_sheets,
        "order": [int(u["unit_id"]) for u in units],
        "sheet_of_unit": {u["unit_id"]: i // per + 1
                          for i, u in enumerate(units)},
    }, indent=2), encoding="utf-8")

    written = []
    for s in range(n_sheets):
        if a.only is not None and s + 1 != a.only:
            continue
        chunk = units[s * per:(s + 1) * per]
        sheet = Image.new("RGB", (sheet_w, sheet_h), BG)
        d = ImageDraw.Draw(sheet)
        d.text((MARGIN, MARGIN),
               f"contact sheet {s+1}/{n_sheets}   "
               f"(presentation order, seed {a.shuffle_seed})",
               fill=FG, font=title_font)
        for i, u in enumerate(chunk):
            r, c = divmod(i, a.cols)
            x = MARGIN + c * cw
            y = MARGIN + 34 + r * ch
            d.rectangle([x, y, x + cw - 4, y + ch - 4], fill=CELL_BG,
                        outline=(200, 200, 200))
            d.text((x + PAD, y + 4), f"#{u['unit_id']}", fill=FG, font=font)
            src = DATA / "raw" / u["representative_uid"]
            with Image.open(src) as im:
                thumb = fit(im, a.cell)
            ox = x + PAD + (a.cell - thumb.width) // 2
            oy = y + PAD + LABEL_H + (a.cell - thumb.height) // 2
            sheet.paste(thumb, (ox, oy))
        p = OUT / f"sheet_{s+1:03d}.png"
        sheet.save(p, optimize=True)
        written.append(p)

    print(f"units={len(units)} per_sheet={per} sheets={n_sheets}")
    for p in written:
        print(f"  wrote {p.relative_to(REPO)}  ({p.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
