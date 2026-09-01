"""Phase 2: annotation tool for annotator B (human).

Displays one image at a time and records one row per annotation unit.
Blind by construction: the filename, the class folder, and the
diagnosis label are never rendered. Presentation order is the same
fixed shuffle used for annotator A's contact sheets (seed 20260901),
read from contact_sheets/presentation_order.json.

Resumable: progress is written after every keystroke, so the pass can
be stopped and restarted at any point.

Keys
  u / e / d   tag script = urdu / english / digit
  w / i / c   digit glyph = western / eastern_arabic_indic / unclear
              (asked only after choosing 'd')
  a           toggle the ambiguous flag for the current unit
  n           type a note for the current unit
  LEFT        go back one unit (to revise)
  RIGHT       skip forward without tagging
  q           save and quit

Run: python src/annotate.py [--out data/annotations/scripts_annotB.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import tkinter as tk
from pathlib import Path
from tkinter import simpledialog

from PIL import Image, ImageTk

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
ORDER = DATA / "annotations" / "contact_sheets" / "presentation_order.json"
COLS = ["unit_id", "script", "digit_glyph", "ambiguous", "note"]
SCRIPT_KEYS = {"u": "urdu", "e": "english", "d": "digit"}
GLYPH_KEYS = {"w": "western", "i": "eastern_arabic_indic", "c": "unclear"}
CANVAS = 600


class Annotator:
    def __init__(self, out: Path) -> None:
        self.out = out
        units = {int(u["unit_id"]): u for u in
                 csv.DictReader((DATA / "annotation_units.csv")
                                .open(encoding="utf-8"))}
        order = json.loads(ORDER.read_text(encoding="utf-8"))["order"]
        self.units = [units[i] for i in order]
        self.rows: dict[int, dict] = {}
        if out.exists():
            for r in csv.DictReader(out.open(encoding="utf-8")):
                self.rows[int(r["unit_id"])] = r
        self.i = next((k for k, u in enumerate(self.units)
                       if int(u["unit_id"]) not in self.rows), 0)
        self.pending_digit = False

        self.root = tk.Tk()
        self.root.title("Script annotation - annotator B")
        self.status = tk.Label(self.root, font=("Segoe UI", 13), pady=6)
        self.status.pack()
        self.canvas = tk.Label(self.root)
        self.canvas.pack(padx=10)
        self.help = tk.Label(
            self.root, font=("Segoe UI", 10), fg="#444", pady=8,
            text="u urdu   e english   d digit  ->  w western / "
                 "i eastern-arabic-indic / c unclear\n"
                 "a toggle ambiguous    n note    LEFT back    "
                 "RIGHT skip    q save and quit")
        self.help.pack()
        self.root.bind("<Key>", self.on_key)
        self.render()

    # ---------- persistence ----------
    def save(self) -> None:
        with self.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            for uid in sorted(self.rows, key=int):
                w.writerow({c: self.rows[uid].get(c, "") for c in COLS})

    def row(self, uid: int) -> dict:
        return self.rows.setdefault(
            uid, {"unit_id": uid, "script": "", "digit_glyph": "na",
                  "ambiguous": "0", "note": ""})

    # ---------- ui ----------
    def render(self) -> None:
        if self.i >= len(self.units):
            self.status.config(
                text=f"DONE - {len(self.rows)}/{len(self.units)} tagged. "
                     f"Press q to save and quit.")
            return
        u = self.units[self.i]
        uid = int(u["unit_id"])
        img = Image.open(DATA / "raw" / u["representative_uid"]).convert("RGB")
        s = CANVAS / max(img.size)
        img = img.resize((max(1, int(img.width * s)),
                          max(1, int(img.height * s))), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(image=self.photo)
        r = self.rows.get(uid)
        tag = (f"{r['script']}/{r['digit_glyph']}"
               f"{' AMBIG' if r.get('ambiguous') == '1' else ''}"
               if r and r.get("script") else "untagged")
        done = len(self.rows)
        prompt = "  <- choose glyph: w / i / c" if self.pending_digit else ""
        self.status.config(
            text=f"unit {uid}   [{self.i+1}/{len(self.units)}]   "
                 f"tagged {done}/{len(self.units)}   |  {tag}{prompt}")

    def on_key(self, ev) -> None:
        k = ev.keysym
        ch = (ev.char or "").lower()
        if k == "Left":
            self.i = max(0, self.i - 1)
            self.pending_digit = False
        elif k == "Right":
            self.i = min(len(self.units), self.i + 1)
            self.pending_digit = False
        elif ch == "q":
            self.save()
            self.root.destroy()
            return
        elif self.i < len(self.units):
            uid = int(self.units[self.i]["unit_id"])
            if self.pending_digit and ch in GLYPH_KEYS:
                self.row(uid)["digit_glyph"] = GLYPH_KEYS[ch]
                self.pending_digit = False
                self.save()
                self.i += 1
            elif ch in SCRIPT_KEYS:
                r = self.row(uid)
                r["script"] = SCRIPT_KEYS[ch]
                if ch == "d":
                    self.pending_digit = True      # must pick a glyph
                else:
                    r["digit_glyph"] = "na"
                    self.save()
                    self.i += 1
            elif ch == "a":
                r = self.row(uid)
                r["ambiguous"] = "0" if r.get("ambiguous") == "1" else "1"
                self.save()
            elif ch == "n":
                r = self.row(uid)
                note = simpledialog.askstring(
                    "Note", f"unit {uid}", initialvalue=r.get("note", ""),
                    parent=self.root)
                if note is not None:
                    r["note"] = note.replace("\n", " ").strip()
                    self.save()
        self.render()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=DATA / "annotations" / "scripts_annotB.csv")
    a = ap.parse_args()
    app = Annotator(a.out.resolve())
    app.root.mainloop()
    app.save()
    print(f"saved {len(app.rows)} rows to {a.out}")
    print("validate with: python src/validate_annotations.py", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
