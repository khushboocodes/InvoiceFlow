"""Tkinter helper to label the validation set.

Shows each image with a side panel for entering the four text fields and
two presence flags. Saves to ``tests/validation/labels.json`` after each
image and on exit.

Run::

    python -m scripts.annotate_validation
    python -m scripts.annotate_validation --sample-size 30   # default
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


MAX_DISPLAY_WIDTH = 900
MAX_DISPLAY_HEIGHT = 1100


class ValidationLabeler:
    def __init__(self, image_paths: list[Path], labels_path: Path):
        self.image_paths = image_paths
        self.labels_path = labels_path
        self.documents: dict[str, dict] = {}

        self._load_existing()
        self.current_index = self._next_unlabeled_index()

        self.root = tk.Tk()
        self.root.title("InvoiceFlow Validation Labeler")
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass
        self.root.configure(bg="#0f172a")

        self.tk_image: ImageTk.PhotoImage | None = None
        self.scale = 1.0

        self._build_ui()
        self._render_current()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

    # ----------------------------------------------------------------- UI
    def _build_ui(self):
        body_font = tkfont.Font(family="Segoe UI", size=10)
        small_font = tkfont.Font(family="Segoe UI", size=9)

        self.status = tk.Label(
            self.root, text="", bg="#1e293b", fg="#e2e8f0",
            font=tkfont.Font(family="Segoe UI", size=11, weight="bold"),
            anchor="w", padx=14, pady=8,
        )
        self.status.pack(side="top", fill="x")

        main = tk.Frame(self.root, bg="#0f172a")
        main.pack(fill="both", expand=True)

        # Left: image canvas with scrollbars.
        canvas_frame = tk.Frame(main, bg="#0f172a")
        canvas_frame.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg="#1f2937", highlightthickness=0)
        v_scroll = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        h_scroll = tk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True, padx=12, pady=12)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        # Right: form.
        form = tk.Frame(main, bg="#0f172a", width=380)
        form.pack(side="right", fill="y", padx=(0, 12), pady=12)
        form.pack_propagate(False)

        tk.Label(form, text="GROUND TRUTH", bg="#0f172a", fg="#94a3b8", font=small_font).pack(
            anchor="w", padx=8, pady=(8, 4)
        )

        self.entries: dict[str, tk.Entry] = {}
        for label_text, key in (
            ("Dealer Name", "dealer_name"),
            ("Model Name", "model_name"),
            ("Horse Power (int)", "horse_power"),
            ("Asset Cost (int, no commas)", "asset_cost"),
        ):
            tk.Label(form, text=label_text, bg="#0f172a", fg="#cbd5e1", font=body_font).pack(
                anchor="w", padx=8, pady=(8, 2)
            )
            entry = tk.Entry(form, bg="#1e293b", fg="#f1f5f9", insertbackground="#f1f5f9", font=body_font)
            entry.pack(anchor="w", padx=8, fill="x")
            self.entries[key] = entry

        self.signature_present = tk.BooleanVar(value=False)
        self.stamp_present = tk.BooleanVar(value=False)
        tk.Checkbutton(
            form, text="Signature present", variable=self.signature_present,
            bg="#0f172a", fg="#cbd5e1", selectcolor="#1e293b",
            activebackground="#0f172a", activeforeground="#f1f5f9", font=body_font,
        ).pack(anchor="w", padx=8, pady=(12, 2))
        tk.Checkbutton(
            form, text="Stamp present", variable=self.stamp_present,
            bg="#0f172a", fg="#cbd5e1", selectcolor="#1e293b",
            activebackground="#0f172a", activeforeground="#f1f5f9", font=body_font,
        ).pack(anchor="w", padx=8)

        # Buttons.
        btn_frame = tk.Frame(form, bg="#0f172a")
        btn_frame.pack(fill="x", pady=20, padx=8)
        tk.Button(btn_frame, text="◀ Prev", command=self._prev, bg="#334155", fg="#f1f5f9", font=body_font).pack(side="left", padx=(0, 4))
        tk.Button(btn_frame, text="Save & Next ▶", command=self._save_and_next, bg="#6c5ce7", fg="#ffffff", font=body_font).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Skip", command=self._skip, bg="#334155", fg="#f1f5f9", font=body_font).pack(side="left", padx=4)

        tk.Label(form, text="CONTROLS", bg="#0f172a", fg="#94a3b8", font=small_font).pack(
            anchor="w", padx=8, pady=(16, 4)
        )
        controls = (
            "Tab: cycle entry fields\n"
            "Enter: save & next\n"
            "← / →: prev / next\n"
            "Esc: save & quit\n"
            "Empty value = null"
        )
        tk.Label(form, text=controls, bg="#0f172a", fg="#cbd5e1", font=body_font, justify="left").pack(
            anchor="w", padx=8
        )

    def _bind_keys(self):
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._save_and_next())
        self.root.bind("<Return>", lambda e: self._save_and_next())
        self.root.bind("<Escape>", lambda e: self._on_quit())

    # ----------------------------------------------------------- rendering
    def _render_current(self):
        idx = self.current_index
        path = self.image_paths[idx]

        self.status.configure(
            text=f"[{idx + 1}/{len(self.image_paths)}]  {path.name}  —  labeled: {len(self.documents)}"
        )

        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Image load failed", f"{path.name}: {exc}")
            return

        # Scale to fit.
        scale = min(MAX_DISPLAY_WIDTH / img.width, MAX_DISPLAY_HEIGHT / img.height, 1.0)
        if scale < 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        self.scale = scale
        self.tk_image = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, img.width, img.height))

        # Pre-fill the form if we already have a label for this image.
        doc_id = path.stem
        existing = self.documents.get(doc_id)
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        if existing:
            fields = existing.get("fields", {})
            for key, entry in self.entries.items():
                value = fields.get(key)
                if value is not None:
                    entry.insert(0, str(value))
            self.signature_present.set(bool(fields.get("signature_present", False)))
            self.stamp_present.set(bool(fields.get("stamp_present", False)))
        else:
            self.signature_present.set(False)
            self.stamp_present.set(False)

        # Focus the first empty entry.
        for entry in self.entries.values():
            if not entry.get():
                entry.focus_set()
                break

    # ------------------------------------------------------------- actions
    def _save_and_next(self):
        self._save_current()
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self._render_current()
        else:
            messagebox.showinfo("Done", "Reached end of image set. Saved all labels.")

    def _prev(self):
        self._save_current()
        if self.current_index > 0:
            self.current_index -= 1
            self._render_current()

    def _skip(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self._render_current()

    def _on_quit(self):
        self._save_current()
        self.root.destroy()

    def _save_current(self):
        path = self.image_paths[self.current_index]
        doc_id = path.stem

        def _coerce(s: str, *, integer: bool = False):
            s = s.strip()
            if not s:
                return None
            if integer:
                try:
                    return int(s.replace(",", "").replace("₹", "").replace("Rs.", "").strip())
                except ValueError:
                    return None
            return s

        record = {
            "doc_id": doc_id,
            "fields": {
                "dealer_name": _coerce(self.entries["dealer_name"].get()),
                "model_name": _coerce(self.entries["model_name"].get()),
                "horse_power": _coerce(self.entries["horse_power"].get(), integer=True),
                "asset_cost": _coerce(self.entries["asset_cost"].get(), integer=True),
                "signature_present": bool(self.signature_present.get()),
                "stamp_present": bool(self.stamp_present.get()),
            },
        }

        # Only save if at least one field has been provided.
        has_data = any(
            record["fields"][k] is not None
            for k in ("dealer_name", "model_name", "horse_power", "asset_cost")
        ) or record["fields"]["signature_present"] or record["fields"]["stamp_present"]
        if has_data:
            self.documents[doc_id] = record
            self._write_labels()

    def _write_labels(self):
        payload = {
            "_comment": "Hand-labeled ground truth for the validation set.",
            "documents": [self.documents[k] for k in sorted(self.documents.keys())],
        }
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        self.labels_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_existing(self):
        if not self.labels_path.exists():
            return
        try:
            payload = json.loads(self.labels_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        for doc in payload.get("documents", []):
            self.documents[doc["doc_id"]] = doc

    def _next_unlabeled_index(self) -> int:
        for i, p in enumerate(self.image_paths):
            if p.stem not in self.documents:
                return i
        return 0

    def run(self):
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "train_data_idfc" / "train",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "validation" / "labels.json",
    )
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=137)  # different from YOLO seed
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"FAIL: source dir not found: {args.source}", file=sys.stderr)
        return 1

    all_pngs = sorted(args.source.glob("*.png"))
    if not all_pngs:
        print(f"FAIL: no images in {args.source}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    sample = rng.sample(all_pngs, min(args.sample_size, len(all_pngs)))
    sample.sort()
    print(f"Labeling {len(sample)} images from {args.source}")
    print(f"Saving labels to {args.labels}")

    ValidationLabeler(sample, args.labels).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
