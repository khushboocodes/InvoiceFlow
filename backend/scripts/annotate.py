"""Minimal, reliable image annotator for the signature/stamp YOLO dataset.

Built with Tkinter (Python stdlib) so there are no Qt / PyQt5 crash bugs.
Saves labels directly in YOLO format compatible with Ultralytics training.

Controls
--------
* Drag with mouse           : draw a box
* Press 1 (signature) / 2 (stamp) right after drawing : assign class
* Press D / Right Arrow     : next image (auto-saves)
* Press A / Left Arrow      : previous image
* Press U                   : undo last box on current image
* Press C                   : clear all boxes on current image
* Press Q / Escape          : quit (auto-saves)

The right sidebar shows current image index, count of each class on the
image, and total counts across the dataset.

Run::

    python -m scripts.annotate
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox

from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLASSES = ["signature", "stamp"]
CLASS_COLORS = ["#22c55e", "#3b82f6"]  # signature green, stamp blue


@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class ImageState:
    path: Path
    boxes: list[Box] = field(default_factory=list)
    width: int = 0
    height: int = 0


class Annotator:
    def __init__(self, images_dir: Path, labels_dir: Path):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        labels_dir.mkdir(parents=True, exist_ok=True)

        self.image_paths = sorted(images_dir.glob("*.png")) + sorted(
            images_dir.glob("*.jpg")
        ) + sorted(images_dir.glob("*.jpeg"))
        if not self.image_paths:
            raise SystemExit(f"No images found in {images_dir}")

        self.states: list[ImageState] = [ImageState(path=p) for p in self.image_paths]
        self.current_index = 0

        # Tk state — full-screen, dark themed.
        self.root = tk.Tk()
        self.root.title("InvoiceFlow Annotator — signatures + stamps")
        # Maximize the window so big invoice scans are visible without scrolling.
        try:
            self.root.state("zoomed")  # Windows / Linux
        except tk.TclError:
            self.root.attributes("-zoomed", True)  # macOS / others
        self.root.configure(bg="#0f172a")
        self.root.update_idletasks()

        self.image_obj: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.scale = 1.0
        self.user_zoom = 1.0  # multiplier on top of fit-to-viewport scale
        self.draw_start: tuple[int, int] | None = None
        self.preview_id: int | None = None
        self.box_canvas_ids: list[tuple[int, int]] = []  # (rect_id, label_id)
        self.pending_box: tuple[float, float, float, float] | None = None

        self._build_ui()
        self._load_existing_labels()
        # Defer first render until after the window has its real geometry.
        self.root.after(50, self._render_current)
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        title_font = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        body_font = tkfont.Font(family="Segoe UI", size=10)
        small_font = tkfont.Font(family="Segoe UI", size=9)

        # Top status bar.
        self.status_top = tk.Label(
            self.root,
            text="",
            bg="#1e293b",
            fg="#e2e8f0",
            font=title_font,
            anchor="w",
            padx=14,
            pady=8,
        )
        self.status_top.pack(side="top", fill="x")

        # Main split: canvas left, sidebar right.
        main = tk.Frame(self.root, bg="#0f172a")
        main.pack(fill="both", expand=True)

        canvas_frame = tk.Frame(main, bg="#0f172a")
        canvas_frame.pack(side="left", fill="both", expand=True)

        # Scrollable canvas — invoice scans can be much taller than the viewport.
        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#1f2937",
            highlightthickness=0,
            cursor="cross",
        )
        v_scroll = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        h_scroll = tk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True, padx=12, pady=12)

        # Mouse wheel scrolling. Shift+Wheel = horizontal.
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        self.canvas.bind(
            "<Shift-MouseWheel>",
            lambda e: self.canvas.xview_scroll(-1 * (e.delta // 120), "units"),
        )

        # Right sidebar.
        sidebar = tk.Frame(main, bg="#0f172a", width=260)
        sidebar.pack(side="right", fill="y", padx=(0, 12), pady=12)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="STATS", bg="#0f172a", fg="#94a3b8", font=small_font).pack(
            anchor="w", padx=8, pady=(8, 4)
        )
        self.stats_label = tk.Label(
            sidebar,
            text="",
            bg="#0f172a",
            fg="#e2e8f0",
            font=body_font,
            justify="left",
            anchor="nw",
        )
        self.stats_label.pack(anchor="w", padx=8, pady=4, fill="x")

        tk.Label(
            sidebar, text="CURRENT IMAGE", bg="#0f172a", fg="#94a3b8", font=small_font
        ).pack(anchor="w", padx=8, pady=(16, 4))
        self.image_stats_label = tk.Label(
            sidebar,
            text="",
            bg="#0f172a",
            fg="#e2e8f0",
            font=body_font,
            justify="left",
            anchor="nw",
        )
        self.image_stats_label.pack(anchor="w", padx=8, pady=4, fill="x")

        tk.Label(sidebar, text="CONTROLS", bg="#0f172a", fg="#94a3b8", font=small_font).pack(
            anchor="w", padx=8, pady=(16, 4)
        )
        controls = (
            "Drag mouse:  draw a box\n"
            "Press 1:     classify as 'signature'\n"
            "Press 2:     classify as 'stamp'\n"
            "Press D / →: next image\n"
            "Press A / ←: previous image\n"
            "Press U:     undo last box\n"
            "Press C:     clear all boxes\n"
            "+ / -:        zoom in / out\n"
            "0:            reset zoom\n"
            "Mouse wheel:  scroll vertically\n"
            "Shift+wheel:  scroll horizontally\n"
            "Press Q / Esc: quit (auto-saves)"
        )
        tk.Label(
            sidebar,
            text=controls,
            bg="#0f172a",
            fg="#cbd5e1",
            font=body_font,
            justify="left",
            anchor="nw",
        ).pack(anchor="w", padx=8, pady=4, fill="x")

        tk.Label(
            sidebar,
            text="LEGEND",
            bg="#0f172a",
            fg="#94a3b8",
            font=small_font,
        ).pack(anchor="w", padx=8, pady=(16, 4))
        legend_frame = tk.Frame(sidebar, bg="#0f172a")
        legend_frame.pack(anchor="w", padx=8, pady=4, fill="x")
        for i, (name, color) in enumerate(zip(CLASSES, CLASS_COLORS)):
            row = tk.Frame(legend_frame, bg="#0f172a")
            row.pack(anchor="w", pady=2)
            tk.Frame(row, bg=color, width=14, height=14).pack(side="left", padx=(0, 8))
            tk.Label(
                row, text=f"{i+1}: {name}", bg="#0f172a", fg="#e2e8f0", font=body_font
            ).pack(side="left")

    def _bind_keys(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        self.root.bind("<Key-1>", lambda e: self._assign_class(0))
        self.root.bind("<Key-2>", lambda e: self._assign_class(1))
        self.root.bind("<Key-d>", lambda e: self._next_image())
        self.root.bind("<Key-D>", lambda e: self._next_image())
        self.root.bind("<Right>", lambda e: self._next_image())
        self.root.bind("<Key-a>", lambda e: self._prev_image())
        self.root.bind("<Key-A>", lambda e: self._prev_image())
        self.root.bind("<Left>", lambda e: self._prev_image())
        self.root.bind("<Key-u>", lambda e: self._undo_last())
        self.root.bind("<Key-U>", lambda e: self._undo_last())
        self.root.bind("<Key-c>", lambda e: self._clear_all())
        self.root.bind("<Key-C>", lambda e: self._clear_all())
        self.root.bind("<Key-q>", lambda e: self._on_quit())
        self.root.bind("<Key-Q>", lambda e: self._on_quit())
        self.root.bind("<Escape>", lambda e: self._on_quit())

        # Zoom controls.
        self.root.bind("<Key-plus>", lambda e: self._zoom(1.25))
        self.root.bind("<Key-equal>", lambda e: self._zoom(1.25))  # `=` shares key with `+`
        self.root.bind("<Key-minus>", lambda e: self._zoom(0.8))
        self.root.bind("<Key-0>", lambda e: self._zoom_reset())

    # -------------------------------------------------------------- mouse
    def _canvas_xy(self, event) -> tuple[float, float]:
        """Translate widget coords to canvas coords (accounts for scroll)."""
        return self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

    def _on_mouse_down(self, event):
        x, y = self._canvas_xy(event)
        self.draw_start = (x, y)
        if self.preview_id is not None:
            self.canvas.delete(self.preview_id)
            self.preview_id = None

    def _on_mouse_drag(self, event):
        if self.draw_start is None:
            return
        x1, y1 = self.draw_start
        x, y = self._canvas_xy(event)
        if self.preview_id is not None:
            self.canvas.coords(self.preview_id, x1, y1, x, y)
        else:
            self.preview_id = self.canvas.create_rectangle(
                x1, y1, x, y, outline="#fde68a", width=2, dash=(4, 4)
            )

    def _on_mouse_up(self, event):
        if self.draw_start is None:
            return
        x1, y1 = self.draw_start
        x2, y2 = self._canvas_xy(event)
        self.draw_start = None

        # Normalize.
        cx1, cy1 = min(x1, x2), min(y1, y2)
        cx2, cy2 = max(x1, x2), max(y1, y2)

        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            # Too small — probably a misclick.
            if self.preview_id is not None:
                self.canvas.delete(self.preview_id)
                self.preview_id = None
            return

        # Convert canvas coords to image (original-resolution) coords.
        ix1, iy1 = cx1 / self.scale, cy1 / self.scale
        ix2, iy2 = cx2 / self.scale, cy2 / self.scale

        # Clamp to image bounds.
        state = self.states[self.current_index]
        ix1 = max(0, min(state.width, ix1))
        ix2 = max(0, min(state.width, ix2))
        iy1 = max(0, min(state.height, iy1))
        iy2 = max(0, min(state.height, iy2))

        self.pending_box = (ix1, iy1, ix2, iy2)
        self.status_top.configure(
            text=f"Box drawn — press 1 for 'signature' or 2 for 'stamp'."
        )

    # ------------------------------------------------------------- zoom
    def _zoom(self, factor: float) -> None:
        self.user_zoom = max(0.4, min(self.user_zoom * factor, 4.0))
        self._render_current()

    def _zoom_reset(self) -> None:
        self.user_zoom = 1.0
        self._render_current()


    def _assign_class(self, cls: int):
        if self.pending_box is None:
            return
        ix1, iy1, ix2, iy2 = self.pending_box
        state = self.states[self.current_index]
        state.boxes.append(Box(cls=cls, x1=ix1, y1=iy1, x2=ix2, y2=iy2))
        self.pending_box = None
        if self.preview_id is not None:
            self.canvas.delete(self.preview_id)
            self.preview_id = None
        self._save_current()
        self._draw_boxes()
        self._update_status()

    def _undo_last(self):
        state = self.states[self.current_index]
        if state.boxes:
            state.boxes.pop()
            self._save_current()
            self._draw_boxes()
            self._update_status()

    def _clear_all(self):
        state = self.states[self.current_index]
        if not state.boxes:
            return
        if messagebox.askyesno("Clear", f"Remove all {len(state.boxes)} boxes on this image?"):
            state.boxes.clear()
            self._save_current()
            self._draw_boxes()
            self._update_status()

    # ------------------------------------------------------- navigation
    def _next_image(self):
        if self.current_index < len(self.states) - 1:
            self._save_current()  # ensure even an empty label file exists for negatives
            self.current_index += 1
            self._render_current()

    def _prev_image(self):
        if self.current_index > 0:
            self._save_current()
            self.current_index -= 1
            self._render_current()

    def _on_quit(self):
        self._save_current()
        self.root.destroy()

    # --------------------------------------------------------- rendering
    def _render_current(self) -> None:
        state = self.states[self.current_index]
        try:
            img = Image.open(state.path)
            img.load()
            img = img.convert("RGB")
        except Exception as exc:
            messagebox.showerror("Image load failed", f"{state.path.name}: {exc}")
            return

        state.width, state.height = img.size

        # Scale to fit the canvas area at its current size, with optional user zoom.
        # Default: fit longest side to the canvas viewport.
        self.canvas.update_idletasks()
        viewport_w = max(self.canvas.winfo_width(), 600)
        viewport_h = max(self.canvas.winfo_height(), 600)
        fit_scale = min(viewport_w / img.width, viewport_h / img.height, 1.0)
        # Honor user zoom multiplier.
        scale = fit_scale * self.user_zoom
        if abs(scale - 1.0) > 0.01:
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size, Image.LANCZOS)
        self.scale = scale

        self.image_obj = img
        self.tk_image = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")
        # Configure scrollregion so scroll bars activate when the image
        # exceeds the viewport (zoom-in case).
        self.canvas.configure(scrollregion=(0, 0, img.width, img.height))
        self.box_canvas_ids = []
        self._draw_boxes()
        self._update_status()
        self.pending_box = None

    def _draw_boxes(self) -> None:
        # Wipe prior box overlays without redrawing the image.
        for rect_id, label_id in self.box_canvas_ids:
            self.canvas.delete(rect_id)
            self.canvas.delete(label_id)
        self.box_canvas_ids = []

        state = self.states[self.current_index]
        for box in state.boxes:
            x1 = box.x1 * self.scale
            y1 = box.y1 * self.scale
            x2 = box.x2 * self.scale
            y2 = box.y2 * self.scale
            color = CLASS_COLORS[box.cls]
            rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3)
            label = self.canvas.create_text(
                x1 + 4,
                max(y1 - 12, 4),
                text=CLASSES[box.cls].upper(),
                fill=color,
                anchor="nw",
                font=("Segoe UI", 9, "bold"),
            )
            self.box_canvas_ids.append((rect, label))

    # ------------------------------------------------------------ status
    def _update_status(self) -> None:
        total = len(self.states)
        idx = self.current_index + 1
        state = self.states[self.current_index]
        sig = sum(1 for b in state.boxes if b.cls == 0)
        stamp = sum(1 for b in state.boxes if b.cls == 1)
        self.status_top.configure(
            text=f"Image {idx}/{total}  ·  {state.path.name}  ·  signatures: {sig}  stamps: {stamp}"
        )

        # Aggregate stats.
        labeled = sum(1 for s in self.states if s.boxes)
        all_sig = sum(b.cls == 0 for s in self.states for b in s.boxes)
        all_stamp = sum(b.cls == 1 for s in self.states for b in s.boxes)
        with_neither = sum(
            1 for i, s in enumerate(self.states) if i <= self.current_index and not s.boxes
        )
        self.stats_label.configure(
            text=(
                f"Total images:  {total}\n"
                f"Visited:       {idx}\n"
                f"With labels:   {labeled}\n"
                f"Total signatures: {all_sig}\n"
                f"Total stamps:     {all_stamp}\n"
                f"Negatives so far: {with_neither}"
            )
        )

        self.image_stats_label.configure(
            text=(
                f"Filename: {state.path.name}\n"
                f"Size:     {state.width} × {state.height}\n"
                f"Boxes:    {len(state.boxes)}"
            )
        )

    # --------------------------------------------------------- persistence
    def _label_path_for(self, image_path: Path) -> Path:
        return self.labels_dir / (image_path.stem + ".txt")

    def _save_current(self) -> None:
        state = self.states[self.current_index]
        # Determine image dimensions if not yet loaded (first save in a session).
        if state.width == 0:
            with Image.open(state.path) as img:
                state.width, state.height = img.size

        path = self._label_path_for(state.path)
        if not state.boxes:
            # Empty file = valid negative example.
            path.write_text("")
            return

        lines = []
        for b in state.boxes:
            cx = (b.x1 + b.x2) / 2 / state.width
            cy = (b.y1 + b.y2) / 2 / state.height
            bw = (b.x2 - b.x1) / state.width
            bh = (b.y2 - b.y1) / state.height
            lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        path.write_text("\n".join(lines) + "\n")

    def _load_existing_labels(self) -> None:
        for state in self.states:
            label_path = self._label_path_for(state.path)
            if not label_path.exists():
                continue
            text = label_path.read_text().strip()
            if not text:
                continue
            # We need image dims to convert YOLO -> pixel space.
            if state.width == 0:
                with Image.open(state.path) as img:
                    state.width, state.height = img.size
            for line in text.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, bw, bh = (float(x) for x in parts[1:])
                except ValueError:
                    continue
                if cls not in (0, 1):
                    continue
                x1 = (cx - bw / 2) * state.width
                y1 = (cy - bh / 2) * state.height
                x2 = (cx + bw / 2) * state.width
                y2 = (cy + bh / 2) * state.height
                state.boxes.append(Box(cls=cls, x1=x1, y1=y1, x2=x2, y2=y2))

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "train_data_idfc" / "yolo",
    )
    args = parser.parse_args()

    images_dir = args.target / "images"
    labels_dir = args.target / "labels"

    if not images_dir.exists():
        print(f"FAIL: images directory missing: {images_dir}", file=sys.stderr)
        print("Run scripts/prepare_label_set.py first.", file=sys.stderr)
        return 1

    print(f"Loading images from {images_dir}")
    print(f"Saving labels to   {labels_dir}")
    Annotator(images_dir, labels_dir).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
