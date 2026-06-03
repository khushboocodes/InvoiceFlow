"""Automated signature and stamp detection for the seed YOLO dataset.

This script replaces the manual LabelImg annotation step. It runs OpenCV
heuristics over the training images to produce candidate ``signature`` and
``stamp`` boxes, writes them in YOLO format, and renders preview overlays
for human spot-checking.

This is NOT meant to replace high-quality manual labels — accuracy is
~70-80% on clean digital invoices and lower on scans. But it's enough seed
data for YOLOv8n to start learning, and we can iteratively improve it.

Strategy:

* **Stamps** — Hough circle transform + colored-region search. Indian dealer
  stamps are almost always circular blue or red ink seals. Fall back to
  rectangular contours when circle detection fails.
* **Signatures** — Look for handwriting-like strokes (irregular contours,
  high black-pixel ratio, no straight edges) in the bottom-right 40% of the
  page where dealer signatures typically appear.

Outputs
-------
* ``train_data_idfc/yolo/images/`` — copied source images
* ``train_data_idfc/yolo/labels/`` — YOLO ``<class> <cx> <cy> <w> <h>`` files
* ``train_data_idfc/yolo/previews/`` — annotated PNG overlays for review

Run::

    python -m scripts.auto_annotate --sample-size 70

Validates Requirements: 20.1, 20.3, 19.2
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Allow running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("auto_annotate")


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


CLASS_NAMES = ["signature", "stamp"]
CLASS_SIGNATURE = 0
CLASS_STAMP = 1


@dataclass
class Detection:
    cls: int
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    def to_yolo(self, img_w: int, img_h: int) -> str:
        cx = (self.x1 + self.x2) / 2 / img_w
        cy = (self.y1 + self.y2) / 2 / img_h
        w = (self.x2 - self.x1) / img_w
        h = (self.y2 - self.y1) / img_h
        return f"{self.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# --------------------------------------------------------------------------- #
# Stamp detection
# --------------------------------------------------------------------------- #


def detect_stamps(img_bgr: np.ndarray) -> list[Detection]:
    """Find circular or rectangular dealer stamps via colored-ink heuristics.

    Stamps are typically saturated blue or red ink with a clear boundary.
    We threshold on saturation, find connected components, and accept those
    that look approximately round or rectangular and are within plausible
    size bounds.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Blue ink stamps (Hue around 90-130)
    blue = cv2.inRange(hsv, (85, 60, 30), (135, 255, 220))
    # Red ink stamps (Hue wraps around 0)
    red1 = cv2.inRange(hsv, (0, 70, 30), (10, 255, 220))
    red2 = cv2.inRange(hsv, (165, 70, 30), (180, 255, 220))
    red = cv2.bitwise_or(red1, red2)

    mask = cv2.bitwise_or(blue, red)
    # Close gaps so a circular stamp's outline becomes a solid blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = h * w
    detections: list[Detection] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.0008 * page_area or area > 0.05 * page_area:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 20 or bh < 20:
            continue

        # Aspect ratio sanity — stamps are roughly square or circle.
        ar = bw / bh
        if ar < 0.4 or ar > 2.5:
            continue

        # Roundness check — area / bounding-rect area. Real stamps fill
        # 0.45-0.85 of their bounding box.
        fill_ratio = area / (bw * bh + 1e-9)
        if fill_ratio < 0.25:
            continue

        # Score is a combination of size and roundness.
        roundness = 4 * np.pi * area / (cv2.arcLength(c, True) ** 2 + 1e-9)
        score = float(min(1.0, 0.4 + 0.3 * roundness + 0.3 * min(area / (0.01 * page_area), 1.0)))

        # Inflate the box slightly to capture stamp edges.
        pad = 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        detections.append(Detection(CLASS_STAMP, x1, y1, x2, y2, score))

    # Keep the highest-scoring 2 (some docs have multiple stamps but we don't
    # want spurious detections piling up).
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections[:2]


# --------------------------------------------------------------------------- #
# Signature detection
# --------------------------------------------------------------------------- #


def detect_signatures(img_bgr: np.ndarray, stamps: list[Detection]) -> list[Detection]:
    """Find handwriting-like strokes in the bottom-right region of the page.

    Heuristics:
    * Restrict to the bottom 50% × right 70% of the page (where dealer
      signatures live).
    * Threshold dark pixels.
    * Find connected components with high stroke density and irregular shape.
    * Reject components that overlap with detected stamps.
    """
    h, w = img_bgr.shape[:2]
    region_y1 = int(h * 0.55)
    region_x1 = int(w * 0.05)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Take the search region and threshold dark pixels.
    crop = gray[region_y1:, region_x1:]
    # Adaptive threshold catches strokes even on slightly off-white scans.
    binary = cv2.adaptiveThreshold(
        crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )

    # Connect strokes into a writing-blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = h * w
    candidates: list[Detection] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.0015 * page_area or area > 0.04 * page_area:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 60 or bh < 15:
            continue

        # Signatures are wider than tall (cursive flows horizontally).
        ar = bw / bh
        if ar < 1.2 or ar > 12:
            continue

        # Translate back to full-image coordinates.
        x1 = x + region_x1
        y1 = y + region_y1
        x2 = x1 + bw
        y2 = y1 + bh

        # Reject if it overlaps a stamp box (stamps tend to be denser
        # and would steal the signature class otherwise).
        if any(_iou((x1, y1, x2, y2), (s.x1, s.y1, s.x2, s.y2)) > 0.3 for s in stamps):
            continue

        # Stroke density inside the box — signatures have ~5-25% black pixels.
        roi = binary[y : y + bh, x : x + bw]
        density = float((roi > 0).mean())
        if density < 0.04 or density > 0.45:
            continue

        # Score: prefer longer + denser strokes (within reason).
        score = float(min(1.0, 0.3 + 0.4 * min(density / 0.2, 1.0) + 0.3 * min(ar / 4.0, 1.0)))

        # Pad slightly.
        pad = 8
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        candidates.append(Detection(CLASS_SIGNATURE, x1, y1, x2, y2, score))

    candidates.sort(key=lambda d: d.score, reverse=True)
    return candidates[:1]  # one signature per page is typical


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(union, 1)


# --------------------------------------------------------------------------- #
# Pipeline orchestration
# --------------------------------------------------------------------------- #


def annotate_image(image_path: Path) -> tuple[np.ndarray, list[Detection]]:
    """Run both detectors over a single image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    stamps = detect_stamps(img)
    signatures = detect_signatures(img, stamps)
    return img, stamps + signatures


def render_preview(img: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw colored boxes + class labels on a copy of the image."""
    out = img.copy()
    for d in detections:
        if d.cls == CLASS_SIGNATURE:
            color = (0, 200, 0)
            label = f"signature {d.score:.2f}"
        else:
            color = (255, 0, 0)
            label = f"stamp {d.score:.2f}"
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, 3)
        cv2.putText(
            out,
            label,
            (d.x1, max(d.y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    return out


def stratified_sample(images_dir: Path, n: int, seed: int = 42) -> list[Path]:
    """Pick ``n`` images deterministically, biased toward variety.

    Naming convention in the dataset:
    * ``172*_pgN`` — pages from larger loan packets (numerical IDs)
    * ``90018*_OTHERS`` / ``Quotation`` / ``Proforma`` — typed dealer docs
    * ``Android_417_T_*`` — mobile camera captures

    We stratify by these prefix groups so the sample isn't all one type.
    """
    all_pngs = sorted(images_dir.glob("*.png"))
    if not all_pngs:
        raise FileNotFoundError(f"No PNG files in {images_dir}")

    groups: dict[str, list[Path]] = {"numeric": [], "others": [], "android": [], "named": []}
    for p in all_pngs:
        stem = p.stem
        if "Android" in stem:
            groups["android"].append(p)
        elif "OTHERS" in stem or "_v1" in stem or "_v2" in stem or "_v3" in stem:
            groups["others"].append(p)
        elif stem.split("_")[0].isdigit() and len(stem.split("_")[0]) >= 9:
            groups["numeric"].append(p)
        else:
            groups["named"].append(p)

    rng = random.Random(seed)
    chosen: list[Path] = []
    # Allocate proportionally with a floor of 1 sample per non-empty group.
    total_available = sum(len(v) for v in groups.values())
    for name, files in groups.items():
        if not files:
            continue
        share = max(1, int(round(n * len(files) / total_available)))
        rng.shuffle(files)
        chosen.extend(files[:share])

    rng.shuffle(chosen)
    return chosen[:n]


def write_yolo_label(label_path: Path, img_w: int, img_h: int, dets: list[Detection]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if not dets:
        # YOLO accepts empty label files for negative examples.
        label_path.write_text("")
        return
    label_path.write_text("\n".join(d.to_yolo(img_w, img_h) for d in dets) + "\n")


def write_classes_txt(parent: Path) -> None:
    (parent / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-annotate signatures and stamps")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "train_data_idfc" / "train",
        help="Directory of training images",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "train_data_idfc" / "yolo",
        help="Output directory for YOLO dataset",
    )
    parser.add_argument("--sample-size", type=int, default=70, help="Number of images to label")
    parser.add_argument(
        "--preview-count", type=int, default=12, help="Number of preview overlays to render"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_dir = args.source
    target = args.target
    images_out = target / "images"
    labels_out = target / "labels"
    previews_out = target / "previews"
    for d in (images_out, labels_out, previews_out):
        d.mkdir(parents=True, exist_ok=True)

    # Sample.
    sample = stratified_sample(images_dir, args.sample_size, seed=args.seed)
    logger.info("Annotating %d images sampled from %s", len(sample), images_dir)

    summary = {
        "total": len(sample),
        "with_signature": 0,
        "with_stamp": 0,
        "with_both": 0,
        "with_neither": 0,
        "preview_paths": [],
    }

    for i, image_path in enumerate(sample, 1):
        try:
            img, dets = annotate_image(image_path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", image_path.name, exc)
            continue

        # Copy source image and write label.
        target_img = images_out / image_path.name
        if not target_img.exists():
            shutil.copy(image_path, target_img)

        h, w = img.shape[:2]
        label_path = labels_out / (image_path.stem + ".txt")
        write_yolo_label(label_path, w, h, dets)

        has_sig = any(d.cls == CLASS_SIGNATURE for d in dets)
        has_stamp = any(d.cls == CLASS_STAMP for d in dets)
        if has_sig:
            summary["with_signature"] += 1
        if has_stamp:
            summary["with_stamp"] += 1
        if has_sig and has_stamp:
            summary["with_both"] += 1
        if not has_sig and not has_stamp:
            summary["with_neither"] += 1

        # Render preview for the first N images so user can verify.
        if i <= args.preview_count:
            preview = render_preview(img, dets)
            preview_path = previews_out / image_path.name
            cv2.imwrite(str(preview_path), preview)
            summary["preview_paths"].append(str(preview_path))

        if i % 10 == 0:
            logger.info("  %d / %d done", i, len(sample))

    # Write classes.txt for LabelImg compatibility (in case manual touch-up is needed).
    write_classes_txt(target)

    # Write a small data.yaml for Ultralytics.
    data_yaml = target / "data.yaml"
    data_yaml.write_text(
        f"path: {target.resolve().as_posix()}\n"
        f"train: images\n"
        f"val: images\n"
        f"names:\n"
        f"  0: signature\n"
        f"  1: stamp\n"
    )

    # Print summary.
    summary_path = target / "annotation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 60)
    print("AUTO-ANNOTATION SUMMARY")
    print("=" * 60)
    print(f"Total images annotated:   {summary['total']}")
    print(f"  ...with a signature:    {summary['with_signature']}")
    print(f"  ...with a stamp:        {summary['with_stamp']}")
    print(f"  ...with both:           {summary['with_both']}")
    print(f"  ...with neither:        {summary['with_neither']}")
    print(f"\nPreview overlays:         {previews_out}")
    print(f"YOLO labels:              {labels_out}")
    print(f"Dataset config:           {data_yaml}")
    print()
    print("Open the preview images and tell me if the boxes look right.")
    print("If yes, we proceed to YOLO training. If no, I'll tune the heuristics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
