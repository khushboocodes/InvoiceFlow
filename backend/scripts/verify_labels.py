"""Verify the YOLO labels you produced in LabelImg.

Run this after annotation is done to catch:
* Missing label files
* Malformed YOLO lines
* Class IDs outside 0/1
* Boxes outside [0, 1] coords
* Empty class distribution sanity (need both signatures and stamps)
* Renders an overlay sample so you can spot-check 6 random images

Run::

    python -m scripts.verify_labels
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLASSES = {0: "signature", 1: "stamp"}
COLORS = {0: (0, 200, 0), 1: (255, 0, 0)}


def parse_yolo_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) != 5:
        return None
    try:
        cls = int(parts[0])
        cx, cy, bw, bh = (float(x) for x in parts[1:])
    except ValueError:
        return None
    return cls, cx, cy, bw, bh


def render_overlay(img_path: Path, label_path: Path, out_path: Path) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parsed = parse_yolo_line(line)
            if parsed is None:
                continue
            cls, cx, cy, bw, bh = parsed
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = COLORS.get(cls, (0, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                img,
                CLASSES.get(cls, f"cls_{cls}"),
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "train_data_idfc" / "yolo",
    )
    parser.add_argument("--preview-count", type=int, default=6)
    args = parser.parse_args()

    images_dir = args.target / "images"
    labels_dir = args.target / "labels"
    previews_dir = args.target / "previews_manual"

    if not images_dir.exists():
        print(f"FAIL: images directory missing: {images_dir}")
        return 1
    if not labels_dir.exists():
        print(f"FAIL: labels directory missing: {labels_dir}")
        return 1

    # Clean old previews if any.
    if previews_dir.exists():
        for f in previews_dir.glob("*.png"):
            f.unlink()

    images = sorted(images_dir.glob("*.png"))
    n_images = len(images)
    n_with_labels = 0
    n_signatures = 0
    n_stamps = 0
    n_empty = 0
    issues: list[str] = []

    for img_path in images:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            issues.append(f"No label file for {img_path.name}")
            continue
        n_with_labels += 1

        text = label_path.read_text().strip()
        if not text:
            n_empty += 1
            continue

        for line_no, line in enumerate(text.splitlines(), 1):
            parsed = parse_yolo_line(line)
            if parsed is None:
                issues.append(f"{img_path.name}:{line_no} malformed: {line!r}")
                continue
            cls, cx, cy, bw, bh = parsed
            if cls not in CLASSES:
                issues.append(f"{img_path.name}:{line_no} unknown class {cls}")
                continue
            for name, val in (("cx", cx), ("cy", cy), ("bw", bw), ("bh", bh)):
                if not (0.0 <= val <= 1.0):
                    issues.append(f"{img_path.name}:{line_no} {name}={val} outside [0,1]")
            if cls == 0:
                n_signatures += 1
            elif cls == 1:
                n_stamps += 1

    # Render some preview overlays so the user can spot-check.
    rng = random.Random(0)
    sample = rng.sample(images, min(args.preview_count, len(images)))
    for img_path in sample:
        label_path = labels_dir / (img_path.stem + ".txt")
        out = previews_dir / img_path.name
        render_overlay(img_path, label_path, out)

    # Write data.yaml for Ultralytics now that we know labels exist.
    data_yaml = args.target / "data.yaml"
    data_yaml.write_text(
        f"path: {args.target.resolve().as_posix()}\n"
        f"train: images\n"
        f"val: images\n"
        f"names:\n"
        f"  0: signature\n"
        f"  1: stamp\n"
    )

    # Report.
    print()
    print("=" * 60)
    print("LABEL VERIFICATION")
    print("=" * 60)
    print(f"Images in folder:           {n_images}")
    print(f"With label files:           {n_with_labels}")
    print(f"  ...with signatures:       {n_signatures}")
    print(f"  ...with stamps:           {n_stamps}")
    print(f"  ...empty (no objects):    {n_empty}")
    print(f"Issues found:               {len(issues)}")
    if issues:
        print()
        print("ISSUES:")
        for i in issues[:20]:
            print(f"  - {i}")
        if len(issues) > 20:
            print(f"  ...and {len(issues) - 20} more")

    print()
    print(f"Preview overlays written to: {previews_dir}")
    print()
    if not issues and n_signatures > 0 and n_stamps > 0:
        print("✓ Labels look valid. Ready to train YOLO.")
        return 0
    print("✗ Issues found or class distribution skewed. Fix before training.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
