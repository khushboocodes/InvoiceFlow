"""Split the labeled images + labels into train / val / test directories.

Ultralytics expects a directory layout like::

    train_data_idfc/yolo/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml

This script reorganizes the flat ``images/`` and ``labels/`` directories into
that layout deterministically (seeded shuffle so re-runs are stable). Default
split is 70 / 15 / 15.

Run::

    python -m scripts.prepare_yolo_dataset
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "train_data_idfc" / "yolo",
    )
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target = args.target
    flat_images = target / "images"
    flat_labels = target / "labels"

    if not flat_images.exists() or not flat_labels.exists():
        print(f"FAIL: missing {flat_images} or {flat_labels}", file=sys.stderr)
        return 1

    # Detect whether the layout is already split. If so, undo it back to flat
    # before re-splitting so re-runs are clean.
    nested_train = flat_images / "train"
    if nested_train.is_dir():
        print("Detected nested split; flattening before re-split...")
        for split in ("train", "val", "test"):
            for kind_dir in (flat_images / split, flat_labels / split):
                if not kind_dir.is_dir():
                    continue
                for f in kind_dir.iterdir():
                    f.replace(kind_dir.parent / f.name)
                kind_dir.rmdir()

    # Collect all (image, label) pairs.
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(flat_images.glob("*.png")):
        lbl = flat_labels / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))

    if not pairs:
        print("FAIL: no (image, label) pairs found", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(round(n * args.train_frac))
    n_val = int(round(n * args.val_frac))
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    # Move into split sub-directories.
    for split, pair_list in splits.items():
        img_dir = flat_images / split
        lbl_dir = flat_labels / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, lbl in pair_list:
            img.replace(img_dir / img.name)
            lbl.replace(lbl_dir / lbl.name)

    # Write Ultralytics data.yaml.
    data_yaml = target / "data.yaml"
    data_yaml.write_text(
        f"path: {target.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"names:\n"
        f"  0: signature\n"
        f"  1: stamp\n"
    )

    # Summary.
    print()
    print("=" * 60)
    print("YOLO DATASET PREPARED")
    print("=" * 60)
    for split, pair_list in splits.items():
        print(f"  {split:5}  {len(pair_list)} images")
    print(f"\nTotal: {n} images")
    print(f"data.yaml: {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
