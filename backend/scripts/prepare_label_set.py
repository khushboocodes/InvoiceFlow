"""Prepare a clean folder of images for manual LabelImg annotation.

Picks a stratified sample of images across the four filename groups in the
training set (numeric IDs, OTHERS / vN, Android photos, named docs) so the
manual annotation set isn't all one kind of document.

Outputs::

    train_data_idfc/yolo/
    ├── images/        (65 source images, copied)
    ├── classes.txt    (signature, stamp — for LabelImg)
    └── data.yaml      (Ultralytics config, written after labeling)

Run::

    python -m scripts.prepare_label_set --sample-size 65
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


CLASS_NAMES = ["signature", "stamp"]


def stratified_sample(images_dir: Path, n: int, seed: int = 42) -> list[Path]:
    """Pick ``n`` images deterministically across filename groups."""
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
    total_available = sum(len(v) for v in groups.values())
    for files in groups.values():
        if not files:
            continue
        share = max(1, int(round(n * len(files) / total_available)))
        rng.shuffle(files)
        chosen.extend(files[:share])

    rng.shuffle(chosen)
    return chosen[:n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "train_data_idfc" / "train",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "train_data_idfc" / "yolo",
    )
    parser.add_argument("--sample-size", type=int, default=65)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_out = args.target / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    sample = stratified_sample(args.source, args.sample_size, args.seed)
    print(f"Copying {len(sample)} stratified images to {images_out}")

    for p in sample:
        dest = images_out / p.name
        if not dest.exists():
            shutil.copy(p, dest)

    # classes.txt sets LabelImg's class list. Order matters — index 0 is
    # signature, index 1 is stamp, matching the YOLO label file format.
    (args.target / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n")

    print()
    print(f"  Source:        {args.source}")
    print(f"  Sample size:   {len(sample)}")
    print(f"  Image folder:  {images_out}")
    print(f"  classes.txt:   {args.target / 'classes.txt'}")
    print()
    print("Next: run scripts/launch_labelimg.ps1 to start labeling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
