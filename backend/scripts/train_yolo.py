"""Fine-tune YOLOv8n for signature/stamp detection.

This script:

1. Detects the available device (CUDA via PyTorch, else CPU).
2. Loads COCO-pretrained YOLOv8n base weights (downloaded once into
   ``models/base/yolov8n.pt`` by Ultralytics on first invocation).
3. Trains for a configurable number of epochs (default 75) at 640x640 with
   built-in Ultralytics augmentations.
4. Validates and reports mAP@50 and mAP@[50:95] on the held-out test split.
5. Copies the best epoch's weights to ``models/yolov8n_sig_stamp.pt``.

Run::

    python -m scripts.train_yolo

Validates Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.device import detect

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("train_yolo")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "train_data_idfc" / "yolo" / "data.yaml",
    )
    parser.add_argument(
        "--base-weights",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models" / "base" / "yolov8n.pt",
    )
    parser.add_argument(
        "--output-weights",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models" / "yolov8n_sig_stamp.pt",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models" / "training_runs",
    )
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience")
    parser.add_argument("--map50-floor", type=float, default=0.85)
    args = parser.parse_args()

    if not args.data_yaml.exists():
        logger.error("data.yaml not found at %s — run prepare_yolo_dataset.py first", args.data_yaml)
        return 1

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.base_weights.parent.mkdir(parents=True, exist_ok=True)

    device = detect()
    logger.info("Training on device: %s (%s)", device.kind.value, device.description)
    if device.kind.value == "cpu":
        logger.warning("CPU training will be very slow — expect ~6+ hours. GPU is strongly recommended.")

    # Import lazily so the module doesn't pay the ultralytics import cost
    # when the script is just being inspected.
    from ultralytics import YOLO  # type: ignore[import-not-found]

    # Resolve base weights. If not bundled yet, Ultralytics will fetch them
    # one time from GitHub on first run.
    if args.base_weights.exists():
        logger.info("Using bundled base weights: %s", args.base_weights)
        model = YOLO(str(args.base_weights))
    else:
        logger.info("Bundled base weights not found at %s; downloading via Ultralytics", args.base_weights)
        model = YOLO("yolov8n.pt")
        # After a fresh download, copy the cached weights into our bundle path
        # so subsequent runs are fully offline.
        try:
            cached = Path(model.ckpt_path) if hasattr(model, "ckpt_path") else None
            if cached and cached.exists():
                shutil.copy(cached, args.base_weights)
                logger.info("Cached base weights copied to %s for offline reuse", args.base_weights)
        except Exception as exc:
            logger.warning("Could not cache base weights: %s", exc)

    logger.info("Starting training: epochs=%d imgsz=%d batch=%d", args.epochs, args.imgsz, args.batch)

    train_kwargs = dict(
        data=str(args.data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device.torch_device_string(),
        project=str(args.runs_dir),
        name="sig_stamp",
        exist_ok=True,
        # Augmentations chosen to mimic real-world invoice noise.
        degrees=5.0,
        translate=0.05,
        scale=0.10,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.0,  # NEVER flip — invoices are not symmetric
        mosaic=0.5,
        mixup=0.0,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        # Class weighting — both equally important.
        single_cls=False,
        verbose=True,
        patience=args.patience,
    )

    results = model.train(**train_kwargs)
    logger.info("Training finished")

    # Locate the best-epoch weights and copy them into the canonical bundle path.
    run_dir = args.runs_dir / "sig_stamp"
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        logger.error("Could not locate best.pt at %s", best)
        return 2

    args.output_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.output_weights)
    logger.info("Copied best weights to %s", args.output_weights)

    # Run validation on the test split.
    logger.info("Running test-split validation...")
    test_model = YOLO(str(args.output_weights))
    val_results = test_model.val(
        data=str(args.data_yaml),
        split="test",
        device=device.torch_device_string(),
        imgsz=args.imgsz,
        verbose=False,
    )

    # Different Ultralytics versions expose mAP via different attribute names.
    map50 = float(getattr(val_results.box, "map50", 0.0))
    map5095 = float(getattr(val_results.box, "map", 0.0))

    print()
    print("=" * 60)
    print("YOLO TRAINING SUMMARY")
    print("=" * 60)
    print(f"Device:           {device.description}")
    print(f"Epochs:           {args.epochs}")
    print(f"Image size:       {args.imgsz}")
    print(f"Held-out mAP@50:  {map50:.3f}  (floor {args.map50_floor})")
    print(f"Held-out mAP@50-95: {map5095:.3f}")
    print(f"Run directory:    {run_dir}")
    print(f"Final weights:    {args.output_weights}")

    if map50 < args.map50_floor:
        print()
        print(f"FAIL: mAP@50 {map50:.3f} below floor {args.map50_floor}")
        print("Consider: more annotations, more epochs, or hyperparameter tweaks.")
        return 3

    print("\nOK: weights ready for inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
