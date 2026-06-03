"""Bootstrap weak labels from the unlabeled training set.

Runs the full pipeline on every image in ``train_data_idfc/train/`` and
writes a ``data/pseudo_labels.json`` file with the predictions. Documents
where Tier-1 and Tier-2 agree with high confidence are flagged as
``high_confidence`` — these are candidates for future fine-tuning rounds.

Run::

    python -m scripts.pseudo_label
    python -m scripts.pseudo_label --limit 50

Validates Requirement 20.4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("pseudo_label")


HIGH_CONFIDENCE_THRESHOLD = 0.85


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "train_data_idfc" / "train",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "pseudo_labels.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.source.is_dir():
        logger.error("Source dir not found: %s", args.source)
        return 1

    images = sorted(args.source.glob("*.png"))
    if args.limit is not None:
        images = images[:args.limit]
    if not images:
        logger.error("No images to label")
        return 1

    from executable import build_pipeline

    logger.info("Building pipeline...")
    pipeline = build_pipeline()

    pseudo_labels: list[dict] = []
    high_conf_count = 0

    for i, img_path in enumerate(images, 1):
        t0 = time.monotonic()
        try:
            result = pipeline.process_one(img_path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", img_path.name, exc)
            continue
        elapsed = time.monotonic() - t0

        per_field = result.fields
        is_high_conf = (
            per_field.dealer_name.confidence >= HIGH_CONFIDENCE_THRESHOLD
            and per_field.model_name.confidence >= HIGH_CONFIDENCE_THRESHOLD
            and per_field.horse_power.confidence >= HIGH_CONFIDENCE_THRESHOLD
            and per_field.asset_cost.confidence >= HIGH_CONFIDENCE_THRESHOLD
        )
        if is_high_conf:
            high_conf_count += 1

        pseudo_labels.append({
            "doc_id": result.doc_id,
            "fields": {
                "dealer_name": per_field.dealer_name.value,
                "model_name": per_field.model_name.value,
                "horse_power": per_field.horse_power.value,
                "asset_cost": per_field.asset_cost.value,
                "signature_present": per_field.signature.present,
                "stamp_present": per_field.stamp.present,
            },
            "confidences": {
                "dealer_name": round(per_field.dealer_name.confidence, 3),
                "model_name": round(per_field.model_name.confidence, 3),
                "horse_power": round(per_field.horse_power.confidence, 3),
                "asset_cost": round(per_field.asset_cost.confidence, 3),
                "signature": round(per_field.signature.confidence, 3),
                "stamp": round(per_field.stamp.confidence, 3),
                "document": round(result.confidence, 3),
            },
            "high_confidence": is_high_conf,
            "elapsed_sec": round(elapsed, 2),
        })

        if i % 10 == 0:
            print(f"  [{i}/{len(images)}] high_conf so far: {high_conf_count}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"documents": pseudo_labels}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("PSEUDO-LABEL SUMMARY")
    print("=" * 60)
    print(f"Total documents: {len(pseudo_labels)}")
    print(f"High-confidence: {high_conf_count} ({high_conf_count/len(pseudo_labels)*100:.1f}%)")
    print(f"Output:          {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
