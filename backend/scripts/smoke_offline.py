"""Offline-mode smoke test.

Runs the pipeline on a single sample image with ``--offline`` engaged.
Asserts no ``OfflineViolation`` is raised — meaning every model + tokenizer
+ master file loaded from local disk without any outbound HTTP call.

This is the smaller, native-Python equivalent of running inside a
``--network=none`` Docker container. Run it before packaging the
submission.

Run::

    python -m scripts.smoke_offline
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("smoke_offline")


def main() -> int:
    from utils import offline_guard

    # Engage the kill-switch BEFORE we import any heavy modules.
    offline_guard.enable_offline_mode()

    # Now import the pipeline.
    from executable import build_pipeline
    from utils.offline_guard import OfflineViolation

    train_dir = Path(__file__).resolve().parents[2] / "train_data_idfc" / "train"
    samples = sorted(train_dir.glob("*.png"))
    if not samples:
        logger.error("No samples in %s", train_dir)
        return 1

    sample = samples[0]
    logger.info("Building pipeline in offline mode")
    try:
        pipeline = build_pipeline()
    except OfflineViolation as exc:
        logger.error("OFFLINE VIOLATION during pipeline init: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Pipeline init crashed")
        return 1

    logger.info("Processing %s in offline mode", sample.name)
    try:
        result = pipeline.process_one(sample)
    except OfflineViolation as exc:
        logger.error("OFFLINE VIOLATION during inference: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Pipeline raised on inference")
        return 1

    if result.error:
        logger.error("Pipeline returned error: %s", result.error)
        return 1

    logger.info("OK: offline mode succeeded")
    logger.info("  doc_id: %s", result.doc_id)
    logger.info("  confidence: %.3f", result.confidence)
    logger.info("  processing_time: %.2fs", result.processing_time_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
