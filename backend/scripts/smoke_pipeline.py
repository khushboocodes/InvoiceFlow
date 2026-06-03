"""End-to-end smoke test: run the full pipeline against a few real invoices.

Intentionally not under tests/ because:

* It loads every model — torch, paddleocr, ultralytics YOLO, transformers
  Qwen — which together take 30-60 seconds at process startup.
* It runs against the actual ``train_data_idfc`` images, not synthetic ones.
* It exists for human verification, not as a CI gate.

Run::

    python -m scripts.smoke_pipeline
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executable import build_pipeline


def main() -> int:
    train_dir = Path(__file__).resolve().parents[2] / "train_data_idfc" / "train"
    if not train_dir.exists():
        print(f"FAIL: train dir not found: {train_dir}", file=sys.stderr)
        return 1

    samples = sorted(train_dir.glob("*.png"))[:5]
    if not samples:
        print(f"FAIL: no samples in {train_dir}", file=sys.stderr)
        return 1

    print(f"Building pipeline...")
    t0 = time.monotonic()
    pipeline = build_pipeline()
    init_secs = time.monotonic() - t0
    print(f"Pipeline ready in {init_secs:.1f}s\n")

    print(f"Processing {len(samples)} samples\n" + "=" * 60)
    for path in samples:
        print(f"\n>>> {path.name}")
        t0 = time.monotonic()
        result = pipeline.process_one(path)
        elapsed = time.monotonic() - t0
        timings = pipeline.last_timings

        # Pretty fields summary.
        print(f"  total: {elapsed:.2f}s  (ocr {timings.ocr:.2f}, vision {timings.detection:.2f}, "
              f"tier1 {timings.extraction_tier1:.2f}, tier2 {timings.extraction_tier2:.2f})")
        if pipeline.last_slm_invoked:
            print(f"  SLM invoked")
        print(f"  dealer:    {result.fields.dealer_name.value!r:60} (conf {result.fields.dealer_name.confidence:.2f})")
        print(f"  model:     {result.fields.model_name.value!r:60} (conf {result.fields.model_name.confidence:.2f})")
        print(f"  horse_pow: {result.fields.horse_power.value!r:60} (conf {result.fields.horse_power.confidence:.2f})")
        print(f"  cost:      {result.fields.asset_cost.value!r:60} (conf {result.fields.asset_cost.confidence:.2f})")
        print(f"  signature: present={result.fields.signature.present}  bbox={result.fields.signature.bbox}")
        print(f"  stamp:     present={result.fields.stamp.present}  bbox={result.fields.stamp.bbox}")
        print(f"  doc conf:  {result.confidence:.3f}")
        if result.error:
            print(f"  ERROR:     {result.error}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
