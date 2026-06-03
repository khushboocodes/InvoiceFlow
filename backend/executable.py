"""InvoiceFlow CLI — single entry point for IDFC submission.

Usage::

    # Single document → JSON to stdout, plus sample_output/result.json
    python executable.py path/to/invoice.png

    # Specify output path
    python executable.py path/to/invoice.png --output result.json

    # Batch mode: process every supported file in a directory
    python executable.py --batch path/to/directory

    # Engage offline kill-switch (refuses any non-loopback network call)
    python executable.py path/to/invoice.png --offline

    # PS-reference flat output shape (collapses TextField wrappers)
    python executable.py path/to/invoice.png --legacy

Exit codes
----------
* 0  — success
* 1  — schema validation or output assembly failure
* 2  — unsupported input format / nonexistent path
* 3  — offline-mode network violation

Validates Requirements: 11.1-11.7, 16.5, 22.1-22.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

# CRITICAL Windows DLL ordering: import torch BEFORE paddle. The utils
# package handles this in __init__.py, but executing this file directly
# bypasses package init, so we force-import here too.
try:
    import torch  # noqa: F401  type: ignore[import-not-found]
except ImportError:
    pass

# Now import our modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import offline_guard
from utils.detection import VisionDetector
from utils.device import detect as detect_device
from utils.ingestion import SUPPORTED_EXTENSIONS
from utils.masters import load as load_masters
from utils.ocr import OcrEngine
from utils.pipeline import Pipeline
from utils.schema import ExtractionResult, to_legacy_dict
from utils.slm import try_load_slm

logger = logging.getLogger("executable")


BACKEND_ROOT = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_ROOT / "models"
DATA_DIR = BACKEND_ROOT / "data"
DEFAULT_OUTPUT = BACKEND_ROOT / "sample_output" / "result.json"

# Bundled model paths (must match the layout in scripts/download_models.py
# and scripts/train_yolo.py).
YOLO_WEIGHTS = MODELS_DIR / "yolov8n_sig_stamp.pt"
DETECTION_CONFIG = MODELS_DIR / "detection.yaml"
QWEN_DIR = MODELS_DIR / "qwen2.5-1.5b-instruct"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="InvoiceFlow: offline document AI for invoice field extraction.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Path to a PDF / PNG / JPG / JPEG file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: sample_output/result.json for single-doc mode).",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        default=None,
        help="Process every supported file in this directory; emits one JSON per line on stdout.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Engage the network kill-switch — any outbound call raises OfflineViolation.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Emit the flat PS-reference JSON shape (no per-field confidence wrappers).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level logs to stderr.",
    )
    return parser.parse_args(argv)


def configure_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=level,
        stream=sys.stderr,
        force=True,
    )


def build_pipeline() -> Pipeline:
    """Instantiate every model and master once. Reused across all docs."""
    print("[pipeline] detecting device", flush=True)
    device = detect_device()
    logger.info("Device: %s (%s)", device.kind.value, device.description)
    print(f"[pipeline] device = {device.kind.value}", flush=True)

    masters = load_masters(DATA_DIR)
    logger.info(
        "Loaded masters: dealer=%d entries, asset=%d entries",
        len(masters.dealer),
        len(masters.asset),
    )
    print(f"[pipeline] masters loaded ({len(masters.dealer)} dealers, {len(masters.asset)} assets)", flush=True)

    print("[pipeline] loading OCR engine...", flush=True)
    ocr = OcrEngine(device, MODELS_DIR / "paddleocr")
    print("[pipeline] OCR ready", flush=True)

    print("[pipeline] loading YOLO detector...", flush=True)
    detector = VisionDetector(device, YOLO_WEIGHTS, DETECTION_CONFIG)
    print("[pipeline] YOLO ready", flush=True)

    # SLM is optional — degrades gracefully when missing.
    print("[pipeline] loading SLM (Qwen 1.5B safetensors)...", flush=True)
    slm = try_load_slm(device, QWEN_DIR)
    if slm is None:
        logger.info("SLM disabled — Tier-2 fallback unavailable")
        print("[pipeline] SLM unavailable — Tier-1 only", flush=True)
    else:
        logger.info("SLM ready (Qwen2.5-1.5B-Instruct)")
        print("[pipeline] SLM ready", flush=True)

    print("[pipeline] all components loaded", flush=True)
    return Pipeline(device=device, ocr=ocr, detector=detector, slm=slm, masters=masters)


def iter_supported_files(directory: Path) -> Iterable[Path]:
    """Yield supported files in a directory, sorted for deterministic output."""
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield p


def emit_json(result: ExtractionResult, *, legacy: bool, output_path: Optional[Path], stdout: bool) -> None:
    """Print to stdout and optionally write to a file."""
    if legacy:
        payload = to_legacy_dict(result)
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        text = result.model_dump_json(indent=2)

    if stdout:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.quiet)

    # Engage kill-switch BEFORE any model is loaded so transitive imports
    # can't sneak in a sneaky download.
    if args.offline:
        offline_guard.enable_offline_mode()

    # Always set HF / Transformers offline env vars defensively.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # Validate args.
    if args.batch is None and args.input is None:
        sys.stderr.write("ERROR: provide either an input path or --batch <directory>\n")
        return 2

    if args.batch is not None:
        if not args.batch.is_dir():
            sys.stderr.write(f"ERROR: batch path is not a directory: {args.batch}\n")
            return 2
    else:
        assert args.input is not None
        if not args.input.exists():
            sys.stderr.write(f"ERROR: input file not found: {args.input}\n")
            return 2
        if args.input.suffix.lower() not in SUPPORTED_EXTENSIONS:
            sys.stderr.write(
                f"ERROR: unsupported extension {args.input.suffix!r}; "
                f"expected one of {sorted(SUPPORTED_EXTENSIONS)}\n"
            )
            return 2

    # Build pipeline once.
    try:
        pipeline = build_pipeline()
    except Exception as exc:
        from utils.offline_guard import OfflineViolation

        if isinstance(exc, OfflineViolation):
            sys.stderr.write(f"ERROR: offline violation during init: {exc}\n")
            return 3
        logger.exception("Pipeline init failed")
        sys.stderr.write(f"ERROR: pipeline init failed: {exc}\n")
        return 1

    # Process.
    if args.batch is not None:
        return _run_batch(pipeline, args)
    return _run_single(pipeline, args)


def _run_single(pipeline: Pipeline, args: argparse.Namespace) -> int:
    output_path = args.output if args.output is not None else DEFAULT_OUTPUT
    try:
        result = pipeline.process_one(args.input)
    except Exception as exc:
        logger.exception("Pipeline raised on single doc")
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1

    try:
        emit_json(result, legacy=args.legacy, output_path=output_path, stdout=True)
    except Exception as exc:
        logger.exception("JSON emission failed")
        sys.stderr.write(f"ERROR: JSON emission failed: {exc}\n")
        return 1

    return 0


def _run_batch(pipeline: Pipeline, args: argparse.Namespace) -> int:
    files = list(iter_supported_files(args.batch))
    logger.info("Batch mode: %d files in %s", len(files), args.batch)

    if not files:
        sys.stderr.write(f"WARNING: no supported files in {args.batch}\n")
        return 0

    output_dir: Optional[Path] = args.output
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for i, path in enumerate(files, 1):
        logger.info("[%d/%d] %s", i, len(files), path.name)
        try:
            result = pipeline.process_one(path)
        except Exception as exc:
            logger.exception("Batch entry %s failed", path.name)
            failures += 1
            continue

        per_doc_output = (output_dir / f"{path.stem}.json") if output_dir is not None else None
        try:
            # In batch mode emit one compact JSON per line on stdout.
            if args.legacy:
                payload = to_legacy_dict(result)
                line = json.dumps(payload, ensure_ascii=False)
            else:
                line = result.model_dump_json()
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

            if per_doc_output is not None:
                per_doc_output.write_text(line + "\n", encoding="utf-8")
        except Exception as exc:
            logger.exception("JSON emission failed for %s", path.name)
            failures += 1

    if failures:
        sys.stderr.write(f"WARNING: {failures}/{len(files)} batch entries had errors\n")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
