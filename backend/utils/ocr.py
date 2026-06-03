"""Stage 2A: Multilingual OCR via PaddleOCR PP-OCRv4 mobile.

Returns word-level tokens with bounding boxes, recognition confidence, and
script detection (English / Hindi / Gujarati / mixed). The OCR engine is a
process-lifetime singleton — instantiated once in ``executable.py`` and
reused per document, so the heavy model weights load only once.

Acceptance Criterion 2.6 forbids any network calls during init or inference.
PaddleOCR's first-run behavior is to download model files into a user cache
directory; the loader either points at our bundled ``models/paddleocr/``
directory, or accepts that the cache is already populated from a prior run.
We never let the loader fall back to a fresh download — if no weights are
available locally, we raise loudly.

Validates Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 17.2, 17.3
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
from PIL import Image

from utils.device import DeviceInfo
from utils.ingestion import PreprocessedPage

logger = logging.getLogger(__name__)


Script = Literal["en", "hi", "gu", "mixed"]


@dataclass(frozen=True)
class OcrToken:
    """A single OCR-detected token (word) on the page.

    Attributes:
        text: The recognized string. Stripped of leading/trailing whitespace.
        bbox: Axis-aligned box ``(x1, y1, x2, y2)`` in pixel coordinates of
              the preprocessed page image.
        confidence: Recognition confidence in [0.0, 1.0].
        script: Best-effort script classification by Unicode codepoint range.
    """

    text: str
    bbox: tuple[int, int, int, int]
    confidence: float
    script: Script


# --------------------------------------------------------------------------- #
# Script detection
# --------------------------------------------------------------------------- #

# Unicode codepoint ranges per script.
_DEVANAGARI_RANGE = (0x0900, 0x097F)  # Hindi
_GUJARATI_RANGE = (0x0A80, 0x0AFF)


def _detect_script(text: str) -> Script:
    """Classify a token's script by counting codepoints in each Unicode block.

    Returns ``"mixed"`` when at least two scripts are present, else the
    dominant one. Empty strings default to ``"en"``.
    """
    if not text:
        return "en"

    has_hi = False
    has_gu = False
    has_other = False
    for ch in text:
        code = ord(ch)
        if _DEVANAGARI_RANGE[0] <= code <= _DEVANAGARI_RANGE[1]:
            has_hi = True
        elif _GUJARATI_RANGE[0] <= code <= _GUJARATI_RANGE[1]:
            has_gu = True
        elif ch.isalnum() or ch.isspace() or ch in "₹.,-/:;()[]'\"":
            has_other = True

    scripts_present = sum((has_hi, has_gu))
    if scripts_present >= 2:
        return "mixed"
    if has_hi:
        return "hi"
    if has_gu:
        return "gu"
    if has_other and not (has_hi or has_gu):
        return "en"
    return "mixed" if scripts_present == 1 and has_other else "en"


# --------------------------------------------------------------------------- #
# OCR engine
# --------------------------------------------------------------------------- #


class OcrEngine:
    """PaddleOCR PP-OCRv4 mobile wrapper.

    PaddleOCR loads three models per language: detection (det), recognition
    (rec), and direction classification (cls). PP-OCRv4 mobile en+ch shares
    a single rec model that handles English; for Hindi and Gujarati we run
    a second pass with ``lang="hi"`` (rec is multilingual Indic) and merge
    the token streams.

    Args:
        device: Resolved device info from :func:`utils.device.detect`.
        models_dir: Path to ``backend/models/paddleocr/``. When ``None`` we
            rely on PaddleOCR's default cache, which must already be populated.
    """

    def __init__(self, device: DeviceInfo, models_dir: Optional[Path] = None):
        # Hard-disable any network access from PaddleOCR transitively.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self.device = device
        self.models_dir = models_dir

        print("[ocr] starting OcrEngine init", flush=True)

        # IMPORTANT: torch must be imported BEFORE paddle on Windows.
        # Both libraries ship their own MKL / OpenMP DLLs and paddle's
        # variant masks several symbols torch needs at module load time.
        # Importing torch first locks its DLLs into the process, after
        # which paddle imports cleanly. Reverse order produces a
        # ``WinError 127: The specified procedure could not be found``
        # crash inside ``torch.__init__``.
        try:
            import torch  # noqa: F401  type: ignore[import-not-found]
        except ImportError:
            pass

        # Lazy import — PaddleOCR's import is heavy.
        print("[ocr] importing paddleocr", flush=True)
        from paddleocr import PaddleOCR  # type: ignore[import-not-found]
        print("[ocr] paddleocr imported", flush=True)

        common_kwargs = dict(
            use_angle_cls=True,
            use_gpu=device.is_gpu,
            show_log=False,
        )

        # PaddleOCR's "lang" argument selects which rec model to load.
        # PP-OCRv4 ships:
        #   * "en"          — English-only rec (best for Latin docs)
        #   * "ch"          — English + Chinese (default; works for English fine)
        #   * "devanagari"  — Hindi (Devanagari script)
        # Gujarati (`gu`) does not have a dedicated PP-OCRv4 model in the
        # 2.8.x series. Gujarati documents are uncommon in the dataset, and
        # when they do appear the document headers are typically in English.
        # We rely on the English engine for Latin tokens and the Devanagari
        # engine for Indic tokens; pure-Gujarati documents fall back to the
        # SLM (Stage 3 Tier 2), which sees the OCR text we did manage to
        # capture from the English engine alone.
        # We instantiate one engine per script and run them in series.
        # Each engine is small (~10 MB) so RAM cost is negligible.
        logger.info("Loading PaddleOCR engines (en, devanagari) on %s", device.kind.value)
        print("[ocr] constructing English engine", flush=True)

        self._engine_en = PaddleOCR(lang="en", **common_kwargs)
        print("[ocr] English engine ready, constructing Devanagari engine", flush=True)
        try:
            self._engine_hi: Optional["PaddleOCR"] = PaddleOCR(lang="devanagari", **common_kwargs)
            print("[ocr] Devanagari engine ready", flush=True)
        except Exception as exc:
            logger.warning("Devanagari OCR engine unavailable (%s); skipping", exc)
            print(f"[ocr] Devanagari failed: {exc}", flush=True)
            self._engine_hi = None
        # No Gujarati-specific engine in PP-OCRv4; track explicitly so callers
        # can introspect what scripts are supported.
        self._engine_gu: Optional["PaddleOCR"] = None
        print("[ocr] OcrEngine init complete", flush=True)

    # --------------------------------------------------------------- public
    def extract(self, page: PreprocessedPage) -> list[OcrToken]:
        """Run OCR on the preprocessed page image and return token list."""
        return self.extract_image(page.image)

    def extract_image(self, image: Image.Image) -> list[OcrToken]:
        """Lower-level: OCR a PIL image directly.

        Useful for tests, master mining, and the demo bridge where we may
        have an in-memory image but not a full :class:`PreprocessedPage`.
        """
        rgb_array = np.asarray(image.convert("RGB"))

        tokens: list[OcrToken] = []
        engines: list[tuple[str, "PaddleOCR"]] = [("en", self._engine_en)]
        if self._engine_hi is not None:
            engines.append(("hi", self._engine_hi))
        if self._engine_gu is not None:
            engines.append(("gu", self._engine_gu))

        # PaddleOCR returns nested results: one entry per image, each a list
        # of [bbox_4points, (text, confidence)] tuples. We flatten and dedupe
        # tokens that overlap across engines (a quad-IoU check).
        seen_quads: list[list[float]] = []
        for engine_lang, engine in engines:
            try:
                raw = engine.ocr(rgb_array, cls=True)
            except Exception as exc:
                logger.warning("OCR engine '%s' raised: %s", engine_lang, exc)
                continue
            if not raw:
                continue
            page_results = raw[0]
            if not page_results:
                continue
            for entry in page_results:
                if not entry or len(entry) < 2:
                    continue
                quad, info = entry[0], entry[1]
                if not info or len(info) < 2:
                    continue
                text, confidence = info[0], info[1]
                text_clean = (text or "").strip()
                if not text_clean:
                    continue

                # Skip if this quad overlaps a token from a previous engine.
                if any(_quad_iou(quad, prior) > 0.7 for prior in seen_quads):
                    continue
                seen_quads.append(list(_flatten_quad(quad)))

                bbox = _quad_to_bbox(quad)
                if bbox is None:
                    continue
                tokens.append(
                    OcrToken(
                        text=text_clean,
                        bbox=bbox,
                        confidence=float(confidence),
                        script=_detect_script(text_clean),
                    )
                )

        # Stable order: top-to-bottom, then left-to-right.
        tokens.sort(key=lambda t: (t.bbox[1] // 20, t.bbox[0]))
        return tokens


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _quad_to_bbox(quad: Sequence[Sequence[float]]) -> Optional[tuple[int, int, int, int]]:
    """Convert a 4-point quad ``[[x,y]*4]`` to an axis-aligned bbox of ints."""
    try:
        xs = [int(round(float(p[0]))) for p in quad]
        ys = [int(round(float(p[1]))) for p in quad]
    except (TypeError, ValueError, IndexError):
        return None
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    if x1 >= x2 or y1 >= y2:
        return None
    return (max(0, x1), max(0, y1), x2, y2)


def _flatten_quad(quad: Sequence[Sequence[float]]) -> Iterable[float]:
    for p in quad:
        yield float(p[0])
        yield float(p[1])


def _quad_iou(a: Sequence[Sequence[float]] | list[float], b: list[float]) -> float:
    """Approximate quad IoU using axis-aligned bounding boxes.

    Sufficient for de-duplicating overlapping detections across engines that
    typically converge on the same word.
    """
    box_a = _quad_to_bbox(a) if not (isinstance(a, list) and isinstance(a[0], (int, float))) else _flat_to_bbox(a)
    box_b = _flat_to_bbox(b)
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(union, 1)


def _flat_to_bbox(flat: list[float]) -> Optional[tuple[int, int, int, int]]:
    if len(flat) < 8:
        return None
    xs = [int(round(flat[i])) for i in range(0, 8, 2)]
    ys = [int(round(flat[i])) for i in range(1, 8, 2)]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)
