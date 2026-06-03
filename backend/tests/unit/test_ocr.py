"""Unit tests for utils.ocr — Stage 2A (PaddleOCR wrapper).

Validates Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 17.2, 17.3
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tests.fixtures.generate import make_invoice_image
from utils.ocr import OcrToken, _detect_script, _quad_to_bbox


# --------------------------------------------------------------------------- #
# Pure tests (no PaddleOCR required) — script detection + quad helpers
# --------------------------------------------------------------------------- #


def test_detect_script_english_only():
    assert _detect_script("ABC TRACTORS PVT LTD") == "en"
    assert _detect_script("Mahindra 575 DI") == "en"
    assert _detect_script("Total Cost: Rs. 5,25,000") == "en"
    assert _detect_script("50 HP") == "en"


def test_detect_script_hindi_only():
    # Pure Devanagari
    assert _detect_script("ट्रैक्टर मॉडल") == "hi"
    assert _detect_script("एचपी") == "hi"


def test_detect_script_gujarati_only():
    assert _detect_script("ટ્રેક્ટર") == "gu"
    assert _detect_script("બળ") == "gu"


def test_detect_script_mixed_returns_mixed():
    """A token with two distinct Indic scripts should classify as mixed."""
    assert _detect_script("ट्रैक्टर ટ્રેક્ટર") == "mixed"


def test_detect_script_empty_string_defaults_to_english():
    assert _detect_script("") == "en"


def test_quad_to_bbox_axis_aligned():
    quad = [[10, 20], [100, 22], [102, 60], [12, 58]]
    bbox = _quad_to_bbox(quad)
    assert bbox == (10, 20, 102, 60)


def test_quad_to_bbox_clamps_negative_coords():
    quad = [[-5, -3], [50, -2], [52, 30], [-3, 28]]
    bbox = _quad_to_bbox(quad)
    assert bbox == (0, 0, 52, 30)


def test_quad_to_bbox_returns_none_for_degenerate_quad():
    # Zero-area quad
    assert _quad_to_bbox([[10, 20], [10, 20], [10, 20], [10, 20]]) is None
    # Malformed input
    assert _quad_to_bbox([[10, 20], [10, 20]]) is None


def test_ocr_token_is_frozen():
    """OcrToken dataclass is immutable — downstream code shouldn't mutate
    OCR results out from under each other."""
    t = OcrToken(text="hello", bbox=(0, 0, 10, 10), confidence=0.9, script="en")
    with pytest.raises(Exception):
        t.confidence = 0.1  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Live engine tests — skipped when PaddleOCR isn't installed yet
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def ocr_engine():
    """Try to load a real OcrEngine; skip the live tests if unavailable.

    PaddleOCR + paddlepaddle are heavy installs and are sometimes not yet in
    the venv during partial runs. Skipping cleanly keeps the rest of the
    suite green.
    """
    try:
        import paddleocr  # noqa: F401
    except ImportError:
        pytest.skip("paddleocr not installed in this environment")

    from utils.device import detect
    from utils.ocr import OcrEngine

    info = detect()
    try:
        return OcrEngine(info)
    except Exception as exc:
        pytest.skip(f"PaddleOCR engine failed to initialize: {exc}")


def test_ocr_extracts_tokens_from_synthetic_invoice(ocr_engine):
    """The fixture invoice contains canonical tokens — model_name, total cost,
    HP — and OCR must find at least the most prominent ones."""
    img = make_invoice_image()
    tokens = ocr_engine.extract_image(img)

    assert len(tokens) > 0, "OCR returned no tokens"
    text_blob = " ".join(t.text for t in tokens).lower()

    # We expect at least *some* of the prominent strings to be recognized.
    # PP-OCRv4 mobile is not perfect on synthetic Pillow renders, so we use
    # an OR over high-frequency anchors rather than asserting every token.
    anchors = ["abc", "tractors", "mahindra", "575", "50", "hp", "total"]
    hits = sum(1 for a in anchors if a in text_blob)
    assert hits >= 3, f"expected at least 3 anchor hits, got {hits} in: {text_blob[:200]}"


def test_ocr_tokens_have_well_formed_bboxes(ocr_engine):
    """Every returned token must have a well-formed bbox and a confidence in [0,1]."""
    img = make_invoice_image()
    tokens = ocr_engine.extract_image(img)
    assert tokens

    for t in tokens:
        x1, y1, x2, y2 = t.bbox
        assert all(isinstance(c, int) for c in t.bbox)
        assert x1 < x2 and y1 < y2
        assert x1 >= 0 and y1 >= 0
        assert 0.0 <= t.confidence <= 1.0
        assert t.script in ("en", "hi", "gu", "mixed")


def test_ocr_tokens_sorted_top_to_bottom(ocr_engine):
    """Output ordering is top-to-bottom, then left-to-right within rows.

    The relevance scorer and Tier-1 anchor matching depend on this being
    stable so anchor proximity calculations are meaningful.
    """
    img = make_invoice_image()
    tokens = ocr_engine.extract_image(img)
    assert len(tokens) >= 4

    rows = [t.bbox[1] // 20 for t in tokens]
    assert rows == sorted(rows), "tokens not sorted by row"


def test_ocr_against_real_train_image(ocr_engine):
    """Run OCR against one real image from the train set as a smoke test."""
    train_dir = Path(__file__).resolve().parents[3] / "train_data_idfc" / "train"
    if not train_dir.exists():
        pytest.skip("Real train images not available")

    pngs = sorted(train_dir.glob("*.png"))
    if not pngs:
        pytest.skip("No images in train_dir")

    img = Image.open(pngs[0]).convert("RGB")
    tokens = ocr_engine.extract_image(img)

    # Real invoices should produce non-trivial output. We don't assert exact
    # token counts because real-world quality varies wildly.
    assert isinstance(tokens, list)
    if tokens:
        for t in tokens:
            assert 0.0 <= t.confidence <= 1.0
