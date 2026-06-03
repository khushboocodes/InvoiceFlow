"""Unit tests for utils.pipeline — Stage assembly + output building.

These tests use mocked stage components so they run fast (no real OCR /
YOLO / SLM). The end-to-end smoke against real models lives in
``tests/integration/test_pipeline_e2e.py``.

Validates Requirements: 8.1, 8.2, 8.3, 9.1, 9.2, 9.3, 14.4, 14.5,
15.1, 15.2, 15.3, 15.4, 15.5, 15.6
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from utils.detection import Detection
from utils.device import Device, DeviceInfo
from utils.extraction import FieldExtraction
from utils.ingestion import PreprocessedPage
from utils.masters import Masters
from utils.ocr import OcrToken
from utils.pipeline import Pipeline
from utils.schema import ExtractionResult


def _device() -> DeviceInfo:
    return DeviceInfo(kind=Device.CPU, cuda_index=None, description="CPU")


def _make_pipeline(
    *,
    ocr_tokens: list[OcrToken],
    detections: list[Detection],
    slm_response_values: dict | None = None,
    masters: Masters | None = None,
) -> Pipeline:
    """Build a Pipeline with mocked OCR + Vision + (optional) SLM stages."""
    ocr = MagicMock()
    ocr.extract.return_value = ocr_tokens

    detector = MagicMock()
    detector.detect.return_value = detections

    slm = None
    if slm_response_values is not None:
        slm = MagicMock()
        slm_response = MagicMock()
        slm_response.values = slm_response_values
        slm_response.parsed = True
        slm_response.latency_sec = 0.1
        slm.refine.return_value = slm_response

    return Pipeline(
        device=_device(),
        ocr=ocr,
        detector=detector,
        slm=slm,
        masters=masters or Masters(),
    )


def _make_page(tmp_path: Path) -> Path:
    """Save a tiny synthetic invoice and return its path."""
    img_path = tmp_path / "invoice.png"
    Image.new("RGB", (400, 600), color="white").save(img_path)
    return img_path


def _tok(text: str, x: int = 0, y: int = 0, conf: float = 0.95) -> OcrToken:
    return OcrToken(text=text, bbox=(x, y, x + 100, y + 20), confidence=conf, script="en")


# --------------------------------------------------------------------------- #
# Happy path: all four text fields + both visuals resolve in Tier-1
# --------------------------------------------------------------------------- #


def test_pipeline_full_invoice_resolves_all_fields(tmp_path: Path):
    img_path = _make_page(tmp_path)
    pipeline = _make_pipeline(
        ocr_tokens=[
            _tok("ABC", x=200, y=10),
            _tok("TRACTORS", x=300, y=10),
            _tok("Tractor", x=0, y=100),
            _tok("Model", x=80, y=100),
            _tok(":", x=160, y=100),
            _tok("Mahindra", x=190, y=100),
            _tok("575", x=300, y=100),
            _tok("DI", x=350, y=100),
            _tok("Horse", x=0, y=200),
            _tok("Power", x=80, y=200),
            _tok(":", x=160, y=200),
            _tok("50", x=190, y=200),
            _tok("HP", x=240, y=200),
            _tok("Grand", x=0, y=400),
            _tok("Total", x=80, y=400),
            _tok(":", x=160, y=400),
            _tok("525000", x=190, y=400),
        ],
        detections=[
            Detection(cls="signature", bbox=(50, 500, 200, 540), confidence=0.92),
            Detection(cls="stamp", bbox=(220, 480, 340, 580), confidence=0.88),
        ],
    )

    result = pipeline.process_one(img_path)

    assert isinstance(result, ExtractionResult)
    assert result.doc_id == "invoice"
    assert result.error is None

    # All four text fields populated.
    assert result.fields.horse_power.value == 50
    assert result.fields.asset_cost.value == 525000
    assert result.fields.model_name.value is not None and "575" in result.fields.model_name.value

    # Visuals match detection.
    assert result.fields.signature.present is True
    assert result.fields.signature.bbox == (50, 500, 200, 540)
    assert result.fields.stamp.present is True
    assert result.fields.stamp.bbox == (220, 480, 340, 580)

    # Doc-level confidence is non-trivial.
    assert result.confidence > 0.5

    # Timing fields populated.
    assert result.processing_time_sec > 0


# --------------------------------------------------------------------------- #
# Empty / missing visuals → present=False, bbox=None
# --------------------------------------------------------------------------- #


def test_pipeline_emits_present_false_when_no_visual_detections(tmp_path: Path):
    img_path = _make_page(tmp_path)
    pipeline = _make_pipeline(ocr_tokens=[], detections=[])
    result = pipeline.process_one(img_path)

    assert result.fields.signature.present is False
    assert result.fields.signature.bbox is None
    assert result.fields.signature.confidence == 0.0
    assert result.fields.stamp.present is False
    assert result.fields.stamp.bbox is None


# --------------------------------------------------------------------------- #
# Picks highest-confidence detection per class
# --------------------------------------------------------------------------- #


def test_pipeline_picks_highest_confidence_signature(tmp_path: Path):
    img_path = _make_page(tmp_path)
    pipeline = _make_pipeline(
        ocr_tokens=[],
        detections=[
            Detection(cls="signature", bbox=(10, 10, 100, 50), confidence=0.6),
            Detection(cls="signature", bbox=(200, 200, 350, 250), confidence=0.95),
        ],
    )
    result = pipeline.process_one(img_path)
    assert result.fields.signature.bbox == (200, 200, 350, 250)
    assert result.fields.signature.confidence == pytest.approx(0.95)


# --------------------------------------------------------------------------- #
# Stage isolation — Property 10
# --------------------------------------------------------------------------- #


def test_pipeline_returns_empty_on_unsupported_extension(tmp_path: Path):
    bad = tmp_path / "doc.txt"
    bad.write_text("not an image")
    pipeline = _make_pipeline(ocr_tokens=[], detections=[])
    result = pipeline.process_one(bad)

    assert result.error is not None and "unsupported_format" in result.error
    assert result.fields.dealer_name.value is None
    assert result.confidence == 0.0


def test_pipeline_returns_empty_on_corrupt_image(tmp_path: Path):
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"this is not a PNG")
    pipeline = _make_pipeline(ocr_tokens=[], detections=[])
    result = pipeline.process_one(bad)

    assert result.error is not None and "corrupt_input" in result.error
    assert result.confidence == 0.0


def test_pipeline_continues_when_ocr_raises(tmp_path: Path):
    """An OCR crash must not kill the pipeline — vision still runs."""
    img_path = _make_page(tmp_path)

    pipeline = _make_pipeline(
        ocr_tokens=[],
        detections=[Detection(cls="stamp", bbox=(0, 0, 50, 50), confidence=0.9)],
    )
    # Patch OCR to raise.
    pipeline.ocr.extract = MagicMock(side_effect=RuntimeError("synthetic OCR failure"))

    result = pipeline.process_one(img_path)
    assert result.error is None  # OCR failure is non-fatal
    assert result.fields.stamp.present is True  # vision still ran
    assert result.fields.dealer_name.value is None  # text fields are null


def test_pipeline_continues_when_vision_raises(tmp_path: Path):
    """A Vision crash must not kill the pipeline — text fields still extract."""
    img_path = _make_page(tmp_path)
    pipeline = _make_pipeline(
        ocr_tokens=[
            _tok("Tractor", x=0, y=0),
            _tok("Model", x=80, y=0),
            _tok(":", x=160, y=0),
            _tok("Mahindra", x=190, y=0),
            _tok("575", x=300, y=0),
        ],
        detections=[],
    )
    pipeline.detector.detect = MagicMock(side_effect=RuntimeError("vision failure"))

    result = pipeline.process_one(img_path)
    assert result.error is None
    assert result.fields.signature.present is False
    # Text extraction still resolved the model name.
    assert result.fields.model_name.value is not None


# --------------------------------------------------------------------------- #
# SLM hallucination guard — Property 6
# --------------------------------------------------------------------------- #


def test_pipeline_rejects_slm_hallucination_for_text_field(tmp_path: Path):
    """SLM returns a dealer name not in the OCR text → must be rejected."""
    img_path = _make_page(tmp_path)
    pipeline = _make_pipeline(
        # Tokens deliberately positioned far from page top so the letterhead
        # heuristic doesn't pick them up — Tier-1 dealer extraction stays null
        # which forces Tier-2 SLM invocation.
        ocr_tokens=[
            _tok("body", x=0, y=400),
            _tok("paragraph", x=80, y=400),
            _tok("text", x=200, y=400),
        ],
        detections=[],
        slm_response_values={
            "dealer_name": "ZYZ MOTORS LIMITED",  # NOT in OCR text
            "model_name": None,
            "horse_power": None,
            "asset_cost": None,
        },
    )

    result = pipeline.process_one(img_path)
    # Hallucinated dealer rejected → field stays null.
    assert result.fields.dealer_name.value is None


def test_pipeline_accepts_slm_value_when_substring_of_ocr(tmp_path: Path):
    """SLM value that IS a substring of OCR text → accepted."""
    img_path = _make_page(tmp_path)
    pipeline = _make_pipeline(
        # Body tokens (not letterhead) so Tier-1 doesn't get a head start.
        ocr_tokens=[
            _tok("body", x=0, y=400),
            _tok("section", x=80, y=400),
            _tok("XYZ", x=200, y=400),
            _tok("Limited", x=280, y=400),
            _tok("more", x=400, y=400),
        ],
        detections=[],
        slm_response_values={
            "dealer_name": "XYZ Limited",
            "model_name": None,
            "horse_power": None,
            "asset_cost": None,
        },
    )

    result = pipeline.process_one(img_path)
    assert result.fields.dealer_name.value == "XYZ Limited"


# --------------------------------------------------------------------------- #
# Schema round-trip — Property 1
# --------------------------------------------------------------------------- #


def test_pipeline_output_round_trips_through_pydantic(tmp_path: Path):
    img_path = _make_page(tmp_path)
    pipeline = _make_pipeline(
        ocr_tokens=[
            _tok("Horse", x=0, y=0),
            _tok("Power", x=80, y=0),
            _tok(":", x=160, y=0),
            _tok("50", x=190, y=0),
            _tok("HP", x=240, y=0),
        ],
        detections=[],
    )
    result = pipeline.process_one(img_path)

    serialized = result.model_dump_json()
    rebuilt = ExtractionResult.model_validate_json(serialized)
    assert rebuilt == result
