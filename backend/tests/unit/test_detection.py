"""Unit tests for utils.detection — Stage 2B (YOLOv8n inference).

Validates Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 17.2, 17.3
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from utils.detection import (
    CLASS_INDEX,
    CLASS_NAMES,
    Detection,
    VisionDetector,
    _DetectorConfig,
)
from utils.device import Device, DeviceInfo

WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "models" / "yolov8n_sig_stamp.pt"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "models" / "detection.yaml"


# ---------------------------------------------------------------------------
# Pure tests (no model required)
# ---------------------------------------------------------------------------


def test_class_names_match_training_data_yaml():
    """The class index/name mapping must match what data.yaml declared
    during YOLO training (signature=0, stamp=1)."""
    assert CLASS_NAMES == ("signature", "stamp")
    assert CLASS_INDEX == {"signature": 0, "stamp": 1}


def test_detector_config_defaults_when_missing(tmp_path: Path):
    """Missing detection.yaml must not raise — fall back to defaults."""
    config = _DetectorConfig.load(tmp_path / "missing.yaml")
    assert config.signature_threshold == pytest.approx(0.35)
    assert config.stamp_threshold == pytest.approx(0.40)
    assert config.nms_iou == pytest.approx(0.45)


def test_detector_config_loads_from_yaml(tmp_path: Path):
    yaml_path = tmp_path / "detection.yaml"
    yaml_path.write_text(
        "confidence_thresholds:\n"
        "  signature: 0.55\n"
        "  stamp: 0.65\n"
        "nms_iou: 0.50\n"
    )
    config = _DetectorConfig.load(yaml_path)
    assert config.signature_threshold == pytest.approx(0.55)
    assert config.stamp_threshold == pytest.approx(0.65)
    assert config.nms_iou == pytest.approx(0.50)
    assert config.threshold_for("signature") == pytest.approx(0.55)
    assert config.threshold_for("stamp") == pytest.approx(0.65)


def test_detection_dataclass_is_frozen():
    """Property 4 — bbox shape invariant. Detection is immutable so callers
    can't tweak coords after the schema's validator has run."""
    det = Detection(cls="signature", bbox=(10, 20, 100, 60), confidence=0.9)
    with pytest.raises(Exception):
        det.confidence = 0.1  # type: ignore[misc]


def test_missing_weights_raises_filenotfound(tmp_path: Path):
    """Constructor must fail loudly if the .pt file is missing."""
    info = DeviceInfo(kind=Device.CPU, cuda_index=None, description="CPU")
    with pytest.raises(FileNotFoundError):
        VisionDetector(info, tmp_path / "ghost.pt")


# ---------------------------------------------------------------------------
# Live model tests — skipped automatically if weights aren't bundled yet
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def detector():
    """Load the trained detector once per test module."""
    if not WEIGHTS_PATH.exists():
        pytest.skip(f"Weights not yet trained: {WEIGHTS_PATH}")

    from utils.device import detect

    info = detect()
    return VisionDetector(info, WEIGHTS_PATH, CONFIG_PATH)


def test_detector_returns_list_for_blank_image(detector):
    """A blank white image should produce zero or near-zero detections."""
    blank = Image.new("RGB", (640, 640), color="white")
    detections = detector._detect_image(blank)
    # The model could theoretically hallucinate on noise; we just assert the
    # API contract: detections is a list of Detection objects.
    assert isinstance(detections, list)
    for det in detections:
        assert isinstance(det, Detection)
        assert det.cls in ("signature", "stamp")
        assert 0.0 <= det.confidence <= 1.0


def test_detector_runs_against_real_train_image(detector):
    """Run the detector against a real labeled image and verify schema-valid output.

    We don't assert mAP here (that's the trainer's job). We assert:
    * ``detect()`` returns a list of well-formed :class:`Detection`s.
    * Bounding boxes satisfy the Property-4 invariant (x1<x2, y1<y2, ints).
    * Per-class threshold filtering is applied.
    """
    images_dir = Path(__file__).resolve().parents[3] / "train_data_idfc" / "yolo" / "images"
    # Try test split first, then val, then train as fallback.
    for split in ("test", "val", "train"):
        candidate_dir = images_dir / split
        if not candidate_dir.exists():
            continue
        candidates = sorted(candidate_dir.glob("*.png"))
        if candidates:
            sample = candidates[0]
            break
    else:
        pytest.skip("No labeled images available")

    img = Image.open(sample).convert("RGB")
    detections = detector._detect_image(img)

    assert isinstance(detections, list)
    for det in detections:
        assert isinstance(det, Detection)
        assert det.cls in ("signature", "stamp")
        x1, y1, x2, y2 = det.bbox
        assert all(isinstance(c, int) for c in det.bbox)
        assert x1 < x2
        assert y1 < y2
        assert x1 >= 0 and y1 >= 0
        # Per-class threshold should have been enforced.
        threshold = detector.config.threshold_for(det.cls)
        assert det.confidence >= threshold

    # Detections sorted by confidence descending.
    confs = [d.confidence for d in detections]
    assert confs == sorted(confs, reverse=True)


def test_best_per_class_picks_top_detection_per_class(detector):
    """``best_per_class`` returns at most one detection per class."""
    fake = [
        Detection(cls="signature", bbox=(10, 20, 100, 60), confidence=0.91),
        Detection(cls="signature", bbox=(200, 200, 300, 240), confidence=0.55),
        Detection(cls="stamp", bbox=(400, 500, 500, 600), confidence=0.80),
    ]
    best = detector.best_per_class(fake)
    assert best["signature"] is not None
    assert best["signature"].confidence == pytest.approx(0.91)
    assert best["stamp"] is not None
    assert best["stamp"].confidence == pytest.approx(0.80)


def test_best_per_class_returns_none_for_missing(detector):
    only_signature = [Detection(cls="signature", bbox=(0, 0, 10, 10), confidence=0.9)]
    best = detector.best_per_class(only_signature)
    assert best["signature"] is not None
    assert best["stamp"] is None
