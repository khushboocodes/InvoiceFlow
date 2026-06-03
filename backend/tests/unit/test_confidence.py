"""Unit tests for utils.confidence — Stage 5.

Validates Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 14.1, 14.2, 14.3, 14.4, 14.5
"""

from __future__ import annotations

import pytest

from utils.confidence import (
    DEFAULT_WEIGHTS,
    REVIEW_THRESHOLD,
    aggregate,
)
from utils.detection import Detection
from utils.normalization import NormalizedField


def _all_text(values: dict[str, tuple[str | int | None, float]]) -> dict[str, NormalizedField]:
    """Helper: build a dict of NormalizedField from (value, confidence) tuples."""
    return {
        name: NormalizedField(value=value, confidence=conf) for name, (value, conf) in values.items()
    }


def _det(cls: str, conf: float) -> Detection:
    return Detection(cls=cls, bbox=(10, 20, 100, 60), confidence=conf)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Per-field confidences passthrough — Requirement 14.1
# --------------------------------------------------------------------------- #


def test_per_field_confidences_clamped_to_unit_interval():
    text_fields = _all_text(
        {
            "dealer_name": ("ABC", 0.91),
            "model_name": ("M1", 0.85),
            "horse_power": (50, 0.7),
            "asset_cost": (525000, 0.6),
        }
    )
    visual = {"signature": _det("signature", 0.94), "stamp": _det("stamp", 0.88)}
    report = aggregate(text_fields, visual)

    for v in report.per_field.values():
        assert 0.0 <= v <= 1.0


def test_per_field_confidences_match_inputs():
    text_fields = _all_text(
        {
            "dealer_name": ("ABC", 0.91),
            "model_name": ("M1", 0.85),
            "horse_power": (50, 0.7),
            "asset_cost": (525000, 0.6),
        }
    )
    visual = {"signature": _det("signature", 0.94), "stamp": _det("stamp", 0.88)}
    report = aggregate(text_fields, visual)

    assert report.per_field["dealer_name"] == pytest.approx(0.91)
    assert report.per_field["horse_power"] == pytest.approx(0.7)
    assert report.per_field["signature"] == pytest.approx(0.94)
    assert report.per_field["stamp"] == pytest.approx(0.88)


def test_missing_visual_detection_yields_zero_confidence():
    text_fields = _all_text(
        {
            "dealer_name": ("ABC", 0.9),
            "model_name": ("M1", 0.9),
            "horse_power": (50, 0.9),
            "asset_cost": (525000, 0.9),
        }
    )
    visual = {"signature": None, "stamp": _det("stamp", 0.8)}
    report = aggregate(text_fields, visual)

    assert report.per_field["signature"] == 0.0
    assert report.per_field["stamp"] == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# Document-level aggregation — Requirement 14.3
# --------------------------------------------------------------------------- #


def test_document_confidence_is_weighted_average_of_present_fields():
    """All six fields present, all at 0.9 → doc conf = 0.9."""
    text_fields = _all_text(
        {
            "dealer_name": ("ABC", 0.9),
            "model_name": ("M1", 0.9),
            "horse_power": (50, 0.9),
            "asset_cost": (525000, 0.9),
        }
    )
    visual = {"signature": _det("signature", 0.9), "stamp": _det("stamp", 0.9)}
    report = aggregate(text_fields, visual)

    assert report.document == pytest.approx(0.9)


def test_document_confidence_when_all_fields_zero():
    text_fields = _all_text(
        {
            "dealer_name": (None, 0.0),
            "model_name": (None, 0.0),
            "horse_power": (None, 0.0),
            "asset_cost": (None, 0.0),
        }
    )
    visual = {"signature": None, "stamp": None}
    report = aggregate(text_fields, visual)

    assert report.document == 0.0
    assert report.needs_review is True


# --------------------------------------------------------------------------- #
# Weight redistribution — Property 10 / Acceptance Criterion 14.3
# --------------------------------------------------------------------------- #


def test_weight_redistribution_when_one_field_is_null():
    """A null field doesn't drag the doc score; its weight gets ignored."""
    text_fields = _all_text(
        {
            "dealer_name": ("ABC", 0.9),
            "model_name": ("M1", 0.9),
            "horse_power": (50, 0.9),
            "asset_cost": (None, 0.0),  # null — should not pull doc down
        }
    )
    visual = {"signature": _det("signature", 0.9), "stamp": _det("stamp", 0.9)}
    report = aggregate(text_fields, visual)

    # All present fields are 0.9, so doc score should be 0.9 (not 0.72).
    assert report.document == pytest.approx(0.9)


def test_weight_redistribution_with_mixed_present_fields():
    """Two present fields at 0.5 and 0.9 → doc score is the weight-normalized avg."""
    text_fields = _all_text(
        {
            "dealer_name": ("ABC", 0.5),
            "model_name": (None, 0.0),
            "horse_power": (None, 0.0),
            "asset_cost": (525000, 0.9),
        }
    )
    visual = {"signature": None, "stamp": None}
    report = aggregate(text_fields, visual)

    # dealer weight 0.20, cost weight 0.20 → equal share → average = 0.7
    expected = (0.20 * 0.5 + 0.20 * 0.9) / (0.20 + 0.20)
    assert report.document == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Review threshold — Acceptance Criterion 9.5
# --------------------------------------------------------------------------- #


def test_needs_review_when_below_threshold():
    text_fields = _all_text(
        {
            "dealer_name": ("ABC", 0.5),
            "model_name": ("M1", 0.5),
            "horse_power": (50, 0.5),
            "asset_cost": (525000, 0.5),
        }
    )
    visual = {"signature": _det("signature", 0.5), "stamp": _det("stamp", 0.5)}
    report = aggregate(text_fields, visual)

    assert report.document == pytest.approx(0.5)
    assert report.needs_review is True


def test_does_not_need_review_when_above_threshold():
    text_fields = _all_text(
        {
            "dealer_name": ("ABC", 0.95),
            "model_name": ("M1", 0.95),
            "horse_power": (50, 0.95),
            "asset_cost": (525000, 0.95),
        }
    )
    visual = {"signature": _det("signature", 0.95), "stamp": _det("stamp", 0.95)}
    report = aggregate(text_fields, visual)

    assert report.document == pytest.approx(0.95)
    assert report.needs_review is False


# --------------------------------------------------------------------------- #
# Weights contract
# --------------------------------------------------------------------------- #


def test_default_weights_sum_to_one():
    """The weight contract — summing to 1.0 keeps doc scores in [0, 1] when
    all fields are present."""
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_default_weights_have_all_six_field_keys():
    expected = {"dealer_name", "model_name", "horse_power", "asset_cost", "signature", "stamp"}
    assert set(DEFAULT_WEIGHTS.keys()) == expected


def test_review_threshold_is_reasonable():
    """The threshold is documented in the PRD; this test pins the chosen value."""
    assert 0.5 <= REVIEW_THRESHOLD <= 0.9
