"""Unit tests for utils.schema — Pydantic output contract.

Validates Requirements: 10.1, 10.2, 10.3, 10.7, 10.8, 10.9, 15.1, 15.2
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from utils.schema import (
    ExtractionResult,
    Fields,
    NumericField,
    TextField,
    VisualField,
    empty_result,
    to_legacy_dict,
)


def _make_full_result() -> ExtractionResult:
    return ExtractionResult(
        doc_id="invoice_001",
        fields=Fields(
            dealer_name=TextField(value="ABC Tractors Pvt Ltd", confidence=0.92),
            model_name=TextField(value="Mahindra 575 DI", confidence=0.98),
            horse_power=NumericField(value=50, confidence=0.95),
            asset_cost=NumericField(value=525000, confidence=0.91),
            signature=VisualField(present=True, bbox=(100, 200, 300, 250), confidence=0.94),
            stamp=VisualField(present=True, bbox=(400, 500, 500, 550), confidence=0.91),
        ),
        confidence=0.94,
        processing_time_sec=3.8,
        cost_estimate_usd=0.0002,
    )


# --------------------------------------------------------------------------- #
# Property 1: round-trip
# --------------------------------------------------------------------------- #


def test_roundtrip_full_result():
    """model_validate_json(model_dump_json(r)) == r — Property 1."""
    original = _make_full_result()
    rebuilt = ExtractionResult.model_validate_json(original.model_dump_json())
    assert rebuilt == original


def test_roundtrip_empty_result():
    original = empty_result("missing_doc", error="file not found")
    rebuilt = ExtractionResult.model_validate_json(original.model_dump_json())
    assert rebuilt == original


# --------------------------------------------------------------------------- #
# Property 3: confidence bounds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -5])
def test_confidence_must_be_in_unit_interval(bad):
    with pytest.raises(ValidationError):
        TextField(value="x", confidence=bad)
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(
            {
                **_make_full_result().model_dump(),
                "confidence": bad,
            }
        )


# --------------------------------------------------------------------------- #
# Property 4: bbox shape invariant
# --------------------------------------------------------------------------- #


def test_bbox_must_be_well_formed_when_present():
    # x1 >= x2 — invalid
    with pytest.raises(ValidationError):
        VisualField(present=True, bbox=(300, 200, 100, 250), confidence=0.9)
    # y1 >= y2 — invalid
    with pytest.raises(ValidationError):
        VisualField(present=True, bbox=(100, 250, 300, 200), confidence=0.9)


def test_bbox_can_be_null_when_not_present():
    field = VisualField(present=False, bbox=None, confidence=0.0)
    assert field.bbox is None
    assert field.present is False


# --------------------------------------------------------------------------- #
# Property 2: missing values are JSON null, not absent
# --------------------------------------------------------------------------- #


def test_null_numeric_field_serializes_as_json_null():
    """A null horse_power must serialize to JSON null, not 0 — Property 2."""
    field = NumericField(value=None, confidence=0.0)
    payload = json.loads(field.model_dump_json())
    assert payload == {"value": None, "confidence": 0.0}


def test_empty_result_serializes_with_all_keys_present():
    er = empty_result("doc_X")
    payload = json.loads(er.model_dump_json())
    assert set(payload.keys()) >= {"doc_id", "fields", "confidence", "processing_time_sec", "cost_estimate_usd"}
    assert set(payload["fields"].keys()) == {
        "dealer_name",
        "model_name",
        "horse_power",
        "asset_cost",
        "signature",
        "stamp",
    }
    assert payload["fields"]["horse_power"]["value"] is None
    assert payload["fields"]["asset_cost"]["value"] is None
    assert payload["fields"]["signature"]["bbox"] is None


# --------------------------------------------------------------------------- #
# Legacy / PS-reference shape
# --------------------------------------------------------------------------- #


def test_legacy_shape_matches_ps_example():
    """to_legacy_dict produces the flat shape from the PS example."""
    result = _make_full_result()
    legacy = to_legacy_dict(result)

    assert legacy["doc_id"] == "invoice_001"
    assert legacy["fields"]["dealer_name"] == "ABC Tractors Pvt Ltd"
    assert legacy["fields"]["model_name"] == "Mahindra 575 DI"
    assert legacy["fields"]["horse_power"] == 50
    assert legacy["fields"]["asset_cost"] == 525000
    assert legacy["fields"]["signature"] == {"present": True, "bbox": [100, 200, 300, 250]}
    assert legacy["fields"]["stamp"] == {"present": True, "bbox": [400, 500, 500, 550]}
    assert legacy["confidence"] == 0.94
    assert legacy["processing_time_sec"] == 3.8
    assert legacy["cost_estimate_usd"] == 0.0002
    assert "error" not in legacy


def test_legacy_shape_includes_error_when_set():
    er = empty_result("bad_doc", error="file corrupted")
    legacy = to_legacy_dict(er)
    assert legacy["error"] == "file corrupted"
    assert legacy["fields"]["horse_power"] is None


# --------------------------------------------------------------------------- #
# Extra fields are forbidden — defensive contract
# --------------------------------------------------------------------------- #


def test_extra_fields_forbidden_at_top_level():
    base = _make_full_result().model_dump()
    base["bogus_extra_key"] = "should fail"
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(base)
