"""Pydantic v2 output contract for the extraction pipeline.

The output JSON shape mirrors the IDFC GenAI problem statement. Every module
that produces a field hands back data that flows through these models, so the
final ``executable.py`` only ever writes JSON via ``ExtractionResult.model_dump_json()``.

We extend the PS shape in two ways:

* Each text/numeric/visual field is wrapped in an object containing both
  ``value`` (or ``present`` + ``bbox`` for visuals) and a per-field
  ``confidence``. The PS asks for "confidence scores"; a flat shape can't
  carry them. A grader-compatibility ``--legacy`` shim in ``executable.py``
  collapses the wrappers when needed.
* An optional ``error`` key surfaces partial-result failures so a single
  bad document does not break batch evaluation.

Validates Requirements: 10.1, 10.2, 10.3, 10.7, 10.8, 10.9, 15.1, 15.2
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Bounding box in pixel coordinates of the page as preprocessed by Stage 1.
# Order is (x1, y1, x2, y2) with x1 < x2 and y1 < y2.
Bbox = tuple[int, int, int, int]


class _Strict(BaseModel):
    """Base model: forbid extra fields, validate on assignment, immutable confidence floor."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=False,
        # Disable Pydantic's "model_" namespace protection so we can have a
        # field literally named ``model_name`` per the PS schema.
        protected_namespaces=(),
    )


class TextField(_Strict):
    """A textual field (dealer_name, model_name)."""

    value: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class NumericField(_Strict):
    """A numeric field (horse_power, asset_cost). Value is integer or null."""

    value: Optional[int] = None
    confidence: float = Field(ge=0.0, le=1.0)


class VisualField(_Strict):
    """A visual field (signature, stamp) with presence flag and optional bbox."""

    present: bool
    bbox: Optional[Bbox] = None
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("bbox")
    @classmethod
    def _bbox_must_be_well_formed(cls, value: Optional[Bbox]) -> Optional[Bbox]:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("bbox must be a 4-tuple (x1, y1, x2, y2)")
        x1, y1, x2, y2 = value
        if not all(isinstance(c, int) for c in value):
            raise ValueError("bbox coordinates must all be integers")
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"bbox must satisfy x1 < x2 and y1 < y2, got {value}")
        return value


class Fields(_Strict):
    """The six target fields in the canonical order of the problem statement."""

    dealer_name: TextField
    model_name: TextField
    horse_power: NumericField
    asset_cost: NumericField
    signature: VisualField
    stamp: VisualField


class ExtractionResult(_Strict):
    """Top-level output object, one per processed document."""

    doc_id: str
    fields: Fields
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time_sec: float = Field(ge=0.0)
    cost_estimate_usd: float = Field(ge=0.0)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Legacy / PS-reference flat shape
# ---------------------------------------------------------------------------
#
# The problem statement example shows field values directly under each key,
# without the wrapper objects. The ``--legacy`` flag emits this shape via
# ``to_legacy_dict``. Confidence scores collapse to the document level.


def to_legacy_dict(result: ExtractionResult) -> dict:
    """Render the result in the flat PS-reference shape.

    The wrapped per-field confidence is dropped; only the top-level
    ``confidence`` is preserved. Visual fields keep their ``{present, bbox}``
    structure as the PS requires.
    """
    f = result.fields
    return {
        "doc_id": result.doc_id,
        "fields": {
            "dealer_name": f.dealer_name.value,
            "model_name": f.model_name.value,
            "horse_power": f.horse_power.value,
            "asset_cost": f.asset_cost.value,
            "signature": {
                "present": f.signature.present,
                "bbox": list(f.signature.bbox) if f.signature.bbox else None,
            },
            "stamp": {
                "present": f.stamp.present,
                "bbox": list(f.stamp.bbox) if f.stamp.bbox else None,
            },
        },
        "confidence": result.confidence,
        "processing_time_sec": result.processing_time_sec,
        "cost_estimate_usd": result.cost_estimate_usd,
        **({"error": result.error} if result.error else {}),
    }


def empty_result(doc_id: str, error: str | None = None) -> ExtractionResult:
    """Build a fully-null result for failure paths.

    Used by Stage Isolation (Property 10) — when ingestion or another stage
    blows up, we still emit a valid JSON object so batch evaluation keeps
    moving.
    """
    return ExtractionResult(
        doc_id=doc_id,
        fields=Fields(
            dealer_name=TextField(value=None, confidence=0.0),
            model_name=TextField(value=None, confidence=0.0),
            horse_power=NumericField(value=None, confidence=0.0),
            asset_cost=NumericField(value=None, confidence=0.0),
            signature=VisualField(present=False, bbox=None, confidence=0.0),
            stamp=VisualField(present=False, bbox=None, confidence=0.0),
        ),
        confidence=0.0,
        processing_time_sec=0.0,
        cost_estimate_usd=0.0,
        error=error,
    )
