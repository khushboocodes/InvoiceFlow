"""Stage 5: Confidence aggregation.

Computes per-field confidences and a single document-level confidence by
combining the normalized field scores with the visual-detection scores.

The weighting is:

* dealer_name: 0.20
* model_name:  0.20
* horse_power: 0.15
* asset_cost:  0.20
* signature:   0.125
* stamp:       0.125

Weights sum to 1.0. When a field has confidence == 0 (i.e. value is null),
its weight is redistributed equally to the others before computing the
document-level score — this prevents one missing field from dragging the
doc confidence below useful operational thresholds.

Validates Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 14.1, 14.2, 14.3, 14.4, 14.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from utils.detection import Detection
from utils.normalization import NormalizedField


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


# Default field weights (sum to 1.0). Tuned so the four text fields together
# carry 75% of the score and the two visual fields together carry 25% — the
# visual fields are easier to score above 0.5 IoU once trained, so they
# shouldn't dominate the doc-level signal.
DEFAULT_WEIGHTS: dict[str, float] = {
    "dealer_name": 0.20,
    "model_name": 0.20,
    "horse_power": 0.15,
    "asset_cost": 0.20,
    "signature": 0.125,
    "stamp": 0.125,
}

# Document is flagged for manual review when the doc-level confidence is
# below this floor. The orchestrator surfaces this via the JSON output.
REVIEW_THRESHOLD: float = 0.70


# --------------------------------------------------------------------------- #
# Output type
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConfidenceReport:
    """Per-field and aggregated confidences ready for the output JSON."""

    per_field: dict[str, float]
    document: float
    needs_review: bool


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _safe_unit(value: float) -> float:
    """Clamp to [0.0, 1.0]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def aggregate(
    text_fields: Mapping[str, NormalizedField],
    visual_detections: Mapping[str, Optional[Detection]],
    *,
    weights: Optional[dict[str, float]] = None,
    review_threshold: float = REVIEW_THRESHOLD,
) -> ConfidenceReport:
    """Compute per-field and document-level confidence.

    Args:
        text_fields: Output of :func:`utils.normalization.normalize`. Must
            contain all four text-field keys.
        visual_detections: Mapping of class name → highest-confidence
            :class:`utils.detection.Detection` or None when not detected.
        weights: Optional override for the default weight dict. Must contain
            all six field keys.
        review_threshold: Confidence below which the document gets flagged
            for human review.

    Returns:
        A :class:`ConfidenceReport`. The per-field confidences are clamped
        to [0.0, 1.0]; the document confidence is a weight-renormalized
        average that ignores fields with confidence 0 (i.e. nulls).
    """
    w = weights or DEFAULT_WEIGHTS

    # Per-field confidences.
    per_field: dict[str, float] = {}
    for name in ("dealer_name", "model_name", "horse_power", "asset_cost"):
        field = text_fields.get(name)
        per_field[name] = _safe_unit(field.confidence) if field is not None else 0.0
    for name in ("signature", "stamp"):
        det = visual_detections.get(name)
        per_field[name] = _safe_unit(det.confidence) if det is not None else 0.0

    # Document-level: weight-renormalized average.
    # Property: when every field has confidence 0, doc score is 0 (not NaN).
    contributing_keys = [k for k, c in per_field.items() if c > 0.0]
    if not contributing_keys:
        doc = 0.0
    else:
        total_weight = sum(w[k] for k in contributing_keys)
        if total_weight <= 0:
            doc = 0.0
        else:
            doc = sum(w[k] * per_field[k] for k in contributing_keys) / total_weight
    doc = _safe_unit(doc)

    return ConfidenceReport(
        per_field=per_field,
        document=doc,
        needs_review=doc < review_threshold,
    )
