"""Stage 4: Field normalization and validation.

Takes the raw extractions from Stage 3 and the visual detections from Stage
2B and produces canonical, validated field values. Operations:

* ``dealer_name`` — fuzzy match against ``Dealer_Master`` using RapidFuzz
  token-set ratio. Score ≥ 90 → swap raw for canonical and bump confidence;
  70-89 → keep raw, dampen confidence; < 70 → keep raw, heavy dampen.
* ``model_name`` — case-insensitive whitespace-normalized exact match
  against ``Asset_Master``. Match → swap canonical; miss → keep raw,
  dampen confidence.
* ``horse_power`` — int coercion + range gate [15, 150].
* ``asset_cost`` — int coercion + range gate [100k, 5M] + cross-field
  consistency check against ``horse_power``.

Validates Requirements: 4.4, 4.5, 4.6, 5.5, 5.6, 6.5, 7.1-7.7, 14.1-14.5
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from utils.extraction import COST_MAX, COST_MIN, FieldExtraction, HP_MAX, HP_MIN
from utils.masters import Masters

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Output type
# --------------------------------------------------------------------------- #


@dataclass
class NormalizedField:
    """A normalized + validated field value."""

    value: str | int | None
    confidence: float
    canonical_match: Optional[str] = None  # name of the master entry matched
    match_score: Optional[float] = None  # fuzzy ratio 0-100, or None
    notes: tuple[str, ...] = ()  # human-readable diagnostics


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


# Confidence-empirical (HP, cost) band: tractors of N HP typically cost
# 8000-15000 INR per HP. Anything wildly outside this range gets a soft
# penalty rather than rejection.
HP_TO_COST_FLOOR = 6000   # ₹/HP minimum plausible
HP_TO_COST_CEILING = 20000  # ₹/HP maximum plausible


def normalize(
    extractions: dict[str, FieldExtraction],
    masters: Masters,
) -> dict[str, NormalizedField]:
    """Run normalization across all four text fields.

    Args:
        extractions: Output of :func:`utils.extraction.extract_text_fields`.
            Must contain all four field names; missing keys result in
            null normalized fields.
        masters: Loaded dealer + asset masters.

    Returns:
        Dict keyed by field name with :class:`NormalizedField` values.
    """
    result: dict[str, NormalizedField] = {}

    result["dealer_name"] = _normalize_dealer_name(
        extractions.get("dealer_name"), masters
    )
    result["model_name"] = _normalize_model_name(
        extractions.get("model_name"), masters
    )
    result["horse_power"] = _normalize_horse_power(extractions.get("horse_power"))
    result["asset_cost"] = _normalize_asset_cost(extractions.get("asset_cost"))

    # Cross-field consistency check (Acceptance Criterion 7.6).
    _apply_cross_field_consistency(result)

    return result


# --------------------------------------------------------------------------- #
# Per-field normalizers
# --------------------------------------------------------------------------- #


def _normalize_dealer_name(
    fx: Optional[FieldExtraction], masters: Masters
) -> NormalizedField:
    if fx is None or not fx.value:
        return NormalizedField(value=None, confidence=0.0)
    raw = str(fx.value).strip()
    if not raw:
        return NormalizedField(value=None, confidence=0.0)

    # If no master is loaded, just return the raw value with the Tier-1 conf.
    if not masters.dealer:
        return NormalizedField(
            value=raw,
            confidence=fx.confidence,
            notes=("no_dealer_master_loaded",),
        )

    from rapidfuzz import fuzz, process

    # Build a flat list of (string, canonical) pairs so we match against
    # canonical names AND aliases, but report the canonical as the swap.
    candidate_strings: list[str] = []
    canonical_for: dict[str, str] = {}
    for entry in masters.dealer:
        for form in entry.all_forms():
            candidate_strings.append(form)
            canonical_for[form] = entry.canonical

    best = process.extractOne(
        raw,
        candidate_strings,
        scorer=fuzz.token_set_ratio,
        score_cutoff=0,
    )
    if best is None:
        return NormalizedField(
            value=raw,
            confidence=fx.confidence * 0.5,
            notes=("no_dealer_match",),
        )

    matched_string, score, _ = best
    canonical = canonical_for[matched_string]

    if score >= 90:
        # Strong match — swap to canonical and bump confidence by 10% (capped at 1.0).
        return NormalizedField(
            value=canonical,
            confidence=min(1.0, fx.confidence + 0.10),
            canonical_match=canonical,
            match_score=float(score),
        )
    if score >= 70:
        return NormalizedField(
            value=raw,
            confidence=fx.confidence * 0.85,
            canonical_match=canonical,
            match_score=float(score),
            notes=("dealer_partial_match",),
        )
    return NormalizedField(
        value=raw,
        confidence=fx.confidence * 0.5,
        match_score=float(score),
        notes=("dealer_weak_match",),
    )


def _normalize_model_name(
    fx: Optional[FieldExtraction], masters: Masters
) -> NormalizedField:
    if fx is None or not fx.value:
        return NormalizedField(value=None, confidence=0.0)
    raw = str(fx.value).strip()
    if not raw:
        return NormalizedField(value=None, confidence=0.0)

    if not masters.asset:
        return NormalizedField(
            value=raw,
            confidence=fx.confidence,
            notes=("no_asset_master_loaded",),
        )

    # Normalize for exact-match comparison.
    normalized_raw = _normalize_for_exact_match(raw)
    for entry in masters.asset:
        normalized_canonical = _normalize_for_exact_match(entry.full_name)
        if normalized_raw == normalized_canonical:
            return NormalizedField(
                value=entry.full_name,
                confidence=min(1.0, fx.confidence + 0.05),
                canonical_match=entry.full_name,
                match_score=100.0,
            )

    # No exact match — keep raw with reduced confidence (Acceptance Criterion 5.6).
    return NormalizedField(
        value=raw,
        confidence=fx.confidence * 0.7,
        notes=("model_no_exact_match",),
    )


def _normalize_for_exact_match(text: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation for comparison."""
    cleaned = re.sub(r"[^\w\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip().lower()


def _normalize_horse_power(fx: Optional[FieldExtraction]) -> NormalizedField:
    if fx is None or fx.value is None:
        return NormalizedField(value=None, confidence=0.0)
    try:
        value = int(fx.value)
    except (TypeError, ValueError):
        return NormalizedField(value=None, confidence=0.0, notes=("hp_coercion_failed",))
    if not (HP_MIN <= value <= HP_MAX):
        return NormalizedField(value=None, confidence=0.0, notes=("hp_out_of_range",))
    return NormalizedField(value=value, confidence=fx.confidence)


def _normalize_asset_cost(fx: Optional[FieldExtraction]) -> NormalizedField:
    if fx is None or fx.value is None:
        return NormalizedField(value=None, confidence=0.0)
    try:
        value = int(fx.value)
    except (TypeError, ValueError):
        return NormalizedField(value=None, confidence=0.0, notes=("cost_coercion_failed",))
    if not (COST_MIN <= value <= COST_MAX):
        return NormalizedField(value=None, confidence=0.0, notes=("cost_out_of_range",))
    return NormalizedField(value=value, confidence=fx.confidence)


# --------------------------------------------------------------------------- #
# Cross-field consistency
# --------------------------------------------------------------------------- #


def _apply_cross_field_consistency(fields: dict[str, NormalizedField]) -> None:
    """Mutate ``fields`` in place to reduce ``asset_cost`` confidence when
    HP and cost don't track the empirical band.

    Acceptance Criterion 7.6.
    """
    hp_field = fields.get("horse_power")
    cost_field = fields.get("asset_cost")
    if hp_field is None or cost_field is None:
        return
    hp = hp_field.value
    cost = cost_field.value
    if not isinstance(hp, int) or not isinstance(cost, int) or hp <= 0:
        return

    cost_per_hp = cost / hp
    if HP_TO_COST_FLOOR <= cost_per_hp <= HP_TO_COST_CEILING:
        return

    # Apply 20% confidence dampening + diagnostic note.
    cost_field.confidence *= 0.8
    cost_field.notes = cost_field.notes + ("hp_cost_inconsistent",)
    logger.debug(
        "Cross-field check: HP=%d, cost=%d, ratio=%.0f outside [%d, %d]",
        hp,
        cost,
        cost_per_hp,
        HP_TO_COST_FLOOR,
        HP_TO_COST_CEILING,
    )
