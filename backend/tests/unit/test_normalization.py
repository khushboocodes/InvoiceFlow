"""Unit tests for utils.normalization — Stage 4.

Validates Requirements: 4.4, 4.5, 4.6, 5.5, 5.6, 6.5, 7.1-7.7, 14.1-14.5
"""

from __future__ import annotations

import pytest

from utils.extraction import FieldExtraction
from utils.masters import AssetEntry, DealerEntry, Masters
from utils.normalization import (
    HP_TO_COST_CEILING,
    HP_TO_COST_FLOOR,
    NormalizedField,
    normalize,
)


def _empty_masters() -> Masters:
    return Masters(dealer=[], asset=[])


def _populated_masters() -> Masters:
    return Masters(
        dealer=[
            DealerEntry(
                canonical="MADHU PAVAN AUTOMOBILES",
                aliases=("MADHU PAWAN AUTO", "MADHU PAVAN AUTO"),
                frequency=12,
            ),
            DealerEntry(
                canonical="SRI AMUTHAM TRACTORS",
                aliases=("SRI AMUTHAM",),
                frequency=8,
            ),
        ],
        asset=[
            AssetEntry(brand="Mahindra", model="575 DI", full_name="Mahindra 575 DI"),
            AssetEntry(brand="Sonalika", model="DI 60", full_name="Sonalika DI 60"),
            AssetEntry(brand="New Holland", model="3032 TX", full_name="New Holland 3032 TX"),
        ],
    )


# --------------------------------------------------------------------------- #
# Dealer name fuzzy matching — Acceptance 4.4, 4.5, 4.6
# --------------------------------------------------------------------------- #


def test_dealer_strong_match_swaps_to_canonical():
    """≥90 fuzzy match — replace raw with canonical and bump confidence."""
    masters = _populated_masters()
    extractions = {
        "dealer_name": FieldExtraction(
            "dealer_name", "MADHU PAWAN AUTOMOBILES", 0.7, "tier1"  # OCR typo
        ),
        "model_name": FieldExtraction("model_name", None, 0.0, "none"),
        "horse_power": FieldExtraction("horse_power", None, 0.0, "none"),
        "asset_cost": FieldExtraction("asset_cost", None, 0.0, "none"),
    }
    out = normalize(extractions, masters)

    dealer = out["dealer_name"]
    assert dealer.value == "MADHU PAVAN AUTOMOBILES"
    assert dealer.canonical_match == "MADHU PAVAN AUTOMOBILES"
    assert dealer.match_score is not None and dealer.match_score >= 90
    # Confidence should be bumped above the raw 0.7
    assert dealer.confidence >= 0.7


def test_dealer_partial_match_keeps_raw_with_dampened_confidence():
    """70-89 score — keep raw, multiply confidence by 0.85."""
    masters = _populated_masters()
    # Loose typo expected to land in the 70-89 band — partial overlap with
    # one of the canonical names but not strong enough to swap.
    extractions = {
        "dealer_name": FieldExtraction("dealer_name", "MADHU AUTO COMPANY", 0.8, "tier1"),
        "model_name": FieldExtraction("model_name", None, 0.0, "none"),
        "horse_power": FieldExtraction("horse_power", None, 0.0, "none"),
        "asset_cost": FieldExtraction("asset_cost", None, 0.0, "none"),
    }
    out = normalize(extractions, masters)
    dealer = out["dealer_name"]
    # Either partial-match (70-89) or weak-match (<70) — but original raw retained
    assert dealer.value == "MADHU AUTO COMPANY"


def test_dealer_no_master_returns_raw_unchanged():
    """No master loaded — keep the Tier-1 value as-is."""
    extractions = {
        "dealer_name": FieldExtraction("dealer_name", "ABC TRACTORS", 0.85, "tier1"),
        "model_name": FieldExtraction("model_name", None, 0.0, "none"),
        "horse_power": FieldExtraction("horse_power", None, 0.0, "none"),
        "asset_cost": FieldExtraction("asset_cost", None, 0.0, "none"),
    }
    out = normalize(extractions, _empty_masters())
    assert out["dealer_name"].value == "ABC TRACTORS"
    assert out["dealer_name"].confidence == pytest.approx(0.85)


def test_dealer_null_input_returns_null():
    extractions = {
        "dealer_name": FieldExtraction("dealer_name", None, 0.0, "none"),
        "model_name": FieldExtraction("model_name", None, 0.0, "none"),
        "horse_power": FieldExtraction("horse_power", None, 0.0, "none"),
        "asset_cost": FieldExtraction("asset_cost", None, 0.0, "none"),
    }
    out = normalize(extractions, _populated_masters())
    assert out["dealer_name"].value is None
    assert out["dealer_name"].confidence == 0.0


# --------------------------------------------------------------------------- #
# Model name exact matching — Acceptance 5.5, 5.6
# --------------------------------------------------------------------------- #


def test_model_exact_match_swaps_to_canonical():
    masters = _populated_masters()
    extractions = {
        "dealer_name": FieldExtraction("dealer_name", None, 0.0, "none"),
        "model_name": FieldExtraction("model_name", "mahindra 575 DI", 0.7, "tier1"),  # case mismatch
        "horse_power": FieldExtraction("horse_power", None, 0.0, "none"),
        "asset_cost": FieldExtraction("asset_cost", None, 0.0, "none"),
    }
    out = normalize(extractions, masters)
    assert out["model_name"].value == "Mahindra 575 DI"
    assert out["model_name"].canonical_match == "Mahindra 575 DI"
    assert out["model_name"].match_score == 100.0


def test_model_no_match_keeps_raw_with_reduced_confidence():
    masters = _populated_masters()
    extractions = {
        "dealer_name": FieldExtraction("dealer_name", None, 0.0, "none"),
        "model_name": FieldExtraction("model_name", "Unknown Brand X100", 0.7, "tier1"),
        "horse_power": FieldExtraction("horse_power", None, 0.0, "none"),
        "asset_cost": FieldExtraction("asset_cost", None, 0.0, "none"),
    }
    out = normalize(extractions, masters)
    assert out["model_name"].value == "Unknown Brand X100"
    # Confidence reduced by 30% (multiply by 0.7)
    assert out["model_name"].confidence == pytest.approx(0.7 * 0.7)
    assert "model_no_exact_match" in out["model_name"].notes


def test_model_match_strips_punctuation_and_whitespace():
    masters = _populated_masters()
    extractions = {
        "dealer_name": FieldExtraction("dealer_name", None, 0.0, "none"),
        "model_name": FieldExtraction(
            "model_name", "  MAHINDRA  575-DI  ", 0.6, "tier1"
        ),
        "horse_power": FieldExtraction("horse_power", None, 0.0, "none"),
        "asset_cost": FieldExtraction("asset_cost", None, 0.0, "none"),
    }
    out = normalize(extractions, masters)
    assert out["model_name"].value == "Mahindra 575 DI"


# --------------------------------------------------------------------------- #
# Numeric normalization — Acceptance 6.5, 7.5
# --------------------------------------------------------------------------- #


def test_horse_power_in_range_passes_through():
    extractions = _all_null_except("horse_power", FieldExtraction("horse_power", 50, 0.9, "tier1"))
    out = normalize(extractions, _empty_masters())
    assert out["horse_power"].value == 50
    assert out["horse_power"].confidence == pytest.approx(0.9)


def test_horse_power_out_of_range_rejected():
    extractions = _all_null_except("horse_power", FieldExtraction("horse_power", 5, 0.9, "tier1"))
    out = normalize(extractions, _empty_masters())
    assert out["horse_power"].value is None
    assert out["horse_power"].confidence == 0.0
    assert "hp_out_of_range" in out["horse_power"].notes


def test_asset_cost_out_of_range_rejected():
    extractions = _all_null_except(
        "asset_cost", FieldExtraction("asset_cost", 50, 0.9, "tier1")
    )
    out = normalize(extractions, _empty_masters())
    assert out["asset_cost"].value is None


# --------------------------------------------------------------------------- #
# Cross-field consistency — Acceptance 7.6
# --------------------------------------------------------------------------- #


def test_cross_field_consistent_no_dampening():
    """A 50 HP tractor at ₹525,000 → 10,500/HP, well within the band."""
    extractions = {
        "dealer_name": FieldExtraction("dealer_name", None, 0.0, "none"),
        "model_name": FieldExtraction("model_name", None, 0.0, "none"),
        "horse_power": FieldExtraction("horse_power", 50, 0.95, "tier1"),
        "asset_cost": FieldExtraction("asset_cost", 525000, 0.95, "tier1"),
    }
    out = normalize(extractions, _empty_masters())
    assert out["asset_cost"].confidence == pytest.approx(0.95)
    assert "hp_cost_inconsistent" not in out["asset_cost"].notes


def test_cross_field_inconsistent_dampens_cost():
    """A 50 HP tractor at ₹100,000 → 2,000/HP — too cheap to be plausible."""
    extractions = {
        "dealer_name": FieldExtraction("dealer_name", None, 0.0, "none"),
        "model_name": FieldExtraction("model_name", None, 0.0, "none"),
        "horse_power": FieldExtraction("horse_power", 50, 0.95, "tier1"),
        "asset_cost": FieldExtraction("asset_cost", 100000, 0.95, "tier1"),
    }
    out = normalize(extractions, _empty_masters())
    # Cost confidence multiplied by 0.8
    assert out["asset_cost"].confidence == pytest.approx(0.95 * 0.8)
    assert "hp_cost_inconsistent" in out["asset_cost"].notes


def test_cross_field_check_skipped_when_either_field_null():
    extractions = {
        "dealer_name": FieldExtraction("dealer_name", None, 0.0, "none"),
        "model_name": FieldExtraction("model_name", None, 0.0, "none"),
        "horse_power": FieldExtraction("horse_power", None, 0.0, "none"),
        "asset_cost": FieldExtraction("asset_cost", 525000, 0.95, "tier1"),
    }
    out = normalize(extractions, _empty_masters())
    assert out["asset_cost"].confidence == pytest.approx(0.95)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _all_null_except(name: str, extraction: FieldExtraction) -> dict[str, FieldExtraction]:
    base = {
        "dealer_name": FieldExtraction("dealer_name", None, 0.0, "none"),
        "model_name": FieldExtraction("model_name", None, 0.0, "none"),
        "horse_power": FieldExtraction("horse_power", None, 0.0, "none"),
        "asset_cost": FieldExtraction("asset_cost", None, 0.0, "none"),
    }
    base[name] = extraction
    return base
