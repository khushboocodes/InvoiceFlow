"""
Unit tests for SpecMatrixValidator
"""
from document_profiles.tractor_quotation.spec_matrix import SpecMatrixValidator

def test_valid_hp_for_model():
    model_meta = {"hp": 50, "price_range_inr": [700000, 850000]}
    assert SpecMatrixValidator.is_valid_hp_for_model(model_meta, 50.0) is True
    assert SpecMatrixValidator.is_valid_hp_for_model(model_meta, 52.0) is True
    assert SpecMatrixValidator.is_valid_hp_for_model(model_meta, 20.0) is False

def test_valid_cost_for_model():
    model_meta = {"hp": 50, "price_range_inr": [700000, 850000]}
    assert SpecMatrixValidator.is_valid_cost_for_model(model_meta, 750000.0) is True
    assert SpecMatrixValidator.is_valid_cost_for_model(model_meta, 50000.0) is False

def test_reconcile_hp():
    model_meta = {"hp": 48.0}
    # If extraction is 47.5, should reconcile to 48.0
    assert SpecMatrixValidator.reconcile_hp(47.5, model_meta) == 48.0
    # If extraction is None, impute from metadata
    assert SpecMatrixValidator.reconcile_hp(None, model_meta) == 48.0
