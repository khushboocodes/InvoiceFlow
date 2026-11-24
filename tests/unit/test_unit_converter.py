"""Unit tests for unit converter (kW to HP)."""
import pytest
from core.normalization.unit_converter import extract_and_normalize_hp

def test_extract_explicit_hp():
    assert extract_and_normalize_hp("Engine Capacity: 45 HP") == 45.0
    assert extract_and_normalize_hp("Model 50 H.P. DI") == 50.0
    assert extract_and_normalize_hp("Tractor 48.5 Horse Power") == 48.5

def test_convert_kw_to_hp():
    # 35.0 kW * 1.34102 = 46.9357 -> 46.9 HP
    hp = extract_and_normalize_hp("Engine Power: 35.0 kW")
    assert hp == 46.9

def test_reject_out_of_bounds():
    assert extract_and_normalize_hp("Model 744") is None  # 744 is out of bounds (>120)
    assert extract_and_normalize_hp("Displacement: 2979 CC") is None
