"""Unit tests for currency parser."""
import pytest
from core.normalization.currency_parser import clean_currency_string, is_valid_asset_cost

def test_clean_currency_string_formatted():
    assert clean_currency_string("Rs. 7,45,000/-") == 745000.0
    assert clean_currency_string("₹ 8,20,500.00") == 820500.0
    assert clean_currency_string("INR 6,50,000") == 650000.0

def test_clean_currency_ocr_typos():
    assert clean_currency_string("Rs. 7,45,OOO/-") == 745000.0
    assert clean_currency_string("Rs. 745000") == 745000.0

def test_is_valid_asset_cost():
    assert is_valid_asset_cost(745000.0) is True
    assert is_valid_asset_cost(150000.0) is False  # Too low
    assert is_valid_asset_cost(3000000.0) is False  # Too high
