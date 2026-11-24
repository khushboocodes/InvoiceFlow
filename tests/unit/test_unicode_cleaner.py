"""Unit tests for Indic numeral cleaning."""
import pytest
from core.normalization.unicode_cleaner import normalize_text_and_numerals

def test_devanagari_numerals():
    assert normalize_text_and_numerals("४५ एचपी") == "45 एचपी"
    assert normalize_text_and_numerals("७४५०००") == "745000"

def test_gujarati_numerals():
    assert normalize_text_and_numerals("૫૦ હોર્સપાવર") == "50 હોર્સપાવર"
    assert normalize_text_and_numerals("૭૫૦૦૦૦") == "750000"
