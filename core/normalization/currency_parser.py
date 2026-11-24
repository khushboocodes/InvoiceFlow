"""
InvoiceFlow Indian Currency Normalization & Arithmetic Engine
"""
import re
from typing import Optional, List

INDIAN_CURRENCY_REGEX = re.compile(
    r'(?:Rs\.?|INR|₹|रु\.?|રૂ\.)?\s*([1-9]\d{0,2}(?:,\d{2})*,\d{3}(?:\.\d{2})?|[1-9]\d{4,7}(?:\.\d{2})?)',
    re.IGNORECASE
)

WORDS_TO_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "hazar": 1000, "lakh": 100000, "lac": 100000, "lakhs": 100000,
    "crore": 10000000, "crores": 10000000
}

def clean_currency_string(raw_str: str) -> Optional[float]:
    """
    Cleans raw monetary text string (e.g. 'Rs. 7,45,000/-', '7 20 ooq') and returns float value.
    """
    if not raw_str:
        return None
    
    # Strip common noise
    cleaned = raw_str.replace('/-', '').replace('—', '').replace('=', '').strip()
    
    # Strip spaces around commas, dots, and hyphens (e.g. '8, 6 0,000' -> '8,60,000')
    cleaned = re.sub(r'\s*([,.\-])\s*', r'\1', cleaned)

    # Replace OCR digit confusions ONLY when adjacent to digits or numerals (prevents turning 'Total' into 'Tota1')
    cleaned = re.sub(r'(?<=[,\d])[oOqQ]+(?=[,\d\s\-/]|$)', lambda m: '0' * len(m.group(0)), cleaned)
    cleaned = re.sub(r'(?<=[,\d])[lI|]+(?=[,\d\s\-/]|$)', lambda m: '1' * len(m.group(0)), cleaned)
    cleaned = re.sub(r'(?<=\s)[oOqQ]+(?=[,\d])', lambda m: '0' * len(m.group(0)), cleaned)
    cleaned = re.sub(r'(?<=\s)[lI|]+(?=[,\d])', lambda m: '1' * len(m.group(0)), cleaned)
    
    # Collapse remaining spaces between adjacent digits (e.g. '7 20 000' -> '720000')
    cleaned = re.sub(r'(\d)\s+(\d)', r'\1\2', cleaned)
    cleaned = re.sub(r'(\d)\s+(\d)', r'\1\2', cleaned)
    
    match = INDIAN_CURRENCY_REGEX.search(cleaned)
    if match:
        num_str = match.group(1).replace(',', '').strip()
        try:
            val = float(num_str)
            return val
        except ValueError:
            return None
    return None

def is_valid_asset_cost(cost: float) -> bool:
    """Enforces Indian tractor commercial price boundaries (2 Lakhs to 25 Lakhs)."""
    return 200000.0 <= cost <= 2500000.0
