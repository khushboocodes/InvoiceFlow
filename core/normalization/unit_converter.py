"""
InvoiceFlow Unit Converter Engine
Converts engine power ratings (kW to HP) and validates physical bounds.
"""
import re
from typing import Optional, Tuple

KW_TO_HP_FACTOR = 1.34102

HP_PATTERNS = [
    re.compile(r'(?i)(?:\[|\(|\b)(\d{2}(?:\.\d{1,2})?)\s*(?:hp|h\.p\.|horse\s*power|एचपी|हॉर्सपावर|હોર્સપાવર)(?:\]|\)|\b|\s|$)'),
    re.compile(r'(?i)(?:hp|h\.p\.|एचपी)\s*[:\-\.]?\s*(\d{2}(?:\.\d{1,2})?)(?:\]|\)|\b|\s|$)')
]

KW_PATTERNS = [
    re.compile(r'(?i)\b(\d{2}(?:\.\d{1,2})?)\s*(?:kw|k\.w\.|किलोवाट)\b')
]

def extract_and_normalize_hp(text: str) -> Optional[float]:
    """
    Extracts HP or kW from string, converts kW to HP, and validates range [15, 120].
    """
    # 1. Check for explicit HP patterns
    for pat in HP_PATTERNS:
        match = pat.search(text)
        if match:
            try:
                hp_val = float(match.group(1))
                if 15.0 <= hp_val <= 120.0:
                    return round(hp_val, 1)
            except ValueError:
                pass

    # 2. Check for kW patterns and convert
    for pat in KW_PATTERNS:
        match = pat.search(text)
        if match:
            try:
                kw_val = float(match.group(1))
                hp_val = kw_val * KW_TO_HP_FACTOR
                if 15.0 <= hp_val <= 120.0:
                    return round(hp_val, 1)
            except ValueError:
                pass

    return None
