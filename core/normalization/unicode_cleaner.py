"""
InvoiceFlow Unicode & Numeral Cleaner
Normalizes NFC Unicode and maps Indic numerals (Devanagari, Gujarati) to ASCII digits.
"""
import unicodedata

# Devanagari numerals ०..९ to 0..9
DEVANAGARI_DIGITS = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

# Gujarati numerals ૦..૯ to 0..9
GUJARATI_DIGITS = {
    '૦': '0', '૧': '1', '૨': '2', '૩': '3', '૪': '4',
    '૫': '5', '૬': '6', '૭': '7', '૮': '8', '૯': '9'
}

def normalize_text_and_numerals(text: str) -> str:
    """
    Converts Indic numeral glyphs to standard ASCII digits and normalizes Unicode.
    """
    if not text:
        return ""
    # NFC normalization
    norm = unicodedata.normalize("NFC", text)
    # Translate Devanagari digits
    for dev_char, ascii_char in DEVANAGARI_DIGITS.items():
        norm = norm.replace(dev_char, ascii_char)
    # Translate Gujarati digits
    for guj_char, ascii_char in GUJARATI_DIGITS.items():
        norm = norm.replace(guj_char, ascii_char)
    return norm.strip()
