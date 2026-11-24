"""
Dealer Name Extractor
Extracts dealership legal and trade names from top header zone with negative keyword filtering.
"""
import re
from typing import Optional, Tuple, List
from rapidfuzz import fuzz
from core.layout.spatial_graph import SpatialWordGraph, WordToken

BUYER_PATTERNS = [
    re.compile(r'(?i)\b(?:customer|buyer|farmer|purchaser|shri|smt)\b'),
    re.compile(r'(?i)\b(?:नाम|ग्राहक|किसान|श्री|पता|ग्राम|तहसील)\b'),
    re.compile(r'(?i)\b(?:to\s*:\s*|sold\s*to\s*:)\b')
]

FINANCIER_BLOCKLIST = [
    "idfc", "idfc first", "hdfc", "hdfc bank", "state bank", "sbi",
    "mahindra finance", "chola", "cholamandalam", "kotak", "tvs credit"
]

HEADER_NOISE_BLOCKLIST = [
    "quotation", "proforma invoice", "tax invoice", "estimate", "delivery challan",
    "order permit", "permit order", "bill of supply", "retail invoice", "cash memo"
]

CORPORATE_AFFILIATION_PATTERNS = [
    re.compile(r'(?i)\b(?:subsidiary|wholly\s*owned|a\s*unit\s*of)\b')
]

DEALER_ANCHORS = [
    "dealer", "dealership", "tractors", "tractor", "motors", "motor",
    "agencies", "agency", "automobiles", "automobile", "auto", "agro",
    "traders", "trader", "enterprises", "enterprise", "corporation",
    "industries", "industry", "pvt ltd", "ltd", "limited", "works", "sales",
    "kendra", "krishi", "sewa", "service", "farm"
]

DATE_TIME_RE = re.compile(r'(?:\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b|https?://|\.com\b|\.in\b|page\s+\d+)', re.IGNORECASE)

OEM_BRAND_NAMES = [
    "eicher", "mahindra", "swaraj", "sonalika", "farmtrac", "powertrac",
    "massey ferguson", "massey", "tafe", "john deere", "kubota", "new holland",
    "escorts", "preet", "indo farm"
]

class DealerNameExtractor:
    def __init__(self, known_dealers: Optional[List[str]] = None):
        self.known_dealers = known_dealers or []

    def extract(self, graph: SpatialWordGraph) -> Tuple[str, float, Optional[Tuple[int, int, int, int]]]:
        """
        Extracts dealership name from Page 1 header region.
        Returns (dealer_name, confidence, bounding_box).
        """
        # Crop header zone (top 35% of page)
        header_tokens = graph.find_tokens_in_region(ymin=0, xmin=0, ymax=350, xmax=1000)
        if not header_tokens:
            return "UNKNOWN_DEALER", 0.0, None

        # Group tokens into lines based on baseline vertical proximity
        lines: List[List[WordToken]] = []
        sorted_tokens = sorted(header_tokens, key=lambda t: (t.ymin, t.xmin))
        
        for t in sorted_tokens:
            placed = False
            for line in lines:
                base_y = sum(lt.ymin for lt in line) / len(line)
                if abs(t.ymin - base_y) <= 8:
                    line.append(t)
                    placed = True
                    break
            if not placed:
                lines.append([t])

        candidates = []
        for line in lines:
            line_text = " ".join([t.text for t in line]).strip()
            if len(line_text) < 4:
                continue

            # Skip lines dominated by dates, times, URLs or page numbers
            if DATE_TIME_RE.search(line_text):
                # If the line is primarily a timestamp/header, discard it
                cleaned_line = DATE_TIME_RE.sub('', line_text).strip()
                if len(cleaned_line) < 4 or re.match(r'^[\s,.:;/-]+$', cleaned_line):
                    continue
                line_text = cleaned_line

            # Check if line is purely a document title (e.g. "TAX INVOICE", "QUOTATION")
            clean_lower = line_text.lower().strip(" :.-")
            if clean_lower in HEADER_NOISE_BLOCKLIST:
                continue

            if any(p.search(line_text) for p in CORPORATE_AFFILIATION_PATTERNS):
                continue

            # Check negative buyer patterns
            is_buyer = any(p.search(line_text) for p in BUYER_PATTERNS)
            if is_buyer:
                continue

            # Check financier blocklist
            is_fin = any(b in line_text.lower() for b in FINANCIER_BLOCKLIST)
            if is_fin:
                continue

            # Check digit ratio: dealer name should not be mostly numbers (GSTIN, Phone, etc.)
            num_digits = sum(c.isdigit() for c in line_text)
            if num_digits / max(1, len(line_text)) > 0.35:
                continue

            # Score prominence
            ymin = min(t.ymin for t in line)
            xmin = min(t.xmin for t in line)
            ymax = max(t.ymax for t in line)
            xmax = max(t.xmax for t in line)
            avg_height = sum(t.height for t in line) / len(line)

            score = avg_height * 1.5 - (ymin * 0.1)
            # Heavy bonus for dealer anchor keywords
            has_anchor = any(k in line_text.lower() for k in DEALER_ANCHORS)
            if has_anchor:
                score += 50.0

            # Huge bonus for explicit business indicators like "M/s" or "Messrs" (including OCR variations Mls, M|s, M/s)
            if re.search(r'(?i)\b(?:m[/\\|1l!]?s\.?|messrs|m/s:?)\b', line_text):
                score += 55.0

            # Disqualify/heavily penalize pure OEM brand banners (e.g. "EICHER TRACTORS", "MAHINDRA TRACTORS")
            clean_low = re.sub(r'[^a-z ]', '', line_text.lower()).strip()
            is_pure_oem = any(
                clean_low == f"{b} tractors" or clean_low == b or clean_low == f"{b} tractor"
                for b in OEM_BRAND_NAMES
            )
            if is_pure_oem:
                score -= 80.0

            candidates.append((score, line_text, (ymin, xmin, ymax, xmax)))

        if not candidates:
            return "UNKNOWN_DEALER", 0.0, None

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_text, bbox = candidates[0]

        # Clean phone numbers, mobiles, and sales contact info
        cleaned = re.sub(r'(?i)\b(?:\+?91[\-\s]?)?[6-9]\d{4}[\s\-]?\d{5}\b', '', best_text)
        cleaned = re.sub(r'\b\d{10}\b', '', cleaned)
        cleaned = re.sub(r'(?i)\b(?:mob|mobile|cell|phone|ph|tel|sales|sales\s*:\s*\d+|w|m|mo)\s*[:.\-]?\s*\d+.*$', '', cleaned)
        # Clean franchise declarations like "Authorised Dealers: EICHER TRACTORS"
        cleaned = re.sub(r'(?i)\b(?:authorised|authorized)\s+[A-Za-z]+\s*[:.\-]?\s*.*$', '', cleaned)
        cleaned = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', cleaned)
        cleaned = re.sub(r'(?i)\b(?:mail|email|e-mail)\s*[:.\-]?\s*\S+', '', cleaned)
        # Strip trailing parenthetical noise (e.g. '(W)', '(M)', etc.)
        cleaned = re.sub(r'\s*\([A-Za-z0-9\s]*\)?', '', cleaned)
        cleaned = re.sub(r'\(.*$', '', cleaned)
        
        # Clean leading prefixes (M/s, Mls, Messrs, Quotation, Estimate, etc.)
        cleaned = re.sub(
            r'(?i)^(?:quotation|estimate|tax invoice|proforma invoice|m[/\\|1l!]?s\.?|messrs|m/s:?)\s*[-:–]?\s*',
            '',
            cleaned
        ).strip(" :;,-_()")

        # Strip trailing OEM brand badges if dealership name already has distinct identity (e.g. "SABAR AGROTECH TAFE" -> "SABAR AGROTECH")
        if len(cleaned.split()) >= 2:
            cleaned = re.sub(
                r'(?i)\s+\b(?:tafe|mahindra|swaraj|sonalika|eicher|massey\s*ferguson|john\s*deere)\b\s*$',
                '',
                cleaned
            ).strip(" :;,-_")

        confidence = min(1.0, max(0.5, best_score / 60.0))
        return cleaned, round(confidence, 2), bbox
