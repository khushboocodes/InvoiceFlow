"""Stage 3 Tier 1: Deterministic regex + anchor-based field extraction.

Pulls the four text fields (``dealer_name``, ``model_name``, ``horse_power``,
``asset_cost``) directly from the OCR token stream using anchor keywords and
proximity scoring. The majority of well-structured invoices resolve here
without ever touching the SLM (Stage 3 Tier 2), keeping latency and cost low.

Each extractor returns a :class:`FieldExtraction` carrying the value, a
confidence score in [0, 1], the source tier, and the indices of the OCR
tokens that contributed to the match. Confidence is computed as::

    tier1_conf = anchor_precision * ocr_token_conf * proximity_bonus * sanity_bonus

Where:

* ``anchor_precision`` ∈ [0.5, 1.0] — hand-tuned per anchor based on how
  often it produces wrong values in the wild. ``Total`` alone is noisier
  than ``Grand Total``, so the latter scores higher.
* ``ocr_token_conf`` — minimum recognition confidence among the tokens
  forming the extracted value. We take the worst because one bad character
  can ruin a numeric.
* ``proximity_bonus`` — 1.0 when the value sits adjacent to the anchor;
  decays with token distance. Stops the rule from grabbing arbitrary
  numbers from far down the page.
* ``sanity_bonus`` — 1.0 if the value passes the field's domain range
  (HP in [15, 150], cost in [100k, 5M]); 0.5 if just outside; 0.0 rejects.

Validates Requirements: 4.1, 4.2, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 6.1, 6.2, 6.3, 7.1, 7.2, 7.3
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Sequence

from utils.ocr import OcrToken

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Output type
# --------------------------------------------------------------------------- #


FieldName = Literal["dealer_name", "model_name", "horse_power", "asset_cost"]
SourceTier = Literal["tier1", "tier2", "none"]


@dataclass
class FieldExtraction:
    """A single field's extracted value with provenance and confidence."""

    name: FieldName
    value: str | int | None
    confidence: float
    source: SourceTier
    evidence_token_ids: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #

# Per-field confidence threshold below which Tier-2 fallback fires.
TIER1_CONFIDENCE_THRESHOLD = 0.55

# Numeric sanity ranges (Requirements 6.5 and 7.5).
HP_MIN, HP_MAX = 15, 150
COST_MIN, COST_MAX = 100_000, 5_000_000


# --------------------------------------------------------------------------- #
# Anchor patterns
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Anchor:
    """A keyword anchor and how much we trust it."""

    pattern: re.Pattern[str]
    precision: float  # in [0.5, 1.0], hand-tuned

    @classmethod
    def make(cls, regex: str, precision: float = 0.85, flags: int = re.IGNORECASE) -> "Anchor":
        return cls(pattern=re.compile(regex, flags), precision=precision)


# Horse-power anchors — Latin + Devanagari + Gujarati variants.
# Note: HP can appear glued to the value with no whitespace ("50HP"), so we
# include a "(?<=\d)" lookbehind variant that fires when HP follows a digit
# directly. This is essential for OCR output that doesn't insert spaces.
_HP_ANCHORS: tuple[Anchor, ...] = (
    Anchor.make(r"\bH\.?P\.?\b", precision=0.95),
    Anchor.make(r"(?<=\d)\s*H\.?P\.?\b", precision=0.95),
    Anchor.make(r"\bhp\b", precision=0.92),
    Anchor.make(r"(?<=\d)\s*hp\b", precision=0.92),
    # HP glued to the end of a token (e.g. "50HP", "SOHP")
    Anchor.make(r"HP\b", precision=0.85),
    Anchor.make(r"horse\s*power", precision=0.95),
    Anchor.make(r"एचपी", precision=0.92),
    Anchor.make(r"बल", precision=0.85),
    Anchor.make(r"બળ", precision=0.85),
)

# Asset-cost anchors. "Grand Total" beats "Total" because plain "Total"
# also appears beside subtotals.
_COST_ANCHORS: tuple[Anchor, ...] = (
    Anchor.make(r"grand\s*total", precision=0.97),
    Anchor.make(r"net\s*amount", precision=0.95),
    Anchor.make(r"net\s*total", precision=0.94),
    Anchor.make(r"total\s*cost", precision=0.94),
    Anchor.make(r"total\s*amount", precision=0.92),
    Anchor.make(r"\btotal\b", precision=0.78),
    Anchor.make(r"रकम", precision=0.85),
    Anchor.make(r"कुल", precision=0.85),
    Anchor.make(r"કુલ", precision=0.85),
    Anchor.make(r"₹", precision=0.70),
    Anchor.make(r"\bRs\.?\b", precision=0.70),
    Anchor.make(r"\bINR\b", precision=0.78),
)

# Dealer-name anchors. Letterhead-region heuristic is applied separately.
_DEALER_ANCHORS: tuple[Anchor, ...] = (
    Anchor.make(r"authorized\s+dealer", precision=0.95),
    Anchor.make(r"authorised\s+dealer", precision=0.95),
    Anchor.make(r"\bdealer\s*[:\-]", precision=0.85),
    Anchor.make(r"\bM/s\.?\b", precision=0.90),
    Anchor.make(r"मेसर्स", precision=0.85),
)

# Model-name anchors. The brand keyword list is built dynamically from the
# asset master so we don't hard-code dealer-specific brands here.
_MODEL_ANCHORS: tuple[Anchor, ...] = (
    Anchor.make(r"tractor\s*model", precision=0.95),
    Anchor.make(r"\bmodel\s*[:\-]", precision=0.92),
    Anchor.make(r"\bmodel\s+name\b", precision=0.94),
    Anchor.make(r"asset\s*model", precision=0.92),
)


# --------------------------------------------------------------------------- #
# Number / currency helpers
# --------------------------------------------------------------------------- #

# Matches numbers like 50, 50.5, 5,25,000 (Indian comma style), 525000.
_NUMERIC_RE = re.compile(r"[\d][\d,]*(?:\.\d+)?")
# Strip everything that isn't a digit before integer-coerce.
_DIGITS_ONLY = re.compile(r"[^\d]")


def _to_int(text: str) -> Optional[int]:
    """Convert a numeric string to int, ignoring commas, currency symbols, decimal zeros."""
    cleaned = _DIGITS_ONLY.sub("", text)
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def _proximity_bonus(distance_px: int) -> float:
    """Decay confidence with distance from the anchor.

    Tokens within 80 px get full credit; the bonus drops linearly to ~0.7 by
    300 px. Beyond that we don't trust the association at all.
    """
    if distance_px <= 80:
        return 1.0
    if distance_px >= 300:
        return 0.55
    return 1.0 - (distance_px - 80) / 220 * 0.45


def _sanity_bonus_numeric(value: int, lo: int, hi: int) -> float:
    if lo <= value <= hi:
        return 1.0
    # Within 30% of the band — worth keeping but heavily penalized.
    if lo * 0.7 <= value <= hi * 1.3:
        return 0.5
    return 0.0


# --------------------------------------------------------------------------- #
# Anchor matching primitives
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _AnchorHit:
    """A single anchor match in the OCR token stream."""

    token_index: int
    anchor: Anchor


def _find_anchor_hits(tokens: Sequence[OcrToken], anchors: Iterable[Anchor]) -> list[_AnchorHit]:
    """Return every (token_index, anchor) pair where an anchor pattern matches.

    Real OCR splits multi-word phrases like "Grand Total" or "Authorized Dealer"
    into separate tokens, so we test single tokens AND joined sliding windows
    of 2 and 3 adjacent tokens. When a match comes from a joined window, the
    reported ``token_index`` is the LAST token in the window — downstream
    extractors scan forward from there, which is where the value lives in
    "<anchor> <separator> <value>" style invoices.

    Tokens joined for matching only when they sit on the same approximate row
    (within 30 px y-tolerance), to avoid spurious cross-row matches.
    """
    anchor_list = list(anchors)
    hits: list[_AnchorHit] = []

    def _same_row(a: OcrToken, b: OcrToken, tol: int = 30) -> bool:
        a_cy = (a.bbox[1] + a.bbox[3]) / 2
        b_cy = (b.bbox[1] + b.bbox[3]) / 2
        return abs(a_cy - b_cy) <= tol

    for i, tok in enumerate(tokens):
        # 1-token check.
        for anchor in anchor_list:
            if anchor.pattern.search(tok.text):
                hits.append(_AnchorHit(token_index=i, anchor=anchor))

        # 2-token check (only if next token is on the same row).
        if i + 1 < len(tokens) and _same_row(tok, tokens[i + 1]):
            joined2 = f"{tok.text} {tokens[i + 1].text}"
            for anchor in anchor_list:
                if anchor.pattern.search(joined2):
                    hits.append(_AnchorHit(token_index=i + 1, anchor=anchor))

        # 3-token check.
        if (
            i + 2 < len(tokens)
            and _same_row(tok, tokens[i + 1])
            and _same_row(tok, tokens[i + 2])
        ):
            joined3 = f"{tok.text} {tokens[i + 1].text} {tokens[i + 2].text}"
            for anchor in anchor_list:
                if anchor.pattern.search(joined3):
                    hits.append(_AnchorHit(token_index=i + 2, anchor=anchor))

    # Deduplicate identical (index, anchor) pairs that the joined windows
    # might re-produce.
    seen: set[tuple[int, int]] = set()
    unique: list[_AnchorHit] = []
    for h in hits:
        key = (h.token_index, id(h.anchor))
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique


def _bbox_distance_px(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Euclidean distance between bbox centroids in pixels."""
    ax = (a[0] + a[2]) / 2
    ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2
    by = (b[1] + b[3]) / 2
    return int(round(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5))


def _same_row(a: OcrToken, b: OcrToken, *, row_tolerance: int = 30) -> bool:
    """True when the two tokens sit on roughly the same horizontal row."""
    a_cy = (a.bbox[1] + a.bbox[3]) / 2
    b_cy = (b.bbox[1] + b.bbox[3]) / 2
    return abs(a_cy - b_cy) <= row_tolerance


def _same_row_or_after(anchor: OcrToken, candidate: OcrToken, *, row_tolerance: int = 30) -> bool:
    """True if the candidate token sits on the same row as the anchor (right of)
    or on the next few rows (below)."""
    a_y_center = (anchor.bbox[1] + anchor.bbox[3]) / 2
    c_y_center = (candidate.bbox[1] + candidate.bbox[3]) / 2
    if abs(a_y_center - c_y_center) <= row_tolerance:
        # Same row — must be to the right of the anchor.
        return candidate.bbox[0] >= anchor.bbox[0] - 5
    # Next row(s) — must be below the anchor.
    return c_y_center > a_y_center


# --------------------------------------------------------------------------- #
# Field-specific extractors
# --------------------------------------------------------------------------- #


def _ocr_digit_repair(text: str) -> str:
    """Repair common OCR digit confusions for HP context.

    Only applied when the source token already contains at least one digit
    (i.e. it's clearly a degraded numeric like "2q" or "5o", not a plain word
    like "tomato"). Letters near digits get converted: q→9, o/O→0, l/I→1,
    z/Z→2, s/S→5, b/B→8, g/G→9, t/T→7.
    """
    if not re.search(r"\d", text):
        return text
    table = str.maketrans({"q": "9", "Q": "9", "o": "0", "O": "0",
                           "l": "1", "I": "1", "z": "2", "Z": "2",
                           "s": "5", "S": "5", "b": "8", "B": "8",
                           "g": "9", "G": "9", "t": "7", "T": "7"})
    return text.translate(table)


def _ocr_digit_repair_loose(text: str) -> str:
    """Aggressive variant: applies digit repair even without a digit anchor.

    Used when the surrounding context (e.g. an HP keyword inside the token)
    proves the token is a numeric, so any letters are likely OCR confusions.
    Converts only the most common confusions (S/O/I/L/B/Z/Q/G/T).
    """
    table = str.maketrans({"q": "9", "Q": "9", "o": "0", "O": "0",
                           "l": "1", "I": "1", "z": "2", "Z": "2",
                           "s": "5", "S": "5", "b": "8", "B": "8",
                           "g": "9", "G": "9"})
    return text.translate(table)


def _extract_horse_power(tokens: Sequence[OcrToken]) -> FieldExtraction:
    """HP is a small integer near an HP anchor (left, right, or embedded).

    Also runs an OCR-digit-repair pass on tokens that look like degraded
    numerics ('2q HP' → '29 HP') because horsepower digits are routinely
    misread on stamped invoices.
    """
    candidates: list[tuple[int, float, list[int]]] = []  # (value, score, evidence)

    # Permissive numeric pattern that allows a single letter-as-digit confusion.
    permissive_num_re = re.compile(r"[\d][\dqQoOlIzZsSbBgGtT]{0,2}")

    for hit in _find_anchor_hits(tokens, _HP_ANCHORS):
        anchor_token = tokens[hit.token_index]
        # The HP value can be embedded in the anchor token itself (e.g. "50HP")
        # or live in an adjacent token. Try both directions because in an
        # invoice the value can sit BEFORE the HP keyword ("50 HP") or AFTER
        # the HP label ("HP: 50").
        for anchor_text in (anchor_token.text, _ocr_digit_repair_loose(anchor_token.text)):
            in_token_match = _NUMERIC_RE.search(anchor_text)
            if in_token_match:
                value = _to_int(in_token_match.group(0))
                if value is not None and HP_MIN <= value <= HP_MAX:
                    base_conf = hit.anchor.precision * anchor_token.confidence * 1.0
                    base_conf *= _sanity_bonus_numeric(value, HP_MIN, HP_MAX)
                    if anchor_text != anchor_token.text:
                        base_conf *= 0.85  # penalty for OCR repair
                    candidates.append((value, base_conf, [hit.token_index]))
                    break

        # Otherwise scan a small window in both directions.
        for offset in (-3, -2, -1, 1, 2, 3):
            j = hit.token_index + offset
            if not (0 <= j < len(tokens)):
                continue
            other = tokens[j]
            # Same-row check (left or right of anchor).
            if not _same_row(anchor_token, other):
                continue
            num_match = _NUMERIC_RE.search(other.text)
            if not num_match:
                # Try with OCR digit repair — token like "2q" → "29".
                repaired = _ocr_digit_repair(other.text)
                num_match = _NUMERIC_RE.search(repaired)
                if not num_match:
                    continue
            value = _to_int(num_match.group(0))
            if value is None:
                continue
            sanity = _sanity_bonus_numeric(value, HP_MIN, HP_MAX)
            if sanity == 0.0:
                continue
            distance = _bbox_distance_px(anchor_token.bbox, other.bbox)
            conf = (
                hit.anchor.precision
                * min(anchor_token.confidence, other.confidence)
                * _proximity_bonus(distance)
                * sanity
                * 0.9  # slight penalty since we may have repaired an OCR char
            )
            candidates.append((value, conf, [hit.token_index, j]))

    return _select_best_numeric(candidates, "horse_power")


def _extract_asset_cost(tokens: Sequence[OcrToken]) -> FieldExtraction:
    """Asset cost is a large integer next to a Total/cost anchor.

    Currency symbols, commas, and decimal-zero suffixes are stripped before
    integer coercion (Acceptance Criterion 7.2). We also tolerate trailing
    OCR garbage like "680,000/-", "5,50,000=00", "1106506.00".
    """
    candidates: list[tuple[int, float, list[int]]] = []

    # Permissive numeric regex: long digit groups separated by commas/periods,
    # optionally followed by /- or =00. The trailing /-, =00, or stray digit
    # noise is stripped during sanity check.
    permissive_re = re.compile(r"\d{1,3}(?:,\d{2,3})+|\d{4,}")

    for hit in _find_anchor_hits(tokens, _COST_ANCHORS):
        anchor_token = tokens[hit.token_index]

        for j_offset in range(-1, 7):
            j = hit.token_index + j_offset
            if j < 0 or j >= len(tokens):
                continue
            other = tokens[j]
            if j != hit.token_index and not _same_row_or_after(anchor_token, other):
                continue

            for num_match in permissive_re.finditer(other.text):
                value = _to_int(num_match.group(0))
                if value is None:
                    continue
                sanity = _sanity_bonus_numeric(value, COST_MIN, COST_MAX)
                if sanity == 0.0:
                    continue
                distance = (
                    0 if j == hit.token_index else _bbox_distance_px(anchor_token.bbox, other.bbox)
                )
                conf = (
                    hit.anchor.precision
                    * min(anchor_token.confidence, other.confidence)
                    * _proximity_bonus(distance)
                    * sanity
                )
                evidence = [hit.token_index] if j == hit.token_index else [hit.token_index, j]
                candidates.append((value, conf, evidence))

    # Fallback: scan tokens for a "rupee-shaped" number when no anchor
    # produced a result. We only consider tokens in the BOTTOM HALF of the
    # OCR stream and prefer comma-formatted values, with plain
    # 6-7 digit numbers as a secondary option.
    if not candidates:
        if tokens:
            max_y = max(t.bbox[3] for t in tokens)
            mid_y = max_y * 0.30  # bottom 70%
        else:
            mid_y = 0
        comma_re = re.compile(r"\d{1,3}(?:,\d{2,3})+")
        # Also handle Indian-format with periods/spaces as separators (OCR
        # often confuses commas with periods or full stops): "7.00,000",
        # "11,20,000", "1 06 506".
        mixed_sep_re = re.compile(r"\d{1,3}(?:[\.,\s]\d{2,3}){2,}")
        plain_re = re.compile(r"\b\d{6,7}\b")
        plain_candidates: list[tuple[int, float, list[int]]] = []
        for j, tok in enumerate(tokens):
            if tok.bbox[1] < mid_y:
                continue
            text = tok.text
            # Prefer comma-formatted matches first.
            for num_match in comma_re.finditer(text):
                value = _to_int(num_match.group(0))
                if value is None or not (COST_MIN <= value <= COST_MAX):
                    continue
                conf = 0.55 * tok.confidence
                candidates.append((value, conf, [j]))
            # Mixed-separator (period/comma/space) — common OCR noise.
            for num_match in mixed_sep_re.finditer(text):
                value = _to_int(num_match.group(0))
                if value is None or not (COST_MIN <= value <= COST_MAX):
                    continue
                conf = 0.50 * tok.confidence
                candidates.append((value, conf, [j]))
            # Plain large digit groups (6-7 digits) as fallback.
            for num_match in plain_re.finditer(text):
                raw = num_match.group(0)
                # Skip phone-number-like patterns (start with 9, 8, 7).
                if len(raw) == 7 and raw[0] in "789":
                    continue
                value = _to_int(raw)
                if value is None or not (COST_MIN <= value <= COST_MAX):
                    continue
                conf = 0.40 * tok.confidence
                plain_candidates.append((value, conf, [j]))

        # If no comma-formatted hits, fall back to the LATEST plain candidate
        # (real costs sit at the document's totals row, which is the lowest
        # text on the page).
        if not candidates and plain_candidates:
            plain_candidates.sort(key=lambda x: tokens[x[2][0]].bbox[1], reverse=True)
            candidates.append(plain_candidates[0])

    return _select_best_numeric(candidates, "asset_cost")


def _extract_dealer_name(
    tokens: Sequence[OcrToken],
    page_height: Optional[int] = None,
) -> FieldExtraction:
    """Dealer name from anchors plus a letterhead-region heuristic.

    Three strategies tried in order:
    1. Letterhead: a long uppercase token in the top 15% of the page
       (the strongest signal — a real dealer name almost always appears
       prominently at the top).
    2. Anchored: text after "Authorized Dealer", "M/s", "Dealer:" etc.
       (Authorized Dealer typically refers to the BRAND the dealer
       represents — Mahindra, Eicher — not the dealer's own name. We
       use this as a fallback when no letterhead is found.)
    3. Fallback: longest mostly-uppercase token in the top 25%.
    """
    candidates: list[tuple[str, float, list[int]]] = []

    # Words that signal we've drifted off the dealer name into surrounding
    # boilerplate.
    bridge_words = {"authorised", "authorized", "dealer", "dealers", "for"}
    boilerplate_stops = {
        "implement", "implements", "spares", "spare", "parts",
        "service", "services", "sales", "division", "branch",
        "gstin", "gstino", "gst", "pan", "pano", "address", "phone", "mob", "mobile",
        "tel", "tin", "cin", "subject", "to", "jurisdiction",
    }
    soft_stop_words = {"ltd", "ltd.", "limited", "pvt", "pvt.", "private", "co.", "company"}

    # Strategy 1: letterhead region (top 30% of the page) FIRST. This is the
    # strongest signal — a real dealer name almost always sits prominently
    # at the top of the page in larger type than body text.
    if page_height is not None:
        top_threshold = int(page_height * 0.30)

        def _looks_like_dealer_token(text: str) -> bool:
            """Reject obvious non-dealer tokens (numbers, single chars, GSTIN…)."""
            if len(text) < 3:
                return False
            if not any(c.isalpha() for c in text):
                return False
            # Reject GSTIN/PAN-style alphanumeric blobs.
            if re.fullmatch(r"\d+|[A-Z0-9]{10,}", text):
                return False
            # Reject if mostly digits.
            digits = sum(1 for c in text if c.isdigit())
            letters = sum(1 for c in text if c.isalpha())
            if digits > letters and digits >= 4:
                return False
            return True

        letterhead = [
            (i, tok)
            for i, tok in enumerate(tokens)
            if tok.bbox[1] < top_threshold
            and _looks_like_dealer_token(tok.text)
        ]
        if letterhead:
            line_value, line_evidence = _join_top_line(letterhead)
            if line_value:
                avg_conf = (
                    sum(tokens[i].confidence for i in line_evidence) / len(line_evidence)
                )
                # Letterhead gets a strong prior (0.95) when the line is
                # entirely uppercase letters — that's a hallmark of a logo.
                line_text = " ".join(tokens[i].text for i in line_evidence)
                upper_letters = sum(1 for c in line_text if c.isupper())
                total_letters = sum(1 for c in line_text if c.isalpha())
                upper_frac = (upper_letters / total_letters) if total_letters else 0
                # Don't accept letterhead lines that are obviously non-dealer
                # boilerplate or page metadata.
                if any(stop in line_text.lower() for stop in ("quotation", "invoice", "bill of", "performa")):
                    pass
                else:
                    if upper_frac >= 0.7 and total_letters >= 6:
                        prior = 0.95
                    else:
                        prior = 0.80
                    conf = prior * avg_conf
                    candidates.append((line_value, conf, line_evidence))

    # Strategy 2: anchored extraction. Demoted vs letterhead because
    # "AUTHORIZED DEALER FOR" almost always names a tractor BRAND, not the
    # dealer's own legal name.
    for hit in _find_anchor_hits(tokens, _DEALER_ANCHORS):
        anchor_token = tokens[hit.token_index]
        names_parts: list[str] = []
        evidence_ids: list[int] = [hit.token_index]
        for offset in range(1, 8):
            j = hit.token_index + offset
            if j >= len(tokens):
                break
            tok = tokens[j]
            if not _same_row_or_after(anchor_token, tok):
                break
            text_lower = tok.text.lower().strip(".,:;-")
            if _NUMERIC_RE.fullmatch(tok.text.replace(",", "")):
                break
            if text_lower in boilerplate_stops:
                break
            if text_lower in bridge_words and not names_parts:
                evidence_ids.append(j)
                continue
            if text_lower in bridge_words and names_parts:
                break
            names_parts.append(tok.text)
            evidence_ids.append(j)
            if text_lower in soft_stop_words:
                break
            if len(names_parts) >= 6:
                break
        if names_parts:
            value = " ".join(names_parts).strip(" .,:-")
            value = _dedupe_repeated_phrase(value)
            if 4 <= len(value) <= 80:
                avg_conf = sum(tokens[i].confidence for i in evidence_ids) / len(evidence_ids)
                # Anchored "Authorized Dealer FOR" is downweighted because
                # it usually references the BRAND, not the dealer.
                prior = (
                    0.55 if "authoris" in tokens[hit.token_index].text.lower()
                    or "authoriz" in tokens[hit.token_index].text.lower()
                    else hit.anchor.precision
                )
                conf = prior * avg_conf
                candidates.append((value, conf, evidence_ids))

    return _select_best_text(candidates, "dealer_name")


def _dedupe_repeated_phrase(value: str) -> str:
    """Collapse OCR ghosts where the same phrase appears twice in a row.

    Two passes:
      1) Exact-token-window: ``A B C A B C`` → ``A B C``.
      2) Fuzzy-token: trailing tokens within Levenshtein distance ≤2 of an
         earlier token (case insensitive) are dropped — handles OCR variants
         like "LTD" vs "LtD" or "GORPORATION" vs "CORPORATION".
    """
    parts = value.split()
    if len(parts) < 2:
        return value

    # Exact-window dedupe.
    n = len(parts)
    for w in range(n // 2, 0, -1):
        if parts[-w:] == parts[-2 * w:-w]:
            parts = parts[:-w]
            return _dedupe_repeated_phrase(" ".join(parts))

    # Fuzzy-tail dedupe: if the last token approximately matches one of the
    # earlier tokens (LCS-style or RapidFuzz ratio), drop it.
    try:
        from rapidfuzz import fuzz as _fuzz

        if len(parts) >= 2:
            last = parts[-1].lower().strip(".,:;-")
            for prior in parts[:-1]:
                prior_l = prior.lower().strip(".,:;-")
                if not prior_l or not last:
                    continue
                if _fuzz.ratio(last, prior_l) >= 80:
                    parts.pop()
                    return _dedupe_repeated_phrase(" ".join(parts))
    except ImportError:
        pass

    return " ".join(parts)


def _join_top_line(letterhead: list[tuple[int, OcrToken]]) -> tuple[str, list[int]]:
    """Pick the topmost real text line from candidate letterhead tokens.

    Strategy:
    1. Group tokens within 30 px vertically as one line.
    2. Walk lines from top to bottom; return the FIRST line that is:
       - At least 4 alpha chars
       - Not a skip word (Quotation/Invoice)
       - Not GSTIN/PAN/Mobile/Address pattern
       - Not pure non-Latin (Devanagari noise)
    """
    if not letterhead:
        return "", []

    skip_words = {
        "quotation", "invoice", "performa", "proforma", "bill", "challan",
        "delivery", "order", "estimate", "receipt", "ouotation", "ouotaton",
        "guotation", "qoutation", "estimation", "office", "branch", "dist.",
        "district", "village", "market", "address", "ouotation/estimate",
        # "Authorised Dealer ..." typically describes the BRAND, not the dealer
        "authorised", "authorized", "auth.dealer", "auth-dealer",
        # Address keywords
        "road", "near", "behind", "opposite", "main", "stand", "bus",
        "barhalganj", "barhalgan", "barhalgan,", "pin", "pin-",
        # Contact keywords
        "phone", "phone:", "mobile", "mobile:", "mob", "mob.", "tel",
        "telephone", "fax", "email", "e-mail", "website",
    }

    # Reject lines that look like GSTIN/PAN/address rows.
    skip_patterns = (
        re.compile(r"\bGSTIN", re.IGNORECASE),
        re.compile(r"\bPAN\b", re.IGNORECASE),
        re.compile(r"\bMob[ile.]*\s*[:\-]", re.IGNORECASE),
        re.compile(r"\bPh[one.]*\s*[:\-]", re.IGNORECASE),
        re.compile(r"\bTel[ephone.]*\s*[:\-]", re.IGNORECASE),
        re.compile(r"\bemail", re.IGNORECASE),
        re.compile(r"\bWebsite", re.IGNORECASE),
        re.compile(r"\bAddress", re.IGNORECASE),
        re.compile(r"\bSubject\s*to", re.IGNORECASE),
        re.compile(r"Sub[ej]ec[lt]?\s+[lt]o", re.IGNORECASE),  # "Subect to", "Subecl lo"
        re.compile(r"[JI][un][rni][nri]sdiction", re.IGNORECASE),  # "Jurisdiction" + OCR variants
        re.compile(r"\d{6}"),  # 6-digit pin code or large numeric ID
        re.compile(r"\d{4,}-\d{4,}"),  # phone numbers like 98123-10100
        re.compile(r"Off[il]ce", re.IGNORECASE),  # "Office" / "Offlce" OCR variant
        re.compile(r"Estimate", re.IGNORECASE),
        re.compile(r"\bPin[\s\-,.]?\d", re.IGNORECASE),
        re.compile(r"\b(Road|Street|Lane|Bazar|Bazaar|Nagar|Marg|Pin|City)\b", re.IGNORECASE),
        re.compile(r"\bDealer[\s\-:]*", re.IGNORECASE),
    )

    lines: dict[int, list[tuple[int, OcrToken]]] = {}
    for i, tok in letterhead:
        bucket = tok.bbox[1] // 30
        lines.setdefault(bucket, []).append((i, tok))

    def line_qualifies(bucket_tokens: list[tuple[int, OcrToken]]) -> bool:
        text = " ".join(t.text for _, t in bucket_tokens)
        if any(w in text.lower() for w in skip_words):
            return False
        if any(p.search(text) for p in skip_patterns):
            return False
        # Need at least one Latin alpha word of 4+ chars and decent
        # letter-density.  Single-word lines like "Dnahindra Y/?" are too
        # ambiguous to be a dealer name.
        latin_words = re.findall(r"[A-Za-z]{4,}", text)
        if len(latin_words) < 2:
            return False
        # Need decent letter density.
        alpha = sum(1 for c in text if c.isalpha())
        if alpha < 8:
            return False
        return True

    # Walk buckets top→bottom and return the first qualifying one.
    for bucket_y in sorted(lines.keys()):
        bts = lines[bucket_y]
        if not line_qualifies(bts):
            continue
        bts.sort(key=lambda x: x[1].bbox[0])

        # Within the bucket, prefer the longest token if it covers most of
        # the line; else fall back to a joined unique-word string.
        longest_token = max(bts, key=lambda x: len(x[1].text))
        longest_text = longest_token[1].text.strip(" .,:-")
        other_words: list[str] = []
        for _, tok in bts:
            if tok is longest_token[1]:
                continue
            w = tok.text.strip(" .,:-")
            if w and w.lower() not in longest_text.lower():
                other_words.append(w)
        if longest_text and len(longest_text) >= 4:
            raw_text = longest_text + (" " + " ".join(other_words) if other_words else "")
        else:
            raw_text = " ".join(tok.text for _, tok in bts).strip(" .,:-")
        raw_text = _dedupe_exact_window(raw_text)
        evidence = [i for i, _ in bts]
        if 4 <= len(raw_text) <= 80:
            return raw_text, evidence
    return "", []


def _dedupe_exact_window(value: str) -> str:
    """Conservative dedupe: only collapse exactly-repeated token windows."""
    parts = value.split()
    if len(parts) < 2:
        return value
    n = len(parts)
    for w in range(n // 2, 0, -1):
        if parts[-w:] == parts[-2 * w:-w]:
            parts = parts[:-w]
            return _dedupe_exact_window(" ".join(parts))
    return " ".join(parts)


def _extract_model_name(
    tokens: Sequence[OcrToken],
    brand_keywords: Sequence[str] = (),
) -> FieldExtraction:
    """Model name lives next to a Model/Tractor anchor or starts with a brand keyword."""
    candidates: list[tuple[str, float, list[int]]] = []

    # Strategy 1: explicit anchors.
    for hit in _find_anchor_hits(tokens, _MODEL_ANCHORS):
        anchor_token = tokens[hit.token_index]
        parts: list[str] = []
        evidence_ids: list[int] = [hit.token_index]
        for offset in range(1, 5):
            j = hit.token_index + offset
            if j >= len(tokens):
                break
            tok = tokens[j]
            if not _same_row_or_after(anchor_token, tok):
                break
            parts.append(tok.text)
            evidence_ids.append(j)
            if len(parts) >= 4:
                break
        value = " ".join(parts).strip(" .,:-")
        value = _strip_model_trailers(value)
        if not value or len(value) < 3:
            continue
        # A real model name should contain at least one digit run.
        if not re.search(r"\d", value):
            continue
        avg_conf = sum(tokens[i].confidence for i in evidence_ids) / len(evidence_ids)
        conf = hit.anchor.precision * avg_conf
        candidates.append((value, conf, evidence_ids))

    # Strategy 2: brand keyword match.
    if brand_keywords:
        brand_re = re.compile(
            r"\b(" + "|".join(re.escape(b) for b in brand_keywords) + r")\b",
            flags=re.IGNORECASE,
        )
        # Words that signal we've drifted into dealer / address text rather
        # than a model identifier.
        model_stop_words = {
            "dist", "dist.", "district", "tractors", "tractor", "implement",
            "implements", "spares", "spare", "parts", "service", "services",
            "sales", "division", "branch", "office", "ltd", "ltd.", "limited",
            "pvt", "pvt.", "private", "agro", "industries", "corporation",
            "authorised", "authorized", "dealer", "dealers", "for", "address",
            "phone", "mob", "mobile", "tel", "gstin", "gstino", "pan", "tin",
            "cin", "subject", "to", "jurisdiction", "h.p.", "hp",
            "horsepower", "engine", "accessories",
        }
        for i, tok in enumerate(tokens):
            if not brand_re.search(tok.text):
                continue
            # Consume up to 6 following tokens on the same row.
            parts = [tok.text]
            evidence_ids = [i]
            saw_digit = bool(re.search(r"\d", tok.text))
            for offset in range(1, 7):
                j = i + offset
                if j >= len(tokens):
                    break
                t2 = tokens[j]
                if not _same_row_or_after(tok, t2):
                    break
                # Stop on dealer/address-y words.
                t2_lower = t2.text.lower().strip(".,:;-()")
                if t2_lower in model_stop_words:
                    break
                # Stop on the same brand keyword reappearing — that signals
                # a duplicated logo or letterhead echo, not real model text.
                if j > i and brand_re.fullmatch(t2.text.strip()):
                    break
                parts.append(t2.text)
                evidence_ids.append(j)
                if re.search(r"\d", t2.text):
                    saw_digit = True
                if len(parts) >= 6:
                    break
            value = " ".join(parts).strip(" .,:-")
            value = _strip_model_trailers(value)
            value = _dedupe_repeated_phrase(value)
            if len(value) < 3:
                continue
            # A real model name should contain at least one digit; without
            # one, the candidate gets a much lower confidence so anchored or
            # numeric-bearing brand matches win first.
            if not re.search(r"\d", value):
                # Allow brand-only hits as last-resort with conf=0.40.
                if not saw_digit:
                    avg_conf = sum(tokens[k].confidence for k in evidence_ids) / len(evidence_ids)
                    candidates.append((value, 0.40 * avg_conf, evidence_ids))
                continue
            # Reject values containing stop words even if they snuck through.
            if any(w in value.lower() for w in ("dist.", "ahmedabad", "spares", "implement")):
                continue
            avg_conf = sum(tokens[k].confidence for k in evidence_ids) / len(evidence_ids)
            # Brand match is a strong signal but no anchor — give it 0.85 prior.
            conf = 0.85 * avg_conf
            candidates.append((value, conf, evidence_ids))

    return _select_best_text(candidates, "model_name")


def _strip_model_trailers(value: str) -> str:
    """Trim trailing OCR garbage from a model name candidate.

    Drops trailing tokens that are obvious noise: pure punctuation, single
    Devanagari/Gujarati glyphs, '!' or '#' followed by digits, etc.
    """
    parts = value.split()
    while parts:
        last = parts[-1]
        # Pure punctuation or single non-Latin char.
        if not re.search(r"[A-Za-z0-9]", last) or len(last) <= 1:
            parts.pop()
            continue
        # Token contains non-Latin script (e.g. Devanagari) — likely OCR noise.
        if re.search(r"[\u0900-\u097F\u0A80-\u0AFF]", last):
            parts.pop()
            continue
        # Trailing tokens like "HP." or "39!" or "#" — strip.
        if re.fullmatch(r"[hH][pP]\W*", last) or re.fullmatch(r"\W+\d+\W*", last):
            parts.pop()
            continue
        break
    return " ".join(parts).strip(" .,:-")


# --------------------------------------------------------------------------- #
# Top-level orchestrator
# --------------------------------------------------------------------------- #


def extract_text_fields(
    tokens: Sequence[OcrToken],
    embedded_text: Optional[str] = None,
    brand_keywords: Sequence[str] = (),
    page_height: Optional[int] = None,
) -> dict[str, FieldExtraction]:
    """Run all four Tier-1 extractors and return a dict keyed by field name.

    Always returns all four keys; missing fields have ``value=None``,
    ``confidence=0.0``, ``source="none"``.

    Args:
        tokens: OCR tokens from :class:`utils.ocr.OcrEngine.extract`.
        embedded_text: Optional embedded text layer from a digital PDF
            (currently unused — included for forward compatibility).
        brand_keywords: List of tractor brand strings from the asset master.
            Used by ``_extract_model_name`` for keyword-based fallback.
        page_height: Optional page height in pixels. Enables the
            letterhead-region dealer-name heuristic.

    Returns:
        Dict with keys ``dealer_name``, ``model_name``, ``horse_power``,
        ``asset_cost``.
    """
    return {
        "dealer_name": _extract_dealer_name(tokens, page_height=page_height),
        "model_name": _extract_model_name(tokens, brand_keywords=brand_keywords),
        "horse_power": _extract_horse_power(tokens),
        "asset_cost": _extract_asset_cost(tokens),
    }


def derive_hp_from_model(model_value: Optional[str]) -> Optional[int]:
    """Parse an HP integer out of a model_name string when explicit HP fails.

    Many tractor invoices encode HP inside the model designation:
    ``"MAHINDRA YUVO TECH + 405 DI (HP-39)"`` → 39
    ``"MAHINDRA Tech # 405 D1 HP. 39!"`` → 39
    ``"3805P+ 40HP"`` → 40
    ``"MF 245 DI 50 HP"`` → 50
    ``"PT-439"`` → 41 (last 2 digits of model number, ONLY for known patterns)

    Only fires when the value is plausibly an HP (15–150).
    """
    if not model_value:
        return None
    # Common forms — order matters, more specific patterns first.
    patterns = (
        re.compile(r"\bH[\.\s]*P[\.\-:\s]+(\d{2,3})\b", re.IGNORECASE),  # "HP. 39", "H.P. 39"
        re.compile(r"\(\s*HP[-:\s]*(\d{2,3})\s*\)", re.IGNORECASE),       # "(HP-39)"
        re.compile(r"\b(\d{2,3})\s*H[\.\s]*P\b", re.IGNORECASE),          # "39 HP", "39 H.P"
        re.compile(r"\b(\d{2,3})HP\b", re.IGNORECASE),                    # "40HP"
    )
    for pat in patterns:
        m = pat.search(model_value)
        if m:
            try:
                v = int(m.group(1))
            except ValueError:
                continue
            if HP_MIN <= v <= HP_MAX:
                return v
    return None


# --------------------------------------------------------------------------- #
# Helpers used by all extractors
# --------------------------------------------------------------------------- #


def _select_best_numeric(
    candidates: list[tuple[int, float, list[int]]], field_name: FieldName
) -> FieldExtraction:
    if not candidates:
        return FieldExtraction(name=field_name, value=None, confidence=0.0, source="none")
    candidates.sort(key=lambda x: x[1], reverse=True)
    value, confidence, evidence = candidates[0]
    return FieldExtraction(
        name=field_name,
        value=value,
        confidence=min(1.0, max(0.0, confidence)),
        source="tier1",
        evidence_token_ids=evidence,
    )


def _select_best_text(
    candidates: list[tuple[str, float, list[int]]], field_name: FieldName
) -> FieldExtraction:
    if not candidates:
        return FieldExtraction(name=field_name, value=None, confidence=0.0, source="none")
    candidates.sort(key=lambda x: x[1], reverse=True)
    value, confidence, evidence = candidates[0]
    return FieldExtraction(
        name=field_name,
        value=value,
        confidence=min(1.0, max(0.0, confidence)),
        source="tier1",
        evidence_token_ids=evidence,
    )


def fields_below_threshold(
    extractions: dict[str, FieldExtraction],
    threshold: float = TIER1_CONFIDENCE_THRESHOLD,
) -> list[str]:
    """Return field names whose Tier-1 confidence is below the threshold.

    Used by the orchestrator to decide which fields to retry via the
    Tier-2 SLM fallback (Stage 3 Tier 2, Task 18).
    """
    return [name for name, fx in extractions.items() if fx.confidence < threshold]
