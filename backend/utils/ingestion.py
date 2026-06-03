"""Stage 1: Document ingestion and preprocessing.

Accepts PDF, PNG, JPG, and JPEG inputs and returns a single
:class:`PreprocessedPage`. The page image has been deskewed, denoised, and
contrast-normalized so downstream OCR and vision stages see consistent input
regardless of source format.

For multi-page PDFs the module scores each page for invoice-quotation
relevance and only emits the highest-scoring page — the train set contains
documents with up to 51 pages where the actual quotation is buried deep
inside a loan packet, so processing every page would blow the latency budget.

PyMuPDF handles every PDF case:

* Digital PDFs — text layer is preserved on :attr:`PreprocessedPage.embedded_text`
  for the chosen page; the page image is rasterized at 300 DPI.
* Scanned PDFs — same rasterization; ``embedded_text`` is None when the text
  layer is empty.

This avoids the ``pdf2image`` + ``poppler`` native dependency entirely while
still satisfying every Stage 1 acceptance criterion.

Validates Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# Acceptance Criterion 1.1: supported extensions
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg"})

# Default rasterization DPI (Acceptance Criterion 1.3).
DEFAULT_DPI = 300

# Maximum pages to consider for relevance scoring. Beyond this we sample
# uniformly to keep a 51-page packet from blowing the latency budget.
MAX_PAGES_TO_SCORE = 12


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class IngestionError(Exception):
    """Base class for all Stage 1 errors."""


class UnsupportedFormatError(IngestionError):
    """Raised when the input extension is not in :data:`SUPPORTED_EXTENSIONS`."""


class CorruptInputError(IngestionError):
    """Raised when the file exists but cannot be parsed by PyMuPDF or Pillow."""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class PreprocessedPage:
    """The single page selected for downstream extraction.

    Attributes:
        image:          Post-preprocess RGB image as a Pillow :class:`PIL.Image.Image`.
        embedded_text:  Text layer extracted by PyMuPDF for the selected page,
                        or ``None`` for image inputs and scanned PDFs.
        page_index:     Zero-based index of the chosen page within the source.
        source_path:    Path of the original input file.
        page_count:     Total page count of the source document (1 for images).
        relevance_score: The internal scoring used to pick the page; useful for
                         debugging multi-page selection.
    """

    image: Image.Image
    embedded_text: Optional[str]
    page_index: int
    source_path: Path
    page_count: int
    relevance_score: float = 0.0


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def load(path: Path | str) -> PreprocessedPage:
    """Load and preprocess the input file, returning the most relevant page.

    Args:
        path: Path to a ``.pdf``, ``.png``, ``.jpg``, or ``.jpeg`` file.

    Raises:
        UnsupportedFormatError: extension not in :data:`SUPPORTED_EXTENSIONS`.
        CorruptInputError: file exists but cannot be opened.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported extension {suffix!r}; expected one of {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if not p.exists():
        raise CorruptInputError(f"Input file does not exist: {p}")

    if suffix == ".pdf":
        return _load_pdf(p)
    return _load_image(p)


# --------------------------------------------------------------------------- #
# Image branch
# --------------------------------------------------------------------------- #


def _load_image(path: Path) -> PreprocessedPage:
    """Load a single image as a one-page document."""
    try:
        with Image.open(path) as raw:
            raw.load()
            rgb = raw.convert("RGB")
    except Exception as exc:
        raise CorruptInputError(f"Failed to open image {path}: {exc}") from exc

    preprocessed = _preprocess_pil(rgb)
    return PreprocessedPage(
        image=preprocessed,
        embedded_text=None,
        page_index=0,
        source_path=path,
        page_count=1,
        relevance_score=1.0,
    )


# --------------------------------------------------------------------------- #
# PDF branch
# --------------------------------------------------------------------------- #


def _load_pdf(path: Path) -> PreprocessedPage:
    """Open a PDF, score pages, and return the highest-scoring one preprocessed."""
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise CorruptInputError(f"Failed to open PDF {path}: {exc}") from exc

    try:
        page_count = doc.page_count
        if page_count == 0:
            raise CorruptInputError(f"PDF has zero pages: {path}")

        # Choose which pages to score. For very large packets we sample.
        candidate_indices = _select_candidate_pages(page_count)

        best_index = candidate_indices[0]
        best_score = -1.0
        best_text: Optional[str] = None

        for idx in candidate_indices:
            page = doc.load_page(idx)
            text = page.get_text("text") or ""
            score = _score_relevance_text(text)
            if score > best_score:
                best_score = score
                best_index = idx
                best_text = text

        # Re-load the chosen page and rasterize it at 300 DPI.
        chosen = doc.load_page(best_index)
        pil_image = _rasterize_page(chosen, dpi=DEFAULT_DPI)
        preprocessed = _preprocess_pil(pil_image)

        # An empty text layer means the page is effectively scanned.
        cleaned_text = best_text.strip() if best_text else ""
        embedded_text = cleaned_text if cleaned_text else None

        return PreprocessedPage(
            image=preprocessed,
            embedded_text=embedded_text,
            page_index=best_index,
            source_path=path,
            page_count=page_count,
            relevance_score=max(best_score, 0.0),
        )
    finally:
        doc.close()


def _select_candidate_pages(page_count: int) -> list[int]:
    """Return the page indices to score for relevance.

    Small docs get every page. Large docs get a uniformly-spaced sample so we
    don't OCR all 51 pages of a loan packet at startup.
    """
    if page_count <= MAX_PAGES_TO_SCORE:
        return list(range(page_count))

    step = page_count / MAX_PAGES_TO_SCORE
    indices = sorted({int(round(i * step)) for i in range(MAX_PAGES_TO_SCORE)})
    return [min(i, page_count - 1) for i in indices]


def _rasterize_page(page: fitz.Page, *, dpi: int) -> Image.Image:
    """Render a PDF page to a Pillow image at the requested DPI.

    Implements Acceptance Criterion 1.3 — the same path is used for both
    digital and scanned PDFs because PyMuPDF rasterizes either correctly.
    """
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    mode = "RGB" if pixmap.n < 4 else "RGBA"
    return Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples).convert("RGB")


# --------------------------------------------------------------------------- #
# Relevance scoring (multi-page PDFs)
# --------------------------------------------------------------------------- #

# Anchor keywords associated with quotation/invoice content. Hindi/Gujarati
# variants included so vernacular pages aren't penalized.
_RELEVANCE_ANCHORS_TEXT = [
    # Currency / totals
    ("total", 2.0),
    ("grand total", 3.0),
    ("net amount", 2.0),
    ("net total", 2.0),
    ("invoice", 2.5),
    ("quotation", 3.0),
    ("proforma", 2.5),
    ("रू", 1.5),  # Hindi rupee
    ("कुल", 2.0),  # Hindi "total"
    ("रकम", 1.5),  # Hindi "amount"
    ("કુલ", 2.0),  # Gujarati "total"
    # Asset / model
    ("model", 1.5),
    ("tractor", 2.5),
    ("hp", 1.5),
    ("h.p", 1.5),
    ("बल", 1.0),  # Hindi HP variant
    ("एचपी", 1.5),  # Hindi "HP"
    ("બળ", 1.0),  # Gujarati HP variant
    # Dealer
    ("dealer", 1.5),
    ("m/s", 1.0),
    ("authorized dealer", 2.0),
    # Currency symbols (single chars, lower weight)
    ("₹", 1.5),
    ("rs.", 1.5),
    ("rs ", 0.8),
    ("inr", 1.0),
]

_DIGIT_BLOCK_RE = re.compile(r"\b\d{4,8}\b")


def _score_relevance_text(text: str) -> float:
    """Score a page's likelihood of being the quotation we want.

    Cheap, keyword-only scoring is used for multi-page selection because we
    cannot afford to OCR every page. A digital PDF's text layer is sufficient;
    for a fully scanned multi-page PDF every page scores 0 here and we fall
    back to picking page 0 (which is the spec's default behavior).
    """
    if not text:
        return 0.0
    haystack = text.lower()

    score = 0.0
    for keyword, weight in _RELEVANCE_ANCHORS_TEXT:
        if keyword in haystack:
            # Multiple hits stack but with diminishing returns — log-linear.
            occurrences = haystack.count(keyword)
            score += weight * (1.0 + 0.4 * (occurrences - 1))

    # Long numeric runs (asset costs, HP figures) are a strong invoice signal.
    digit_blocks = _DIGIT_BLOCK_RE.findall(haystack)
    score += min(len(digit_blocks), 6) * 0.5

    return score


# --------------------------------------------------------------------------- #
# OpenCV preprocessing
# --------------------------------------------------------------------------- #


def _preprocess_pil(image: Image.Image) -> Image.Image:
    """Apply deskew, denoise, and contrast normalization to a PIL image.

    Implements Acceptance Criterion 1.5. Pure-OpenCV implementation; converts
    to BGR for OpenCV, runs the filters, converts back to RGB Pillow.
    """
    rgb = np.asarray(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = _deskew(bgr)
    bgr = _denoise(bgr)
    bgr = _normalize_contrast(bgr)
    rgb_out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_out)


def _deskew(image_bgr: np.ndarray) -> np.ndarray:
    """Detect skew via the minAreaRect of binarized text and rotate to correct.

    Skew angles outside ±10° are treated as false positives (likely rotated
    photos or ID-card images we shouldn't touch).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Invert so text is white on black (foreground for findNonZero).
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = cv2.findNonZero(threshold)
    if coords is None or len(coords) < 50:
        return image_bgr

    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV's minAreaRect angle is in [-90, 0); normalize to small ±°.
    if angle < -45:
        angle = 90 + angle

    # Don't rotate if the angle is trivial or absurd.
    if abs(angle) < 0.5 or abs(angle) > 10:
        return image_bgr

    h, w = image_bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image_bgr,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _denoise(image_bgr: np.ndarray) -> np.ndarray:
    """Lightweight color denoising — preserves edge sharpness for OCR."""
    return cv2.fastNlMeansDenoisingColored(image_bgr, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21)


def _normalize_contrast(image_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the Y channel of YCrCb — boosts faint scans without blowing out colors."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    return cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2BGR)
