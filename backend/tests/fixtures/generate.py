"""Synthetic fixture generator for ingestion tests.

These helpers produce in-memory PIL images and one-page PDFs that look like
tractor invoices to the relevance scorer. They have no external dependencies
beyond Pillow and PyMuPDF (already required by the pipeline) and are safe to
run offline. Used by ``tests/unit/test_ingestion.py`` and integration tests.
"""

from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 18) -> ImageFont.ImageFont:
    """Best-effort font loader. Falls back to PIL's default bitmap font."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_invoice_image(width: int = 800, height: int = 1100) -> Image.Image:
    """A clean, typed tractor quotation as a PIL image."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    big = _font(24)
    medium = _font(16)
    small = _font(14)

    draw.text((40, 40), "ABC TRACTORS PVT LTD", font=big, fill=(0, 0, 0))
    draw.text((40, 75), "Authorized Mahindra Dealer", font=medium, fill=(0, 0, 0))
    draw.text((40, 100), "QUOTATION", font=big, fill=(0, 0, 0))
    draw.line([(40, 140), (width - 40, 140)], fill=(0, 0, 0), width=1)

    draw.text((40, 170), "Quote No.: Q-2024-0142", font=medium, fill=(0, 0, 0))
    draw.text((40, 200), "Date     : 14 Apr 2024", font=medium, fill=(0, 0, 0))

    draw.text((40, 260), "Tractor Model : Mahindra 575 DI", font=medium, fill=(0, 0, 0))
    draw.text((40, 290), "Horse Power   : 50 HP", font=medium, fill=(0, 0, 0))
    draw.text((40, 320), "Engine        : 4-cyl, 2730 cc", font=medium, fill=(0, 0, 0))

    draw.line([(40, 370), (width - 40, 370)], fill=(0, 0, 0), width=1)
    draw.text((40, 390), "Total Cost: Rs. 5,25,000", font=big, fill=(0, 0, 0))

    draw.text((40, height - 200), "Dealer Signature", font=small, fill=(120, 120, 120))
    draw.text((40, height - 180), "_________________", font=medium, fill=(0, 0, 0))
    draw.text((width - 200, height - 200), "Authorized Stamp", font=small, fill=(120, 120, 120))
    draw.ellipse(
        [(width - 200, height - 170), (width - 80, height - 50)],
        outline=(0, 0, 0),
        width=2,
    )
    return img


def make_noise_page() -> Image.Image:
    """A page that is intentionally NOT a quotation — used to test relevance scoring."""
    img = Image.new("RGB", (400, 560), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "Loan Application", font=_font(16), fill=(0, 0, 0))
    draw.text((20, 60), "Customer photograph attached on next page.", font=_font(12), fill=(0, 0, 0))
    return img


def make_skewed_invoice_image(angle_degrees: float = 4.0) -> Image.Image:
    """An invoice rotated by the specified angle to exercise deskew."""
    base = make_invoice_image()
    return base.rotate(angle_degrees, fillcolor=(255, 255, 255), expand=False)


def write_invoice_image(path: Path) -> Path:
    """Write a synthetic invoice image to disk and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    make_invoice_image().save(path)
    return path


def write_three_page_pdf(path: Path, *, quotation_at_index: int = 1) -> Path:
    """Write a 3-page PDF with the quotation at the given page index.

    The other two pages contain irrelevant noise so the relevance scorer
    has a clear winner to pick. Pages are kept small (300x420 pt ≈ A6) so
    the fixture is fast to rasterize even at 300 DPI.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    try:
        for i in range(3):
            img = make_invoice_image(width=400, height=560) if i == quotation_at_index else make_noise_page()
            from io import BytesIO

            buf = BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            page = doc.new_page(width=300, height=420)
            page.insert_image(page.rect, stream=buf.getvalue())
        doc.save(str(path))
    finally:
        doc.close()
    return path


def write_invoice_pdf_with_text_layer(path: Path) -> Path:
    """Write a 1-page PDF with an embedded text layer (no rasterization needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    try:
        page = doc.new_page(width=595, height=842)  # A4 in points
        page.insert_text((50, 60), "ABC TRACTORS PVT LTD")
        page.insert_text((50, 90), "Tractor Model: Mahindra 575 DI")
        page.insert_text((50, 110), "Horse Power: 50 HP")
        page.insert_text((50, 140), "Total Cost: Rs. 5,25,000")
        page.insert_text((50, 200), "Authorized Dealer Signature")
        doc.save(str(path))
    finally:
        doc.close()
    return path
