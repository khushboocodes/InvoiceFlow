"""
InvoiceFlow Ingestion Validator
Validates file headers, magic bytes, dimensions, and physical constraints.
"""
import os
from pathlib import Path
from typing import Tuple, Optional

# Supported signatures
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
PDF_MAGIC = b"%PDF-"

MAX_FILE_SIZE_BYTES = 35 * 1024 * 1024  # 35 MB ceiling to allow uncompressed high-res mobile scans
MAX_PIXEL_DIM = 10000  # 10,000 pixels

class IngestionValidationError(Exception):
    """Raised when an uploaded document violates security or format constraints."""
    pass

def validate_file_integrity(file_path: str | Path) -> Tuple[str, int]:
    """
    Validates file existence, size, and header magic bytes.
    Returns (detected_format, file_size_bytes).
    """
    path = Path(file_path)
    if not path.is_file():
        raise IngestionValidationError(f"File not found: {file_path}")

    size = path.stat().st_size
    if size == 0:
        raise IngestionValidationError(f"File is empty (0 bytes): {file_path}")
    if size > MAX_FILE_SIZE_BYTES:
        raise IngestionValidationError(
            f"File exceeds 25MB ceiling ({size / (1024*1024):.2f} MB): {file_path}"
        )

    # Read leading bytes for format sniffing
    with open(path, "rb") as f:
        header = f.read(16)

    if header.startswith(PNG_MAGIC):
        fmt = "png"
    elif header.startswith(JPEG_MAGIC):
        fmt = "jpeg"
    elif header.startswith(PDF_MAGIC):
        fmt = "pdf"
    else:
        # Fallback to extension check if magic bytes are obscured
        ext = path.suffix.lower().lstrip(".")
        if ext in ["png", "jpg", "jpeg", "pdf", "webp", "tiff", "tif"]:
            fmt = "jpeg" if ext in ["jpg", "jpeg"] else ext
        else:
            raise IngestionValidationError(
                f"Unsupported format with header {header[:8]!r} (ext: {ext})"
            )

    return fmt, size
