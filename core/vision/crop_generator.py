"""
InvoiceFlow Localized Bounding-Crop Generator (Phase 8, Task 8.2)
Extracts targeted regions from invoice images with spatial padding for VLM analysis.
"""
from typing import Tuple, Optional
import numpy as np

def crop_bounding_region(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding_pct: float = 0.15
) -> np.ndarray:
    """
    Crops normalized [ymin, xmin, ymax, xmax] (range 0-1000) region from image with padding.
    """
    h, w = image.shape[:2]
    ymin_n, xmin_n, ymax_n, xmax_n = bbox

    # Convert to pixel coordinates
    y1 = int((ymin_n / 1000.0) * h)
    x1 = int((xmin_n / 1000.0) * w)
    y2 = int((ymax_n / 1000.0) * h)
    x2 = int((xmax_n / 1000.0) * w)

    # Apply padding
    pad_y = int((y2 - y1) * padding_pct)
    pad_x = int((x2 - x1) * padding_pct)

    crop_y1 = max(0, y1 - pad_y)
    crop_x1 = max(0, x1 - pad_x)
    crop_y2 = min(h, y2 + pad_y)
    crop_x2 = min(w, x2 + pad_x)

    return image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
