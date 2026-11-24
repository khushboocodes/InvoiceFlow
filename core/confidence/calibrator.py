"""
InvoiceFlow Confidence Calibrator (Task 7.8)
Computes multi-factor calibrated confidence scores in [0.0, 1.0].
"""
from typing import Optional, Dict, Any

class ConfidenceCalibrator:
    @staticmethod
    def calibrate_field(
        ocr_conf: float = 0.85,
        spatial_conf: float = 0.80,
        catalogue_conf: float = 0.90,
        cross_field_valid: bool = True
    ) -> float:
        """
        Calibrated confidence formula:
        C_final = 0.30 * C_ocr + 0.30 * C_spatial + 0.25 * C_catalogue + 0.15 * C_cross
        """
        cross_score = 1.0 if cross_field_valid else 0.40
        c_final = (
            0.30 * ocr_conf +
            0.30 * spatial_conf +
            0.25 * catalogue_conf +
            0.15 * cross_score
        )
        return round(min(1.0, max(0.10, c_final)), 2)
