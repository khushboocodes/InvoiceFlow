"""
Pydantic v2 Output Data Contracts for Tractor Loan Quotation Intelligence.
"""
from typing import Optional, Tuple, List
from pydantic import BaseModel, Field, field_validator

class VisualEntityDetection(BaseModel):
    """Detection contract for visual objects (stamps, signatures)."""
    present: bool = Field(..., description="Binary presence indicator")
    bounding_box: Optional[Tuple[int, int, int, int]] = Field(
        None,
        description="Coordinates formatted as [ymin, xmin, ymax, xmax] normalized to integer range [0, 1000]"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated detection confidence score")
    page_index: int = Field(0, ge=0, description="Zero-based page index where the object is located")

    @field_validator("bounding_box")
    @classmethod
    def validate_bbox_coordinates(cls, v, values):
        present = values.data.get("present", False)
        if present and v is None:
            raise ValueError("Bounding box must be provided when entity is detected as present")
        if v is not None:
            ymin, xmin, ymax, xmax = v
            if not (0 <= ymin <= ymax <= 1000 and 0 <= xmin <= xmax <= 1000):
                raise ValueError(
                    f"Invalid coordinates {v}: coordinates must be in [0, 1000] and ymin<=ymax, xmin<=xmax"
                )
        return v

class DocumentMetadata(BaseModel):
    """Pipeline execution metadata and performance telemetry."""
    execution_time_seconds: float = Field(..., description="Total end-to-end processing latency in seconds")
    pipeline_version: str = Field("1.0.0", description="InvoiceFlow pipeline version")
    page_count: int = Field(1, ge=1, description="Number of pages in the processed document")
    ocr_engine: str = Field("DualTrack-PaddleOCR/EasyOCR", description="Primary OCR engine utilized")
    fallback_triggered: bool = Field(False, description="Whether VLM fallback was invoked")

class TractorQuotationExtraction(BaseModel):
    """
    Official InvoiceFlow Output Schema for Tractor Loan Quotations.
    Contains the 6 mandatory extraction fields.
    """
    document_id: str = Field(..., description="Unique filename or document identifier")
    dealer_name: str = Field(..., min_length=2, max_length=250, description="Normalized name of the tractor dealership")
    model_name: str = Field(..., min_length=2, max_length=150, description="Canonical OEM model designation")
    horse_power: float = Field(..., ge=10.0, le=160.0, description="Rated tractor engine horsepower")
    asset_cost: float = Field(..., ge=50000.0, le=5000000.0, description="Total quotation asset value in INR")
    dealer_signature: VisualEntityDetection = Field(..., description="Authorized signature detection and location")
    dealer_stamp: VisualEntityDetection = Field(..., description="Official dealership stamp detection and location")
    metadata: DocumentMetadata = Field(..., description="Execution telemetry and audit provenance")

class ControlledErrorResponse(BaseModel):
    """Standardized error contract for corrupted or unreadable documents."""
    document_id: str
    status: str = "FAILED"
    error_code: str
    error_message: str
    execution_time_seconds: float
