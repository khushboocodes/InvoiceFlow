"""Unit tests for Pydantic v2 schemas and bounding box validation."""
import pytest
from document_profiles.tractor_quotation.schema import (
    VisualEntityDetection,
    DocumentMetadata,
    TractorQuotationExtraction
)

def test_valid_tractor_extraction():
    data = {
        "document_id": "test_01.png",
        "dealer_name": "Patel Tractors & Farm Machinery",
        "model_name": "Mahindra 575 DI SP Plus",
        "horse_power": 47.0,
        "asset_cost": 745000.0,
        "dealer_signature": {
            "present": True,
            "bounding_box": [820, 680, 910, 890],
            "confidence": 0.94,
            "page_index": 0
        },
        "dealer_stamp": {
            "present": True,
            "bounding_box": [800, 650, 930, 850],
            "confidence": 0.92,
            "page_index": 0
        },
        "metadata": {
            "execution_time_seconds": 2.15,
            "pipeline_version": "1.0.0",
            "page_count": 1,
            "ocr_engine": "EasyOCR",
            "fallback_triggered": False
        }
    }
    extracted = TractorQuotationExtraction(**data)
    assert extracted.horse_power == 47.0
    assert extracted.dealer_signature.present is True
    assert extracted.dealer_signature.bounding_box == (820, 680, 910, 890)

def test_invalid_bbox_raises_error():
    with pytest.raises(ValueError):
        VisualEntityDetection(
            present=True,
            bounding_box=[1200, 0, 800, 500], # Out of bounds (>1000)
            confidence=0.9
        )
