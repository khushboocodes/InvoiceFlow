"""
InvoiceFlow Integration Test Suite (Task 10.2)
Verifies end-to-end extraction on real documents.
"""
from pathlib import Path
import pytest
from core.pipeline import InvoiceFlowPipeline
from document_profiles.tractor_quotation.schema import TractorQuotationExtraction

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_DOC = PROJECT_ROOT / "datasets" / "raw" / "172427893_3_pg11.png"

@pytest.fixture(scope="module")
def pipeline():
    return InvoiceFlowPipeline(languages=["en"])

def test_full_pipeline_end_to_end(pipeline):
    if not SAMPLE_DOC.is_file():
        pytest.skip(f"Sample file not found: {SAMPLE_DOC}")

    result = pipeline.process_document(SAMPLE_DOC)
    
    # 1. Output type contract
    assert isinstance(result, TractorQuotationExtraction)
    assert result.document_id == SAMPLE_DOC.name

    # 2. Field assertions
    assert len(result.dealer_name) > 3
    assert "The Odisha Agro" in result.dealer_name or len(result.dealer_name) > 5
    assert "Swaraj" in result.model_name
    assert 40.0 <= result.horse_power <= 60.0
    assert 500000.0 <= result.asset_cost <= 1500000.0

    # 3. Visual entity assertions
    assert result.dealer_signature.present is True
    assert result.dealer_signature.bounding_box is not None
    ymin, xmin, ymax, xmax = result.dealer_signature.bounding_box
    assert 0 <= ymin <= ymax <= 1000
    assert 0 <= xmin <= xmax <= 1000

    assert result.dealer_stamp.present is True
    assert result.dealer_stamp.bounding_box is not None
    s_ymin, s_xmin, s_ymax, s_xmax = result.dealer_stamp.bounding_box
    assert 0 <= s_ymin <= s_ymax <= 1000
    assert 0 <= s_xmin <= s_xmax <= 1000

    # 4. Telemetry assertions
    assert result.metadata.execution_time_seconds > 0.0
    assert result.metadata.pipeline_version == "1.0.0"
