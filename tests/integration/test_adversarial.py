"""
InvoiceFlow Adversarial Robustness Test Suite (Task 10.3)
Verifies containment of malformed, 0-byte, and malicious inputs.
"""
from pathlib import Path
import tempfile
import pytest
from core.pipeline import InvoiceFlowPipeline
from document_profiles.tractor_quotation.schema import ControlledErrorResponse
from core.ingestion.validator import IngestionValidationError

@pytest.fixture(scope="module")
def pipeline():
    return InvoiceFlowPipeline(languages=["en"])

def test_empty_zero_byte_file(pipeline, tmp_path):
    empty_file = tmp_path / "empty_doc.png"
    empty_file.write_bytes(b"")

    # Ingestion validator should raise IngestionValidationError
    with pytest.raises(IngestionValidationError):
        pipeline.process_document(empty_file)

    # Safe method returns ControlledErrorResponse
    res = pipeline.process_document_safe(empty_file)
    assert isinstance(res, ControlledErrorResponse)
    assert res.status == "FAILED"
    assert "empty" in res.error_message.lower()

def test_nonexistent_file(pipeline, tmp_path):
    missing_file = tmp_path / "ghost_file.pdf"
    
    with pytest.raises(IngestionValidationError):
        pipeline.process_document(missing_file)

    res = pipeline.process_document_safe(missing_file)
    assert isinstance(res, ControlledErrorResponse)
    assert res.status == "FAILED"
    assert "not found" in res.error_message.lower()

def test_unsupported_file_format(pipeline, tmp_path):
    txt_file = tmp_path / "malicious.exe"
    txt_file.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00")

    with pytest.raises(IngestionValidationError):
        pipeline.process_document(txt_file)

    res = pipeline.process_document_safe(txt_file)
    assert isinstance(res, ControlledErrorResponse)
    assert res.status == "FAILED"

def test_corrupted_image_header(pipeline, tmp_path):
    corrupt_png = tmp_path / "corrupt.png"
    # Valid PNG magic but truncated corrupted body
    corrupt_png.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRgarbage")

    res = pipeline.process_document_safe(corrupt_png)
    assert isinstance(res, ControlledErrorResponse)
    assert res.status == "FAILED"
