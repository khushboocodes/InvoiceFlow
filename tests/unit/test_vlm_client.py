"""
Unit tests for MultimodalVLMClient (Phase 8, Tasks 8.3 & 8.4)
"""
from core.ocr.vlm_client import MultimodalVLMClient

def test_vlm_client_offline_availability():
    client = MultimodalVLMClient(api_key="")
    assert client.is_available() is False
    assert client.extract_document("non_existent_file.png") is None

def test_vlm_client_output_sanitization():
    client = MultimodalVLMClient()
    raw_data = {
        "dealer_name": "  Renuka Motors  ",
        "model_name": "Eicher 548",
        "horse_power": "48.0",
        "asset_cost": "860000",
        "dealer_signature_present": True,
        "dealer_stamp_present": True
    }
    sanitized = client._sanitize_vlm_output(raw_data)
    assert sanitized["dealer_name"] == "Renuka Motors"
    assert sanitized["model_name"] == "Eicher 548"
    assert sanitized["horse_power"] == 48.0
    assert sanitized["asset_cost"] == 860000.0
    assert sanitized["confidence"] == 0.98
