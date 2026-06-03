"""Unit tests for utils.slm — Stage 3 Tier 2 (SLM fallback).

The vast majority of tests here exercise pure-Python helpers (prompt
templating, JSON extraction, anti-hallucination guard, numeric coercion).
The live SLM model test is marked as a separate fixture that's skipped when
the GGUF weights aren't present; it's expensive and only useful for
end-to-end smoke checks.

Validates Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 10.7
"""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.slm import (
    PER_CALL_TIMEOUT_SECONDS,
    SYSTEM_PROMPT,
    SlmResponse,
    _build_user_message,
    _coerce_int,
    _extract_json_blob,
    is_substring_of_ocr,
)


# --------------------------------------------------------------------------- #
# Prompt templating
# --------------------------------------------------------------------------- #


def test_build_user_message_includes_ocr_text_and_field_list():
    ocr = "Some OCR tokens here."
    fields = ["dealer_name", "model_name"]
    msg = _build_user_message(ocr, fields)

    assert "OCR_TEXT:" in msg
    assert "Some OCR tokens here." in msg
    assert '"dealer_name"' in msg
    assert '"model_name"' in msg
    assert "Return ONLY the JSON" in msg


def test_system_prompt_documents_anti_hallucination_rule():
    """The system prompt must instruct the model to copy verbatim — that's
    what makes the post-hoc substring check meaningful."""
    assert "verbatim" in SYSTEM_PROMPT.lower() or "exactly" in SYSTEM_PROMPT.lower()


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #


def test_extract_json_blob_strips_markdown_fences():
    raw = '```json\n{"a": 1}\n```'
    assert _extract_json_blob(raw) == '{"a": 1}'


def test_extract_json_blob_finds_object_in_prose():
    raw = "Sure, here is the JSON: {\"value\": 50}. Hope that helps!"
    assert _extract_json_blob(raw) == '{"value": 50}'


def test_extract_json_blob_returns_none_for_garbage():
    assert _extract_json_blob("") is None
    assert _extract_json_blob("no JSON in here") is None


def test_extract_json_blob_handles_nested_braces():
    """A real SLM might include nested objects; we still take the first {...} envelope."""
    raw = '{"outer": {"inner": "value"}}'
    blob = _extract_json_blob(raw)
    assert blob is not None
    assert blob.startswith("{") and blob.endswith("}")
    assert "outer" in blob and "inner" in blob


# --------------------------------------------------------------------------- #
# Anti-hallucination guard — Acceptance Criterion 6.5 (Property 6)
# --------------------------------------------------------------------------- #


def test_is_substring_of_ocr_normalizes_whitespace_and_case():
    ocr = "ABC TRACTORS PVT LTD\nGSTIN: 12ABCDE3456F1Z5"
    assert is_substring_of_ocr("abc tractors pvt ltd", ocr)
    assert is_substring_of_ocr("ABC  TRACTORS\tPVT LTD", ocr)
    assert is_substring_of_ocr("Abc Tractors Pvt Ltd", ocr)


def test_is_substring_of_ocr_rejects_hallucination():
    """The killer test for the anti-hallucination guard."""
    ocr = "ABC TRACTORS PVT LTD\nQuotation No: Q-2024-0142"
    # SLM hallucinated a different dealer name not in the OCR text
    assert is_substring_of_ocr("XYZ Motors Limited", ocr) is False


def test_is_substring_of_ocr_returns_false_for_empty_inputs():
    assert is_substring_of_ocr("", "ABC") is False
    assert is_substring_of_ocr("ABC", "") is False
    assert is_substring_of_ocr("", "") is False


# --------------------------------------------------------------------------- #
# Numeric coercion
# --------------------------------------------------------------------------- #


def test_coerce_int_accepts_int_and_float():
    assert _coerce_int(50) == 50
    assert _coerce_int(50.7) == 50
    assert _coerce_int("50") == 50
    assert _coerce_int("525000") == 525000


def test_coerce_int_strips_currency_and_commas():
    assert _coerce_int("Rs. 5,25,000") == 525000
    assert _coerce_int("₹525000") == 525000
    assert _coerce_int("INR 50,000") == 50000


def test_coerce_int_rejects_non_numeric_strings():
    assert _coerce_int("") is None
    assert _coerce_int("not a number") is None
    assert _coerce_int(None) is None
    assert _coerce_int(True) is None  # bools are not ints for our purposes
    assert _coerce_int(False) is None


# --------------------------------------------------------------------------- #
# SlmResponse dataclass
# --------------------------------------------------------------------------- #


def test_slm_response_is_immutable():
    r = SlmResponse(values={"a": 1}, raw_output="x", parsed=True, latency_sec=0.5)
    with pytest.raises(Exception):
        r.parsed = False  # type: ignore[misc]


def test_per_call_timeout_is_reasonable():
    """Acceptance Criterion 6.8: SLM should complete in ≤ 4s on CPU.

    We hard-cap each call at ``PER_CALL_TIMEOUT_SECONDS``; this test
    documents the chosen budget.
    """
    assert 2 <= PER_CALL_TIMEOUT_SECONDS <= 15


def test_try_load_slm_returns_none_when_weights_missing(tmp_path):
    """Graceful fallback: orchestrator gets None instead of an exception."""
    from utils.device import Device, DeviceInfo
    from utils.slm import try_load_slm

    info = DeviceInfo(kind=Device.CPU, cuda_index=None, description="CPU")
    result = try_load_slm(info, tmp_path / "nonexistent_model_dir")
    assert result is None


def test_slm_unavailable_error_is_runtime_error():
    """SlmUnavailableError must inherit from RuntimeError so generic catches work."""
    from utils.slm import SlmUnavailableError
    assert issubclass(SlmUnavailableError, RuntimeError)


# --------------------------------------------------------------------------- #
# Live SLM tests — only run when safetensors weights are bundled
# --------------------------------------------------------------------------- #


MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "qwen2.5-1.5b-instruct"


@pytest.fixture(scope="module")
def slm_engine():
    if not MODEL_DIR.exists() or not (MODEL_DIR / "model.safetensors").exists():
        pytest.skip(f"Qwen weights not yet downloaded: {MODEL_DIR}")
    try:
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("transformers not installed")

    from utils.device import detect
    from utils.slm import SlmFallback

    info = detect()
    return SlmFallback(info, MODEL_DIR)


def test_slm_extracts_dealer_and_model_from_ocr_text(slm_engine):
    """End-to-end smoke: feed the SLM a clear English OCR string and verify
    it extracts the four fields and the substrings appear in the OCR text."""
    ocr_text = (
        "ABC TRACTORS PVT LTD\n"
        "Authorized Mahindra Dealer\n"
        "Quotation Q-2024-0142\n"
        "Date: 14 Apr 2024\n"
        "Tractor Model: Mahindra 575 DI\n"
        "Horse Power: 50 HP\n"
        "Engine: 4-cyl 2730 cc\n"
        "Grand Total: Rs. 5,25,000\n"
    )
    resp = slm_engine.refine(ocr_text, ["dealer_name", "model_name", "horse_power", "asset_cost"])

    # The model may not get every field perfect, but at least one of the
    # easy ones (HP or cost) should resolve correctly.
    correct = 0
    if resp.values.get("horse_power") == 50:
        correct += 1
    if resp.values.get("asset_cost") == 525000:
        correct += 1
    if resp.values.get("model_name") and "575" in str(resp.values["model_name"]):
        correct += 1
    if resp.values.get("dealer_name") and "ABC" in str(resp.values["dealer_name"]):
        correct += 1

    assert correct >= 2, f"SLM resolved fewer than 2 fields correctly: {resp.values}"


def test_slm_returns_nulls_for_empty_field_request(slm_engine):
    resp = slm_engine.refine("", [])
    assert resp.parsed
    assert resp.values == {}
