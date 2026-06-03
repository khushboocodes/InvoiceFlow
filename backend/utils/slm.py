"""Stage 3 Tier 2: Local SLM fallback for low-confidence text extractions.

Wraps Qwen2.5-1.5B-Instruct via Hugging Face ``transformers``. The model is
invoked only when one or more Tier-1 rule extractions fall below the
confidence threshold; it receives OCR text only (no images) and returns a
strict-JSON object covering the four text fields.

We intentionally use the safetensors / transformers path instead of the
GGUF / llama.cpp path that the original spec called for. Reason: prebuilt
``llama-cpp-python`` wheels for Windows tend to require AVX-512, and not
every dev or eval machine has it. ``transformers`` rides on the same torch
install we already use for YOLO and PaddleOCR, so it's a known-good code
path with no extra native dependencies.

Anti-hallucination guard
------------------------
For text fields (``dealer_name``, ``model_name``), any value the SLM
returns is rejected if it does not appear (after whitespace + case
normalization) as a substring of the OCR text. Numeric fields are
sanity-checked against the same domain ranges as Tier-1.

Validates Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 10.1, 10.7
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.device import DeviceInfo
from utils.extraction import COST_MAX, COST_MIN, HP_MAX, HP_MIN

logger = logging.getLogger(__name__)


# Per-call hard timeout; if generation runs over this, return all-nulls.
PER_CALL_TIMEOUT_SECONDS = 12

# Maximum new tokens the SLM may emit per call. Keep tight — the output is
# a tiny JSON object, anything more than ~200 tokens is the model rambling.
MAX_NEW_TOKENS = 200


@dataclass(frozen=True)
class SlmResponse:
    """A single SLM call result.

    Attributes:
        values:        Mapping of field name to extracted value (or None when
                       the SLM declined to answer).
        raw_output:    Raw text the model produced; useful for debugging /
                       error analysis. Always non-empty when ``parsed`` is
                       True.
        parsed:        True when JSON parsing succeeded.
        latency_sec:   Wall-clock seconds spent in the model call.
    """

    values: dict[str, str | int | None]
    raw_output: str
    parsed: bool
    latency_sec: float


class SlmUnavailableError(RuntimeError):
    """Raised when the SLM cannot be loaded.

    The orchestrator catches this and runs without Tier-2 fallback —
    Tier-1 rules alone produce the final extraction.
    """


def try_load_slm(device: DeviceInfo, model_dir: Path) -> Optional["SlmFallback"]:
    """Construct an :class:`SlmFallback`, or return None if unavailable.

    This is the orchestrator-facing entry point: callers get a None-or-engine
    result and never have to handle :class:`SlmUnavailableError` themselves.
    """
    try:
        return SlmFallback(device, model_dir)
    except (SlmUnavailableError, FileNotFoundError) as exc:
        logger.warning("SLM unavailable: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #


SYSTEM_PROMPT = (
    "You are a data extraction engine. From the OCR text of a tractor invoice, "
    "extract ONLY the requested fields. "
    "Return strictly valid JSON. Do not invent values. "
    "If a field is not present in the text, return null. "
    "For text fields, return the value EXACTLY as it appears in the OCR text "
    "(verbatim substring, no paraphrasing). "
    "For numeric fields, return integers only. "
    "No currency symbols, no commas, no decimals."
)


def _build_user_message(ocr_text: str, missing_fields: list[str]) -> str:
    schema_lines = [
        '"dealer_name": "string or null"',
        '"model_name":  "string or null"',
        '"horse_power": "integer or null"',
        '"asset_cost":  "integer or null"',
    ]
    schema = "{\n  " + ",\n  ".join(schema_lines) + "\n}"
    return (
        f'OCR_TEXT:\n"""\n{ocr_text}\n"""\n\n'
        f"REQUESTED_FIELDS: {json.dumps(missing_fields)}\n\n"
        f"OUTPUT_SCHEMA:\n{schema}\n\n"
        f"Return ONLY the JSON object for the REQUESTED_FIELDS. No prose, no markdown."
    )


# Strip markdown fences and other framing the model sometimes wraps around
# the JSON despite our instructions.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_blob(raw: str) -> Optional[str]:
    if not raw:
        return None
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return None
    return match.group(0)


# --------------------------------------------------------------------------- #
# SLM engine
# --------------------------------------------------------------------------- #


class SlmFallback:
    """Local-only SLM extractor backed by transformers + safetensors.

    Args:
        device: Resolved device info from :func:`utils.device.detect`.
            On CUDA we move the model to GPU at fp16; on CPU we use fp32.
        model_dir: Directory containing the safetensors model and tokenizer
            (typically ``backend/models/qwen2.5-1.5b-instruct/``).
    """

    def __init__(self, device: DeviceInfo, model_dir: Path):
        if not model_dir.exists() or not model_dir.is_dir():
            raise FileNotFoundError(
                f"Qwen model directory not found at {model_dir}. "
                "Run `python -m scripts.download_models` once with internet to populate."
            )
        # Spot-check a couple of expected files so we fail loudly here rather
        # than deep inside transformers' loader.
        required = ["config.json", "tokenizer.json", "model.safetensors"]
        missing = [f for f in required if not (model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Qwen model directory is missing files: {missing}. "
                "Re-run `python -m scripts.download_models`."
            )

        # Hard-disable network per the offline contract.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # Lazy import — transformers is heavy.
        try:
            import torch  # noqa: F401  (DLL ordering on Windows)
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SlmUnavailableError(
                "transformers / torch not installed; SLM fallback disabled."
            ) from exc

        self.device = device
        self.model_dir = model_dir

        logger.info("Loading Qwen2.5-1.5B from %s on %s", model_dir, device.kind.value)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                local_files_only=True,
                use_fast=True,
            )
            # FP16 on GPU saves both VRAM and time; FP32 on CPU.
            torch_dtype = "auto" if device.is_gpu else None
            self._model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                local_files_only=True,
                torch_dtype=torch_dtype,
            )
            target_device = device.torch_device_string()
            self._model.to(target_device)
            self._model.eval()
            self._target_device = target_device
        except (OSError, RuntimeError, ValueError) as exc:
            raise SlmUnavailableError(
                f"transformers failed to load Qwen: {exc}. "
                "Pipeline will fall back to Tier-1 rules only."
            ) from exc

        # Note: the chat template is bundled with the tokenizer, so
        # apply_chat_template handles the <|im_start|>...<|im_end|> framing.

    # ------------------------------------------------------------- public
    def refine(self, ocr_text: str, missing_fields: list[str]) -> SlmResponse:
        """Extract the requested fields from OCR text.

        Args:
            ocr_text: Concatenated OCR token stream for the document.
            missing_fields: Fields needing Tier-2 — names from
                ``("dealer_name", "model_name", "horse_power", "asset_cost")``.

        Returns:
            An :class:`SlmResponse`. Even on JSON-parse or timeout failures
            the returned object is structurally valid (with all-nulls).
        """
        if not missing_fields:
            return SlmResponse(values={}, raw_output="", parsed=True, latency_sec=0.0)

        cleaned_fields = [
            f
            for f in missing_fields
            if f in ("dealer_name", "model_name", "horse_power", "asset_cost")
        ]
        if not cleaned_fields:
            return SlmResponse(values={}, raw_output="", parsed=True, latency_sec=0.0)

        user_msg = _build_user_message(ocr_text, cleaned_fields)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # Try once with a tiny non-zero temperature; retry once at greedy.
        first = self._call_with_timeout(messages, temperature=0.1, do_sample=True)
        if first.parsed:
            return self._sanitize(first, cleaned_fields)

        logger.warning("SLM first-pass JSON parse failed; retrying greedy")
        second = self._call_with_timeout(messages, temperature=0.0, do_sample=False)
        if second.parsed:
            return self._sanitize(second, cleaned_fields)

        logger.warning("SLM retry also failed; returning all-nulls")
        return SlmResponse(
            values={f: None for f in cleaned_fields},
            raw_output=second.raw_output or first.raw_output,
            parsed=False,
            latency_sec=first.latency_sec + second.latency_sec,
        )

    # ------------------------------------------------------------ internal
    def _call_with_timeout(
        self,
        messages: list[dict],
        *,
        temperature: float,
        do_sample: bool,
    ) -> SlmResponse:
        """Run one generate call with a wall-clock timeout."""

        def _run() -> str:
            import torch

            prompt_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(prompt_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self._target_device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._target_device)

            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=do_sample,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                if do_sample:
                    gen_kwargs["temperature"] = max(0.05, temperature)
                    gen_kwargs["top_p"] = 0.95
                output_ids = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

            # Strip the prompt prefix from the output.
            new_tokens = output_ids[0, input_ids.shape[1]:]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            try:
                raw = future.result(timeout=PER_CALL_TIMEOUT_SECONDS)
            except FuturesTimeoutError:
                logger.warning("SLM call timed out after %ds", PER_CALL_TIMEOUT_SECONDS)
                return SlmResponse(values={}, raw_output="", parsed=False, latency_sec=PER_CALL_TIMEOUT_SECONDS)
            except Exception as exc:
                logger.warning("SLM call raised: %s", exc)
                return SlmResponse(values={}, raw_output="", parsed=False, latency_sec=time.monotonic() - start)

        latency = time.monotonic() - start

        blob = _extract_json_blob(raw)
        if blob is None:
            return SlmResponse(values={}, raw_output=raw, parsed=False, latency_sec=latency)

        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            return SlmResponse(values={}, raw_output=raw, parsed=False, latency_sec=latency)

        if not isinstance(parsed, dict):
            return SlmResponse(values={}, raw_output=raw, parsed=False, latency_sec=latency)

        return SlmResponse(values=parsed, raw_output=raw, parsed=True, latency_sec=latency)

    # -------------------------------------------------------- post-processing
    def _sanitize(self, response: SlmResponse, requested: list[str]) -> SlmResponse:
        """Apply numeric sanity checks. Substring guard runs in the orchestrator
        because it needs the raw OCR text, which the response object doesn't
        carry.
        """
        cleaned: dict[str, str | int | None] = {}
        for field_name in requested:
            raw_value = response.values.get(field_name)
            if raw_value is None or raw_value == "":
                cleaned[field_name] = None
                continue

            if field_name in ("dealer_name", "model_name"):
                if not isinstance(raw_value, str):
                    cleaned[field_name] = None
                    continue
                cleaned[field_name] = raw_value.strip() or None

            elif field_name == "horse_power":
                value = _coerce_int(raw_value)
                cleaned[field_name] = (
                    value if value is not None and HP_MIN <= value <= HP_MAX else None
                )

            elif field_name == "asset_cost":
                value = _coerce_int(raw_value)
                cleaned[field_name] = (
                    value if value is not None and COST_MIN <= value <= COST_MAX else None
                )

        return SlmResponse(
            values=cleaned,
            raw_output=response.raw_output,
            parsed=response.parsed,
            latency_sec=response.latency_sec,
        )


# --------------------------------------------------------------------------- #
# Public helpers used by the orchestrator
# --------------------------------------------------------------------------- #


def is_substring_of_ocr(value: str, ocr_text: str) -> bool:
    """Anti-hallucination check (Acceptance Criterion 6.5).

    Returns True when ``value`` is grounded in ``ocr_text``. We allow some
    OCR-induced fragmentation: an exact substring match wins immediately,
    otherwise we fall back to a RapidFuzz partial-ratio comparison that
    tolerates missing punctuation / spaces.
    """
    if not value or not ocr_text:
        return False
    normalized_value = re.sub(r"\s+", " ", value).strip().lower()
    normalized_text = re.sub(r"\s+", " ", ocr_text).strip().lower()
    if not normalized_value or not normalized_text:
        return False
    if normalized_value in normalized_text:
        return True
    # RapidFuzz partial-ratio gives the best match of the shorter string
    # anywhere inside the longer; ≥85 catches legitimate OCR-fragmented
    # values without admitting hallucinations.
    try:
        from rapidfuzz import fuzz
        score = fuzz.partial_ratio(normalized_value, normalized_text)
        return score >= 85
    except ImportError:
        return False


def _coerce_int(raw: object) -> Optional[int]:
    """Coerce an SLM-returned value to int. Tolerates strings with currency."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        digits = re.sub(r"[^\d]", "", raw)
        if not digits:
            return None
        try:
            return int(digits)
        except ValueError:
            return None
    return None
