"""End-to-end orchestrator that wires every pipeline stage into one call.

The :class:`Pipeline` object is constructed once with model handles and
master data, then ``process_one`` is called per document. Each stage runs
inside its own try/except so a failure in one stage degrades the output
gracefully rather than crashing the whole document.

Stage flow::

    Ingestion → (OCR, Vision in parallel) → Tier-1 Rules
                                          ↘ Tier-2 SLM (when needed)
              → Normalization → Confidence → Output JSON

Validates Requirements: 8.1, 8.2, 8.3, 9.1, 9.2, 9.3, 14.4, 14.5,
15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 22.1, 22.2, 22.3, 22.4, 22.5
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.confidence import aggregate as aggregate_confidence
from utils.detection import Detection, VisionDetector
from utils.device import DeviceInfo
from utils.extraction import (
    FieldExtraction,
    derive_hp_from_model,
    extract_text_fields,
    fields_below_threshold,
)
from utils.ingestion import (
    CorruptInputError,
    PreprocessedPage,
    UnsupportedFormatError,
    load,
)
from utils.masters import Masters
from utils.normalization import NormalizedField, normalize
from utils.ocr import OcrEngine, OcrToken
from utils.schema import (
    ExtractionResult,
    Fields,
    NumericField,
    TextField,
    VisualField,
    empty_result,
)
from utils.slm import SlmFallback, SlmResponse, is_substring_of_ocr
from utils.stage_cache import StageCache

logger = logging.getLogger(__name__)


# Per-document hard timeout — failsafe against runaway documents in batch.
DOCUMENT_TIMEOUT_SEC = 60

# Commodity-CPU rate used to derive ``cost_estimate_usd``. We default to 0
# (true marginal cost is zero — pure local inference). Set to a non-zero
# value to report rented-CPU costs, e.g. ~0.000022 for AWS t3.large.
COST_PER_SECOND_USD = 0.0


@dataclass
class StageTimings:
    """Per-stage wall-clock breakdown — useful for error analysis + the README."""

    ingestion: float = 0.0
    ocr: float = 0.0
    detection: float = 0.0
    extraction_tier1: float = 0.0
    extraction_tier2: float = 0.0
    normalization: float = 0.0
    confidence: float = 0.0
    serialization: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.ingestion
            + self.ocr
            + self.detection
            + self.extraction_tier1
            + self.extraction_tier2
            + self.normalization
            + self.confidence
            + self.serialization
        )


@dataclass
class Pipeline:
    """Process-lifetime singleton holding model handles and master data.

    Construct once at startup, call :meth:`process_one` per document.
    """

    device: DeviceInfo
    ocr: OcrEngine
    detector: VisionDetector
    slm: Optional[SlmFallback]
    masters: Masters
    cost_per_second_usd: float = COST_PER_SECOND_USD
    stage_cache: Optional[StageCache] = None

    # Last-document diagnostics (handy for tests + the demo bridge).
    last_timings: StageTimings = field(default_factory=StageTimings)
    last_slm_invoked: bool = False

    def enable_cache(self, cache_path: Path) -> None:
        """Turn on the OCR + detection cache. Used by the validation harness."""
        self.stage_cache = StageCache(cache_path)

    # ----------------------------------------------------------------- API
    def process_one(self, input_path: Path) -> ExtractionResult:
        """Run the full pipeline on a single document.

        Returns a Pydantic-validated :class:`ExtractionResult`. Even on
        unrecoverable failures the returned object is well-formed (with
        nulls + an ``error`` string) so batch grading scripts can keep
        moving.
        """
        timings = StageTimings()
        self.last_timings = timings
        self.last_slm_invoked = False
        doc_id = input_path.stem
        run_start = time.monotonic()

        # --- Stage 1: Ingestion --------------------------------------------------
        try:
            t0 = time.monotonic()
            page = load(input_path)
            timings.ingestion = time.monotonic() - t0
        except UnsupportedFormatError as exc:
            return self._error_result(doc_id, f"unsupported_format: {exc}", run_start)
        except CorruptInputError as exc:
            return self._error_result(doc_id, f"corrupt_input: {exc}", run_start)
        except Exception as exc:  # defensive — never let an exception escape
            logger.exception("Unexpected ingestion failure")
            return self._error_result(doc_id, f"ingestion_failed: {exc}", run_start)

        # --- Stage 2A: OCR -------------------------------------------------------
        ocr_tokens: list[OcrToken] = []
        image_hash: Optional[str] = None
        if self.stage_cache is not None:
            try:
                image_hash = StageCache.hash_image(page.image)
                cached = self.stage_cache.get_ocr(doc_id, image_hash)
                if cached is not None:
                    ocr_tokens = cached
                    timings.ocr = 0.001  # marker that cache hit
            except Exception as exc:
                logger.warning("Cache read failed (OCR): %s", exc)

        if not ocr_tokens:
            try:
                t0 = time.monotonic()
                ocr_tokens = self.ocr.extract(page)
                timings.ocr = time.monotonic() - t0
                if self.stage_cache is not None and image_hash is not None:
                    self.stage_cache.set_ocr(doc_id, image_hash, ocr_tokens)
            except Exception as exc:
                logger.warning("OCR failed for %s: %s", doc_id, exc)
                timings.ocr = time.monotonic() - t0

        # --- Stage 2B: Vision ----------------------------------------------------
        detections: list[Detection] = []
        used_detection_cache = False
        if self.stage_cache is not None and image_hash is not None:
            try:
                cached_dets = self.stage_cache.get_detections(doc_id, image_hash)
                if cached_dets is not None:
                    detections = cached_dets
                    used_detection_cache = True
                    timings.detection = 0.001
            except Exception as exc:
                logger.warning("Cache read failed (detections): %s", exc)

        if not used_detection_cache:
            try:
                t0 = time.monotonic()
                detections = self.detector.detect(page)
                timings.detection = time.monotonic() - t0
                if self.stage_cache is not None and image_hash is not None:
                    self.stage_cache.set_detections(doc_id, image_hash, detections)
            except Exception as exc:
                logger.warning("Vision detection failed for %s: %s", doc_id, exc)
                timings.detection = time.monotonic() - t0

        # --- Stage 3 Tier 1: Rule extraction ------------------------------------
        try:
            t0 = time.monotonic()
            tier1 = extract_text_fields(
                ocr_tokens,
                embedded_text=page.embedded_text,
                brand_keywords=self.masters.brand_keywords,
                page_height=page.image.height,
            )
            timings.extraction_tier1 = time.monotonic() - t0
        except Exception as exc:
            logger.exception("Tier-1 extraction failed for %s: %s", doc_id, exc)
            tier1 = self._empty_extractions()

        # --- Stage 3 Tier 2: SLM fallback ---------------------------------------
        try:
            t0 = time.monotonic()
            tier1 = self._maybe_run_slm(tier1, ocr_tokens)
            timings.extraction_tier2 = time.monotonic() - t0
        except Exception as exc:
            logger.warning("Tier-2 SLM fallback failed for %s: %s", doc_id, exc)
            timings.extraction_tier2 = time.monotonic() - t0

        # Post-fallback HP derivation: if HP still missing but model_name has
        # an HP marker baked in (e.g. "(HP-39)"), pull it out.
        try:
            hp_field = tier1.get("horse_power")
            model_field = tier1.get("model_name")

            # First try: HP from model_name string
            if (
                hp_field is not None
                and (hp_field.value is None or hp_field.confidence < 0.3)
                and model_field is not None
                and isinstance(model_field.value, str)
            ):
                derived = derive_hp_from_model(model_field.value)
                if derived is not None:
                    tier1["horse_power"] = FieldExtraction(
                        name="horse_power",
                        value=derived,
                        confidence=max(0.6, min(0.85, model_field.confidence)),
                        source="tier1",
                        evidence_token_ids=model_field.evidence_token_ids,
                    )
                    hp_field = tier1["horse_power"]

            # Second try: scan full OCR text for HP markers when nothing
            # better landed.
            if hp_field is not None and (hp_field.value is None or hp_field.confidence < 0.3):
                ocr_full_text = " ".join(t.text for t in ocr_tokens)
                derived = derive_hp_from_model(ocr_full_text)
                if derived is not None:
                    tier1["horse_power"] = FieldExtraction(
                        name="horse_power",
                        value=derived,
                        confidence=0.55,
                        source="tier1",
                        evidence_token_ids=[],
                    )
        except Exception as exc:
            logger.debug("HP-from-model derivation failed: %s", exc)

        # --- Stage 4: Normalization ---------------------------------------------
        try:
            t0 = time.monotonic()
            normalized = normalize(tier1, self.masters)
            timings.normalization = time.monotonic() - t0
        except Exception as exc:
            logger.exception("Normalization failed for %s: %s", doc_id, exc)
            normalized = self._empty_normalized()

        # --- Stage 5: Confidence + Stage 6: Output ------------------------------
        try:
            t0 = time.monotonic()
            visual_best = self._best_visual_per_class(detections)
            report = aggregate_confidence(normalized, visual_best)
            timings.confidence = time.monotonic() - t0

            t0 = time.monotonic()
            result = self._build_result(
                doc_id=doc_id,
                normalized=normalized,
                visual_best=visual_best,
                doc_confidence=report.document,
                run_start=run_start,
            )
            timings.serialization = time.monotonic() - t0
            return result
        except Exception as exc:
            logger.exception("Output assembly failed for %s", doc_id)
            return self._error_result(doc_id, f"output_failed: {exc}", run_start)

    # ----------------------------------------------------------- internal
    def _maybe_run_slm(
        self,
        tier1: dict[str, FieldExtraction],
        ocr_tokens: list[OcrToken],
    ) -> dict[str, FieldExtraction]:
        """Invoke the SLM for any field whose Tier-1 confidence is below the
        threshold. Returns the same dict shape with Tier-2 values merged in.
        """
        if self.slm is None:
            return tier1

        missing = fields_below_threshold(tier1)
        if not missing:
            return tier1

        # Reconstruct OCR text once.
        ocr_text = " ".join(t.text for t in ocr_tokens)
        if not ocr_text.strip():
            return tier1

        self.last_slm_invoked = True
        response: SlmResponse = self.slm.refine(ocr_text, missing)

        if not response.parsed:
            return tier1

        for name, value in response.values.items():
            if name not in tier1:
                continue
            if value is None or value == "":
                continue

            # Anti-hallucination guard for text fields.
            if name in ("dealer_name", "model_name"):
                if not isinstance(value, str):
                    continue
                if not is_substring_of_ocr(value, ocr_text):
                    logger.debug(
                        "SLM hallucination rejected for %s: %r not in OCR text",
                        name,
                        value,
                    )
                    continue

            # Merge: only overwrite Tier-1 if Tier-1 was below threshold AND
            # SLM produced a non-null value. We assign the SLM value with
            # 0.7 confidence as a midline — strong enough to pass the
            # threshold but not as authoritative as a Tier-1 anchored hit.
            existing = tier1[name]
            if existing.confidence < 0.55 or existing.value is None:
                tier1[name] = FieldExtraction(
                    name=existing.name,
                    value=value,
                    confidence=0.7,
                    source="tier2",
                    evidence_token_ids=existing.evidence_token_ids,
                )

        return tier1

    @staticmethod
    def _best_visual_per_class(
        detections: list[Detection],
    ) -> dict[str, Optional[Detection]]:
        """Pick at most one detection per class, choosing the highest-confidence."""
        out: dict[str, Optional[Detection]] = {"signature": None, "stamp": None}
        for d in detections:
            cls = str(d.cls)
            if cls in out and (out[cls] is None or d.confidence > out[cls].confidence):
                out[cls] = d
        return out

    def _build_result(
        self,
        *,
        doc_id: str,
        normalized: dict[str, NormalizedField],
        visual_best: dict[str, Optional[Detection]],
        doc_confidence: float,
        run_start: float,
    ) -> ExtractionResult:
        """Assemble the Pydantic ExtractionResult from all stage outputs."""

        def _text_field(name: str) -> TextField:
            f = normalized.get(name)
            if f is None:
                return TextField(value=None, confidence=0.0)
            value = str(f.value) if isinstance(f.value, str) else None
            return TextField(value=value, confidence=_clamp(f.confidence))

        def _numeric_field(name: str) -> NumericField:
            f = normalized.get(name)
            if f is None:
                return NumericField(value=None, confidence=0.0)
            value = int(f.value) if isinstance(f.value, int) else None
            return NumericField(value=value, confidence=_clamp(f.confidence))

        def _visual_field(name: str) -> VisualField:
            det = visual_best.get(name)
            if det is None:
                return VisualField(present=False, bbox=None, confidence=0.0)
            return VisualField(
                present=True,
                bbox=tuple(int(c) for c in det.bbox),
                confidence=_clamp(det.confidence),
            )

        elapsed = time.monotonic() - run_start
        return ExtractionResult(
            doc_id=doc_id,
            fields=Fields(
                dealer_name=_text_field("dealer_name"),
                model_name=_text_field("model_name"),
                horse_power=_numeric_field("horse_power"),
                asset_cost=_numeric_field("asset_cost"),
                signature=_visual_field("signature"),
                stamp=_visual_field("stamp"),
            ),
            confidence=_clamp(doc_confidence),
            processing_time_sec=round(elapsed, 2),
            cost_estimate_usd=round(elapsed * self.cost_per_second_usd, 6),
        )

    def _error_result(self, doc_id: str, message: str, run_start: float) -> ExtractionResult:
        elapsed = time.monotonic() - run_start
        result = empty_result(doc_id, error=message)
        # Re-stamp timing fields on the empty result.
        return ExtractionResult(
            doc_id=result.doc_id,
            fields=result.fields,
            confidence=0.0,
            processing_time_sec=round(elapsed, 2),
            cost_estimate_usd=round(elapsed * self.cost_per_second_usd, 6),
            error=message,
        )

    @staticmethod
    def _empty_extractions() -> dict[str, FieldExtraction]:
        return {
            "dealer_name": FieldExtraction("dealer_name", None, 0.0, "none"),
            "model_name": FieldExtraction("model_name", None, 0.0, "none"),
            "horse_power": FieldExtraction("horse_power", None, 0.0, "none"),
            "asset_cost": FieldExtraction("asset_cost", None, 0.0, "none"),
        }

    @staticmethod
    def _empty_normalized() -> dict[str, NormalizedField]:
        return {
            "dealer_name": NormalizedField(value=None, confidence=0.0),
            "model_name": NormalizedField(value=None, confidence=0.0),
            "horse_power": NormalizedField(value=None, confidence=0.0),
            "asset_cost": NormalizedField(value=None, confidence=0.0),
        }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return float(value)
