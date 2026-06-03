"""On-disk cache for the heavy stages (OCR, YOLO).

The OCR + YOLO passes account for ~95% of pipeline wall time. During
iteration on the rule extractors we don't want to re-run them on every
change — a simple JSON cache keyed by ``(doc_id, image_sha256)`` lets us
turn a 12-minute eval into a 30-second one.

The cache is OFF by default in production (``executable.py`` doesn't enable
it). It's enabled by the validation harness via ``Pipeline.enable_cache()``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from utils.detection import Detection
from utils.ocr import OcrToken

logger = logging.getLogger(__name__)


class StageCache:
    """JSON-backed cache for OCR tokens and YOLO detections.

    Thread-safe for incremental writes.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._data: dict[str, dict] = {}
        self._dirty = False
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
                logger.info("Loaded stage cache from %s (%d entries)", path, len(self._data))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Stage cache at %s is corrupt (%s); starting fresh", path, exc)
                self._data = {}

    @staticmethod
    def hash_image(image) -> str:
        """SHA256 of a PIL image's raw RGB bytes — stable per identical input."""
        rgb = image.convert("RGB")
        return hashlib.sha256(rgb.tobytes()).hexdigest()[:16]

    def _key(self, doc_id: str, image_hash: str) -> str:
        return f"{doc_id}::{image_hash}"

    # -------------------------------------------------------------- OCR
    def get_ocr(self, doc_id: str, image_hash: str) -> Optional[list[OcrToken]]:
        with self._lock:
            entry = self._data.get(self._key(doc_id, image_hash))
            if not entry or "ocr" not in entry:
                return None
            return [
                OcrToken(
                    text=row["text"],
                    bbox=tuple(row["bbox"]),
                    confidence=float(row["confidence"]),
                    script=row.get("script", "en"),
                )
                for row in entry["ocr"]
            ]

    def set_ocr(self, doc_id: str, image_hash: str, tokens: list[OcrToken]) -> None:
        with self._lock:
            entry = self._data.setdefault(self._key(doc_id, image_hash), {})
            entry["ocr"] = [
                {
                    "text": t.text,
                    "bbox": list(t.bbox),
                    "confidence": t.confidence,
                    "script": t.script,
                }
                for t in tokens
            ]
            self._dirty = True
            self.flush()

    # -------------------------------------------------------- detections
    def get_detections(self, doc_id: str, image_hash: str) -> Optional[list[Detection]]:
        with self._lock:
            entry = self._data.get(self._key(doc_id, image_hash))
            if not entry or "detections" not in entry:
                return None
            return [
                Detection(
                    cls=row["cls"],
                    bbox=tuple(row["bbox"]),
                    confidence=float(row["confidence"]),
                )
                for row in entry["detections"]
            ]

    def set_detections(self, doc_id: str, image_hash: str, dets: list[Detection]) -> None:
        with self._lock:
            entry = self._data.setdefault(self._key(doc_id, image_hash), {})
            entry["detections"] = [
                {"cls": d.cls, "bbox": list(d.bbox), "confidence": d.confidence}
                for d in dets
            ]
            self._dirty = True
            self.flush()

    # -------------------------------------------------------- persistence
    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)
            self._dirty = False
