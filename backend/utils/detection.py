"""Stage 2B: Signature and stamp detection (YOLOv8n inference).

Wraps the fine-tuned YOLOv8n model in a small typed API. The detector is a
process-lifetime singleton — instantiated once in ``executable.py`` and
reused per document, so the model weights load only once.

Output is a list of :class:`Detection` records with the class label, an
axis-aligned bounding box in pixel coordinates of the page image as
preprocessed by Stage 1, and a per-detection confidence in [0.0, 1.0].

Validates Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 17.2, 17.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from PIL import Image

from utils.device import DeviceInfo
from utils.ingestion import PreprocessedPage

logger = logging.getLogger(__name__)


# Class index mapping must match the YOLO data.yaml used during training.
CLASS_NAMES: tuple[str, str] = ("signature", "stamp")
CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class Detection:
    """A single signature or stamp detection on a page."""

    cls: Literal["signature", "stamp"]
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float


@dataclass
class _DetectorConfig:
    """Per-class confidence floors and NMS settings."""

    signature_threshold: float
    stamp_threshold: float
    nms_iou: float

    @classmethod
    def load(cls, path: Path) -> "_DetectorConfig":
        if not path.exists():
            logger.warning("Detection config %s not found, using defaults", path)
            return cls(signature_threshold=0.35, stamp_threshold=0.40, nms_iou=0.45)
        data = yaml.safe_load(path.read_text())
        thresholds = data.get("confidence_thresholds", {})
        return cls(
            signature_threshold=float(thresholds.get("signature", 0.35)),
            stamp_threshold=float(thresholds.get("stamp", 0.40)),
            nms_iou=float(data.get("nms_iou", 0.45)),
        )

    def threshold_for(self, cls_name: str) -> float:
        return self.signature_threshold if cls_name == "signature" else self.stamp_threshold


class VisionDetector:
    """YOLOv8n wrapper for signature and stamp detection.

    Args:
        device: Resolved device info from :func:`utils.device.detect`.
        weights_path: Path to the fine-tuned ``.pt`` weights (typically
            ``models/yolov8n_sig_stamp.pt``).
        config_path: Optional path to ``models/detection.yaml`` overriding the
            per-class confidence thresholds. Defaults to ``weights_path.parent / "detection.yaml"``.
    """

    def __init__(
        self,
        device: DeviceInfo,
        weights_path: Path,
        config_path: Path | None = None,
    ):
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found at {weights_path}")

        # Lazy import — keeps test contexts cheap when the detector is mocked.
        from ultralytics import YOLO  # type: ignore[import-not-found]

        self.device = device
        self.weights_path = weights_path
        self.config = _DetectorConfig.load(
            config_path if config_path is not None else weights_path.parent / "detection.yaml"
        )

        logger.info("Loading YOLOv8n weights from %s", weights_path)
        self.model = YOLO(str(weights_path))

        # Pre-warm the model so the first real call is not the slowest.
        # YOLO's lazy init defers the actual model materialization until the
        # first predict call; running a tiny dummy ensures latency budgets are
        # accurate for the first document of a batch run.
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(
                dummy,
                imgsz=640,
                device=device.torch_device_string(),
                verbose=False,
            )
            logger.info("Detector warmed up on %s", device.torch_device_string())
        except Exception as exc:
            logger.warning("Detector warm-up failed (non-fatal): %s", exc)

    def detect(self, page: PreprocessedPage) -> list[Detection]:
        """Run inference on the preprocessed page and return filtered detections.

        Args:
            page: The :class:`PreprocessedPage` produced by Stage 1.

        Returns:
            All detections passing the per-class confidence threshold,
            sorted by confidence (highest first).
        """
        return self._detect_image(page.image)

    def _detect_image(self, pil_image: Image.Image) -> list[Detection]:
        """Lower-level: run inference on a PIL image directly.

        Useful for tests and the demo bridge where we have a PIL image but
        not a full PreprocessedPage. We pass the lowest per-class threshold
        as the model-side ``conf`` filter to cut output size, then re-filter
        in Python so each class gets its own threshold.
        """
        min_threshold = min(self.config.signature_threshold, self.config.stamp_threshold)

        results = self.model.predict(
            pil_image,
            imgsz=640,
            conf=min_threshold,
            iou=self.config.nms_iou,
            device=self.device.torch_device_string(),
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        # Boxes come back as a tensor; pull to CPU numpy for downstream code.
        boxes = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        detections: list[Detection] = []
        for box, cls_id, conf in zip(boxes, cls_ids, confs):
            if not 0 <= cls_id < len(CLASS_NAMES):
                continue
            cls_name = CLASS_NAMES[cls_id]
            # Per-class threshold filter (the model conf=min was a coarse cut).
            if float(conf) < self.config.threshold_for(cls_name):
                continue
            x1, y1, x2, y2 = box
            # Clamp to non-negative integers and ensure the bbox is well-formed
            # (Property 4 — schema validator will reject otherwise).
            ix1 = max(0, int(round(float(x1))))
            iy1 = max(0, int(round(float(y1))))
            ix2 = max(ix1 + 1, int(round(float(x2))))
            iy2 = max(iy1 + 1, int(round(float(y2))))
            detections.append(
                Detection(
                    cls=cls_name,  # type: ignore[arg-type]
                    bbox=(ix1, iy1, ix2, iy2),
                    confidence=float(conf),
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def best_per_class(self, detections: list[Detection]) -> dict[str, Detection | None]:
        """Return the single highest-confidence detection per class, or None.

        Stage 6 needs at most one signature box and at most one stamp box
        per document. This helper produces that selection deterministically.
        """
        result: dict[str, Detection | None] = {"signature": None, "stamp": None}
        for det in detections:
            if result[det.cls] is None:
                result[det.cls] = det
        return result
