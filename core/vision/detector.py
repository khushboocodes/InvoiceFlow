"""
InvoiceFlow Visual Intelligence Detector (Phase 4)
Detects rubber stamps and authorized signatures using trained YOLOv8 model
with automatic fallback to HSV chromatic segmentation and stroke analysis.
"""
from pathlib import Path
from typing import Tuple, Optional, Any
import cv2
import numpy as np
from document_profiles.tractor_quotation.schema import VisualEntityDetection

YOLO_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "yolo_stamp_sig.pt"
_YOLO_MODEL = None

def get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None and YOLO_WEIGHTS_PATH.is_file():
        try:
            from ultralytics import YOLO
            import torch
            _YOLO_MODEL = YOLO(str(YOLO_WEIGHTS_PATH))
        except Exception:
            _YOLO_MODEL = False
    return _YOLO_MODEL if _YOLO_MODEL is not False else None

def detect_stamp_and_signature(
    image: np.ndarray, page_index: int = 0, word_graph: Optional[Any] = None
) -> Tuple[VisualEntityDetection, VisualEntityDetection]:
    """
    Detects dealer rubber stamps and authorized signatures on an invoice page.
    Returns (dealer_signature, dealer_stamp).
    """
    from typing import Any
    h, w = image.shape[:2]
    stamp_detection = None
    sig_detection = None

    # --- Tier 1: YOLOv8 Deep Object Detection ---
    yolo = get_yolo_model()
    if yolo is not None:
        try:
            results = yolo.predict(image, imgsz=640, conf=0.20, verbose=False)
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                best_stamp_conf = 0.0
                best_sig_conf = 0.0
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy

                    ymin_norm = max(0, min(1000, int((y1 / h) * 1000)))
                    xmin_norm = max(0, min(1000, int((x1 / w) * 1000)))
                    ymax_norm = max(0, min(1000, int((y2 / h) * 1000)))
                    xmax_norm = max(0, min(1000, int((x2 / w) * 1000)))
                    bbox = (ymin_norm, xmin_norm, ymax_norm, xmax_norm)

                    box_h = ymax_norm - ymin_norm
                    box_w = xmax_norm - xmin_norm
                    aspect = float(box_w) / max(1, box_h)

                    # Class 1: Stamp - Real rubber seals are circular, oval, or compact rectangular (aspect 0.4 to 3.5, height >= 35)
                    if cls_id == 1 and conf > best_stamp_conf:
                        if conf >= 0.50 and ymin_norm >= 350 and box_h >= 35 and box_w >= 40 and 0.40 <= aspect <= 3.5:
                            best_stamp_conf = conf
                            stamp_detection = VisualEntityDetection(
                                present=True,
                                bounding_box=bbox,
                                confidence=round(conf, 2),
                                page_index=page_index
                            )
                    # Class 0: Signature - Must be in lower portion (ymin >= 350), sensible size
                    elif cls_id == 0 and conf > best_sig_conf:
                        if conf >= 0.35 and ymin_norm >= 350 and box_h >= 20 and box_w >= 35 and 0.30 <= aspect <= 6.0:
                            best_sig_conf = conf
                            sig_detection = VisualEntityDetection(
                                present=True,
                                bounding_box=bbox,
                                confidence=round(conf, 2),
                                page_index=page_index
                            )
        except Exception:
            pass

    # --- Tier 2: HSV Chromatic Ink Segmentation & Morphological Fallback ---
    if stamp_detection is None or sig_detection is None:
        # Restrict heuristic search to bottom 35% of the page
        y_start = int(h * 0.65)
        footer_crop = image[y_start:h, 0:w]
        fh, fw = footer_crop.shape[:2]

        if stamp_detection is None:
            hsv = cv2.cvtColor(footer_crop, cv2.COLOR_BGR2HSV)
            mask_blue_violet = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([165, 255, 255]))
            mask_red1 = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([15, 255, 255]))
            mask_red2 = cv2.inRange(hsv, np.array([165, 60, 50]), np.array([180, 255, 255]))
            stamp_mask = cv2.bitwise_or(mask_blue_violet, cv2.bitwise_or(mask_red1, mask_red2))
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            stamp_closed = cv2.morphologyEx(stamp_mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(stamp_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_stamp_area = 0
            best_stamp_box = None
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 800 < area < (fh * fw * 0.35):
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    aspect_ratio = float(cw) / ch
                    if 0.5 <= aspect_ratio <= 2.2:
                        if area > best_stamp_area:
                            best_stamp_area = area
                            ymin_norm = int(((y_start + y) / h) * 1000)
                            xmin_norm = int((x / w) * 1000)
                            ymax_norm = int(((y_start + y + ch) / h) * 1000)
                            xmax_norm = int(((x + cw) / w) * 1000)
                            best_stamp_box = (ymin_norm, xmin_norm, ymax_norm, xmax_norm)

            if best_stamp_box:
                # Check if this box overlaps with the detected signature
                is_overlapping_sig = False
                if sig_detection and sig_detection.present and sig_detection.bounding_box:
                    s_ymin, s_xmin, s_ymax, s_xmax = sig_detection.bounding_box
                    b_ymin, b_xmin, b_ymax, b_xmax = best_stamp_box
                    if not (b_ymax < s_ymin or b_ymin > s_ymax or b_xmax < s_xmin or b_xmin > s_xmax):
                        is_overlapping_sig = True

                if not is_overlapping_sig:
                    b_ymin, b_xmin, b_ymax, b_xmax = best_stamp_box
                    cy = (b_ymin + b_ymax) // 2
                    cx = (b_xmin + b_xmax) // 2
                    half_h = max(45, (b_ymax - b_ymin) // 2)
                    half_w = max(55, (b_xmax - b_xmin) // 2)
                    sanitized_stamp_box = (
                        max(450, cy - half_h),
                        max(100, cx - half_w),
                        min(980, cy + half_h),
                        min(980, cx + half_w)
                    )
                    stamp_detection = VisualEntityDetection(
                        present=True,
                        bounding_box=sanitized_stamp_box,
                        confidence=0.82,
                        page_index=page_index
                    )
                else:
                    stamp_detection = VisualEntityDetection(
                        present=False,
                        bounding_box=None,
                        confidence=0.85,
                        page_index=page_index
                    )
            else:
                stamp_detection = VisualEntityDetection(
                    present=False,
                    bounding_box=None,
                    confidence=0.85,
                    page_index=page_index
                )

        if sig_detection is None:
            gray_footer = cv2.cvtColor(footer_crop, cv2.COLOR_BGR2GRAY)
            _, binary_footer = cv2.threshold(gray_footer, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            sig_contours, _ = cv2.findContours(binary_footer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_sig_box = None
            best_sig_score = 0
            for cnt in sig_contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                aspect = float(cw) / max(1, ch)
                # Signature in Indian invoices is typically in right-hand signatory zone:
                # Must be a plausible signature stroke (min 35px height, min 60px width, aspect 0.6-3.5)
                # and located within the right-hand signatory quadrant (y between 30% and 85% of footer, x > 45%)
                if 800 < area < 20000 and 60 < cw < (fw * 0.35) and 35 < ch < (fh * 0.40) and 0.6 <= aspect <= 3.5:
                    in_signatory_zone = (x > fw * 0.45) and (y > fh * 0.25) and ((y + ch) < fh * 0.85)
                    if in_signatory_zone:
                        score = area * 2.0
                        if score > best_sig_score:
                            best_sig_score = score
                            ymin_norm = int(((y_start + y) / h) * 1000)
                            xmin_norm = int((x / w) * 1000)
                            ymax_norm = int(((y_start + y + ch) / h) * 1000)
                            xmax_norm = int(((x + cw) / w) * 1000)
                            best_sig_box = (ymin_norm, xmin_norm, ymax_norm, xmax_norm)

            if best_sig_box:
                sig_detection = VisualEntityDetection(
                    present=True,
                    bounding_box=best_sig_box,
                    confidence=0.89,
                    page_index=page_index
                )
            else:
                sig_detection = VisualEntityDetection(
                    present=False,
                    bounding_box=None,
                    confidence=0.85,
                    page_index=page_index
                )

    return sig_detection, stamp_detection
