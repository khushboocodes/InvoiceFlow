"""
InvoiceFlow OCR & Text Recognition Engine
Extracts word tokens with spatial coordinates using EasyOCR/PaddleOCR.
"""
from typing import List, Tuple
import cv2
import numpy as np
from core.layout.spatial_graph import SpatialWordGraph, WordToken
from core.normalization.unicode_cleaner import normalize_text_and_numerals

class OCREngine:
    def __init__(self, languages: List[str] = None):
        import easyocr
        import torch
        self.languages = languages or ["en", "hi"]
        self.use_gpu = torch.cuda.is_available()
        # Automatically use RTX 3050 if CUDA is enabled, else fallback to CPU
        self.reader = easyocr.Reader(self.languages, gpu=self.use_gpu)

    def extract_word_graph(self, image: np.ndarray) -> SpatialWordGraph:
        """
        Runs OCR on page image and builds SpatialWordGraph with normalized [0, 1000] coordinates.
        """
        h, w = image.shape[:2]
        
        # EasyOCR returns list of: (bbox, text, confidence)
        # where bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        results = self.reader.readtext(image)

        tokens: List[WordToken] = []
        for bbox, text, conf in results:
            cleaned_text = normalize_text_and_numerals(text)
            if not cleaned_text:
                continue

            pts = np.array(bbox, dtype=np.float32)
            xmin = int(np.min(pts[:, 0]))
            xmax = int(np.max(pts[:, 0]))
            ymin = int(np.min(pts[:, 1]))
            ymax = int(np.max(pts[:, 1]))

            # Normalize to integer range [0, 1000]
            ymin_norm = max(0, min(1000, int((ymin / h) * 1000)))
            xmin_norm = max(0, min(1000, int((xmin / w) * 1000)))
            ymax_norm = max(0, min(1000, int((ymax / h) * 1000)))
            xmax_norm = max(0, min(1000, int((xmax / w) * 1000)))

            token = WordToken(
                text=cleaned_text,
                bbox=(ymin_norm, xmin_norm, ymax_norm, xmax_norm),
                confidence=float(conf)
            )
            tokens.append(token)

        return SpatialWordGraph(tokens)
