"""
InvoiceFlow 2D Spatial Word Graph
Spatial bounding box indexing and geometric proximity queries.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class WordToken:
    def __init__(self, text: str, bbox: Tuple[int, int, int, int], confidence: float = 1.0):
        # bbox format: (ymin, xmin, ymax, xmax) normalized to [0, 1000]
        self.text = text
        self.ymin, self.xmin, self.ymax, self.xmax = bbox
        self.confidence = confidence
        self.cy = (self.ymin + self.ymax) / 2.0
        self.cx = (self.xmin + self.xmax) / 2.0
        self.height = self.ymax - self.ymin
        self.width = self.xmax - self.xmin

class SpatialWordGraph:
    """Manages geometric queries over recognized word tokens on a document page."""
    def __init__(self, tokens: List[WordToken]):
        self.tokens = tokens

    def find_tokens_in_region(
        self, ymin: int, xmin: int, ymax: int, xmax: int
    ) -> List[WordToken]:
        """Finds all tokens intersecting or contained inside specified bounds."""
        res = []
        for t in self.tokens:
            if not (t.ymax < ymin or t.ymin > ymax or t.xmax < xmin or t.xmin > xmax):
                res.append(t)
        return res

    def find_tokens_near_anchor(
        self, anchor_token: WordToken, max_dx: int = 400, max_dy: int = 100, direction: str = "right"
    ) -> List[WordToken]:
        """Finds tokens positioned to the right or below an anchor label."""
        res = []
        for t in self.tokens:
            if t == anchor_token:
                continue
            if direction == "right":
                # Right of anchor on approximately the same horizontal line
                if 0 <= (t.xmin - anchor_token.xmax) <= max_dx and abs(t.cy - anchor_token.cy) <= max_dy:
                    res.append(t)
            elif direction == "below":
                # Directly beneath anchor
                if 0 <= (t.ymin - anchor_token.ymax) <= max_dy and abs(t.cx - anchor_token.cx) <= max_dx:
                    res.append(t)
        # Sort spatially
        if direction == "right":
            res.sort(key=lambda x: x.xmin)
        else:
            res.sort(key=lambda x: x.ymin)
        return res

    def get_full_text(self) -> str:
        """Assembles full page text in reading order (top-to-bottom, left-to-right)."""
        sorted_tokens = sorted(self.tokens, key=lambda t: (t.ymin // 20, t.xmin))
        return " ".join([t.text for t in sorted_tokens])
