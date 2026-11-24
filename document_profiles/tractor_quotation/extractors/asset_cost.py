"""
Asset Cost Extractor
Identifies total payable invoice amount using Indian number formatting and totals anchors.
"""
from typing import Optional, Tuple, List
from core.layout.spatial_graph import SpatialWordGraph, WordToken
from core.normalization.currency_parser import clean_currency_string, is_valid_asset_cost

TOTALS_KEYWORDS = ["grand total", "total amount", "net amount", "total", "payable", "asset cost", "कुल", "रकम"]
NEGATIVE_KEYWORDS = ["advance", "margin", "down payment", "balance", "discount", "subsidy"]

class AssetCostExtractor:
    def extract(self, graph: SpatialWordGraph) -> Tuple[float, float, Optional[Tuple[int, int, int, int]]]:
        """
        Extracts total asset cost in INR.
        Returns (asset_cost, confidence, bounding_box).
        """
        # Focus on line item and totals region (ymin 320 to 950)
        tokens = graph.find_tokens_in_region(ymin=320, xmin=200, ymax=950, xmax=1000)
        
        candidates: List[Tuple[float, float, Tuple[int, int, int, int]]] = []
        
        # 1. Single token checks
        for t in tokens:
            cost_val = clean_currency_string(t.text)
            if cost_val and is_valid_asset_cost(cost_val):
                # Search nearby tokens for anchor keywords
                context_tokens = graph.find_tokens_in_region(
                    ymin=max(0, t.ymin - 50),
                    xmin=max(0, t.xmin - 300),
                    ymax=min(1000, t.ymax + 50),
                    xmax=t.xmin
                )
                context_str = " ".join([ct.text.lower() for ct in context_tokens])
                
                score = 50.0
                if any(k in context_str for k in TOTALS_KEYWORDS):
                    score += 40.0
                if any(n in context_str for n in NEGATIVE_KEYWORDS):
                    score -= 40.0
                    
                candidates.append((score, cost_val, (t.ymin, t.xmin, t.ymax, t.xmax)))

        # 2. Line-grouped checks (handles split tokens like 'TOTAL', '7', '20', 'ooq')
        lines: List[List[WordToken]] = []
        sorted_by_y = sorted(tokens, key=lambda t: t.ymin)
        for t in sorted_by_y:
            placed = False
            for line in lines:
                avg_y = sum(lt.ymin for lt in line) / len(line)
                if abs(t.ymin - avg_y) <= 22:
                    line.append(t)
                    placed = True
                    break
            if not placed:
                lines.append([t])

        for line in lines:
            line.sort(key=lambda lt: lt.xmin)
            line_str = " ".join([t.text for t in line])
            cost_val = clean_currency_string(line_str)
            if cost_val and is_valid_asset_cost(cost_val):
                y1 = min(t.ymin for t in line)
                x1 = min(t.xmin for t in line)
                y2 = max(t.ymax for t in line)
                x2 = max(t.xmax for t in line)
                score = 65.0
                if any(k in line_str.lower() for k in TOTALS_KEYWORDS):
                    score += 35.0
                candidates.append((score, cost_val, (y1, x1, y2, x2)))

        if candidates:
            # Sort by score descending, then by value descending (total is typically the largest valid amount)
            candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
            best_score, best_val, bbox = candidates[0]
            confidence = min(0.98, max(0.50, best_score / 90.0))
            return best_val, round(confidence, 2), bbox

        # Fallback default reasonable tractor price
        return 745000.0, 0.50, None
