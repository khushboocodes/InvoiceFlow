"""
Tractor Horse Power (HP) Extractor
Combines regex, kW-to-HP conversion, and OEM catalogue imputation fallback.
"""
from typing import Optional, Tuple, Dict, Any
from core.layout.spatial_graph import SpatialWordGraph
from core.normalization.unit_converter import extract_and_normalize_hp

class HorsePowerExtractor:
    def extract(
        self, graph: SpatialWordGraph, matched_model: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float, Optional[Tuple[int, int, int, int]]]:
        """
        Extracts numeric horsepower value.
        Returns (horse_power, confidence, bounding_box).
        """
        # 1. Scan item table tokens for explicit HP
        table_tokens = graph.find_tokens_in_region(ymin=250, xmin=50, ymax=750, xmax=950)
        # Check individual tokens
        for t in table_tokens:
            hp = extract_and_normalize_hp(t.text)
            if hp is not None:
                return hp, 0.95, (t.ymin, t.xmin, t.ymax, t.xmax)

        # Check line-grouped tokens in table (handles separated "35" and "HP Tractor")
        if table_tokens:
            # Group tokens with similar ymin (within 20px)
            sorted_t = sorted(table_tokens, key=lambda t: (t.ymin // 20, t.xmin))
            table_lines = []
            curr_line = []
            last_y = -1
            for t in sorted_t:
                if last_y == -1 or abs(t.ymin - last_y) < 20:
                    curr_line.append(t)
                    last_y = t.ymin
                else:
                    if curr_line:
                        table_lines.append(curr_line)
                    curr_line = [t]
                    last_y = t.ymin
            if curr_line:
                table_lines.append(curr_line)

            for line in table_lines:
                line_str = " ".join([t.text for t in line])
                hp = extract_and_normalize_hp(line_str)
                if hp is not None:
                    ymin = min(t.ymin for t in line)
                    xmin = min(t.xmin for t in line)
                    ymax = max(t.ymax for t in line)
                    xmax = max(t.xmax for t in line)
                    return hp, 0.95, (ymin, xmin, ymax, xmax)

        # 2. Scan full page text
        full_text = graph.get_full_text()
        hp_from_text = extract_and_normalize_hp(full_text)
        if hp_from_text is not None:
            return hp_from_text, 0.90, None

        # 3. Fallback: Impute from verified OEM model catalogue
        if matched_model and "hp" in matched_model:
            imputed_hp = float(matched_model["hp"])
            return imputed_hp, 0.85, None

        # Default standard rural tractor HP
        return 47.0, 0.50, None
