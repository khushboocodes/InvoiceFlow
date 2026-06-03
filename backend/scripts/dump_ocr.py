"""Dump cached OCR tokens for each labeled doc, with ground truth annotated.

Used to manually scan the OCR quality and identify which fields are even
recoverable from the rule extractors vs. unrecoverable noise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


BACKEND = Path(__file__).resolve().parent.parent
CACHE_PATH = BACKEND / "models" / ".eval_cache.json"
LABELS_PATH = BACKEND / "tests" / "validation" / "labels.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", type=str, default=None)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))["documents"]

    for label in labels:
        doc_id = label["doc_id"]
        if args.doc and doc_id != args.doc:
            continue
        gt = label["fields"]
        entry = None
        for k, v in cache.items():
            if k.startswith(f"{doc_id}::"):
                entry = v
                break
        if not entry:
            continue
        tokens = entry.get("ocr", [])

        # Reconstruct full text by row.
        rows: dict[int, list] = {}
        for t in tokens:
            y = t["bbox"][1]
            row = y // 25
            rows.setdefault(row, []).append(t)
        for r in rows.values():
            r.sort(key=lambda t: t["bbox"][0])

        print(f"\n{'='*80}\n{doc_id}\n{'='*80}")
        print(f"GT dealer: {gt.get('dealer_name')!r}")
        print(f"GT model:  {gt.get('model_name')!r}")
        print(f"GT HP:     {gt.get('horse_power')}")
        print(f"GT cost:   {gt.get('asset_cost')}")
        print()

        gt_dealer_words = set()
        gt_model_words = set()
        if gt.get("dealer_name"):
            gt_dealer_words = {w.lower().strip(".,()") for w in gt["dealer_name"].split() if len(w) > 2}
        if gt.get("model_name"):
            gt_model_words = {w.lower().strip(".,()") for w in gt["model_name"].split() if len(w) > 2}

        for row_y in sorted(rows.keys()):
            row_tokens = rows[row_y]
            line = " | ".join(t["text"] for t in row_tokens)
            y_real = row_tokens[0]["bbox"][1]

            # Highlight rows containing GT material
            tag = ""
            line_lower = line.lower()
            if any(w in line_lower for w in gt_dealer_words):
                tag = " [DEALER?]"
            elif any(w in line_lower for w in gt_model_words):
                tag = " [MODEL?]"

            print(f"  y={y_real:4d}  {line[:200]}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
