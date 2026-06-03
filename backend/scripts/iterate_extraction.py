"""Fast iteration loop for the Tier-1 extractors.

Reads the cached OCR tokens from ``models/.eval_cache.json`` and runs only
the rule extractors against the labeled validation set. No model loading,
no GPU work — pure rule iteration. Each pass takes <1 second.

Run::

    python -m scripts.iterate_extraction
    python -m scripts.iterate_extraction --doc 172448470_3_pg15  # single doc
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz import fuzz

from utils.extraction import (
    derive_hp_from_model,
    extract_text_fields,
)
from utils.masters import load as load_masters
from utils.ocr import OcrToken
from utils.normalization import normalize

logging.basicConfig(level=logging.WARNING)


BACKEND = Path(__file__).resolve().parent.parent
CACHE_PATH = BACKEND / "models" / ".eval_cache.json"
LABELS_PATH = BACKEND / "tests" / "validation" / "labels.json"


def _norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _match_dealer(predicted: Optional[str], gt: Optional[str]) -> bool:
    if gt is None:
        return predicted is None
    if predicted is None:
        return False
    p, g = _norm(predicted), _norm(gt)
    if fuzz.token_set_ratio(p, g) >= 85:
        return True
    if fuzz.partial_ratio(p, g) >= 95 and len(p) >= max(8, len(g) // 2):
        return True
    return False


def _match_model(predicted: Optional[str], gt: Optional[str]) -> bool:
    if gt is None:
        return predicted is None
    if predicted is None:
        return False
    p, g = _norm(predicted), _norm(gt)
    if p == g:
        return True
    gt_nums = re.findall(r"\d+", g)
    gt_alpha = {t for t in g.split() if t.isalpha() and len(t) >= 3}
    p_alpha = {t for t in p.split() if t.isalpha() and len(t) >= 3}
    alpha_overlap = bool(gt_alpha & p_alpha)
    if gt_nums and all(num in p for num in gt_nums) and alpha_overlap:
        return True
    if alpha_overlap:
        long_gt_nums = [n for n in gt_nums if len(n) >= 3]
        if any(num in p for num in long_gt_nums):
            return True
    if fuzz.token_set_ratio(p, g) >= 60 and alpha_overlap:
        return True
    if fuzz.partial_ratio(p, g) >= 80 and alpha_overlap:
        return True
    return False


def _match_num(predicted, gt) -> bool:
    if gt is None:
        return predicted is None
    if predicted is None:
        return False
    if gt == 0:
        return predicted == 0
    return abs(predicted - gt) / abs(gt) <= 0.05


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", type=str, default=None)
    parser.add_argument("--show-tokens", action="store_true")
    args = parser.parse_args()

    # Force UTF-8 output so Devanagari OCR garbage doesn't crash the print loop.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))["documents"]

    masters = load_masters(BACKEND / "data")
    brand_keywords = masters.brand_keywords

    score = {"dealer_name": 0, "model_name": 0, "horse_power": 0, "asset_cost": 0}
    total = 0
    failures: list[dict] = []

    for label in labels:
        doc_id = label["doc_id"]
        if args.doc and doc_id != args.doc:
            continue
        gt = label["fields"]

        # Find cache entry by doc_id prefix
        entry = None
        for key, v in cache.items():
            if key.startswith(f"{doc_id}::"):
                entry = v
                break
        if entry is None:
            continue
        tokens = [
            OcrToken(
                text=row["text"],
                bbox=tuple(row["bbox"]),
                confidence=float(row["confidence"]),
                script=row.get("script", "en"),
            )
            for row in entry.get("ocr", [])
        ]

        if args.show_tokens:
            print(f"\n=== {doc_id} ===")
            for i, t in enumerate(tokens):
                print(f"  [{i:3d}] y={t.bbox[1]:4d} x={t.bbox[0]:4d}  conf={t.confidence:.2f}  '{t.text}'")
            continue

        # Page height: estimate from max y bbox
        page_height = max((t.bbox[3] for t in tokens), default=2000)

        tier1 = extract_text_fields(
            tokens,
            embedded_text=None,
            brand_keywords=brand_keywords,
            page_height=page_height,
        )

        # HP-from-model derivation
        hp_field = tier1.get("horse_power")
        model_field = tier1.get("model_name")
        if (
            hp_field is not None
            and (hp_field.value is None or hp_field.confidence < 0.3)
            and model_field is not None
            and isinstance(model_field.value, str)
        ):
            derived = derive_hp_from_model(model_field.value)
            if derived is not None:
                from utils.extraction import FieldExtraction
                tier1["horse_power"] = FieldExtraction(
                    name="horse_power",
                    value=derived,
                    confidence=max(0.6, min(0.85, model_field.confidence)),
                    source="tier1",
                    evidence_token_ids=model_field.evidence_token_ids,
                )
                hp_field = tier1["horse_power"]

        # HP from full OCR text fallback
        if hp_field is not None and (hp_field.value is None or hp_field.confidence < 0.3):
            ocr_full = " ".join(t.text for t in tokens)
            derived = derive_hp_from_model(ocr_full)
            if derived is not None:
                from utils.extraction import FieldExtraction
                tier1["horse_power"] = FieldExtraction(
                    name="horse_power",
                    value=derived,
                    confidence=0.55,
                    source="tier1",
                    evidence_token_ids=[],
                )

        normalized = normalize(tier1, masters)

        dealer_pred = normalized["dealer_name"].value
        model_pred = normalized["model_name"].value
        hp_pred = normalized["horse_power"].value
        cost_pred = normalized["asset_cost"].value

        matches = {
            "dealer_name": _match_dealer(dealer_pred, gt.get("dealer_name")),
            "model_name": _match_model(model_pred, gt.get("model_name")),
            "horse_power": _match_num(hp_pred, gt.get("horse_power")),
            "asset_cost": _match_num(cost_pred, gt.get("asset_cost")),
        }
        for k, v in matches.items():
            if v:
                score[k] += 1
        total += 1

        any_fail = not all(matches.values())
        if any_fail:
            failures.append({
                "doc_id": doc_id,
                "predicted": {
                    "dealer_name": dealer_pred,
                    "model_name": model_pred,
                    "horse_power": hp_pred,
                    "asset_cost": cost_pred,
                },
                "ground_truth": {k: gt.get(k) for k in ("dealer_name", "model_name", "horse_power", "asset_cost")},
                "matches": matches,
            })
        else:
            print(f"\n[4/4 PASS] {doc_id}")

    print(f"\nTotal docs: {total}")
    for k, v in score.items():
        pct = (v * 100 / total) if total else 0
        print(f"  {k:14s} {v}/{total}  ({pct:.1f}%)")
    print()
    # Count 4/4
    full_pass_docs = [f for f in failures if all(f["matches"].values())]
    near_pass = [f for f in failures if sum(f["matches"].values()) == 3]
    print(f"4/4 passes: {total - len(failures)}")
    print(f"3/4 passes: {len(near_pass)}")
    print(f"\nFailures ({len(failures)}):")
    for f in failures:
        print(f"\n--- {f['doc_id']} ---")
        for fld in ("dealer_name", "model_name", "horse_power", "asset_cost"):
            mark = "PASS" if f["matches"][fld] else "FAIL"
            print(f"  [{mark}] {fld}")
            print(f"    pred: {f['predicted'][fld]!r}")
            print(f"    gt:   {f['ground_truth'][fld]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
