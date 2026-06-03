"""Mine the dealer + asset masters from the unlabeled training set.

Runs PaddleOCR on every image in ``train_data_idfc/train/`` once, caches the
token streams, then derives:

* ``data/dealer_master.json`` — clusters of dealer-name candidates from
  letterhead regions and ``Dealer:`` / ``M/s`` anchored extractions.
* ``data/asset_master.json`` — ``(brand, model)`` triples union'd with the
  curated seed-brand list in :mod:`utils.masters`.

Both outputs are byte-stable (Property 9): re-running the script on the
same input produces identical JSON files.

Run::

    python -m scripts.mine_masters
    python -m scripts.mine_masters --limit 100   # quick partial run
    python -m scripts.mine_masters --no-cache    # bypass OCR cache

Validates Requirements: 8.1-8.7, 12.1-12.4, 13.1-13.4
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("mine_masters")


@dataclass
class _CachedDoc:
    """A document's OCR result for reuse across mining runs."""

    filename: str
    tokens: list[dict]  # serialized OcrToken records
    page_height: int


def _load_or_run_ocr(
    train_dir: Path, cache_path: Path, *, limit: Optional[int], use_cache: bool
) -> list[_CachedDoc]:
    """Return per-image OCR results, loading from cache when possible.

    The cache lives at ``models/.ocr_cache.json`` so a re-run on the same
    images skips the OCR pass entirely. When ``--no-cache`` is set we
    always re-OCR.
    """
    if use_cache and cache_path.exists():
        logger.info("Loading OCR cache from %s", cache_path)
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cached = [_CachedDoc(**doc) for doc in payload.get("docs", [])]
            if limit is not None:
                cached = cached[:limit]
            logger.info("Cache loaded: %d documents", len(cached))
            return cached
        except Exception as exc:
            logger.warning("Cache load failed: %s — falling back to fresh OCR", exc)

    # Fresh OCR pass.
    from utils.device import detect
    from utils.ingestion import load
    from utils.ocr import OcrEngine

    images = sorted(train_dir.glob("*.png"))
    if limit is not None:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"No PNG files in {train_dir}")

    logger.info("OCR-ing %d images (this is the slow part — ~5-10s each on GPU)", len(images))
    print(f"OCR-ing {len(images)} images...", flush=True)
    device = detect()
    logger.info("Constructing OcrEngine (this loads paddle models — first time can be slow)")
    print("Constructing OcrEngine...", flush=True)
    ocr_engine = OcrEngine(device)
    logger.info("OcrEngine ready, starting per-image loop")
    print("OcrEngine ready, starting per-image loop", flush=True)

    docs: list[_CachedDoc] = []
    t0 = time.monotonic()

    def _flush_cache() -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"docs": [doc.__dict__ for doc in docs]},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # GPU memory tends to creep up across many OCR calls — periodic
    # `torch.cuda.empty_cache()` releases unused fragments.
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    for i, img_path in enumerate(images, 1):
        try:
            t_img = time.monotonic()
            page = load(img_path)
            tokens = ocr_engine.extract(page)
            doc_elapsed = time.monotonic() - t_img
            print(f"  [{i}/{len(images)}] {img_path.name} — {len(tokens)} tokens in {doc_elapsed:.1f}s", flush=True)
            docs.append(
                _CachedDoc(
                    filename=img_path.name,
                    tokens=[
                        {
                            "text": t.text,
                            "bbox": list(t.bbox),
                            "confidence": t.confidence,
                            "script": t.script,
                        }
                        for t in tokens
                    ],
                    page_height=page.image.height,
                )
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", img_path.name, exc)

        # Flush cache after every doc so partial progress survives kills.
        _flush_cache()

        # Free unused GPU fragments every 10 docs to slow the OCR creep.
        if cuda_available and i % 10 == 0:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        if i % 5 == 0:
            elapsed = time.monotonic() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(images) - i) / rate if rate > 0 else 0
            logger.info("  %d/%d done (%.1fs elapsed, ETA %.0fs)", i, len(images), elapsed, eta)

    elapsed = time.monotonic() - t0
    logger.info("OCR complete: %d docs in %.1fs", len(docs), elapsed)

    _flush_cache()
    logger.info("OCR cache saved to %s", cache_path)
    return docs


# --------------------------------------------------------------------------- #
# Dealer name candidates
# --------------------------------------------------------------------------- #


_DEALER_ANCHOR_RE = re.compile(
    r"(?:authori[sz]ed\s+dealer|\bM/s\b|\bdealer\s*[:\-])",
    flags=re.IGNORECASE,
)


def _harvest_dealer_candidates(docs: list[_CachedDoc]) -> list[str]:
    """Pull candidate dealer-name strings from each document.

    Two heuristics:
    1. Letterhead region (top 12% of the page) — concatenate the topmost
       multi-token line that's mostly uppercase / proper-cased.
    2. Anchored extraction — text after Authorized Dealer / M/s anchors.
    """
    candidates: list[str] = []

    for doc in docs:
        if not doc.tokens:
            continue

        top_threshold = int(doc.page_height * 0.12)
        # Strategy 1: letterhead — group top-region tokens by row bucket.
        top_tokens = [
            (i, t) for i, t in enumerate(doc.tokens) if t["bbox"][1] < top_threshold
        ]
        if top_tokens:
            buckets: dict[int, list[tuple[int, dict]]] = {}
            for i, t in top_tokens:
                bucket = t["bbox"][1] // 30
                buckets.setdefault(bucket, []).append((i, t))

            def _row_score(bucket_tokens: list[tuple[int, dict]]) -> float:
                joined = " ".join(t["text"] for _, t in bucket_tokens)
                upper_chars = sum(1 for c in joined if c.isupper())
                # Prefer rows with multiple tokens (real letterheads are
                # usually a few words).
                return upper_chars + len(bucket_tokens) * 2

            top_row = max(buckets.values(), key=_row_score)
            top_row.sort(key=lambda x: x[1]["bbox"][0])
            joined = " ".join(t["text"] for _, t in top_row).strip(" .,:;-")
            if 4 <= len(joined) <= 80 and any(c.isalpha() for c in joined):
                candidates.append(joined.upper())

        # Strategy 2: anchored extraction.
        for i, tok in enumerate(doc.tokens):
            if not _DEALER_ANCHOR_RE.search(tok["text"]):
                continue
            # Collect 1-5 tokens to the right on the same row.
            anchor = tok
            same_row_tokens: list[str] = []
            for j in range(i + 1, min(i + 6, len(doc.tokens))):
                t2 = doc.tokens[j]
                a_cy = (anchor["bbox"][1] + anchor["bbox"][3]) / 2
                t_cy = (t2["bbox"][1] + t2["bbox"][3]) / 2
                if abs(a_cy - t_cy) > 30:
                    break
                if re.fullmatch(r"[\d,]+", t2["text"]):
                    break
                same_row_tokens.append(t2["text"])
            if same_row_tokens:
                joined = " ".join(same_row_tokens).strip(" .,:;-")
                if 4 <= len(joined) <= 80:
                    candidates.append(joined.upper())

    return candidates


# --------------------------------------------------------------------------- #
# Asset (brand, model) candidates
# --------------------------------------------------------------------------- #

_MODEL_ID_RE = re.compile(
    r"\b(\d{2,4}[A-Z]{0,3}(?:[-\s]?\d{0,3})?)\b"
)


def _harvest_asset_candidates(
    docs: list[_CachedDoc], seed_brands: tuple[str, ...]
) -> list[tuple[str, str, str]]:
    """Mine ``(brand, model, full_name)`` triples from the OCR streams.

    Strategy: find any token matching a seed brand keyword, then capture the
    next 1-3 tokens on the same row that look like model identifiers
    (digit runs with optional letter suffixes).
    """
    if not seed_brands:
        return []

    brand_pattern = re.compile(
        r"\b(" + "|".join(re.escape(b) for b in seed_brands) + r")\b",
        flags=re.IGNORECASE,
    )

    triples: set[tuple[str, str, str]] = set()
    for doc in docs:
        for i, tok in enumerate(doc.tokens):
            match = brand_pattern.search(tok["text"])
            if not match:
                continue
            brand_canonical = next(
                (b for b in seed_brands if b.lower() == match.group(1).lower()),
                match.group(1),
            )

            # Collect adjacent tokens on the same row to form the model.
            anchor = tok
            model_parts: list[str] = []
            for j in range(i + 1, min(i + 5, len(doc.tokens))):
                t2 = doc.tokens[j]
                a_cy = (anchor["bbox"][1] + anchor["bbox"][3]) / 2
                t_cy = (t2["bbox"][1] + t2["bbox"][3]) / 2
                if abs(a_cy - t_cy) > 30:
                    break
                # Stop on punctuation-only or currency tokens.
                if re.fullmatch(r"[^\w]+", t2["text"]):
                    continue
                if re.fullmatch(r"[\d,]+\.?\d*", t2["text"]) and len(t2["text"]) > 5:
                    break
                model_parts.append(t2["text"])
                if len(model_parts) >= 3:
                    break

            if not model_parts:
                continue
            model = " ".join(model_parts).strip(" .,:;-")
            # Reject if no digit (likely not a model number).
            if not re.search(r"\d", model):
                continue
            if len(model) < 2 or len(model) > 40:
                continue

            full_name = f"{brand_canonical} {model}"
            triples.add((brand_canonical, model, full_name))

    return sorted(triples, key=lambda t: t[2])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "train_data_idfc" / "train",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models" / ".ocr_cache.json",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap docs for a quick run")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the OCR cache")
    args = parser.parse_args()

    if not args.source.is_dir():
        logger.error("Source dir not found: %s", args.source)
        return 1

    docs = _load_or_run_ocr(args.source, args.cache, limit=args.limit, use_cache=not args.no_cache)
    if not docs:
        logger.error("No documents to mine — bailing")
        return 1

    # Mine + cluster dealers.
    from utils.masters import (
        AssetEntry,
        SEED_BRANDS,
        cluster_dealer_candidates,
        merge_asset_entries,
        save_asset_master,
        save_dealer_master,
    )

    logger.info("Harvesting dealer candidates from %d docs", len(docs))
    dealer_candidates = _harvest_dealer_candidates(docs)
    logger.info("Found %d dealer-name candidates (raw)", len(dealer_candidates))

    dealer_entries = cluster_dealer_candidates(dealer_candidates, similarity_threshold=85)
    logger.info("Clustered into %d canonical dealer entries", len(dealer_entries))

    dealer_path = args.data_dir / "dealer_master.json"
    save_dealer_master(dealer_entries, dealer_path)
    logger.info("Wrote %s", dealer_path)

    # Mine assets, union with curated seed.
    logger.info("Harvesting asset (brand, model) triples")
    asset_triples = _harvest_asset_candidates(docs, SEED_BRANDS)
    logger.info("Found %d unique (brand, model) triples", len(asset_triples))

    mined_assets = [AssetEntry(brand=b, model=m, full_name=fn) for b, m, fn in asset_triples]
    # Curated seed: bare brand names without a specific model. Provides a
    # baseline so the brand_keywords helper has data even on a small dataset.
    curated_assets = [
        AssetEntry(brand=b, model="", full_name=b) for b in SEED_BRANDS
    ]
    merged = merge_asset_entries(mined_assets, curated_assets)
    asset_path = args.data_dir / "asset_master.json"
    save_asset_master(merged, asset_path)
    logger.info("Wrote %s (%d entries)", asset_path, len(merged))

    print()
    print("=" * 60)
    print("MASTER MINING SUMMARY")
    print("=" * 60)
    print(f"Documents OCR'd:     {len(docs)}")
    print(f"Dealer entries:      {len(dealer_entries)}")
    print(f"Asset entries:       {len(merged)} ({len(asset_triples)} mined + {len(curated_assets)} curated)")
    print(f"\nOutputs:")
    print(f"  {dealer_path}")
    print(f"  {asset_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
