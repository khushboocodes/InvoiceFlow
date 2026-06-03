"""Trace bucket details."""
import json, sys, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.ocr import OcrToken

cache = json.loads((Path(__file__).resolve().parent.parent / "models" / ".eval_cache.json").read_text(encoding="utf-8"))
doc_id = sys.argv[1]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

for k, v in cache.items():
    if k.startswith(f"{doc_id}::"):
        tokens = [OcrToken(text=row["text"], bbox=tuple(row["bbox"]), confidence=float(row["confidence"]), script=row.get("script","en")) for row in v["ocr"]]
        page_height = max(t.bbox[3] for t in tokens) if tokens else 2000
        top_threshold = int(page_height * 0.30)

        def _looks(text):
            if len(text) < 3: return False
            if not any(c.isalpha() for c in text): return False
            if re.fullmatch(r"\d+|[A-Z0-9]{10,}", text): return False
            digits = sum(1 for c in text if c.isdigit())
            letters = sum(1 for c in text if c.isalpha())
            if digits > letters and digits >= 4: return False
            return True

        letterhead = [(i, t) for i, t in enumerate(tokens) if t.bbox[1] < top_threshold and _looks(t.text)]
        # Show buckets
        from collections import defaultdict
        buckets = defaultdict(list)
        for i, t in letterhead:
            buckets[t.bbox[1] // 30].append((i, t))
        for b in sorted(buckets.keys()):
            text = " | ".join(t.text for _, t in buckets[b])
            print(f"  bucket {b} (y~{b*30}): {text[:120]!r}")
        break
