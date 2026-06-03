"""Quick manual smoke test: load a few real images from the train set and
print metadata. Useful for sanity-checking Stage 1 against real data, not
synthetic fixtures.

Run:
    python -m scripts.smoke_ingestion
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running as a script from the backend root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.ingestion import load


def main() -> int:
    train_dir = Path(__file__).resolve().parents[2] / "train_data_idfc" / "train"
    if not train_dir.exists():
        print(f"Train directory not found: {train_dir}", file=sys.stderr)
        return 1

    samples = sorted(train_dir.glob("*.png"))[:5]
    if not samples:
        print(f"No PNG samples in {train_dir}", file=sys.stderr)
        return 1

    print(f"Loading {len(samples)} samples from {train_dir}\n")
    for path in samples:
        start = time.monotonic()
        try:
            page = load(path)
        except Exception as exc:
            print(f"  FAIL  {path.name}: {exc}")
            continue
        elapsed = time.monotonic() - start
        print(
            f"  OK    {path.name}: {page.image.size} "
            f"({elapsed*1000:.0f}ms, page {page.page_index+1}/{page.page_count})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
