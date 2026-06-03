"""One-time model bundle download.

Fetches Qwen2.5-1.5B-Instruct safetensors weights from the Hugging Face CDN
into ``backend/models/qwen2.5-1.5b-instruct/``. Run this once on a machine
with internet; subsequent pipeline runs load from the local directory with
no network access.

The legacy GGUF (used by the abandoned llama-cpp path) is left alone; this
script does not delete it.

Run::

    python -m scripts.download_models
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("download_models")


# Direct CDN URLs — no Xet, no auth.
QWEN_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_BASE_URL = f"https://huggingface.co/{QWEN_REPO}/resolve/main"

# Files we need for transformers.AutoModelForCausalLM + AutoTokenizer.
# The safetensors is the bulk; tokenizer + config are tiny.
QWEN_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TARGET_DIR = MODELS_DIR / "qwen2.5-1.5b-instruct"


def _human(n: int) -> str:
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024 ** 3):.2f} GB"
    if n >= 1024 * 1024:
        return f"{n / (1024 ** 2):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _download_file(url: str, dest: Path, *, chunk_size: int = 1024 * 1024) -> int:
    """Stream a URL to disk with a progress line. Returns total bytes."""
    req = Request(url, headers={"User-Agent": "invoiceflow-pipeline/0.1 (download_models)"})
    with urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", "0"))
        downloaded = 0
        last_log = time.monotonic()
        with dest.open("wb") as out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                # Log every 2s for big files; suppress for small ones.
                if total > 5 * 1024 * 1024 and now - last_log >= 2:
                    pct = (downloaded / total * 100) if total else 0
                    print(
                        f"    {_human(downloaded)} / {_human(total)} ({pct:.1f}%)",
                        flush=True,
                    )
                    last_log = now
    return downloaded


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0
    for filename in QWEN_FILES:
        dest = TARGET_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            size = dest.stat().st_size
            logger.info("Already present: %s (%s)", dest.name, _human(size))
            continue

        url = f"{QWEN_BASE_URL}/{filename}"
        logger.info("Downloading %s ...", filename)
        partial = dest.with_suffix(dest.suffix + ".partial")
        try:
            n = _download_file(url, partial)
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", filename, exc)
            if partial.exists():
                partial.unlink()
            return 1
        partial.replace(dest)
        total_downloaded += n
        logger.info("Saved %s (%s)", dest.name, _human(n))

    logger.info("All files in %s", TARGET_DIR)
    logger.info("Total downloaded this run: %s", _human(total_downloaded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
