"""Assemble ``submission.zip`` from the backend folder.

Steps:
1. Verify all required files exist (executable, requirements, models, masters).
2. Build a clean staging directory.
3. Copy required files (excluding tests, scripts, demo, virtual envs).
4. Run a sample extraction inside the staging dir to refresh ``sample_output/``.
5. Zip into ``submission.zip``.
6. Print final size and a manifest.

Run::

    python -m scripts.build_submission
    python -m scripts.build_submission --skip-sample   # don't re-run a sample doc
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("build_submission")


BACKEND_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_ROOT.parent

# Files / directories that MUST exist before we can package.
REQUIRED_FILES = (
    "executable.py",
    "requirements.txt",
    "README.md",
    "utils/__init__.py",
    "utils/device.py",
    "utils/offline_guard.py",
    "utils/schema.py",
    "utils/ingestion.py",
    "utils/ocr.py",
    "utils/detection.py",
    "utils/extraction.py",
    "utils/slm.py",
    "utils/normalization.py",
    "utils/masters.py",
    "utils/confidence.py",
    "utils/pipeline.py",
    "models/yolov8n_sig_stamp.pt",
    "models/detection.yaml",
    "models/qwen2.5-1.5b-instruct/config.json",
    "models/qwen2.5-1.5b-instruct/tokenizer.json",
    "models/qwen2.5-1.5b-instruct/model.safetensors",
    "data/dealer_master.json",
    "data/asset_master.json",
)

# Patterns to exclude from the zip (anything matching is skipped).
EXCLUDE_DIRS = (
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "tests",
    "scripts",
    "notebooks",
    "models/training_runs",
    "models/.cache",
    "models/base",
    "models/.ocr_cache.json",
)

EXCLUDE_FILES = (
    ".gitignore",
    "pyproject.toml",
    "conftest.py",
    "pseudo_labels.json",
)


def _verify(missing_action: str = "fail") -> list[str]:
    """Confirm every required file is present. Return list of missing files."""
    missing: list[str] = []
    for relpath in REQUIRED_FILES:
        target = BACKEND_ROOT / relpath
        if not target.exists():
            missing.append(relpath)
    if missing and missing_action == "fail":
        for m in missing:
            logger.error("MISSING: %s", m)
        raise SystemExit(f"Cannot build submission — {len(missing)} required files missing")
    return missing


def _should_include(rel: Path) -> bool:
    """True when this path should land in the zip."""
    parts = rel.as_posix()
    for ex in EXCLUDE_DIRS:
        if parts == ex or parts.startswith(ex + "/"):
            return False
    name = rel.name
    if name in EXCLUDE_FILES:
        return False
    if name.startswith("."):
        return False
    if name.endswith(".pyc"):
        return False
    return True


def _refresh_sample_output() -> None:
    """Run the pipeline on one sample image to refresh sample_output/result.json."""
    samples = sorted((REPO_ROOT / "train_data_idfc" / "train").glob("*.png"))
    if not samples:
        logger.warning("No training samples available — skipping sample_output refresh")
        return

    sample = samples[0]
    output = BACKEND_ROOT / "sample_output" / "result.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Running pipeline on %s to refresh sample_output", sample.name)
    venv_python = BACKEND_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = Path(sys.executable)

    proc = subprocess.run(
        [str(venv_python), "executable.py", str(sample), "--output", str(output), "--quiet"],
        cwd=BACKEND_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        logger.warning("Sample run failed: %s", proc.stderr)
    else:
        logger.info("sample_output/result.json refreshed")


def _build_zip(staging: Path, zip_path: Path) -> int:
    """Walk staging dir and zip everything in. Returns total file count."""
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_dir():
                continue
            arcname = path.relative_to(staging).as_posix()
            zf.write(path, arcname=arcname)
            count += 1
    return count


def _stage_files(staging: Path) -> None:
    """Copy backend files into staging, respecting excludes."""
    for path in sorted(BACKEND_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(BACKEND_ROOT)
        if not _should_include(rel):
            continue
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def _human_size(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "submission.zip",
        help="Path to write submission.zip",
    )
    parser.add_argument("--skip-sample", action="store_true", help="Don't re-run sample extraction")
    parser.add_argument("--skip-verify", action="store_true", help="Don't fail on missing required files")
    args = parser.parse_args()

    logger.info("Building submission from %s", BACKEND_ROOT)

    if not args.skip_verify:
        _verify(missing_action="fail")
        logger.info("All required files present")
    else:
        missing = _verify(missing_action="warn")
        if missing:
            logger.warning("Proceeding with %d missing files", len(missing))

    if not args.skip_sample:
        _refresh_sample_output()

    # Stage to a temp directory.
    staging = REPO_ROOT / ".submission_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    logger.info("Staging files...")
    t0 = time.monotonic()
    _stage_files(staging)
    logger.info("Staged in %.1fs", time.monotonic() - t0)

    # Zip.
    if args.output.exists():
        args.output.unlink()
    logger.info("Compressing to %s ...", args.output)
    t0 = time.monotonic()
    file_count = _build_zip(staging, args.output)
    logger.info("Zipped %d files in %.1fs", file_count, time.monotonic() - t0)

    # Cleanup staging.
    shutil.rmtree(staging)

    final_size = args.output.stat().st_size

    print()
    print("=" * 60)
    print("SUBMISSION BUILT")
    print("=" * 60)
    print(f"Output:        {args.output}")
    print(f"Files in zip:  {file_count}")
    print(f"Size:          {_human_size(final_size)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
