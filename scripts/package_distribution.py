"""
InvoiceFlow Distribution Packaging Script
Packages the complete production pipeline into a clean, air-gapped distribution archive.
"""
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ARCHIVE = PROJECT_ROOT / "invoiceflow_distribution.zip"

INCLUDE_PATHS = [
    "executable.py",
    "requirements.txt",
    "README.md",
    "PRD.md",
    "tasks.md",
    "TRAINING_AND_EXTRACTION_GUIDE.md",
    "sample_output_filtered.json",
    "core",
    "document_profiles",
    "docs",
    "scripts",
    "evaluation",
    "tests"
]

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".venv",
    "datasets/raw",
    "runs",
    "*.pyc"
]

def should_exclude(rel_path: str) -> bool:
    for pat in EXCLUDE_PATTERNS:
        if pat in rel_path.replace("\\", "/"):
            return True
    return False

def package_submission():
    print(f"Creating submission archive: {OUTPUT_ARCHIVE}")
    with zipfile.ZipFile(OUTPUT_ARCHIVE, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in INCLUDE_PATHS:
            item_path = PROJECT_ROOT / item
            if not item_path.exists():
                print(f"Skipping missing: {item}")
                continue

            if item_path.is_file():
                rel = item_path.relative_to(PROJECT_ROOT)
                if not should_exclude(str(rel)):
                    zipf.write(item_path, rel)
                    print(f"Added file: {rel}")
            elif item_path.is_dir():
                for file_p in item_path.rglob("*"):
                    if file_p.is_file():
                        rel = file_p.relative_to(PROJECT_ROOT)
                        if not should_exclude(str(rel)):
                            zipf.write(file_p, rel)

    archive_size_mb = OUTPUT_ARCHIVE.stat().st_size / (1024 * 1024)
    print(f"\n==========================================")
    print(f"SUBMISSION PACKAGED SUCCESSFULLY")
    print(f"Archive: {OUTPUT_ARCHIVE.name}")
    print(f"Size: {archive_size_mb:.2f} MB")
    print(f"==========================================")

if __name__ == "__main__":
    package_submission()
