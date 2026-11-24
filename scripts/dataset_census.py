"""
InvoiceFlow Dataset Census & Manifest Builder (Task 1.1)
Scans datasets/raw/, extracts image dimensions, file sizes, and generates metadata_manifest.json.
"""
import os
import json
import hashlib
from pathlib import Path
import cv2

RAW_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw"
MANIFEST_DIR = Path(__file__).resolve().parent.parent / "datasets" / "manifests"

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def run_census():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in RAW_DIR.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".pdf"]])
    print(f"Total files in raw directory: {len(files)}")

    manifest_records = []
    total_size = 0

    for idx, p in enumerate(files, 1):
        size = p.stat().st_size
        total_size += size
        
        # Read image dimensions
        img = cv2.imread(str(p))
        if img is not None:
            h, w = img.shape[:2]
        else:
            h, w = 0, 0

        sha = compute_sha256(p)
        manifest_records.append({
            "doc_id": p.name,
            "file_name": p.name,
            "file_size_bytes": size,
            "width": w,
            "height": h,
            "sha256": sha
        })
        if idx % 100 == 0 or idx == len(files):
            print(f"Processed [{idx}/{len(files)}] files...")

    output_path = MANIFEST_DIR / "metadata_manifest.json"
    manifest_data = {
        "version": "1.0.0",
        "total_documents": len(manifest_records),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "documents": manifest_records
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\nCensus complete! Manifest written to: {output_path}")
    print(f"Total documents: {len(manifest_records)}, Total size: {manifest_data['total_size_mb']} MB")

if __name__ == "__main__":
    run_census()
