"""
InvoiceFlow Perceptual Deduplication Engine (Task 1.4)
Computes difference hashes (dHash) on downsampled grayscale images to identify duplicate/near-duplicate re-scans.
"""
import json
from pathlib import Path
import cv2
import numpy as np

RAW_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw"
MANIFEST_DIR = Path(__file__).resolve().parent.parent / "datasets" / "manifests"

def compute_dhash(image_path: Path, hash_size: int = 8) -> int:
    """Computes 64-bit difference hash (dHash) using OpenCV and NumPy."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    # Resize to (hash_size + 1, hash_size)
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    # Compute horizontal differences between adjacent pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # Convert bool array to 64-bit integer
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def hamming_distance(hash1: int, hash2: int) -> int:
    """Computes bitwise Hamming distance between two 64-bit hashes."""
    return bin(hash1 ^ hash2).count('1')

def run_deduplication(max_hamming_dist: int = 4):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in RAW_DIR.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".pdf"]])
    print(f"Analyzing {len(files)} documents for perceptual duplicates...")

    hashes = {}
    for idx, f in enumerate(files, 1):
        h = compute_dhash(f)
        hashes[f.name] = h
        if idx % 100 == 0 or idx == len(files):
            print(f"Hashed [{idx}/{len(files)}] images...")

    duplicates = []
    file_names = list(hashes.keys())
    visited = set()

    for i in range(len(file_names)):
        fn1 = file_names[i]
        if fn1 in visited:
            continue
        cluster = [fn1]
        for j in range(i + 1, len(file_names)):
            fn2 = file_names[j]
            dist = hamming_distance(hashes[fn1], hashes[fn2])
            if dist <= max_hamming_dist:
                cluster.append(fn2)
                visited.add(fn2)

        if len(cluster) > 1:
            duplicates.append({
                "canonical": cluster[0],
                "duplicate_count": len(cluster) - 1,
                "cluster": cluster
            })

    output_path = MANIFEST_DIR / "duplicate_clusters.json"
    report = {
        "total_analyzed": len(files),
        "duplicate_clusters_found": len(duplicates),
        "total_near_duplicate_files": sum(c["duplicate_count"] for c in duplicates),
        "clusters": duplicates
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nDeduplication audit complete! Report written to: {output_path}")
    print(f"Duplicate clusters found: {len(duplicates)}")

if __name__ == "__main__":
    run_deduplication()
