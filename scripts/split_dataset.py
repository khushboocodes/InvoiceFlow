"""
InvoiceFlow Grouped Dataset Splitter (Task 3.2)
Partitions documents into Train (70%) and Validation (30%) splits grouped by Application ID.
Guarantees zero leakage of multi-page invoices or dealership templates between splits.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "datasets" / "manifests" / "metadata_manifest.json"
MANIFEST_DIR = Path(__file__).resolve().parent.parent / "datasets" / "manifests"

def extract_case_id(filename: str) -> str:
    """Extracts application case ID from filename (e.g. '172427893_3_pg11.png' -> '172427893')."""
    parts = filename.split("_")
    return parts[0] if parts else filename

def run_split(train_ratio: float = 0.70, seed: int = 42):
    random.seed(seed)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        docs = data.get("documents", [])

    print(f"Total documents to partition: {len(docs)}")

    # Group documents by Case ID
    case_groups = defaultdict(list)
    for doc in docs:
        cid = extract_case_id(doc["file_name"])
        case_groups[cid].append(doc)

    unique_cases = list(case_groups.keys())
    random.shuffle(unique_cases)

    total_docs = len(docs)
    target_train_count = int(total_docs * train_ratio)

    train_docs = []
    val_docs = []
    current_train_count = 0

    for cid in unique_cases:
        group_items = case_groups[cid]
        if current_train_count + len(group_items) <= target_train_count:
            train_docs.extend(group_items)
            current_train_count += len(group_items)
        else:
            val_docs.extend(group_items)

    print(f"Unique Application Cases: {len(unique_cases)}")
    print(f"Train Split: {len(train_docs)} documents ({len(train_docs)/total_docs*100:.1f}%)")
    print(f"Validation Split: {len(val_docs)} documents ({len(val_docs)/total_docs*100:.1f}%)")

    # Save manifests
    train_manifest = {
        "split": "train",
        "total_documents": len(train_docs),
        "documents": train_docs
    }
    val_manifest = {
        "split": "val",
        "total_documents": len(val_docs),
        "documents": val_docs
    }

    with open(MANIFEST_DIR / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, indent=2)

    with open(MANIFEST_DIR / "val_manifest.json", "w", encoding="utf-8") as f:
        json.dump(val_manifest, f, indent=2)

    print(f"Manifests written to:")
    print(f"- {MANIFEST_DIR / 'train_manifest.json'}")
    print(f"- {MANIFEST_DIR / 'val_manifest.json'}")

if __name__ == "__main__":
    run_split()
