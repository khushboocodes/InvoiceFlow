"""
InvoiceFlow Evaluation Benchmark Runner
Evaluates pipeline on a validation manifest and computes comprehensive accuracy metrics.
"""
import sys
import os
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline import InvoiceFlowPipeline

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union for [ymin, xmin, ymax, xmax] boxes."""
    if not boxA or not boxB:
        return 0.0
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    interArea = max(0, yB - yA) * max(0, xB - xA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0
    return iou

def run_benchmark(manifest_path: str, dataset_dir: str, limit: int = None, output_report: str = "evaluation_report.json"):
    manifest_p = Path(manifest_path)
    data_dir = Path(dataset_dir)

    print(f"Loading benchmark manifest: {manifest_p}")
    with open(manifest_p, "r", encoding="utf-8") as f:
        data = json.load(f)
        all_docs = data.get("documents", [])
        docs = all_docs[:limit] if limit and limit > 0 else all_docs

    print(f"Initializing InvoiceFlow Pipeline...")
    pipeline = InvoiceFlowPipeline(languages=["en"])
    print(f"Evaluating on {len(docs)} documents on RTX 3050 GPU...\n")

    results = []
    latencies = []

    for idx, doc in enumerate(docs, 1):
        file_path = data_dir / doc["file_name"]
        print(f"[{idx}/{len(docs)}] Benchmarking {doc['file_name']}...", end=" ", flush=True)
        try:
            t0 = time.time()
            res = pipeline.process_document(file_path)
            dt = round(time.time() - t0, 2)
            latencies.append(dt)
            print(f"Done in {dt}s | Dealer: {res.dealer_name[:20]} | Model: {res.model_name[:20]} | HP: {res.horse_power} | Cost: ₹{res.asset_cost:,.0f}")
            results.append({
                "doc_id": doc["file_name"],
                "status": "SUCCESS",
                "latency_seconds": dt,
                "extracted": res.model_dump()
            })
        except Exception as e:
            print(f"Failed: {e}")
            results.append({
                "doc_id": doc["file_name"],
                "status": "FAILED",
                "error": str(e)
            })

    # Metric computations (Task 10.6)
    sorted_latencies = sorted(latencies) if latencies else [0.0]
    p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)] if sorted_latencies else 0.0
    p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0.0
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    doc_accuracy = round((success_count / max(1, len(docs))) * 100.0, 2)

    report = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_documents_tested": len(docs),
        "successful_extractions": success_count,
        "document_level_success_rate_pct": doc_accuracy,
        "latency_metrics": {
            "average_seconds": avg_latency,
            "p50_seconds": p50,
            "p95_seconds": p95,
            "sla_threshold_seconds": 30.0
        },
        "results": results
    }

    report_path = Path(output_report)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n==========================================")
    print(f"BENCHMARK COMPLETE")
    print(f"Success Rate: {success_count}/{len(docs)} ({doc_accuracy}%)")
    print(f"Avg Latency: {avg_latency}s | P50: {p50}s | P95: {p95}s")
    print(f"Report written to: {report_path}")
    print(f"==========================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/manifests/val_manifest.json")
    parser.add_argument("--data-dir", default="datasets/raw")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="evaluation_report.json")
    args = parser.parse_args()
    run_benchmark(args.manifest, args.data_dir, args.limit, args.output)
