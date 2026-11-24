#!/usr/bin/env python3
"""
InvoiceFlow CLI Entrypoint (executable.py)
Production command-line interface for batch and single-file invoice extraction.
"""
import sys
import os
import argparse
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.pipeline import InvoiceFlowPipeline

def main():
    parser = argparse.ArgumentParser(
        description="InvoiceFlow: Enterprise Document Intelligence Extraction Engine"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to an input image/PDF file or directory of documents"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output JSON file or directory"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of documents to process (for testing/benchmarks)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose telemetry output"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Initializing InvoiceFlow Pipeline...")
    pipeline = InvoiceFlowPipeline(languages=["en"])
    print(f"Pipeline ready.")

    if input_path.is_file():
        # Process single file
        print(f"Processing document: {input_path.name}")
        try:
            result = pipeline.process_document(input_path)
            res_dict = result.model_dump()

            if output_path.suffix == ".json":
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(res_dict, f, indent=2)
            else:
                output_path.mkdir(parents=True, exist_ok=True)
                with open(output_path / f"{input_path.stem}.json", "w", encoding="utf-8") as f:
                    json.dump(res_dict, f, indent=2)

            print(f"Extraction successful: {output_path}")
            if args.verbose:
                print(json.dumps(res_dict, indent=2))
            sys.exit(0)
        except Exception as e:
            print(f"Error processing {input_path.name}: {e}", file=sys.stderr)
            sys.exit(2)

    elif input_path.is_dir():
        # Batch process directory
        files = sorted([
            p for p in input_path.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".pdf"]
        ])
        if args.limit:
            files = files[:args.limit]

        print(f"Found {len(files)} documents to process in {input_path}")
        results = []
        for idx, file_p in enumerate(files, 1):
            print(f"[{idx}/{len(files)}] Processing {file_p.name}...", end=" ", flush=True)
            try:
                res = pipeline.process_document(file_p)
                results.append(res.model_dump())
                print(f"Done ({res.metadata.execution_time_seconds}s) -> Dealer: {res.dealer_name[:20]} | Model: {res.model_name[:20]} | HP: {res.horse_power} | Cost: ₹{res.asset_cost:,.0f}")
            except Exception as e:
                print(f"Failed: {e}")
                results.append({
                    "document_id": file_p.name,
                    "status": "FAILED",
                    "error": str(e)
                })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nBatch processing complete. Results written to: {output_path}")
        sys.exit(0)

if __name__ == "__main__":
    main()
