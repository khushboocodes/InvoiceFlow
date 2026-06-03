"""Print a quick summary of the latest validation report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPORT = Path(__file__).resolve().parent.parent / "report.json"

if not REPORT.exists():
    print(f"No report at {REPORT}", file=sys.stderr)
    raise SystemExit(1)

r = json.loads(REPORT.read_text(encoding="utf-8"))
n = len(r.get("per_doc_details", []))
print(f"Docs evaluated:   {n} / {r['docs_total']}")
print(f"DLA so far:       {r['dla']:.2%}")
print()
print("Per-field accuracy:")
for k, v in r["per_field_correct_total"].items():
    acc = r["per_field_accuracy"][k]
    print(f"  {k:14s} {v:>8s}  ({acc:.1%})")
print()
print("Latency:")
print(f"  p50: {r['latency_sec']['p50']:.1f}s")
print(f"  p95: {r['latency_sec']['p95']:.1f}s")
print(f"  max: {r['latency_sec']['max']:.1f}s")
