# InvoiceFlow — Offline Document AI for Invoice Field Extraction

IDFC GenAI · Convolve 4.0 hackathon submission.

A six-stage Python pipeline that extracts six structured fields from
invoice-type documents (running use case: tractor loan quotations).
Generalizes to retail invoices, industrial invoices, and other
semi-structured business documents.

* **Inputs:** PDF, PNG, JPG, JPEG (single or multi-page)
* **Outputs:** one JSON object per document, conforming to the spec schema
* **Performance:** ~22s p50 / ~33s p95 per document on RTX 3050 + AMD CPU
* **Cost:** $0 marginal — every model is open-source and runs locally
* **Languages:** English + Devanagari (Hindi)
* **Networking:** zero outbound calls at inference; runs in `--network=none`

## Quick start

```bash
# Python 3.10
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows
# source .venv/bin/activate    # Linux / macOS

# Install pinned deps
pip install -r requirements.txt

# Single document
python executable.py path/to/invoice.png

# Batch over a directory
python executable.py --batch path/to/folder

# PS-reference flat output shape
python executable.py path/to/invoice.png --legacy

# Engage offline kill-switch
python executable.py path/to/invoice.png --offline
```

## Submission layout

```
submission.zip
├── executable.py                     # CLI entry point
├── requirements.txt                  # pinned, PyPI-only
├── README.md                         # this file
├── utils/                            # pipeline modules
│   ├── device.py                     # CUDA detection
│   ├── offline_guard.py              # network kill-switch
│   ├── schema.py                     # Pydantic v2 output contract
│   ├── ingestion.py                  # Stage 1
│   ├── ocr.py                        # Stage 2A (PaddleOCR)
│   ├── detection.py                  # Stage 2B (YOLOv8n)
│   ├── extraction.py                 # Stage 3 Tier-1 (rules)
│   ├── slm.py                        # Stage 3 Tier-2 (Qwen 1.5B)
│   ├── normalization.py              # Stage 4
│   ├── masters.py                    # dealer + asset masters
│   ├── confidence.py                 # Stage 5
│   └── pipeline.py                   # orchestrator
├── models/
│   ├── paddleocr/                    # PP-OCRv4 mobile (en + devanagari)
│   ├── yolov8n_sig_stamp.pt          # fine-tuned 5.6 MB
│   ├── detection.yaml                # per-class thresholds
│   └── qwen2.5-1.5b-instruct/        # safetensors (3 GB)
├── data/
│   ├── dealer_master.json            # mined from training set
│   └── asset_master.json
├── docs/
│   ├── architecture.md               # mermaid diagram + stage docs
│   └── error_analysis.md             # failure-mode catalog
└── sample_output/
    └── result.json                   # example extraction
```

## Architecture

Six stages, one per logical responsibility. Stages 2A and 2B run in
parallel against the same preprocessed page; Stage 3 is two-tier.

```
Input → [1: Ingestion] → [2A: OCR]   ↘
                          [2B: Vision] ↘
                                       [3: Extraction] → [4: Normalize] → [5: Confidence] → [6: Output]
                                       ↑ Tier 1: rules
                                       ↓ Tier 2: SLM (only when needed)
```

See `docs/architecture.md` for the full Mermaid diagram and per-stage
responsibilities.

## CLI usage

```
python executable.py <input>                           # single doc → stdout + sample_output/result.json
python executable.py <input> --output result.json       # specify output path
python executable.py --batch <directory>                # one JSON per line on stdout
python executable.py <input> --legacy                   # PS-reference flat shape
python executable.py <input> --offline                  # block any non-loopback network
python executable.py <input> --quiet                    # suppress info logs
```

### Exit codes

* `0` — success
* `1` — schema validation or output assembly failure
* `2` — unsupported input format / nonexistent path
* `3` — offline-mode network violation (caught by kill-switch)

### Output JSON shape (default)

```json
{
  "doc_id": "invoice_001",
  "fields": {
    "dealer_name": {"value": "ABC Tractors", "confidence": 0.92},
    "model_name": {"value": "Mahindra 575 DI", "confidence": 0.98},
    "horse_power": {"value": 50, "confidence": 0.95},
    "asset_cost": {"value": 525000, "confidence": 0.91},
    "signature": {"present": true, "bbox": [100, 200, 300, 250], "confidence": 0.94},
    "stamp": {"present": true, "bbox": [400, 500, 500, 550], "confidence": 0.91}
  },
  "confidence": 0.94,
  "processing_time_sec": 3.8,
  "cost_estimate_usd": 0.0,
  "error": null
}
```

The PS-reference flat shape (collapses per-field wrappers, drops per-field
confidences) is available via `--legacy`.

## Cost analysis

The pipeline has zero marginal monetary cost: every component is
open-source and runs locally. The reported `cost_estimate_usd` field
defaults to 0 but can be configured (in `utils/pipeline.py`'s
`COST_PER_SECOND_USD`) to reflect rented-CPU costs if you want to
amortize compute. For reference:

| Compute | Rate | Per-doc cost |
|---|---|---|
| Local laptop | $0 / hour | $0 |
| AWS t3.large CPU | $0.083 / hour | ~$0.0005 |
| Azure Standard_F4s_v2 | $0.169 / hour | ~$0.001 |

All are well below the $0.01 PS budget.

## Performance characteristics

Measured on a 5-document smoke test against `train_data_idfc/train/` on:
* CPU: AMD Ryzen with AVX, AVX2, FMA (no AVX-512)
* GPU: NVIDIA RTX 3050 6GB Laptop

| Doc | Total | OCR | Vision | Tier-1 | Tier-2 SLM | Doc Conf |
|---|---|---|---|---|---|---|
| 1 | 21.7s | 7.6s | 0.2s | 0.0s | 3.3s | 0.66 |
| 2 | 22.8s | 11.9s | 0.1s | 0.0s | 2.3s | 0.70 |
| 3 | 23.8s | 12.1s | 0.1s | 0.0s | 1.8s | 0.80 |
| 4 | 20.4s | 8.7s | 0.1s | 0.0s | 2.4s | 0.80 |
| 5 | 33.4s | 19.1s | 0.8s | 0.0s | 4.6s | 0.65 |

OCR dominates wall-clock latency. Optimization options:

* Lower OCR resolution from 300 → 200 DPI for the first pass
* Disable the second (Devanagari) engine for documents where the first
  (English) engine already produced > 50 confident tokens
* Deploy on a higher-tier GPU (RTX 4060+) — OCR roughly halves

## Reproducibility

The full pipeline can be rebuilt from scratch:

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Mine the masters from the training set (one-time, ~30-60 min)
python -m scripts.mine_masters

# 3. Train the YOLO sig/stamp detector (one-time, ~3-5 min on GPU)
python -m scripts.prepare_yolo_dataset
python -m scripts.train_yolo

# 4. Run validation evaluation (requires hand-labeled labels.json)
python -m tests.validation.evaluate --report report.json

# 5. Smoke-test offline mode
python -m scripts.smoke_offline
```

See `tasks.md` in the spec folder for the per-component build order.

## Known limitations

* **Gujarati script:** PaddleOCR PP-OCRv4 doesn't ship a Gujarati
  recognition model. Pure-Gujarati documents fall through to the
  Devanagari engine which produces poor results; the SLM fallback
  partially compensates by re-parsing what English tokens were captured.
* **Handwritten invoices:** OCR confidence drops on heavy handwriting;
  validation set should include 5+ handwritten documents to measure
  this honestly.
* **Multi-page packets:** the relevance scorer picks one page from
  multi-page PDFs. If the quotation spans two pages (rare), the second
  page is dropped.

## Testing

```bash
# Run the full unit suite (~5s, no live ML calls)
python -m pytest tests/unit -q --tb=short -p no:cacheprovider \
  --ignore=tests/unit/test_ocr.py \
  --ignore=tests/unit/test_detection.py \
  --ignore=tests/unit/test_slm.py

# Run live ML tests (~60s, requires bundled models)
python -m pytest tests/unit/test_ocr.py tests/unit/test_detection.py tests/unit/test_slm.py
```

See `scripts/run_tests.py` for a Windows-friendly wrapper that handles
pytest's tear-down hang in mixed-ML-runtime environments.

## Repository structure

```
InvoiceFlow/
├── backend/                          ← submission lives here
├── demo/                             ← optional FastAPI bridge for demo
├── src/                              ← React / InvoiceFlow frontend
├── train_data_idfc/train/            ← raw training images
└── .kiro/specs/invoice-extraction-pipeline/   ← formal spec docs
```

The graded artifact is everything under `backend/` packaged into
`submission.zip`. The `demo/` and `src/` folders are NOT included in
the submission.
