# PRD — Intelligent Document AI for Field Extraction from Invoices

**Project codename:** InvoiceFlow
**Hackathon:** IDFC GenAI PS — Convolve 4.0
**Doc owner:** [you]
**Last updated:** 2026-05-28
**Status:** Draft v2 — fully offline, $0 budget

---

## 1. Problem Statement

Banks need to automate extraction of key fields from invoice-type documents (running use case: tractor loan quotations) to accelerate credit decisioning, vendor reconciliation, and loan disbursal. Manual entry is slow and error-prone. Documents vary heavily across:

- **Layout / structure** — every dealer uses a different template
- **Language** — English, Hindi, Gujarati, mixed vernaculars
- **Quality** — clean digital PDFs, scanned, handwritten, mobile photos

The system must extract six structured fields per document and emit a single JSON object.

## 2. Goals & Non-Goals

### Goals
- **100% offline** — no internet calls at inference time, no paid APIs, no cloud services
- **$0 marginal cost** — runs on the evaluator's CPU/low-tier GPU using only open-source weights bundled with the submission
- End-to-end pipeline: PDF/image in → structured JSON out
- ≥95% Document-Level Accuracy on the held-out evaluation set (~100 unseen invoices)
- ≤30s average latency per document on CPU
- Generalize beyond tractor quotations to retail/industrial invoices
- Ship as a self-contained `submission.zip` with `executable.py`, `requirements.txt`, `README.md`, `utils/`, `sample_output/result.json`, and bundled model weights

### Non-Goals
- Any cloud/SaaS/API integration (OpenAI, Anthropic, Google Vision, AWS Textract, HF Inference API — all banned)
- Production-grade auth, multi-tenancy, billing
- Real-time collaboration / streaming inference
- Training large foundation models from scratch
- Frontend integration in the graded submission (React UI stays as separate optional demo)

## 3. Offline-First Constraints (hard rules)

| Constraint | Rule |
|---|---|
| Network | Pipeline must run with `--offline` flag that monkey-patches `urllib`/`requests` to refuse outbound calls. CI test will run inside a network-isolated container. |
| Models | All weights bundled in the zip OR downloaded by an explicit `setup.py` step that the evaluator runs once before evaluation. Default: bundle everything. |
| Tokenizers | All tokenizers cached locally (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` env vars set in the executable). |
| Fonts/locales | Bundle PaddleOCR's `ppocr` keys for en/hi/gu. No font downloads. |
| Submission size | Target <2GB total. Hard cap 5GB. |

## 4. Target Fields & Evaluation Rules

| Field | Datatype | Match Rule | Source |
|---|---|---|---|
| `dealer_name` | string | ≥90% fuzzy match against master file | OCR text + SLM |
| `model_name` | string | exact match against asset master | OCR text + SLM |
| `horse_power` | int | numeric equality within ±5% tolerance | regex on OCR text |
| `asset_cost` | int (digits only, no currency symbols) | numeric equality within ±5% tolerance | regex + SLM fallback |
| `signature` | `{present: bool, bbox: [x1,y1,x2,y2]}` | presence correct & IoU ≥ 0.5 | YOLOv8n fine-tuned |
| `stamp` | `{present: bool, bbox: [x1,y1,x2,y2]}` | presence correct & IoU ≥ 0.5 | YOLOv8n fine-tuned |

**Document-Level Accuracy (DLA):** all six fields correct for a single doc → 1, else 0.

## 5. Dataset

- **Location:** `train_data_idfc/train/`
- **Count:** 495 PNG images (~871 MB)
- **Composition (observed from filenames):**
  - Numeric IDs with `_pgN` suffix → pages extracted from larger loan files (1–51 pages indicates these are deep inside multi-page application packets)
  - `_OTHERS_v1`, `_OTHERS_v2` → multiple resubmissions per loan
  - `_VEHICLE QUOTATION_`, `_Proforma_`, `_Quotation_` → typed doc categories
  - `_Android_417_T_` → mobile-camera captures
- **No ground-truth labels provided** — by design, per the PS

### Annotation Strategy (handling lack of GT)
1. **Manual seed set:** ~50 images hand-labeled for all 6 fields → used as validation set + YOLO training set
2. **Pseudo-labels:** rule-based extraction on the remaining ~445 → confidence-weighted bootstrap
3. **Self-consistency voting:** rules vs SLM agreement → high-confidence pseudo-labels
4. **Synthetic augmentation:** templated invoices for stamp/signature variety if YOLO underfits

## 6. Architecture

```
                     ┌─────────────────────────────────────────┐
                     │          INPUT (PDF or image)           │
                     └────────────────────┬────────────────────┘
                                          ↓
  ┌───────────────────────────────────────────────────────────────────┐
  │  STAGE 1: INGESTION (offline)                                     │
  │  • PyMuPDF (digital PDFs → text + images)                         │
  │  • pdf2image / Pillow (scanned PDFs → 300 DPI PNG)                │
  │  • OpenCV preprocessing: deskew, denoise, contrast normalization  │
  │  • Page relevance scorer (skip non-quotation pages in multi-page) │
  └────────────────────┬──────────────────────────────────────────────┘
                       ↓
  ┌────────────────────┴────────────────────┐
  ↓                                          ↓
  ┌──────────────────────────┐    ┌─────────────────────────────────┐
  │  STAGE 2A: TEXT LAYER    │    │  STAGE 2B: VISION LAYER         │
  │  PaddleOCR (en+hi+gu)    │    │  YOLOv8n fine-tuned             │
  │  ALL LOCAL .pdmodel files│    │  Local .pt weights              │
  │  → words + bboxes + conf │    │  → signature & stamp bboxes     │
  └─────────────┬────────────┘    └─────────────┬───────────────────┘
                ↓                                ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │  STAGE 3: FIELD EXTRACTION (text fields)                        │
  │  Tier 1 — Deterministic regex/anchor rules (free, ~5ms)         │
  │    • HP:  \d+\s*(HP|H\.P\.|hp|बल|एचपी)                          │
  │    • Cost: anchored to "Total", "Grand Total", "₹", "Rs."       │
  │    • Dealer: anchored to "Dealer", "M/s", letterhead region     │
  │    • Model: anchored to "Model", "Tractor", brand keywords      │
  │  Tier 2 — Local SLM fallback (Qwen2.5-1.5B Q4_K_M via llama.cpp)│
  │    Triggered only when Tier 1 confidence < threshold or fails   │
  │    Receives OCR text only (not the image) — CPU-friendly        │
  └─────────────────────────────┬───────────────────────────────────┘
                                ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │  STAGE 4: NORMALIZATION & VALIDATION (offline)                  │
  │  • RapidFuzz against dealer master (mined from dataset)         │
  │  • Exact match against asset master (mined from dataset)        │
  │  • Numeric: strip currency, commas; sanity-check ranges         │
  │    (HP: 15–150, Cost: 100k–5M)                                  │
  │  • Cross-field consistency (HP correlates with cost)            │
  └─────────────────────────────┬───────────────────────────────────┘
                                ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │  STAGE 5: CONFIDENCE & SELF-CHECK                               │
  │  • Per-field confidence = f(OCR conf, rule match strength,      │
  │    fuzzy score, SLM-vs-rule agreement)                          │
  │  • Document-level confidence = weighted aggregate               │
  │  • Flag low-confidence extractions for manual review            │
  └─────────────────────────────┬───────────────────────────────────┘
                                ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │  STAGE 6: OUTPUT                                                │
  │  Structured JSON per spec + processing_time + cost_estimate     │
  └─────────────────────────────────────────────────────────────────┘
```

### Tech Stack — 100% offline, all open-source

| Layer | Choice | License | Offline-ready? |
|---|---|---|---|
| PDF parsing | PyMuPDF | AGPL/Commercial | Yes, pure local |
| Image → PIL | Pillow + pdf2image (poppler) | MIT | Yes |
| OCR | PaddleOCR (PP-OCRv4 mobile, en+hi+gu) | Apache 2.0 | Yes, models bundled |
| Object detection | YOLOv8n (Ultralytics) | AGPL-3.0 | Yes, weights bundled |
| SLM (text fallback) | Qwen2.5-1.5B-Instruct Q4_K_M | Apache 2.0 | Yes, GGUF bundled |
| SLM runtime | llama-cpp-python | MIT | Yes, pure local |
| Fuzzy matching | RapidFuzz | MIT | Yes |
| Image preproc | OpenCV-python-headless + scikit-image | Apache 2.0 / BSD | Yes |
| Validation | Pydantic v2 | MIT | Yes |
| Orchestration | Python 3.10 | PSF | Yes |

### Why these choices
- **PaddleOCR over Tesseract** — Tesseract's Hindi/Gujarati accuracy is poor; PaddleOCR PP-OCRv4 is significantly better on Indic scripts and the mobile variant is only ~10MB per language.
- **Text-only SLM over VLM** — OCR already gives us the text. Running a 2B vision-language model on CPU adds 10–30s per call. A 1.5B text-only model in 4-bit quantization runs in 1–3s on CPU and we don't lose information because the OCR layer is the eyes.
- **YOLOv8n over larger detectors** — n variant is 6MB, runs at ~5 FPS on CPU, sufficient for the IoU ≥ 0.5 bar.
- **llama-cpp-python over Transformers** — Transformers needs PyTorch + CUDA setup hassle. llama.cpp runs purely on CPU with no extra deps and the GGUF format is portable.

## 7. Training & Fine-Tuning

### What we train
**Only the YOLOv8n signature/stamp detector.** Everything else is zero-shot or rule-based.

### Training plan
1. **Annotate 50–80 representative images** in LabelImg (offline desktop tool) with two classes: `signature`, `stamp`
2. **Split:** 40 train / 10 val / 20 held-out test
3. **Augment:** rotation ±5°, brightness, gaussian noise, JPEG compression (mimics scan/photo artifacts)
4. **Fine-tune YOLOv8n** from COCO-pretrained weights, 50–100 epochs, image size 640
5. **Target:** mAP@50 ≥ 0.85 on held-out test (clears IoU ≥ 0.5 requirement)
6. **Training environment:** local RTX 3050 6GB — fully offline, ~20–30 min training run. No Colab dependency.

### Hardware notes
- **Dev machine:** RTX 3050 6GB. Used for: YOLO fine-tuning, EDA acceleration, SLM experimentation. All offline.
- **Eval machine:** unknown (PS says "CPU or low-tier GPU"). Pipeline auto-detects CUDA at runtime via `torch.cuda.is_available()`:
  - PaddleOCR: `use_gpu=True` if CUDA, else CPU
  - YOLOv8n: Ultralytics auto-detects
  - llama.cpp: `n_gpu_layers=-1` if CUDA, else `0` (pure CPU)
- Same GGUF / .pt / Paddle weights work for both modes — no separate builds.

### What we DO NOT train
- OCR model (use PaddleOCR pretrained)
- SLM (use Qwen2.5-1.5B pretrained, zero-shot prompting only)
- Text field extractors (rule-based + SLM fallback with prompt engineering)

## 8. Cost & Latency Budget

| Stage | Target latency (CPU) | Notes |
|---|---|---|
| Ingestion | 1–2s | PDF rasterization is the bottleneck on multi-page |
| OCR (PaddleOCR mobile) | 4–8s | Single page @ 300 DPI on CPU |
| YOLO inference | 0.3–0.5s | YOLOv8n on CPU |
| Rules + normalization | <0.1s | Regex is essentially free |
| SLM fallback (when invoked, ~30% of docs) | 2–4s | Qwen2.5-1.5B Q4 on CPU |
| **Total p50 (no SLM)** | **~8s** | Most docs hit Tier 1 only |
| **Total p95 (with SLM)** | **~16s** | Worst case |

### Cost per doc
- All weights local, all inference local → **$0 marginal cost**
- Even on a paid CPU instance (AWS t3.large @ $0.08/hr): 8s = $0.00018/doc
- Reported in JSON: `"cost_estimate_usd": 0.0002` (compute time × commodity CPU rate)

## 9. Submission Deliverable

```
submission.zip                          (~1.2 GB)
├── executable.py                       # Main entry: takes PDF path, emits JSON
├── requirements.txt                    # Pinned versions, all from PyPI
├── README.md                           # Architecture, pipeline, cost analysis, diagrams
├── setup.py                            # Optional: validates bundled weights
├── utils/
│   ├── __init__.py
│   ├── ingestion.py                    # PDF/image loading + preprocessing
│   ├── ocr.py                          # PaddleOCR wrapper (offline mode)
│   ├── detection.py                    # YOLOv8n wrapper for sig/stamp
│   ├── extraction.py                   # Rules + SLM fallback for text fields
│   ├── slm.py                          # llama.cpp wrapper, prompt templates
│   ├── normalization.py                # Fuzzy matching, validation
│   ├── masters.py                      # Dealer + asset master loaders
│   ├── confidence.py                   # Confidence scoring logic
│   └── schema.py                       # Pydantic models
├── models/
│   ├── paddleocr/                      # ~50 MB — det/rec/cls models for en+hi+gu
│   ├── yolov8n_sig_stamp.pt            # ~6 MB — fine-tuned weights
│   └── qwen2.5-1.5b-q4_k_m.gguf        # ~1 GB — 4-bit text SLM
├── data/
│   ├── dealer_master.json              # Mined from training set
│   └── asset_master.json               # Tractor brand/model master
├── docs/
│   └── architecture.png                # Diagram for README
└── sample_output/
    └── result.json
```

### Output JSON spec (per the PS)

```json
{
  "doc_id": "invoice_001",
  "fields": {
    "dealer_name": "ABC Tractors Pvt Ltd",
    "model_name": "Mahindra 575 DI",
    "horse_power": 50,
    "asset_cost": 525000,
    "signature": {"present": true, "bbox": [100, 200, 300, 250]},
    "stamp": {"present": true, "bbox": [400, 500, 500, 550]}
  },
  "confidence": 0.96,
  "processing_time_sec": 3.8,
  "cost_estimate_usd": 0.0002
}
```

## 10. Evaluation Metrics

### Primary
- **Document-Level Accuracy ≥ 95%** on the unseen 100-doc eval set

### Secondary
- Field-level mAP@[50:95] for signature and stamp
- Average latency per doc (≤30s)
- Cost per doc (<$0.01 — we'll report ~$0.0002)

### Internal validation set
- 30–50 hand-labeled docs from the training pool, stratified by:
  - Language (EN / HI / GU / mixed)
  - Quality (digital / scanned / photo)
  - Layout (typed table / handwritten / form)

## 11. Bonus Deliverables

- **EDA notebook:** state distribution, language distribution (mined via OCR script detection), digital-vs-scanned split, layout clustering, file-size histogram, page-count analysis
- **Error analysis:** confusion matrix per field, failure category breakdown
- **Architecture diagram:** Mermaid + PNG export
- **Local demo (optional, not graded):** existing React/InvoiceFlow UI talking to a FastAPI wrapper around the pipeline. Both run on localhost. Stays out of `submission.zip` — built as a separate `demo/` folder for show-and-tell.

## 12. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PaddleOCR weak on Gujarati handwritten | Medium | High | Add Tesseract as second OCR for voting; SLM can re-parse muddled text |
| Few sig/stamp annotations → YOLO underfits | Medium | Medium | Active learning loop: label hard negatives iteratively; synthetic augmentation |
| SLM hallucinates dealer/model names | Medium | High | Always cross-check SLM output against OCR text; reject if not substring-matching |
| Eval set has unseen tractor brands | High | Medium | Mine master from training set + maintain "fallback to OCR text" path; keep asset master a soft constraint |
| Submission zip too big | Medium | Medium | Use Q4_K_M instead of Q5/Q8; consider Qwen2.5-0.5B if size is critical |
| First-run network attempts | High | High | Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; use llama.cpp's local-only loader; smoke-test in `--no-network` Docker |
| 30s latency exceeded on multi-page PDFs | Medium | Medium | Page relevance classifier first; only deeply process the page most likely to be the quotation |
| llama-cpp-python build fails on evaluator's machine | Medium | High | Ship pre-built wheels in requirements; document Python 3.10 requirement; provide fallback to Transformers if needed |

## 13. Open Questions

1. Authoritative dealer/asset master from organizers, or mine our own from the dataset?
2. For multi-page PDFs in eval set — extract from all pages or just page 1?
3. Annotation tool: LabelImg (offline, lighter) or CVAT (browser, more features)?
4. Frontend in submission or out? — **Decision: out, keep separate as `demo/`**

## 14. Phase Plan

| Phase | Duration | Deliverable | Internet needed? |
|---|---|---|---|
| 0. Setup & EDA | 1 day | Notebook with dataset stats, language detection, layout clustering | First setup only |
| 1. Annotation | 1 day | 50 hand-labeled images (sig/stamp + 6 fields) | No |
| 2. Model bundle | 0.5 day | Download all weights (Paddle, Qwen GGUF, YOLO base) once and stash in `models/` | One-time |
| 3. OCR + rules baseline | 2 days | Tier-1 extraction working on validation set, measure DLA | No |
| 4. YOLO fine-tune | 0.5 day | sig/stamp detector with mAP@50 ≥ 0.85 (local RTX 3050) | No |
| 5. SLM fallback | 1 day | Qwen2.5-1.5B integrated for low-confidence cases | No |
| 6. Master mining + validation | 1 day | Dealer/asset masters + cross-checks | No |
| 7. Confidence calibration | 0.5 day | Per-field + doc-level confidence scoring | No |
| 8. Offline hardening | 0.5 day | `--offline` smoke test in network-isolated container | No |
| 9. Submission packaging | 1 day | Zip with executable, README, diagrams, EDA | No |
| **Total** | **~9 days** | | |

## 15. Success Criteria

- [ ] DLA ≥ 95% on internal validation set
- [ ] Average latency ≤ 30s per doc on CPU
- [ ] Reported cost ≤ $0.01 per doc (actual: $0)
- [ ] Submission zip runs end-to-end on a clean machine via `python executable.py <pdf_path>` with NO internet
- [ ] README has architecture diagram, cost analysis, error analysis
- [ ] Reproducible: `pip install -r requirements.txt` + run = same outputs
- [ ] **Smoke test passes in a `--network=none` Docker container**
