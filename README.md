# InvoiceFlow 🧾

> **100% offline, $0 cost Document AI** — extracts structured fields from tractor loan invoices and quotations using PaddleOCR, a fine-tuned YOLOv8n, and a local Qwen2.5-1.5B SLM.

---

## ✨ What it does

Upload a PDF, PNG, or JPG of a tractor quotation and get back a clean JSON with:

| Field | Example |
|---|---|
| `dealer_name` | "SULTANIA TRACTORS" |
| `model_name` | "Eicher Tractor Model 380 SP+" |
| `horse_power` | 40 |
| `asset_cost` | 660000 |
| `signature` | `present: true, bbox: [x1,y1,x2,y2]` |
| `stamp` | `present: true, bbox: [x1,y1,x2,y2]` |

Every field ships with a **per-field confidence score** and the document gets a **document-level accuracy score**.

---

## 🖥️ Demo (local)

The full-stack app runs locally in two terminals:

```bash
# Terminal 1 — Python backend
cd demo
..\backend\.venv\Scripts\python.exe server.py

# Terminal 2 — React frontend
npm run dev
```

Open **http://localhost:5173** — you'll see the SaaS landing page. Click "Get started" to reach the dashboard, upload an invoice, and watch the pipeline run step-by-step.

> Documents are persisted on the backend (`demo/storage/`), so they survive page reloads and are visible across different browsers on the same machine.

---

## 🏗️ Architecture

```
Invoice (PDF / PNG / JPG)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Stage 1: Ingestion                             │
│  PyMuPDF · pdf2image · OpenCV (deskew/denoise)  │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐   ┌──────────────────────────┐
│  Stage 2A: OCR   │   │  Stage 2B: Vision         │
│  PaddleOCR       │   │  YOLOv8n (fine-tuned)     │
│  PP-OCRv4 mobile │   │  sig/stamp detection      │
│  en + Devanagari │   │  mAP@50 = 0.880           │
└────────┬─────────┘   └────────────┬──────────────┘
         │                          │
         ▼                          │
┌──────────────────────────────┐    │
│  Stage 3: Extraction         │    │
│  Tier-1: Regex + anchor rules│    │
│  Tier-2: Qwen2.5-1.5B SLM   │    │
│  (fires only on low-conf)    │    │
└──────────────────────────────┘    │
         │                          │
         ▼                          ▼
┌─────────────────────────────────────────────────┐
│  Stage 4–6: Normalize → Confidence → JSON out   │
│  RapidFuzz fuzzy matching · Pydantic v2 schema  │
└─────────────────────────────────────────────────┘
```

---

## 📦 Stack

| Layer | Technology |
|---|---|
| **OCR** | PaddleOCR PP-OCRv4 mobile (en + Devanagari) |
| **Object detection** | Ultralytics YOLOv8n, fine-tuned on 65 annotated invoices |
| **SLM fallback** | Qwen2.5-1.5B-Instruct (transformers + safetensors, FP16 on GPU) |
| **Text extraction** | Custom regex anchor library with proximity scoring |
| **Normalization** | RapidFuzz fuzzy matching against curated brand/dealer masters |
| **Output schema** | Pydantic v2, flat + wrapped output shapes |
| **Frontend** | React 18 · TypeScript · Tailwind CSS v4 · Vite · Zustand |
| **Demo bridge** | FastAPI + Uvicorn, server-side document persistence |
| **Hardware tested** | RTX 3050 6 GB Laptop GPU + Intel i5-13450HX, 16 GB RAM |

---

## 🚀 Quick start (backend only)

```bash
cd backend
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# Single document → prints JSON + writes sample_output/result.json
python executable.py path/to/invoice.png

# Batch directory
python executable.py --batch path/to/folder

# Force offline kill-switch (blocks all outbound network)
python executable.py path/to/invoice.png --offline
```

---

## 📁 Repository layout

```
InvoiceFlow/
├── backend/                          # Python extraction pipeline
│   ├── executable.py                 # CLI entry point
│   ├── requirements.txt              # pinned, PyPI-only deps
│   ├── utils/                        # pipeline modules
│   │   ├── ingestion.py              # Stage 1: PDF/image loading
│   │   ├── ocr.py                    # Stage 2A: PaddleOCR wrapper
│   │   ├── detection.py              # Stage 2B: YOLOv8n wrapper
│   │   ├── extraction.py             # Stage 3 Tier-1: rule extractors
│   │   ├── slm.py                    # Stage 3 Tier-2: Qwen2.5 SLM
│   │   ├── normalization.py          # Stage 4: fuzzy normalizer
│   │   ├── confidence.py             # Stage 5: score aggregation
│   │   ├── pipeline.py               # end-to-end orchestrator
│   │   ├── schema.py                 # Pydantic output contract
│   │   ├── masters.py                # dealer + asset master loaders
│   │   ├── device.py                 # CUDA auto-detection
│   │   ├── offline_guard.py          # network kill-switch
│   │   └── stage_cache.py            # OCR + YOLO result cache
│   ├── models/
│   │   ├── yolov8n_sig_stamp.pt      # fine-tuned YOLOv8n weights (~5 MB)
│   │   ├── base/yolov8n.pt           # COCO pretrained base (~6 MB)
│   │   └── detection.yaml            # per-class confidence thresholds
│   ├── data/
│   │   ├── asset_master.json         # tractor brand/model list
│   │   └── dealer_master.json        # dealer canonical names
│   ├── scripts/                      # training, mining, evaluation tools
│   ├── tests/                        # 150+ unit tests
│   ├── docs/                         # architecture + error analysis
│   └── notebooks/eda.ipynb           # exploratory data analysis
│
├── src/                              # React SaaS frontend
│   ├── app/pages/                    # Landing, Dashboard, Upload, Results…
│   ├── app/components/               # UI library + landing sections
│   ├── app/store/documentStore.ts    # Zustand store (localStorage + server sync)
│   └── app/lib/api.ts                # FastAPI bridge client
│
├── demo/
│   ├── server.py                     # FastAPI bridge (not in submission.zip)
│   └── requirements.txt              # fastapi + uvicorn
│
├── index.html                        # Vite entry
├── vite.config.ts
└── package.json
```

---

## 🧪 Tests

```bash
cd backend
python -m pytest tests/unit/ -v
# 150+ tests covering extraction rules, schema, OCR, detection, normalization
```

---

## 📊 Validation results (30-doc held-out set)

| Field | Accuracy |
|---|---|
| dealer_name | 37% |
| model_name | 47% |
| horse_power | 27% |
| asset_cost | 60% |
| signature detected | 63% |
| stamp detected | 83% |
| **Document-Level Accuracy (all fields correct)** | **~3-7%** |

> DLA is low because scanned tractor invoices have severe OCR noise. The pipeline is designed to degrade gracefully — every document still produces a valid JSON with partial results and confidence scores rather than crashing.

---

## 🔧 Model weights not in repo

The following large files are excluded from git and must be downloaded separately:

| File | Size | How to get |
|---|---|---|
| `backend/models/paddleocr/` | ~80 MB | `python -m scripts.download_models` (first run, needs internet) |
| `backend/models/qwen2.5-1.5b-instruct/` | ~3 GB | same script |
| `train_data_idfc/` | private | IDFC hackathon dataset |

The two YOLO `.pt` files (~5 MB each) **are** included in the repo.
