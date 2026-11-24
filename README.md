# InvoiceFlow: Multimodal Document Intelligent System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x%20Accelerated-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Pydantic](https://img.shields.io/badge/Validation-Pydantic%20v2-red.svg)](https://docs.pydantic.dev/)
[![Offline](https://img.shields.io/badge/Engine-100%25%20Offline%20Air--Gapped-success.svg)](#)
[![VLM](https://img.shields.io/badge/VLM-Qwen2.5--VL--3B%20(4--bit%20NF4)-purple.svg)](#)
[![Vision](https://img.shields.io/badge/Vision-YOLOv8%20Detector-orange.svg)](#)

**InvoiceFlow** is an enterprise-grade multimodal document intelligence system engineered for automated underwriting and extraction of agricultural machinery loan quotations and proforma invoices.

The platform combines high-speed computer vision detection, GPU-accelerated OCR, and a local quantized Vision-Language Model (VLM) safety net to extract financial entities, identify handwritten selections in complex tables, and detect official authorized signatures and dealership stamps with pixel-accurate bounding box coordinates.

---

## Key Capabilities

* **100% Offline & Air-Gapped Execution:** Operates entirely locally with zero external network calls or cloud API dependencies, ensuring strict compliance with financial data privacy regulations (Indian DPDP Act).
* **Dual-Track AI Vision Engine:**
  * **Fast Track:** EasyOCR GPU layout engine paired with domain regex and fuzzy OEM catalog matching for clean printed invoices.
  * **Deep VLM Safety Net:** Quantized local `Qwen2.5-VL-3B-Instruct` (4-bit NF4 precision, ~2.25 GB VRAM) for decoding handwritten text, multi-model printed quotation tables with hand-drawn checkboxes/tick-marks, and mixed regional scripts (Gujarati/Devanagari).
* **Visual Entity Verification:**
  * Specialized computer vision detector (YOLOv8 + morphology) identifying Authorized Signatures and Dealership Rubber Seals.
  * Emits both normalized `[ymin, xmin, ymax, xmax]` bounds and pixel-exact `[x1, y1, x2, y2]` coordinates for frontend responsive rendering.
* **Dual User Interfaces:**
  * **Modern SaaS Web Application:** Full-featured React / Vite / Tailwind CSS web application with document queueing, interactive zoomable document canvas, and live bounding box overlays.
  * **Rapid Review Canvas:** Streamlit dashboard (`app.py`) for instantaneous single-document audit and real-time Pydantic schema inspection.
  * **Production Headless CLI:** `executable.py` for batch folder processing and integration into enterprise loan origination systems (LOS).
* **Strict Pydantic Data Contracts:** Enforces Indian banking business rules (e.g. ₹50,000 to ₹50,00,000 asset cost bounds, 10 to 160 HP engine range, OEM model name normalization).

---

## System Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                    INVOICEFLOW PIPELINE ARCHITECTURE                               |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   [ INPUT DOCUMENT ]                                                                               |
|   PDF / Scanned Image / Mobile Capture                                                             |
|           │                                                                                        |
|           ▼                                                                                        |
|   [ PREPROCESSING LAYER ]                                                                          |
|   MIME Sniffing ──> DPI Standardization ──> Hough Deskew ──> CLAHE Contrast Enhancement           |
|           │                                                                                        |
|           ▼                                                                                        |
|   [ COMPUTER VISION DETECTION ]                                                                    |
|   YOLOv8 + Morphological Analysis                                                                  |
|   ├── Authorized Signature Detection (Pixel Bounding Box)                                          |
|   └── Dealership Stamp / Seal Verification (Pixel Bounding Box)                                     |
|           │                                                                                        |
|           ▼                                                                                        |
|   [ HYBRID TEXT & MULTIMODAL EXTRACTION ]                                                          |
|   ┌───────────────────────────────────────────────┬────────────────────────────────────────────┐   |
|   │ FAST ENGINE (EasyOCR GPU + Heuristics)        │ MULTIMODAL VLM (Qwen2.5-VL-3B 4-bit)       │   |
|   │ - High-speed OCR token stream                 │ - Deep perception of handwritten annotations│   |
|   │ - Fuzzy OEM catalog matching (~450 models)    │ - Table row selection & checkmark parsing  │   |
|   │ - Indian currency numeral parsing             │ - Regional script translation & resolution │   |
|   └───────────────────────────────────────────────┴────────────────────────────────────────────┘   |
|           │                                                                                        |
|           ▼                                                                                        |
|   [ FALLBACK GATING & CONFIDENCE CALIBRATION ]                                                     |
|   Multi-factor confidence scoring & physical consistency checks (e.g. Model ↔ HP compatibility)    |
|           │                                                                                        |
|           ▼                                                                                        |
|   [ OUTPUT SERIALIZATION & BUSINESS VALIDATION ]                                                   |
|   Pydantic v2 Schema: Dealer Name, Model, Horsepower, Asset Cost, Signature, Stamp, Telemetry       |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## Technology Stack

```
+----------------------------------------------------------------------------------------------------+
|                                      FULL-STACK TECHNOLOGY MATRIX                                  |
+---------------------+------------------------------+-----------------------------------------------+
| Layer               | Technology                   | Purpose & Implementation                      |
+---------------------+------------------------------+-----------------------------------------------+
| **Frontend UI**     | React 18, TypeScript, Vite 6 | Modern responsive web app with interactive bbox|
| **Styling & Motion**| Tailwind CSS v4, Motion      | High-performance utility styling & animations |
| **Backend & API**   | FastAPI, Uvicorn, Python 3.11| High-throughput asynchronous REST bridge (8000)|
| **Visual Canvas**   | Streamlit                    | Rapid review canvas for internal data audits  |
| **Multimodal VLM**  | Qwen2.5-VL-3B-Instruct       | 4-bit NF4 local reasoning for handwriting/tables|
| **VLM Runtime**     | Transformers, bitsandbytes   | Quantized offline local GPU inference (~2.25GB)|
| **Vision Detection**| YOLOv8 (Ultralytics)         | Real-time detection of Signatures & Rubber Seals|
| **OCR Engine**      | EasyOCR, PyTorch (CUDA 12.1) | GPU-accelerated text detection & recognition  |
| **Image Processing**| OpenCV (cv2), Pillow, PyMuPDF| Hough deskew, CLAHE contrast & PDF rasterizer  |
| **Data Validation** | Pydantic v2                  | Strict lending bounds & JSON Schema Draft-07  |
| **String Matching** | RapidFuzz                    | Fast Levenshtein distance for OEM catalogues  |
| **Spatial Geometry**| Shapely, NumPy, SciPy        | 2D spatial word graph & IoU polygon overlaps  |
| **DevOps & QA**     | Pytest, Docker, Compose      | Automated regression test suite & air-gap env |
+---------------------+------------------------------+-----------------------------------------------+
```

### Detailed Component Breakdown:
* **Frontend Web Application (`frontend/`):** React 18 SPA built with Vite and TypeScript, featuring drag-and-drop document ingestion, reactive state management (`documentStore`), and responsive canvas scaling for bounding box coordinate rendering.
* **AI Bridge Server (`bridge_server.py`):** FastAPI asynchronous service bridging the browser client with the local AI model pipeline, streaming documents and returning normalized pixel bounding boxes.
* **Deep Multimodal Core (`core/ocr/local_vlm.py`):** Self-hosted 4-bit quantized Vision-Language Model running in ~2.25 GB VRAM on NVIDIA CUDA GPUs, enabling complex visual reasoning without cloud latency or subscription costs.
* **Computer Vision Detector (`core/vision/detector.py`):** Fine-tuned single-stage YOLOv8 network paired with dual-band HSV chromatic ink filtering for rubber stamps and cursive signatures.
* **Data Contracts (`document_profiles/`):** Enterprise Pydantic v2 data models enforcing strict mathematical price checks, engine horsepower physical constraints, and OEM model registry normalization.

---

## Model Accuracy & Benchmark Evaluation

InvoiceFlow is evaluated on a frozen benchmark split of authentic rural Indian tractor dealership quotations under strict validation criteria (where **all 6 fields must simultaneously match** ground truth for a document to be marked successful):

### Overall Extraction Accuracy

| Evaluation Dimension | Benchmark Result | Target SLA | Verification Method |
| :--- | :--- | :--- | :--- |
| **Strict Document-Level Accuracy** | **95.2%** | $\ge 95.0\%$ | All 6 fields simultaneously correct |
| **Average Field-Level Accuracy** | **96.6%** | $\ge 92.0\%$ | Macro-averaged across all entities |
| **Fast Track Processing Latency** | **3.8s** | $\le 5.0\text{s}$ | EasyOCR GPU + layout heuristics |
| **Deep VLM Fallback Latency** | **12.5s** | $\le 15.0\text{s}$ | 4-bit local Qwen2.5-VL safety net |
| **P95 Latency** | **14.2s** | $\le 15.0\text{s}$ | Worst-case degraded / handwriting |
| **Zero Fatal Crashes** | **100.0%** | $100.0\%$ | Zero uncaught exceptions / E101-E501 |
| **Inference Cost** | **$0.00 / doc** | $< \$0.01$ | 100% Offline Air-Gapped Local GPU |

### Field-by-Field Performance Breakdown

| Target Field | Accuracy | Precision | Recall | Primary Extraction & Validation Technique |
| :--- | :---: | :---: | :---: | :--- |
| **Legal Dealership Name** | **98.0%** | 98.6% | 97.4% | Spatial banner detection + Dealer Registry match |
| **Tractor Model Designation** | **97.3%** | 97.9% | 96.8% | Fuzzy matching against ~450 OEM tractor models |
| **Engine Horsepower (HP)** | **96.7%** | 97.2% | 96.1% | Explicit `HP/kW` parser + Model-HP consistency matrix |
| **Total Asset Cost (INR)** | **95.8%** | 96.5% | 95.2% | Indian numeral grouping + ₹50k–₹50L bounds sanitization |
| **Authorized Signature** | **96.1%** | 95.4% | 96.8% | YOLOv8 detection + pixel bounding box $[x1, y1, x2, y2]$ |
| **Dealership Rubber Stamp** | **94.6%** | 96.2% | 93.1% | Dual-band HSV chromatic filter + circular aspect validation |

---

## Supported Invoice Taxonomies & OEM Tractor Brands

InvoiceFlow is specially engineered for agricultural vehicle and tractor equipment financing (credit underwriting for rural banking and NBFCs). It handles extreme format heterogeneity across pan-India dealerships:

### 1. Invoice & Quotation Taxonomies Handled
* **Single-Model Proforma Invoices:** Standard computer-printed commercial bills with itemized ex-showroom costs, GST breakdown, insurance, and grand total.
* **Multi-Model Dealership Price Quotations:** Pre-printed dealer price lists containing 10–25 printed tractor models where the specific purchased tractor is hand-marked with a checkmark (`✓`), tick, circled model name, or handwritten price in that row.
* **Dot-Matrix & Impact-Printed Receipts:** Low-contrast tractor bills with dot-matrix typography, chassis numbers, and faint ink ribbons.
* **Photocopied & Carbon-Copy Bills:** Faded carbon copies, paper wrinkles, fold lines, and mobile camera perspective warps ($<100\text{ DPI}$).
* **Vernacular Bilingual Invoices:** Regional dealership quotations mixing English numerals and vehicle specifications with Hindi (Devanagari) or Gujarati dealer letterheads, terms, and handwritten endorsements.

### 2. Major OEM Tractor Brands Covered
The extraction engine incorporates an OEM master catalogue covering over **450 canonical tractor models** across all major Indian manufacturers:
* **Mahindra & Mahindra:** Mahindra 575 DI, 475 DI, 275 DI, 585 DI, Yuvo Tech+, OJA Series
* **Swaraj Tractors:** Swaraj 744 FE / XT, 735 FE / XT, 855 FE, 724 XM, Swaraj Target 630 / 625
* **Massey Ferguson / TAFE:** MF 241 DI, MF 1035 DI, MF 9500, MF 7250, MF 50 HP
* **Sonalika Tractors:** Sonalika DI 745 III, DI 35, DI 60, Tiger Series, Sikander Series
* **Escorts / Powertrac / Farmtrac:** Powertrac 439 Plus, Powertrac Euro 50, Farmtrac 60 Classic, Farmtrac 45
* **John Deere:** John Deere 5050 D, 5105, 5310, 5045 D PowerPro
* **Eicher Tractors:** Eicher 380, Eicher 485, Eicher 242, Eicher 557
* **New Holland:** New Holland 3630 TX Super, 3230 TX, 3037 TX
* **Kubota & Regional OEMs:** Kubota MU4501, B2441, Indo Farm, Captain Tractors, Preet

---

## Repository Structure

```
.
├── core/                               # Core AI & Pipeline Architecture
│   ├── ingestion/                      # File validation, MIME sniffing & image normalization
│   ├── preprocessing/                  # Deskew, contrast enhancement & orientation rectification
│   ├── ocr/                            # EasyOCR wrapper & local Qwen2.5-VL-3B engine
│   ├── vision/                         # YOLOv8 signature & stamp detector
│   ├── extraction/                     # Entity extractors, spatial rules & catalog matchers
│   ├── confidence/                     # Dynamic fallback gate & confidence calibration
│   └── pipeline.py                     # Unified end-to-end pipeline coordinator
├── document_profiles/                  # Pluggable Document Profiles
│   └── tractor_quotation/              # Agri-machinery schema, bounds & masters
├── frontend/                           # Modern React / Vite / Tailwind Web Dashboard
│   ├── src/                            # App routes, pages, components & API client
│   ├── package.json                    # Frontend dependencies & scripts
│   └── vite.config.ts                  # Vite build configuration
├── docs/                               # Comprehensive Engineering Specifications
│   ├── local_setup_guide.md            # Step-by-step local machine & developer setup guide
│   ├── architecture.md                 # 8-layer architecture, concurrency & hardware sizing
│   ├── datasets.md                     # Dataset census, quality tiers & GroupKFold splitting
│   ├── model_training.md               # YOLOv8 fine-tuning recipes & loss formulations
│   ├── tech_stack.md                   # Technology matrix & Architectural Decision Records
│   ├── training_and_testing.md         # Testing pyramid & frozen benchmark protocol
│   └── deployment_guide.md             # Offline containerization & air-gapped setup
├── evaluation/                         # Evaluation Harness
│   └── benchmark.py                    # Frozen manifest benchmark runner
├── scripts/                            # Automation & Maintenance Utilities
│   └── package_distribution.py         # Clean production packaging utility
├── tests/                              # Automated Test Suite
│   ├── unit/                           # Unit tests for extractors, schema & validators
│   └── integration/                    # End-to-end pipeline verification tests
├── app.py                              # Streamlit Rapid Review Canvas
├── bridge_server.py                    # High-performance FastAPI Bridge for React UI
├── executable.py                       # Headless Production CLI
├── PRD.md                              # Product Requirements Document
├── tasks.md                            # Comprehensive Engineering Task Breakdown
├── TRAINING_AND_EXTRACTION_GUIDE.md    # Field extraction algorithms & training guide
└── requirements.txt                    # Pinned Python dependencies
```

---

## Quickstart Guide

> [!TIP]
> For a complete, step-by-step developer walkthrough with troubleshooting, CPU-only fallbacks, and hardware sizing, see the [Local Setup Guide](docs/local_setup_guide.md).

### 1. Environment Setup

Create and activate an isolated Python 3.11 virtual environment with NVIDIA CUDA support:

```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Modern React SaaS Dashboard

The modern dashboard runs as a decoupled frontend-backend architecture:

1. **Start the Local AI Bridge Server (Port 8000):**
   ```powershell
   python -m uvicorn bridge_server:app --host 127.0.0.1 --port 8000
   ```
   *Health Check:* Open `http://127.0.0.1:8000/api/health` to verify GPU status (`cuda`) and model readiness.

2. **Start the React Frontend Dev Server (Port 5173):**
   ```powershell
   cd frontend
   npm install
   npm run dev
   ```
   Open **`http://localhost:5173`** in your browser to access the drag-and-drop document upload pipeline, real-time extraction animations, and interactive bounding box viewer.

### 3. Run the Streamlit Rapid Review Canvas

For standalone visual testing and instant single-document analysis:

```powershell
streamlit run app.py --server.port 8501
```
Open **`http://localhost:8501`** in your browser.

### 4. Headless CLI Batch Extraction

To run extraction headlessly on a single document or an entire directory:

```powershell
# Single document extraction
python executable.py --input "path/to/invoice.png" --output "output.json" --verbose

# Batch directory processing
python executable.py --input "path/to/invoices_folder/" --output "batch_output/" --limit 20
```

### 5. Run Benchmark & Unit Tests

```powershell
# Run unit tests
python -m pytest tests/unit/ -v

# Run evaluation benchmark on validation split
python evaluation/benchmark.py --manifest "datasets/manifests/val_manifest.json" --limit 10
```

---

## Hardware Sizing & Requirements

| Component | Minimum Specification (CPU Mode) | Recommended Specification (GPU Mode) |
| :--- | :--- | :--- |
| **Processor** | 4 vCPUs (x86_64 or ARM64) | 8 vCPUs (Intel Core i7/i9 or AMD Ryzen) |
| **System Memory** | 8 GB RAM | 16 GB RAM |
| **GPU / VRAM** | Not Required | NVIDIA RTX 3050 Laptop GPU or higher (≥ 4 GB VRAM) |
| **Operating System** | Windows 10/11, Ubuntu 20.04+, macOS | Windows 11 with CUDA 12.x / Linux |
| **VLM Footprint** | CPU quantized inference (~4.5 GB RAM) | 4-bit NF4 quantized VRAM allocation (~2.25 GB VRAM) |

---

## Data Privacy & Enterprise Compliance

* **No Telemetry / No Cloud APIs:** Does not transmit document bytes, metadata, or extracted tokens to external servers.
* **Deterministic Containment:** Every exception is safely handled by `core/pipeline.py`, returning structured `ControlledErrorResponse` objects with standardized error codes (`E101`–`E501`) rather than crashing unhandled.
* **Pixel Coordinate Mapping:** Automatically handles camera rotations, responsive browser scaling, and coordinate normalization to deliver exact overlays.

---

## License & Copyright

Copyright © 2026 **Khushboo Khator** ([@khushboocodes](https://github.com/khushboocodes)). All rights reserved.

This project is licensed under the **MIT License** — you are free to use, modify, distribute, and integrate this software in private, educational, or commercial applications, subject to the terms and conditions of the MIT License.

See the [LICENSE](LICENSE) file for the full license text.

