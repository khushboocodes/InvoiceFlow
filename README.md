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

