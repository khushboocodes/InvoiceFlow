# INVOICEFLOW SYSTEM ARCHITECTURE SPECIFICATION
## Comprehensive End-to-End Architectural Blueprint & Technical Design Document

---

| **System Information** | **Specification Details** |
| :--- | :--- |
| **System Name** | **InvoiceFlow Document Intelligence** |
| **Document Classification** | Enterprise Technical Architecture Document (TAD) |
| **Document Version** | **1.0.0-PROD** |
| **Author / Architect** | Lead System Architect / DeepMind Agentic Engineering |
| **Target Workload** | High-Throughput Semi-Structured Document Processing (Tractor Invoices & Quotations) |
| **Execution Environment** | Standalone Python 3.10+ Linux/Windows CLI & Docker Microservice |

---

## 1. Architectural Philosophy & Design Principles

InvoiceFlow is architected according to a **Specialist-First, Universal-Core Paradigm**. The core design philosophy decouples generic document processing infrastructure from domain-specific extraction rules:

```
+----------------------------------------------------------------------------------------------------+
|                                    INVOICEFLOW LAYERED TOPOLOGY                                    |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [ LAYER 1: INGESTION GATEWAY & SECURITY SANDBOX ]                                                 |
|  - Magic byte validation, decompression bomb protection, file quarantine                           |
|                                     │                                                              |
|                                     ▼                                                              |
|  [ LAYER 2: CV PREPROCESSING & NORMALIZATION ]                                                     |
|  - 300 DPI rasterization, 4-way OSD orientation, Hough deskewing, CLAHE contrast                   |
|                                     │                                                              |
|                                     ▼                                                              |
|  [ LAYER 3: DUAL-TRACK TEXT RECOGNITION (OCR) ]                                                    |
|  - Track A: Native Vector Stream (PyMuPDF) | Track B: Multilingual Raster (PaddleOCR v4)           |
|                                     │                                                              |
|                                     ▼                                                              |
|  [ LAYER 4: 2D SPATIAL WORD GRAPH ]                                                                |
|  - R-Tree geometric indexing, label-value spatial proximity scoring                                |
|                                     │                                                              |
|                                     ▼                                                              |
|  [ LAYER 5: COMPUTER VISION INTELLIGENCE ]                                                         |
|  - YOLOv8x fine-tuned stamp & signature detection, HSV chromatic ink filter                        |
|                                     │                                                              |
|                                     ▼                                                              |
|  [ LAYER 6: PLUGGABLE DOMAIN PROFILE (TRACTOR V1) ]                                                |
|  - Field extractors, OEM model catalogue, dealer registry, arithmetic price checker                |
|                                     │                                                              |
|                                     ▼                                                              |
|  [ LAYER 7: CONFIDENCE CALIBRATION & QUALITY ASSURANCE ]                                           |
|  - Platt scaling, cross-field validation matrix, surgical VLM fallback gate                         |
|                                     │                                                              |
|                                     ▼                                                              |
|  [ LAYER 8: SERIALIZATION & TELEMETRY ]                                                            |
|  - Pydantic v2 data models, JSON Schema Draft-07, execution timing & provenance logs               |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### Core Architectural Axioms
1. **Separation of Concerns:** The engine (`core/`) contains zero knowledge of tractor-specific field names, models, or HP ranges. All domain logic resides in `document_profiles/tractor_quotation/`.
2. **Deterministic Over Generative:** Neural generation (LLMs/VLMs) is strictly an exception path ($<5\%$ invocation rate). $95\%+$ of documents execute via deterministic OCR, spatial geometry, and computer vision.
3. **Provable Provenance:** Every extracted value retains:
   * Source page index ($0$-based)
   * Normalized bounding box $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$ in $[0, 1000]$ integer space
   * Raw OCR token text and confidence
   * Extractor method identifier (`"spatial_regex"`, `"catalogue_fuzzy"`, `"vlm_fallback"`)
4. **Zero Silent Corruption:** Ambiguous, unreadable, or conflicting fields emit structured abstention markers (`null` with error code) rather than speculative guesses.

---

## 2. End-to-End Data Flow

```
                                  Incoming Document File
                                            │
                                            ▼
                          ┌───────────────────────────────────┐
                          │     1. Ingestion & Security       │
                          │ - Magic byte check (%PDF, JPEG)   │
                          │ - Max size <= 25MB, Dim <= 10K px │
                          └─────────────────┬─────────────────┘
                                            │
                                            ▼
                          ┌───────────────────────────────────┐
                          │   2. CV Preprocessing Pipeline    │
                          │ - Render PDF pages at 300 DPI     │
                          │ - Auto-orient (0/90/180/270 deg)  │
                          │ - Hough baseline deskew (+/-15 deg│
                          │ - CLAHE contrast enhancement      │
                          └─────────────────┬─────────────────┘
                                            │
                                            ▼
                          ┌───────────────────────────────────┐
                          │    3. Dual-Track Text Engine      │
                          │   Is native vector text valid?    │
                          │   ├── YES ──> PyMuPDF stream (<50ms)
                          │   └── NO  ──> PaddleOCR v4 DBNet++│
                          └─────────────────┬─────────────────┘
                                            │
                                            ▼
                          ┌───────────────────────────────────┐
                          │    4. Spatial 2D Word Graph       │
                          │ - R-Tree spatial indexing         │
                          │ - Script detection (Dev/Guj/Lat)  │
                          └─────────────────┬─────────────────┘
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
        ┌───────────────────────────┐               ┌───────────────────────────┐
        │  5. Visual Intelligence   │               │   6. Semantic Extraction  │
        │ - YOLOv8x stamp detector  │               │ - Dealer Name (Header)    │
        │ - YOLOv8x sig detector    │               │ - Model Name (Table item) │
        │ - HSV ink chromatic filter│               │ - HP (Regex + kW conv)    │
        │ - CCA stroke width filter │               │ - Asset Cost (Comma parse)│
        └─────────────┬─────────────┘               └─────────────┬─────────────┘
                      │                                           │
                      └─────────────────────┬─────────────────────┘
                                            │
                                            ▼
                          ┌───────────────────────────────────┐
                          │  7. Domain Knowledge & Validation │
                          │ - 450+ OEM Model Registry match   │
                          │ - Dealer Master fuzzy match       │
                          │ - Model ↔ HP consistency matrix   │
                          │ - Ex-showroom + Tax arithmetic    │
                          └─────────────────┬─────────────────┘
                                            │
                                            ▼
                          ┌───────────────────────────────────┐
                          │   8. Confidence & Fallback Gate   │
                          │   Is C_final >= 0.75?             │
                          │   ├── YES ──> Proceed to output   │
                          │   └── NO  ──> Crop VLM fallback   │
                          └─────────────────┬─────────────────┘
                                            │
                                            ▼
                          ┌───────────────────────────────────┐
                          │    9. Pydantic v2 Serialization   │
                          │ - Strict JSON Schema compliance   │
                          │ - Telemetry & latency logging     │
                          └───────────────────────────────────┘
```

---

## 3. Detailed Component Decomposition

### 3.1 Layer 1: Ingestion Gateway & Security Sandbox (`core/ingestion/`)
* **`validator.py`:**
  * Checks file headers using `python-magic` (e.g. verifying `\x25\x50\x44\x46` for PDF).
  * Clamps maximum file size to $25\text{ MB}$.
  * Scans PDF object trees to block embedded JavaScript actions (`/JS`, `/JavaScript`) and launch actions (`/Launch`).
* **`sandbox.py`:**
  * Wraps file operations in sandboxed temporary working directories.
  * Enforces maximum page threshold: processes first 10 pages; truncates excess.
  * Clamps raster memory allocations: maximum uncompressed canvas size $10,000 \times 10,000\text{ pixels}$.

### 3.2 Layer 2: Preprocessing & Computer Vision (`core/preprocessing/`)
* **`rasterizer.py`:**
  * Ingests digital or scanned PDF files and renders pages to RGB bitmaps at $300\text{ DPI}$:
    ```python
    scale = 300.0 / 72.0
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    ```
* **`deskew.py`:**
  * Computes Canny edges, performs probabilistic Hough Transform to detect dominant horizontal text baseline lines, computes median angle $\theta \in [-15^\circ, +15^\circ]$, and rotates image with bilinear interpolation.
* **`normalizer.py`:**
  * Converts image to Lab color space.
  * Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on L-channel (`clipLimit=2.0`, `tileGridSize=(8,8)`).
  * Runs Otsu adaptive binarization as a parallel mask for faint dot-matrix receipts.

### 3.3 Layer 3: Dual-Track Text Engine (`core/ocr/`)
* **`native_pdf.py` (Track A - Fast Path):**
  * If the input is a digital PDF with a native font stream, extracts text blocks, font sizes, bounding boxes, and flags in $<50\text{ms}$ with zero character recognition error.
* **`paddle_engine.py` (Track B - Robust Path):**
  * Used for scanned PDFs and mobile photos.
  * Text Detection: DBNet++ (Real-time Differentiable Binarization).
  * Text Recognition: SVTR (Single Visual Model for Text Recognition) loaded with multilingual weights covering Latin, Devanagari (Hindi/Marathi), and Gujarati scripts.
  * Returns word tokens with character-level confidence scores.

### 3.4 Layer 4: Spatial 2D Word Graph (`core/layout/`)
* **`spatial_graph.py`:**
  * Indexes all recognized text tokens into an R-Tree spatial index:
    ```python
    # Token schema: (id, text, xmin, ymin, xmax, ymax, font_size, confidence)
    idx.insert(token_id, (xmin, ymin, xmax, ymax), obj=token)
    ```
  * Provides nearest-neighbor queries:
    * `find_tokens_right_of(anchor_box, max_distance=300)`
    * `find_tokens_below(anchor_box, max_distance=150)`
    * `find_tokens_in_region(ymin, xmin, ymax, xmax)`

### 3.5 Layer 5: Computer Vision Object Detection (`core/vision/`)
* **`detector.py`:**
  * Loads fine-tuned weights `yolo_stamp_sig_v1.pt`.
  * Runs batched forward pass on $1024 \times 1024$ page canvases.
  * Applies Soft-NMS with IoU threshold $0.45$.
* **`color_segmentation.py`:**
  * Performs HSV color filtering on detected stamp bounding boxes.
  * Masks hue in ranges $[100, 160]$ (violet/blue) and $[0, 20]$ (red).
  * Calculates saturation density ratio $S_{\text{ratio}}$. Proposals with $S_{\text{ratio}} < 0.05$ are discarded as printed logos.

### 3.6 Layer 6: Domain Profile Engine (`document_profiles/tractor_quotation/`)
* **`extractors.py`:**
  * Encapsulates candidate generation and scoring for the four text fields:
    * `DealerNameExtractor`
    * `ModelNameExtractor`
    * `HorsePowerExtractor`
    * `AssetCostExtractor`
* **`validators.py`:**
  * Enforces Indian agricultural lending business rules:
    * Tractor HP must fall between $15.0$ and $120.0$.
    * Asset Cost must fall between ₹2,00,000.00 and ₹25,00,000.00.
    * Arithmetic reconciliation: Total = Basic + Taxes + Insurance.
* **`reference_data/`:**
  * Houses canonical lookup databases:
    * `oem_tractor_models.json` ($450+$ OEM models with HP, displacement, and price bounds)
    * `dealer_master.json` ($10,000+$ registered dealerships)

### 3.7 Layer 7: Confidence Calibration & Abstention (`core/confidence/`)
* **`calibrator.py`:**
  * Calculates calibrated multi-factor confidence:
    $$C_{\text{final}} = 0.30 \cdot C_{\text{OCR}} + 0.30 \cdot C_{\text{Spatial}} + 0.25 \cdot C_{\text{Catalogue}} + 0.15 \cdot C_{\text{CrossField}}$$
  * Triggers targeted VLM fallback if $C_{\text{final}} < 0.70$ on critical fields.
  * Emits structured abstention codes if resolution fails.

### 3.8 Layer 8: Output Serialization (`core/output/`)
* **`serializer.py`:**
  * Validates parsed entities against Pydantic v2 `TractorQuotationExtraction` model.
  * Serializes output to JSON Schema Draft-07 compliant format.
  * Attaches execution telemetry (runtime, page count, memory RSS).

---

## 4. Hardware Sizing & Concurrency Architecture

InvoiceFlow is architected for deployment across commodity cloud environments, on-premise servers, and air-gapped sandboxes:

```
+----------------------------------------------------------------------------------------------------+
|                                    DEPLOYMENT TOPOLOGY SIZING                                      |
+---------------------+-------------------------------+----------------------------------------------+
| Resource Component  | Minimum Specification (CPU)   | Recommended Specification (GPU-Accelerated)  |
+---------------------+-------------------------------+----------------------------------------------+
| **CPU**             | 4 vCPUs (x86_64 or ARM64)     | 8 vCPUs (Intel Xeon / AMD EPYC)              |
| **RAM**             | 8 GB RAM                      | 16 GB RAM                                    |
| **GPU Acceleration**| Optional (None required)      | 1x NVIDIA T4 / A10G (16 GB VRAM)             |
| **Disk Space**      | 10 GB SSD (Model weights)     | 25 GB NVMe SSD                               |
| **Throughput**      | 1.8 - 3.2 seconds / document  | 0.45 - 0.85 seconds / document               |
| **Concurrency**     | 2 Parallel Worker Processes   | 8 Parallel Worker Processes                  |
+---------------------+-------------------------------+----------------------------------------------+
```

### Worker Process Concurrency Model
* Ingestion and processing run under **ProcessPoolExecutor** (multi-process, not multi-threaded) to bypass Python's Global Interpreter Lock (GIL) during PyMuPDF rasterization and PaddleOCR inference.
* Workers communicate via non-blocking IPC pipes.
* Memory per worker process is hard-capped at $2.0\text{ GB}$ via `resource.setrlimit(resource.RLIMIT_AS, ...)` on Linux systems.

---

## 5. Failure Recovery & Error Taxonomy

```
+----------------------------------------------------------------------------------------------------+
|                                      SYSTEM ERROR TAXONOMY                                         |
+------------+-----------------------+---------------------------------------------------------------+
| Error Code | Error Classification  | System Action & Recovery Behavior                             |
+------------+-----------------------+---------------------------------------------------------------+
| `E101`     | `CORRUPT_INPUT_FILE`  | Quarantine file; emit JSON with status `FAILED_UNREADABLE`.   |
| `E102`     | `UNSUPPORTED_FORMAT`  | Reject file; emit HTTP 415 / CLI exit code 1.                 |
| `E103`     | `FILE_TOO_LARGE`      | Reject file if >25MB; prevent DoS.                            |
| `E201`     | `ZERO_TEXT_DETECTED`  | Image blank/unreadable; flag for manual underwriting review.  |
| `E301`     | `OEM_MODEL_NOT_FOUND` | Emit best fuzzy candidate; flag confidence = 0.50.            |
| `E302`     | `HP_VALUE_OUT_OF_BOUND| If HP < 15 or > 120, impute from model catalogue.             |
| `E401`     | `PRICE_MATH_MISMATCH` | Mark cost confidence = 0.60; attach arithmetic delta log.    |
| `E501`     | `CUDA_OOM_FALLBACK`   | Automatically transfer YOLO forward pass to CPU and retry.    |
+------------+-----------------------+---------------------------------------------------------------+
```
