# INVOICEFLOW: MASTER ENGINEERING WORK BREAKDOWN STRUCTURE (WBS)
## Complete Implementation Task Manifest: Architecture, Training, Extraction, Testing & Packaging

---

| **Document Information** | **Specification Details** |
| :--- | :--- |
| **Project Name** | **InvoiceFlow Document Intelligence** |
| **Document Classification** | Master Engineering Task Manifest & Work Breakdown Structure (WBS) |
| **Document Version** | **1.0.0-PROD** |
| **Total Engineering Phases** | **12 Sequential Phases (Phase 0 to Phase 11)** |
| **Total Granular Tasks** | **85 Actionable Implementation Tasks** |
| **Acceptance Criteria Target**| $\ge 95\%$ Document Accuracy, $\le 30.0\text{s}$ Latency, $< \$0.01$ Cost, Air-Gapped Ready |

---

## 1. Master Phase & Sprint Schedule Overview

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                              INVOICEFLOW SPRINT TIMELINE                                              |
+-----------------------------------------------------------------------------------------------------------------------+
|                                                                                                                       |
|  [ SPRINT 1: FOUNDATION, DATA FORENSICS & ANNOTATION (Days 1–3) ]                                                     |
|  ├── Phase 0: Environment Setup, Tooling & Scaffolding                                            (Tasks 0.1 - 0.7)   |
|  ├── Phase 1: Dataset Forensics, Ingestion Pipeline & Quality Normalization                       (Tasks 1.1 - 1.8)   |
|  └── Phase 2: Ground-Truth Annotation & Labeling Protocol                                         (Tasks 2.1 - 2.7)   |
|                                                                                                                       |
|  [ SPRINT 2: PARTITIONING & COMPUTER VISION MODEL TRAINING (Days 4–6) ]                                               |
|  ├── Phase 3: Leakage-Free Dataset Partitioning & Manifest Generation                             (Tasks 3.1 - 3.6)   |
|  └── Phase 4: Computer Vision Training (YOLOv8 Stamp & Signature Detectors)                       (Tasks 4.1 - 4.9)   |
|                                                                                                                       |
|  [ SPRINT 3: TEXT ENGINE, MASTER CATALOGUES & 6-FIELD EXTRACTORS (Days 7–10) ]                                        |
|  ├── Phase 5: Dual-Track Text Engine (Vector Stream + PaddleOCR v4) & Spatial 2D Graph            (Tasks 5.1 - 5.8)   |
|  ├── Phase 6: Domain Master Catalogues & Knowledge Graph Construction                             (Tasks 6.1 - 6.6)   |
|  ├── Phase 7: Deterministic Field Extraction Engines (All 6 Target Fields)                        (Tasks 7.1 - 7.9)   |
|  ├── Phase 8: Surgical Multimodal VLM Fallback Routing & Abstention Gate                         (Tasks 8.1 - 8.5)   |
|  └── Phase 9: Pydantic v2 Output Serialization & Telemetry Engine                                 (Tasks 9.1 - 9.5)   |
|                                                                                                                       |
|  [ SPRINT 4: BENCHMARKING, PACKAGING & AIR-GAPPED VERIFICATION (Days 11–13) ]                                         |
|  ├── Phase 10: Comprehensive Testing Pyramid & Frozen Validation Benchmark                         (Tasks 10.1 - 10.8) |
|  └── Phase 11: Production CLI Packaging, Containerization & Air-Gapped Verification               (Tasks 11.1 - 11.7) |
|                                                                                                                       |
+-----------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Phase 0: Environment Setup, Tooling & Scaffolding

### Task 0.1: Virtual Environment & Toolchain Initialization
* **Description:** Initialize isolated Python 3.10/3.11 virtual environment and install build toolchain.
* **Commands:**
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip setuptools wheel uv
  ```
* **Acceptance Criteria:** Python version is $\ge 3.10.0$; `pip` is upgraded to latest.

### Task 0.2: Pinned Dependencies Installation
* **Description:** Install production dependencies from `requirements.txt` with SHA verification.
* **Commands:**
  ```bash
  pip install -r requirements.txt
  ```
* **Acceptance Criteria:** All packages (PyMuPDF, PaddleOCR, Ultralytics, OpenCV, Pydantic, RapidFuzz) install with zero build errors.

### Task 0.3: CUDA / Compute Acceleration Smoke Test
* **Description:** Verify GPU hardware visibility and CUDA toolkit bindings.
* **Script:** `scripts/check_cuda.py`
  ```python
  import torch
  print(f"CUDA Available: {torch.cuda.is_available()}")
  if torch.cuda.is_available():
      print(f"Device Name: {torch.cuda.get_device_name(0)}")
  ```
* **Acceptance Criteria:** Script executes cleanly; prints device profile.

### Task 0.4: Directory Scaffolding & Module Initialization
* **Description:** Create the complete modular directory tree with `__init__.py` files.
* **Structure to Create:**
  ```
  core/
  ├── ingestion/
  ├── preprocessing/
  ├── ocr/
  ├── layout/
  ├── vision/
  │   └── weights/
  ├── normalization/
  ├── confidence/
  └── output/
  document_profiles/
  └── tractor_quotation/
      └── reference_data/
  datasets/
  │   ├── raw/
  │   ├── processed/
  │   └── manifests/
  evaluation/
  tests/
  │   ├── unit/
  │   ├── integration/
  │   └── fixtures/
  docker/
  scripts/
  ```
* **Acceptance Criteria:** All directories created with valid Python package anchors.

### Task 0.5: Git Repository & Hygiene Setup
* **Description:** Initialize Git repository, configure `.gitignore` (blocking raw weights $>100\text{ MB}$, dataset images, cached OCR, and `.env` files).
* **Acceptance Criteria:** `.gitignore` blocks `venv/`, `*.pyc`, `*.pt`, `datasets/raw/`, `__pycache__/`.

### Task 0.6: Pre-Commit Quality Hooks
* **Description:** Configure automated code quality hooks (`ruff`, `black`, `mypy`).
* **Commands:**
  ```bash
  pip install pre-commit ruff black mypy
  pre-commit install
  ```
* **Acceptance Criteria:** Pre-commit passes cleanly on newly created scaffolding.

### Task 0.7: Logging & Correlation ID Infrastructure
* **Description:** Implement `core/logging_config.py` providing structured JSON logging with correlation IDs (`doc_id`, `run_id`, `elapsed_ms`) and PII regex redaction (masking Aadhaar/PAN).
* **Acceptance Criteria:** Logs emit structured JSON; sensitive 10-digit/12-digit tokens are masked with `[REDACTED]`.

---

## 3. Phase 1: Dataset Forensics, Ingestion Pipeline & Normalization

### Task 1.1: Raw Dataset Inventory & Census
* **Description:** Scan and index all $\approx 500$ documents in `datasets/raw/`.
* **Script:** `scripts/dataset_census.py`
* **Deliverable:** Summary report of file count, extensions (`.pdf`, `.jpg`, `.png`), file sizes, and page counts.
* **Acceptance Criteria:** Total file count accurately established; corrupt 0-byte files identified.

### Task 1.2: Magic-Byte MIME Ingestion Validator
* **Description:** Implement `core/ingestion/validator.py` using `python-magic` to inspect file headers and reject disguised or unsupported formats.
* **Acceptance Criteria:** `%PDF-` verified for PDFs; `\xFF\xD8\xFF` verified for JPEG; non-conforming extensions rejected with `E102_UNSUPPORTED_FORMAT`.

### Task 1.3: Anti-Decompression Bomb & Size Clamping Sandbox
* **Description:** Implement `core/ingestion/sandbox.py` enforcing:
  * Maximum file size: $25\text{ MB}$
  * Maximum image canvas: $10,000 \times 10,000\text{ px}$
  * Memory ceiling: $2.0\text{ GB}$ per worker
* **Acceptance Criteria:** Over-sized files safely rejected with `E103_FILE_TOO_LARGE` without process crashing.

### Task 1.4: Perceptual Hash Deduplication Engine
* **Description:** Implement `scripts/deduplicate_dataset.py` using `imagehash.phash`.
* **Logic:** Calculate 64-bit DCT perceptual hash. Group documents with Hamming distance $\le 4$. Retain highest-resolution version.
* **Acceptance Criteria:** Near-identical re-scans identified; duplicate manifest generated.

### Task 1.5: High-Fidelity 300 DPI PDF Rasterizer
* **Description:** Implement `core/preprocessing/rasterizer.py` using PyMuPDF (`fitz`).
* **Logic:** Render all pages to uncompressed RGB numpy arrays at $300\text{ DPI}$ ($\text{Matrix}(4.1667, 4.1667)$).
* **Acceptance Criteria:** Renders 100 pages in $<20\text{ seconds}$ on CPU; produces crisp $300\text{ DPI}$ image arrays.

### Task 1.6: Four-Way Automatic Orientation Rectifier
* **Description:** Implement `core/preprocessing/orientation.py` using Tesseract OSD / MobileNet.
* **Logic:** Detect page orientation angle ($\{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$) and rotate image to upright pose ($0^\circ$).
* **Acceptance Criteria:** Correctly rotates inverted ($180^\circ$) and sideways ($90^\circ/270^\circ$) invoices to $0^\circ$ upright with $>98\%$ accuracy.

### Task 1.7: Hough Transform Baseline Deskewer
* **Description:** Implement `core/preprocessing/deskew.py`.
* **Logic:** Canny edge detection $\to$ Probabilistic Hough Transform on text lines $\to$ Median angle calculation $\to$ Affine rotation for $|\theta| \le 15^\circ$.
* **Acceptance Criteria:** Straightens skewed scans to within $\pm 0.5^\circ$ of horizontal text baseline.

### Task 1.8: Contrast Enhancement & Shadow Removal (CLAHE)
* **Description:** Implement `core/preprocessing/normalizer.py`.
* **Logic:** Convert image to Lab color space. Apply CLAHE on L-channel (`clipLimit=2.0`, `tileGridSize=(8,8)`). Divide by morphological background approximation to eliminate shadows.
* **Acceptance Criteria:** Low-contrast carbon copies and shaded mobile photos restored to clear readable text.

---

## 4. Phase 2: Ground-Truth Annotation & Labeling Protocol

### Task 2.1: Self-Hosted Annotation Tool Deployment
* **Description:** Deploy CVAT or Label Studio via Docker Compose for local collaborative labeling.
* **Configuration:** Define project `InvoiceFlow-Tractor-V1` with bounding box classes: `dealer_stamp`, `dealer_signature`, and key-value text slots.
* **Acceptance Criteria:** Annotator UI accessible at `http://localhost:8080`; test upload functions correctly.

### Task 2.2: Annotation SOP Pilot Calibration
* **Description:** Annotate 10 sample documents following `docs/annotation_guidelines.md`.
* **Acceptance Criteria:** 100% agreement on $[0, 1000]$ normalized coordinate format and canonical value standards between lead and annotators.

### Task 2.3: Stamp & Signature Bounding Box Annotation
* **Description:** Annotate tight bounding boxes for all rubber ink stamps and authorized signatures across the ~350 training documents.
* **Acceptance Criteria:** Every visible physical stamp and signature localized with tight enclosing bounding box.

### Task 2.4: Text Ground-Truth Annotation
* **Description:** Extract and enter canonical ground-truth values for `dealer_name`, `model_name`, `horse_power`, and `asset_cost` for all ~350 documents.
* **Acceptance Criteria:** Ground truth follows strict normalization (e.g. HP as numeric `48.0`, Cost as numeric `745000.0`).

### Task 2.5: Double-Annotation Review & Conflict Resolution
* **Description:** Run secondary audit on all labeled documents; calculate inter-annotator IoU agreement.
* **Acceptance Criteria:** Inter-annotator bounding box $\text{IoU} \ge 0.85$; all field discrepancies resolved by lead auditor.

### Task 2.6: YOLO Darknet Training Format Exporter
* **Description:** Implement `scripts/export_yolo_labels.py`.
* **Logic:** Convert CVAT XML/JSON annotations into YOLO Darknet `.txt` files (`<class_id> <x_center> <y_center> <width> <height>`).
* **Acceptance Criteria:** 100% of labeled training images possess valid matching `.txt` label files.

### Task 2.7: Golden Ground-Truth JSON Manifest Builder
* **Description:** Implement `scripts/build_ground_truth_manifest.py`.
* **Deliverable:** `datasets/manifests/ground_truth_annotations.json` matching the official Pydantic schema.
* **Acceptance Criteria:** Validates cleanly against JSON Schema Draft-07.

---

## 5. Phase 3: Zero-Leakage Dataset Partitioning & Validation Harness

### Task 3.1: Dealership Identity Extraction & Grouping
* **Description:** Extract the canonical dealer name for all ~500 documents and compile into a stratification dataframe.
* **Acceptance Criteria:** Every document linked to a verified `dealer_group` identifier.

### Task 3.2: Dealership-Grouped Stratified Splitting (`GroupKFold`)
* **Description:** Implement `scripts/partition_dataset.py`.
* **Logic:** Use `sklearn.model_selection.GroupKFold(n_splits=5)` on `dealer_group`.
* **Splits:**
  * **Train Split (70%, ~350 docs):** Folds 1, 2, 3, 4.
  * **Validation Split (30%, ~150 docs):** Fold 0 (Frozen benchmark).
* **Acceptance Criteria:** Zero overlap of dealerships between Train and Validation sets.

### Task 3.3: Blind Test Set Quarantine
* **Description:** Move the ~100 official evaluation benchmark documents to an encrypted, air-gapped directory `datasets/holdout_test/`.
* **Acceptance Criteria:** Holdout set isolated; access restricted to final frozen release test.

### Task 3.4: Domain-Specific Data Augmentation Pipeline
* **Description:** Implement `core/preprocessing/augmentations.py` using Albumentations.
* **Augmentations:** Perspective warp, random rotations ($\pm 10^\circ$), HSV inking jitter (faded purple to navy blue), shadow synthesis, paper wrinkle overlay.
* **Acceptance Criteria:** Generates realistic synthetic variations of degraded rural invoices without corrupting labels.

### Task 3.5: Master Dataset Manifest Serialization
* **Description:** Serialize partition metadata into `datasets/manifests/metadata_manifest.json` conforming to `docs/datasets.md`.
* **Acceptance Criteria:** Manifest records `doc_id`, `file_hash_sha256`, `phash`, `split`, `quality_tier`, and `dealer_group`.

### Task 3.6: Automated Leakage Audit Verification Script
* **Description:** Implement `scripts/verify_zero_leakage.py`.
* **Logic:** Assert that $\text{Dealers}(\text{Train}) \cap \text{Dealers}(\text{Val}) == \emptyset$.
* **Acceptance Criteria:** Script outputs `PASS: ZERO DEALER LEAKAGE DETECTED (Exit 0)`.

---

## 6. Phase 4: Visual Intelligence & Object Detection Model Training

### Task 4.1: YOLOv8 Dataset Configuration (`yolo_data.yaml`)
* **Description:** Generate dataset YAML file linking train/val image paths and class mappings:
  ```yaml
  path: ./datasets/yolo_stamps_sigs
  train: images/train
  val: images/val
  names:
    0: signature
    1: stamp
  ```
* **Acceptance Criteria:** YAML validates with Ultralytics data loader.

### Task 4.2: Hyperparameter Configuration & Weight Initialization
* **Description:** Configure training parameters in `scripts/train_yolo.py`:
  * Model: `yolov8x.pt`
  * Image Size: `1024`
  * Epochs: `120`
  * Batch Size: `16` (or `8` with grad accumulation)
  * Optimizer: `AdamW` ($\text{lr}_0 = 0.001$, Cosine Annealing to $0.00001$)
  * Loss: CIoU ($\lambda = 7.5$) + DFL ($\lambda = 1.5$) + BCE ($\lambda = 0.5$)
* **Acceptance Criteria:** Training script initializes weights and builds computational graph without CUDA OOM.

### Task 4.3: Training Execution & Telemetry Logging
* **Description:** Execute training run; log epoch-level train/val loss, mAP@0.50, and mAP@0.50:0.95.
* **Commands:**
  ```bash
  python scripts/train_yolo.py --config configs/yolo_train.yaml
  ```
* **Acceptance Criteria:** 120 epochs complete with early stopping patience $= 20$.

### Task 4.4: Model Checkpoint Selection & Evaluation
* **Description:** Evaluate validation split performance on `best.pt`.
* **Acceptance Gate:**
  * Stamp mAP@0.50 $\ge 0.88$
  * Signature mAP@0.50 $\ge 0.85$
  * Combined mAP@0.50:0.95 $\ge 0.60$
* **Acceptance Criteria:** Metrics exceed minimum acceptance gate; weights saved to `core/vision/weights/yolo_stamp_sig_v1.pt`.

### Task 4.5: HSV Chromatic Inked Seal Segmentation Filter
* **Description:** Implement `core/vision/color_segmentation.py`.
* **Logic:** Convert crop to HSV. Measure pixel density in violet/blue ($H \in [100, 160]$) and red ($H \in [0, 20]$) saturation bands ($S > 0.25$). Discard proposals with $S_{\text{ratio}} < 0.05$.
* **Acceptance Criteria:** Eliminates false-positive monochrome printed logos while retaining genuine rubber stamps.

### Task 4.6: CCA Stroke Width Variance Filter for Signatures
* **Description:** Implement `core/vision/stroke_analysis.py`.
* **Logic:** Connected Component Analysis on signature crop. Measure stroke width variance $\sigma_{\text{stroke}}^2$. Discard proposals with $\sigma_{\text{stroke}}^2 < 0.8$ (printed lines/borders).
* **Acceptance Criteria:** Distinguishes genuine cursive handwriting from printed solid horizontal rules.

### Task 4.7: Stamp & Signature Spatial Overlap Resolver
* **Description:** Implement `core/vision/overlap_resolver.py`.
* **Logic:** If $\text{IoU}(\text{Stamp}, \text{Sig}) > 0.30$, decouple into two independent entities while logging visual co-occurrence flag.
* **Acceptance Criteria:** Overlapping stamp and signature correctly emitted as two distinct bounding boxes.

### Task 4.8: ONNX Runtime Model Export & Validation
* **Description:** Export `yolo_stamp_sig_v1.pt` to ONNX format.
* **Commands:**
  ```bash
  yolo export model=core/vision/weights/yolo_stamp_sig_v1.pt format=onnx imgsz=1024 simplify=True
  ```
* **Acceptance Criteria:** `yolo_stamp_sig_v1.onnx` generated; passes ONNX Runtime model validation test.

### Task 4.9: Visual Inference Latency Benchmark
* **Description:** Measure inference latency across 100 sample documents on CPU and GPU.
* **Acceptance Criteria:** Inference latency $\le 35\text{ms}$ on NVIDIA GPU / $\le 160\text{ms}$ on 4-core CPU.

---

## 7. Phase 5: Dual-Track Text Engine (OCR) & Spatial Graph

### Task 5.1: Track A — Native Vector Text Stream Extractor
* **Description:** Implement `core/ocr/native_pdf.py` using PyMuPDF (`page.get_text("words")`).
* **Logic:** If page has $>50$ vector text characters and valid TrueType fonts, extract tokens directly with exact coordinates and font metadata.
* **Acceptance Criteria:** Digital PDFs process in $<50\text{ms}$ with zero character error rate.

### Task 5.2: Track B — PaddleOCR v4 Multilingual Engine
* **Description:** Implement `core/ocr/paddle_engine.py` using PaddleOCR v4.
* **Logic:** Load DBNet++ text detector and SVTR text recognizer with multilingual support (Latin, Devanagari, Gujarati).
* **Acceptance Criteria:** Scanned PDFs and photos process in $<2.0\text{s}$ on CPU / $<400\text{ms}$ on GPU; outputs words with bounding polygons and confidence scores.

### Task 5.3: Dual-Track Dynamic Router
* **Description:** Implement `core/ocr/router.py`.
* **Logic:** Test for native vector text layer. If valid, route to Track A; otherwise route to Track B.
* **Acceptance Criteria:** 100% of documents routed to optimal engine.

### Task 5.4: Disk-Based OCR Caching Layer
* **Description:** Implement `core/ocr/cache.py`.
* **Logic:** Compute SHA-256 hash of preprocessed page bitmap. Check `cache/ocr/<hash>.json`. If present, load cached OCR tokens; if absent, execute OCR and write cache.
* **Acceptance Criteria:** Avoids redundant OCR during repeated development and testing runs.

### Task 5.5: Indic Numeral & Unicode Glyph Normalizer
* **Description:** Implement `core/normalization/unicode_cleaner.py`.
* **Logic:** NFC Unicode normalization. Map Devanagari (`०, १, २...`) and Gujarati (`૦, ૧, ૨...`) numerals to standard ASCII digits (`0, 1, 2...`).
* **Acceptance Criteria:** Indic numerals reliably converted to standard digits across all text tokens.

### Task 5.6: 2D Spatial Word Graph & R-Tree Indexer
* **Description:** Implement `core/layout/spatial_graph.py` using `rtree`.
* **Logic:** Index all extracted tokens into a 2D spatial tree. Provide geometric query primitives:
  * `get_tokens_right_of(box, max_dist)`
  * `get_tokens_below(box, max_dist)`
  * `get_tokens_in_region(ymin, xmin, ymax, xmax)`
* **Acceptance Criteria:** Spatial neighbor queries execute in $<1.0\text{ms}$.

### Task 5.7: Table Boundary & Column Structure Detector
* **Description:** Implement `core/layout/table_detector.py`.
* **Logic:** Identify table gridlines using morphological horizontal/vertical kernels; locate column headers (`"Particulars"`, `"Description"`, `"Rate"`, `"Amount"`).
* **Acceptance Criteria:** Line-item table region cleanly isolated from letterhead and footer totals.

### Task 5.8: Spatial Layout Integration Test
* **Description:** Write `tests/unit/test_spatial_graph.py` validating word graph construction and spatial queries.
* **Acceptance Criteria:** 100% pass on synthetic spatial test layouts.

---

## 8. Phase 6: Domain Master Catalogues & Knowledge Graph

### Task 6.1: OEM Tractor Model Catalogue Construction
* **Description:** Compile `oem_tractor_models.json` in `document_profiles/tractor_quotation/reference_data/`.
* **Data Sources:** CMVR vehicle certification sheets, manufacturer sales brochures across 10 major OEMs (Mahindra, Swaraj, John Deere, Sonalika, Eicher, Tafe, Kubota, Farmtrac, New Holland, Preet).
* **Contents:** 450+ models with `canonical_name`, `brand`, `series`, `hp`, `valid_hp_range`, `kw`, `typical_price_range_inr`, `aliases`.
* **Acceptance Criteria:** Valid JSON containing 450+ verified models with exact HP ratings.

### Task 6.2: Dealership Master Directory Construction
* **Description:** Compile `dealer_master.json`.
* **Contents:** 10,000+ authorized agricultural equipment dealerships across India with legal business names, trade names, state, district, and authorized OEM brands.
* **Acceptance Criteria:** Valid JSON; covers all dealerships represented in the training corpus.

### Task 6.3: Brand Synonym & Abbreviation Lexicon
* **Description:** Compile `brand_synonyms.json` mapping colloquial shorthand to canonical OEM brands (`"JD"` $\to$ `"John Deere"`, `"MF"` $\to$ `"Massey Ferguson"`, `"M&M"` $\to$ `"Mahindra"`).
* **Acceptance Criteria:** Abbreviation mappings cover all common Indian tractor aliases.

### Task 6.4: Vernacular Agricultural Terms & Number Words Lexicon
* **Description:** Compile `vernacular_lexicon.json` containing Hindi, Gujarati, and English word-number equivalents (`"Lakh"`, `"लाख"`, `"લાખ"`, `"Fifty"`, `"पचास"`, `"પચાસ"`).
* **Acceptance Criteria:** Complete translation table for Indic numbers and financial terms.

### Task 6.5: RapidFuzz SIMD Vector Lookup Index
* **Description:** Implement `document_profiles/tractor_quotation/catalogue_matcher.py` using RapidFuzz.
* **Logic:** Pre-compile token vectors for model and dealer names for sub-millisecond fuzzy search.
* **Acceptance Criteria:** 10,000-dealer fuzzy query executes in $<15\text{ms}$.

### Task 6.6: Model-to-HP Specification Matrix Validator
* **Description:** Implement `document_profiles/tractor_quotation/spec_matrix.py`.
* **Logic:** Provides fast verification: `is_valid_hp_for_model(model_name, hp) -> bool`.
* **Acceptance Criteria:** Accurately validates plausible HP envelopes for all 450+ models.

---

## 9. Phase 7: Deterministic Field Extraction Engines

### Task 7.1: Base Extractor Interface & Contract
* **Description:** Implement abstract base class `core/document_profiles/base_profile.py`:
  ```python
  class BaseFieldExtractor(ABC):
      @abstractmethod
      def extract(self, word_graph: SpatialGraph, image: np.ndarray) -> ExtractionCandidate:
          pass
  ```
* **Acceptance Criteria:** Clean abstract interface enforced across all field extractors.

### Task 7.2: Dealer Name Extractor Implementation
* **Description:** Implement `document_profiles/tractor_quotation/extractors/dealer_name.py`.
* **Logic:**
  1. Crop Page 1 header ($y_{\max} \le 350$).
  2. Strip buyer keywords (`"Customer"`, `"Farmer"`, `"Buyer"`, `"To:"`, `"श्री"`) and bank blocklist.
  3. Compute typographic weight ($W = \text{FontSize} \times 1.5 + \text{Bold} \times 20.0$).
  4. Match candidate with `dealer_master.json` via RapidFuzz `token_set_ratio` ($\ge 85.0$).
  5. Fallback: OCR text inside detected `dealer_stamp`.
* **Acceptance Criteria:** $\ge 97\%$ field accuracy on validation split.

### Task 7.3: Model Name Extractor Implementation
* **Description:** Implement `document_profiles/tractor_quotation/extractors/model_name.py`.
* **Logic:**
  1. Locate line-item table boundary.
  2. Scan for OEM brand anchors (`Mahindra`, `Swaraj`, `John Deere`, etc.).
  3. Extract n-gram token windows ($n \in [2, 6]$).
  4. Match candidates against `oem_tractor_models.json` using weighted Jaro-Winkler distance.
  5. Apply agricultural implement exclusion filter (`"Rotavator"`, `"Trolley"`, `"Plough"`).
* **Acceptance Criteria:** $\ge 98\%$ field accuracy on validation split.

### Task 7.4: Horse Power (HP) Extractor Implementation
* **Description:** Implement `document_profiles/tractor_quotation/extractors/horse_power.py`.
* **Logic:**
  1. Regex scan for HP patterns: `r'(?i)\b(\d{2}(?:\.\d{1,2})?)\s*(?:hp|h\.p\.|horse\s*power|एचपी)\b'`.
  2. Metric conversion for $\text{kW}$ patterns: $\text{HP} = \text{round}(\text{kW} \times 1.34102, 1)$.
  3. Proximity score: minimal distance to `model_name` bounding box.
  4. Physical plausibility check: $15.0 \le \text{HP} \le 120.0$.
  5. Fallback: Impute rated HP from canonical `model_name` in catalogue.
* **Acceptance Criteria:** $\ge 98.5\%$ field accuracy on validation split.

### Task 7.5: Asset Cost Extractor Implementation
* **Description:** Implement `document_profiles/tractor_quotation/extractors/asset_cost.py`.
* **Logic:**
  1. Regex candidate scan matching Indian currency comma notation.
  2. Spatial anchor scoring: right-aligned tokens near `"Grand Total"`, `"Total Amount"`, `"Net Payable"`, `"कुल राशि"`.
  3. Negative keyword penalty: `"Advance"`, `"Margin Money"`, `"Down Payment"`, `"Balance Due"`.
  4. Arithmetic reconciliation: $\text{Total} = \text{Basic} + \text{Taxes} + \text{Insurance}$.
  5. Words-to-numbers validation using Indic number converter.
  6. Price envelope check: $₹2,00,000.0 \le \text{Cost} \le ₹25,00,000.0$.
* **Acceptance Criteria:** $\ge 98\%$ field accuracy on validation split.

### Task 7.6: Dealer Signature Extractor Implementation
* **Description:** Implement `document_profiles/tractor_quotation/extractors/signature.py`.
* **Logic:**
  1. Run YOLOv8 signature detection.
  2. Spatial check: bottom 30% of page ($y_{\min} \ge 700$).
  3. Anchor proximity: near `"Authorized Signatory"`, `"For <Dealer>"`, `"हस्ताक्षर"`.
  4. Stroke width variance filter: $\sigma_{\text{stroke}}^2 \ge 0.8$.
  5. Return `present: bool`, `bounding_box: [ymin, xmin, ymax, xmax]`, `confidence: float`.
* **Acceptance Criteria:** Signature presence F1 $\ge 0.96$; localization IoU $\ge 0.50$.

### Task 7.7: Dealer Stamp Extractor Implementation
* **Description:** Implement `document_profiles/tractor_quotation/extractors/stamp.py`.
* **Logic:**
  1. Run YOLOv8 stamp detection.
  2. HSV chromatic ink filter: $S_{\text{ratio}} \ge 0.05$ in violet/blue/red spectrum.
  3. Overlap resolution with signature.
  4. Return `present: bool`, `bounding_box: [ymin, xmin, ymax, xmax]`, `confidence: float`.
* **Acceptance Criteria:** Stamp presence F1 $\ge 0.96$; localization IoU $\ge 0.50$.

### Task 7.8: Multi-Factor Platt Confidence Calibrator
* **Description:** Implement `core/confidence/calibrator.py`.
* **Logic:**
  $$C_{\text{final}} = 0.30 \cdot C_{\text{OCR}} + 0.30 \cdot C_{\text{Spatial}} + 0.25 \cdot C_{\text{Catalogue}} + 0.15 \cdot C_{\text{CrossField}}$$
* **Acceptance Criteria:** Outputs well-calibrated confidence score in $[0.0, 1.0]$.

### Task 7.9: Cross-Field Consistency Validator
* **Description:** Implement `document_profiles/tractor_quotation/validators.py`.
* **Logic:** Execute Model ↔ HP matrix check, Model ↔ Cost price envelope check, and Dealer ↔ Stamp text consistency.
* **Acceptance Criteria:** Discrepancies caught and auto-corrected where safe; flags logged.

---

## 10. Phase 8: Surgical Multimodal VLM Fallback Routing

### Task 8.1: Confidence Gate Evaluation Trigger
* **Description:** Implement `core/confidence/fallback_gate.py`.
* **Logic:** If $C_{\text{final}} < 0.70$ on any critical field (`dealer_name`, `model_name`, `horse_power`, `asset_cost`), trigger fallback router.
* **Acceptance Criteria:** Triggers strictly on low-confidence cases ($<5\%$ of total documents).

### Task 8.2: Localized Bounding-Crop Generator
* **Description:** Implement `core/vision/crop_generator.py`.
* **Logic:** Rather than sending the full document, crop strictly the candidate region ($+20\%$ spatial padding) to minimize tokens and inference latency.
* **Acceptance Criteria:** Image crop generated in $<5\text{ms}$; file payload $<50\text{ KB}$.

### Task 8.3: Multimodal VLM Client Adapter
* **Description:** Implement `core/ocr/vlm_client.py` supporting Gemini 1.5 Flash (via `google-genai`) and local Qwen2-VL-7B (via Hugging Face).
* **Acceptance Criteria:** Client connects cleanly; supports air-gapped mock mode when offline.

### Task 8.4: Pydantic JSON Schema Constrained Prompt
* **Description:** Implement prompt template enforcing strict JSON output:
  ```json
  {"value": "<extracted>", "confidence": 0.95, "reasoning": "<brief>"}
  ```
* **Acceptance Criteria:** Response parses directly into Pydantic models with zero syntax errors.

### Task 8.5: Fallback Cost & Latency Budget Monitor
* **Description:** Implement `core/confidence/budget_monitor.py`.
* **Logic:** Tracks total API spend and cumulative latency per run. Aborts API calls if cumulative cost exceeds $\$0.005$ per document threshold.
* **Acceptance Criteria:** Cost SLA guaranteed $< \$0.01$ per document.

---

## 11. Phase 9: Pydantic v2 Output Serialization & Telemetry

### Task 9.1: Pydantic v2 Data Models Implementation
* **Description:** Implement `document_profiles/tractor_quotation/schema.py` conforming to `docs/data_dictionary.md`.
* **Models:** `VisualEntityDetection`, `DocumentMetadata`, `TractorQuotationExtraction`.
* **Acceptance Criteria:** Validates field types, ranges, and coordinate constraints.

### Task 9.2: Bounding Box Coordinate Validator
* **Description:** Implement `@field_validator("bounding_box")` ensuring $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$ integers in $[0, 1000]$ and $y_{\min} \le y_{\max}, x_{\min} \le x_{\max}$.
* **Acceptance Criteria:** Invalid coordinates raise explicit validation errors.

### Task 9.3: JSON Schema Draft-07 Export & Verification
* **Description:** Implement `scripts/verify_json_schema.py` exporting Pydantic schema to JSON Schema Draft-07 and validating against production requirements.
* **Acceptance Criteria:** Exported schema matches target specification $100\%$.

### Task 9.4: Execution Telemetry & Audit Logger
* **Description:** Implement `core/output/telemetry.py`.
* **Metrics Tracked:** `execution_time_seconds`, `pipeline_version`, `page_count`, `ocr_engine_used`, `fallback_triggered`, `memory_rss_mb`.
* **Acceptance Criteria:** Telemetry attached to every output JSON record.

### Task 9.5: Controlled Error Response Serializer
* **Description:** Implement `core/output/error_handler.py`.
* **Logic:** If an unreadable or corrupt document fails processing, serialize a standardized error response with appropriate error code (`E101`–`E501`) without crashing.
* **Acceptance Criteria:** 100% of inputs yield valid JSON output.

---

## 12. Phase 10: Testing, Benchmarking & Acceptance Gates

### Task 10.1: Unit Test Suite Implementation
* **Description:** Implement 40+ unit tests across `tests/unit/`:
  * `test_currency_parser.py`
  * `test_hp_normalizer.py`
  * `test_dealer_extractor.py`
  * `test_model_extractor.py`
  * `test_spatial_graph.py`
  * `test_bbox_validator.py`
* **Commands:**
  ```bash
  pytest tests/unit/ -v --cov=core --cov=document_profiles
  ```
* **Acceptance Gate:** 100% unit tests pass; line coverage $\ge 90\%$.

### Task 10.2: Integration Test Suite Implementation
* **Description:** Implement end-to-end integration tests on sample golden fixtures (`tests/integration/test_pipeline.py`).
* **Test Fixtures:**
  * Clean Digital PDF
  * Flatbed Scan
  * Mobile Camera Photo
  * Multi-Page Invoice
* **Acceptance Gate:** All golden fixtures extract all 6 fields correctly.

### Task 10.3: Adversarial Robustness Test Suite
* **Description:** Implement `tests/integration/test_adversarial.py` testing 0-byte files, zip bombs, $180^\circ$ rotated images, corrupt PDFs, and prompt injection strings.
* **Acceptance Gate:** Zero unhandled crashes; all adversarial cases return structured error JSON.

### Task 10.4: Frozen Validation Benchmark Runner
* **Description:** Implement `evaluation/benchmark.py`.
* **Logic:** Ingests `datasets/manifests/val_manifest.json` (~150 documents). Executes full pipeline. Compares against ground truth.
* **Acceptance Criteria:** Evaluates all 150 documents; computes strict Document-Level Accuracy, Field Accuracies, IoUs, and Latency.

### Task 10.5: Accuracy & Tolerance Acceptance Verification
* **Description:** Verify metrics against official acceptance gates:
  $$\text{Document-Level Accuracy} \ge \mathbf{95.0\%}$$
  $$\text{Field Accuracies} \ge \mathbf{97.0\%}$$
  $$\text{Stamp / Signature Presence F1} \ge \mathbf{0.96}$$
  $$\text{Stamp / Signature Localization mAP@0.50} \ge \mathbf{0.88}$$
* **Acceptance Criteria:** Validation report satisfies all gates.

### Task 10.6: Latency & Cost SLA Profiling
* **Description:** Profile execution timing and compute costs:
  * P50 Latency $\le 4.0\text{ seconds}$
  * P95 Latency $\le 15.0\text{ seconds}$
  * Max Timeout $\le 30.0\text{ seconds}$
  * Unit Cost $< \$0.005$ per document
* **Acceptance Criteria:** Performance SLA fully verified.

### Task 10.7: Error Bucket Classification & Failure Analysis
* **Description:** Implement `evaluation/error_analyzer.py`.
* **Logic:** Cluster any validation failures into buckets: OCR typo, catalogue mismatch, stamp false positive, price math conflict.
* **Acceptance Criteria:** Comprehensive error taxonomy report generated.

### Task 10.8: Pre-Commit Automated Regression Suite
* **Description:** Configure automated regression runner executing before any code freeze:
  ```bash
  python evaluation/regression_suite.py --threshold 0.95
  ```
* **Acceptance Criteria:** Ensures zero regressions on previously passing validation documents.

---

## 13. Phase 11: Production CLI Packaging, Containerization & Air-Gapped Verification

### Task 11.1: Standalone CLI Entrypoint (`executable.py`)
* **Description:** Implement production CLI entrypoint using `argparse` conforming to `docs/api_reference.md`:
  ```bash
  python executable.py --input <path> --output <path> [--device cpu|cuda] [--batch-size 4] [--strict-eval]
  ```
* **Acceptance Criteria:** CLI supports single file and batch directory processing; returns standard exit codes (`0`, `1`, `2`, `3`).

### Task 11.2: Clean Virtual Environment Installation Test
* **Description:** Create fresh temporary virtual environment; install dependencies; run CLI on sample invoice.
* **Commands:**
  ```bash
  python -m venv test_env
  source test_env/bin/activate
  pip install -r requirements.txt
  python executable.py --input tests/fixtures/sample_invoice.pdf --output test_out.json
  ```
* **Acceptance Gate:** Runs cleanly from clean install with zero manual patching.

### Task 11.3: Production Dockerfile Implementation
* **Description:** Implement `docker/Dockerfile` based on `nvidia/cuda:11.8.0-runtime-ubuntu22.04` with Python 3.10, OpenCV, Tesseract, and PaddleOCR dependencies.
* **Acceptance Criteria:** Container builds cleanly; image size $< 3.5\text{ GB}$.

### Task 11.4: Pre-Cached Model Weights Bundling
* **Description:** Package fine-tuned YOLO weights (`weights/yolo_stamp_sig_v1.pt`) and pre-downloaded PaddleOCR model caches into the container image.
* **Acceptance Criteria:** Model runs without attempting any runtime internet download.

### Task 11.5: Air-Gapped Zero-Network Execution Test
* **Description:** Run Docker container with networking completely disabled (`--network none`).
* **Commands:**
  ```bash
  docker run --rm --network none -v $(pwd)/data:/data invoiceflow:latest \
      --input /data/sample_invoice.pdf \
      --output /data/output.json
  ```
* **Acceptance Gate:** Pipeline executes successfully and produces valid JSON with zero network access.

### Task 11.6: Final Packaging & Submission Archive
* **Description:** Implement `scripts/package_submission.py`.
* **Logic:** Assemble clean ZIP archive containing:
  * `executable.py`
  * `requirements.txt`
  * `core/`
  * `document_profiles/`
  * `docs/`
  * `README.md`
  * `sample_output.json`
* **Exclusions:** Exclude raw private training PDFs, `.git/`, `cache/`, `*.pyc`.
* **Acceptance Criteria:** Submission archive verified; unzips and runs cleanly in fresh sandbox.

### Task 11.7: Final Pre-Flight 20-Point Checklist Sign-Off
* **Description:** Execute formal review against the 20-point pre-flight checklist in `docs/prd.md`.
* **Acceptance Gate:** 100% checklist items signed off by lead engineer.
