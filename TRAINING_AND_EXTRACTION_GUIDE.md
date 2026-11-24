# INVOICEFLOW: MODEL TRAINING & FIELD EXTRACTION SPECIFICATION
## Comprehensive Engineering Manual: Dataset Training Pipeline & 6-Field Multimodal Extraction Engine

---

| **Document Metadata** | **Specification Details** |
| :--- | :--- |
| **System Name** | **InvoiceFlow Document Intelligence** |
| **Target Document Type** | **Tractor Loan Quotations & Proforma Invoices (IDFC / Multi-Bank Rural Portfolio)** |
| **Document Classification** | Technical Implementation Blueprint & Machine Learning Specification |
| **Version** | **1.0.0-PROD** |
| **Performance SLA** | $\ge 95\%$ Document-Level Accuracy, $\le 30.0\text{s}$ Latency, $< \$0.01$ Cost/Document |
| **Target Hardware** | Commodity CPU (4–8 vCPU) / Optional GPU (NVIDIA T4 / A10G / RTX 3060+) |

---

## 1. Architectural Overview & System Flow

InvoiceFlow solves tractor invoice extraction by pairing a **modular, task-specialized training loop** with a **deterministic-first, dual-track extraction engine**. Rather than relying on an unconstrained end-to-end LLM that can hallucinate or fail on unseen layouts, InvoiceFlow decomposes the problem into decoupled, specialized subsystems:

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                        INVOICEFLOW END-TO-END SYSTEM PIPELINE                                         |
+-----------------------------------------------------------------------------------------------------------------------+
|                                                                                                                       |
|  [ INGESTION & CV PREPROCESSING ]                                                                                     |
|  Input File (PDF / Image) ──> MIME Sniffing ──> 300 DPI Render ──> Orientation Rectify ──> Hough Deskew ──> CLAHE     |
|                                                                                                  │                    |
|                                                                                                  ▼                    |
|  [ DUAL-TRACK TEXT RECOGNITION ]                                                                                      |
|  ┌───────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────┐  |
|  │ TRACK A: Native Vector Stream (PyMuPDF)                      │ TRACK B: Multilingual Raster OCR (PaddleOCR v4)  │  |
|  │ - Direct character glyph & font extraction                    │ - Text Detection (DBNet++)                      │  |
|  │ - Zero OCR error, sub-50ms execution                          │ - Text Recognition (SVTR: Devanagari/Gujarati)  │  |
|  └───────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────┘  |
|                                                                  │                                                    |
|                                                                  ▼                                                    |
|  [ UNIFIED 2D SPATIAL WORD GRAPH ]                                                                                    |
|  Token stream indexed via 2D R-Tree: {text, bbox: [ymin, xmin, ymax, xmax], font_size, script, confidence}          |
|                                                                  │                                                    |
|                                 ┌────────────────────────────────┴────────────────────────────────┐                   |
|                                 ▼                                                                 ▼                   |
|  [ COMPUTER VISION ENGINE ]                                     [ TEXT & SEMANTIC EXTRACTION ENGINE ]                 |
|  - Fine-Tuned YOLOv8x Detector                                  - Dealer Name Extractor (Header + Master Registry)    |
|    * Class 0: Signature Detection                               - Model Name Extractor (Table Item + OEM Catalogue)   |
|    * Class 1: Rubber Stamp Detection                            - Horse Power Extractor (Regex + kW Converter)        |
|  - HSV Inked Seal Chromatic Filter                              - Asset Cost Extractor (Indian Notation + Math Check) |
|  - Contour Variance Handwriting Filter                          - RapidFuzz Multi-Token Catalogue Alignment           |
|                                 │                                                                 │                   |
|                                 └────────────────────────────────┬────────────────────────────────┘                   |
|                                                                  ▼                                                    |
|  [ CROSS-FIELD RECONCILIATION & CONFIDENCE GATE ]                                                                     |
|  - Model ↔ HP Matrix Verification (e.g. Swaraj 744 must have 44-50 HP)                                               |
|  - Mathematical Price Breakdown (Ex-Showroom + Taxes + Insurance == Grand Total)                                      |
|  - Platt-Scaled Multi-Factor Confidence Scoring                                                                       |
|                                                                  │                                                    |
|                                        Is Overall Confidence >= 0.75?                                                 |
|                                        ├── YES ──> Serialize to Pydantic v2 JSON Schema                               |
|                                        └── NO  ──> Targeted Bounding-Crop Multimodal Fallback (Gemini Flash / Qwen-VL)|
|                                                                                                                       |
+-----------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Part I: The Dataset Training Pipeline

The training dataset comprises $\approx 500$ tractor loan quotations exhibiting severe real-world noise (carbon copies, folds, mobile camera perspective warp, mixed English-Hindi-Gujarati scripts). The training workflow transforms this raw corpus into production-ready visual detectors, normalization registries, and calibrated heuristic weights.

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                              DATASET TRAINING WORKFLOW                                                |
+-----------------------------------------------------------------------------------------------------------------------+
|                                                                                                                       |
|   +---------------------------------+        +----------------------------------+                                     |
|   | 1. Raw Invoices (~500 files)    | ─────> | 2. Forensic Ingestion & Cleaning  |                                     |
|   | - Digital PDFs, Scans, Photos   |        | - pHash Deduplication            |                                     |
|   +---------------------------------+        | - 300 DPI Rendering & Deskew     |                                     |
|                                              +----------------------------------+                                     |
|                                                                │                                                      |
|                                                                ▼                                                      |
|   +---------------------------------+        +----------------------------------+                                     |
|   | 4. GroupKFold Splitting         | <───── | 3. Dual-Track Annotation (CVAT)  |                                     |
|   | - Grouped strictly by Dealer    |        | - Bounding Boxes: Stamp, Sig     |                                     |
|   | - 70% Train (~350 docs)         |        | - Field Ground Truth: 4 fields   |                                     |
|   | - 30% Val (~150 docs)           |        +----------------------------------+                                     |
|   +---------------------------------+                                                                                 |
|                   │                                                                                                   |
|         ┌─────────┴────────────────────────────────────────┐                                                          |
|         ▼                                                  ▼                                                          |
|   +---------------------------------+        +----------------------------------+                                     |
|   | 5. YOLOv8x Vision Fine-Tuning   |        | 6. Knowledge Base Mining         |                                     |
|   | - 120 Epochs, Cosine Annealing  |        | - OEM Tractor Models (~450)      |                                     |
|   | - CIoU + DFL + BCE Loss         |        | - Dealership Master Directory    |                                     |
|   | - Stamp HSV Calibration         |        | - Devanagari/Gujarati Terms      |                                     |
|   +---------------------------------+        +----------------------------------+                                     |
|                   │                                                  │                                                |
|                   └─────────────────────────┬────────────────────────┘                                                |
|                                             ▼                                                                         |
|                              +----------------------------------+                                                     |
|                              | 7. Confidence Calibration        |                                                     |
|                              | - Platt Scaling on Val Split     |                                                     |
|                              | - Weight Optimization via LogReg |                                                     |
|                              +----------------------------------+                                                     |
|                                                                                                                       |
+-----------------------------------------------------------------------------------------------------------------------+
```

---

### Step 1: Forensic Ingestion & Quality Cleaning
1. **Perceptual Deduplication:**
   Compute perceptual hashes (`imagehash.phash`) on every document page:
   ```python
   import imagehash
   from PIL import Image

   def compute_page_hash(image: Image.Image) -> str:
       # Resizes to 32x32, converts to grayscale, computes 64-bit DCT hash
       return str(imagehash.phash(image, hash_size=8))
   ```
   * Any two documents with Hamming distance $\le 4$ are flagged as duplicate/near-duplicate re-scans. Only the higher-resolution copy is preserved in the active training set.
2. **Canonical Rasterization:**
   Convert all input pages (regardless of source format) into standardized $300\text{ DPI}$ uncompressed RGB images:
   $$\text{Scale Factor} = \frac{300}{72} \approx 4.1667 \implies \text{fitz.Matrix}(4.1667, 4.1667)$$
3. **Four-Way Orientation Rectification:**
   Run an orientation classifier to identify rotation ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) and rotate to upright position ($0^\circ$).
4. **Hough Transform Deskewing:**
   Detect text baseline skew using Canny edge detection followed by probabilistic Hough transform:
   $$\theta_{\text{deskew}} = \text{median}(\{\theta_i \mid |\theta_i| \le 15^\circ, \text{length}(L_i) \ge 50\text{px}\})$$
   Apply an affine rotation matrix around the image center to rectify angles within $[-15^\circ, +15^\circ]$.

---

### Step 2: Annotation Protocol & Ground-Truth Schema
Annotations are performed in **CVAT (Computer Vision Annotation Tool)** or **Label Studio**:

#### A. Visual Bounding Box Annotations
* `dealer_stamp`: Bounding box enclosing the complete boundary of any rubber ink stamp or seal.
* `dealer_signature`: Bounding box enclosing any handwritten signature, authorized initials, or endorsement mark.
* **Coordinate Standard:** All boxes normalized to integer space $[0, 1000]$:
  $$y_{\min}^{\text{norm}} = \text{round}\left(\frac{y_{\min}}{H} \times 1000\right), \quad x_{\min}^{\text{norm}} = \text{round}\left(\frac{x_{\min}}{W} \times 1000\right)$$
  $$y_{\max}^{\text{norm}} = \text{round}\left(\frac{y_{\max}}{H} \times 1000\right), \quad x_{\max}^{\text{norm}} = \text{round}\left(\frac{x_{\max}}{W} \times 1000\right)$$

#### B. Text Entity Annotations
Each document has a matching ground-truth JSON record containing canonicalized field representations:
```json
{
  "document_id": "IDFC_TRACTOR_0042.pdf",
  "dealer_name": "Patel Tractors & Farm Machinery",
  "model_name": "Mahindra 575 DI SP Plus",
  "horse_power": 47.0,
  "asset_cost": 745000.0,
  "dealer_signature": {
    "present": true,
    "bounding_box": [820, 680, 910, 890],
    "page_index": 0
  },
  "dealer_stamp": {
    "present": true,
    "bounding_box": [800, 650, 930, 850],
    "page_index": 0
  }
}
```

---

### Step 3: Leakage-Free Dataset Partitioning
* **The Leakage Risk:** A random split leaks dealer letterhead layout, logo graphics, and dot-matrix table formats between training and validation splits. A model could memorize that a specific header geometry corresponds to "Patel Tractors", yielding falsely optimistic validation scores.
* **The GroupKFold Solution:**
  Partition the dataset using `GroupKFold(n_splits=5)` grouped strictly by `dealer_name`:
  ```python
  from sklearn.model_selection import GroupKFold
  import pandas as pd

  df = pd.read_csv("metadata_manifest.csv") # contains doc_id, dealer_name, quality_tier
  gkf = GroupKFold(n_splits=5)

  for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df["dealer_name"])):
      df.loc[val_idx, "fold"] = fold
      # Fold 0 selected as primary frozen validation benchmark (30%, ~150 docs)
  ```
* **Evaluation Set Quarantine:** The official 100-document evaluation set remains in an air-gapped, encrypted directory; no engineer or training loop accesses it prior to final freeze.

---

### Step 4: Fine-Tuning the Computer Vision Models (Stamps & Signatures)
We train an object detector on the visual entities:
* **Base Architecture:** `YOLOv8x` (pre-trained on COCO / Document Object Detection).
* **Classes:** `0: signature`, `1: stamp`.
* **Dataset Format (`yolo_data.yaml`):**
  ```yaml
  path: ./datasets/yolo_stamps_sigs
  train: images/train
  val: images/val
  names:
    0: signature
    1: stamp
  ```
* **Data Augmentation Configuration:**
  ```python
  AUGMENTATIONS = {
      "degrees": 10.0,        # Random rotation +/- 10 deg
      "translate": 0.10,      # Translation +/- 10%
      "scale": 0.15,          # Scaling +/- 15%
      "shear": 2.0,           # Shear transform
      "perspective": 0.0005,  # 3D perspective warp
      "hsv_h": 0.015,         # Hue variation (violet to dark blue)
      "hsv_s": 0.70,          # Saturation variation (faint to dense ink)
      "hsv_v": 0.40,          # Brightness variation
      "mosaic": 1.0,          # 4-image mosaic composition
      "mixup": 0.15           # Image blending
  }
  ```
* **Loss Function Formulation:**
  $$\mathcal{L} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}}$$
  where Complete IoU (CIoU) penalizes bounding box center distance, aspect ratio discrepancy, and overlap:
  $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{\text{gt}})}{c^2} + \alpha v$$
* **Training Hyperparameters:**
  * Optimizer: `AdamW` (weight decay $= 0.0005$, momentum $= 0.937$)
  * Initial Learning Rate: $\eta_0 = 1 \times 10^{-3}$, Final: $\eta_f = 1 \times 10^{-5}$ (Cosine Annealing)
  * Epochs: $120$ with patience $20$ (early stopping monitored on validation $\text{mAP@0.50}$)
  * Batch Size: $16$ (on 16GB GPU) or $8$ (with gradient accumulation $= 2$)
* **Model Serialization:** Export trained model to ONNX and FP16 PyTorch weights (`weights/yolo_stamp_sig_v1.pt`).

---

### Step 5: Mining Domain Knowledge Catalogues
We build external domain databases to empower deterministic validation:

#### 1. OEM Tractor Model Catalogue (`oem_tractor_models.json`)
Compiled from CMVR (Central Motor Vehicles Rules) certification sheets and manufacturer catalogues across 10 major OEMs ($N \approx 450$ variants):
```json
{
  "models": [
    {
      "canonical_name": "Mahindra 575 DI SP Plus",
      "brand": "Mahindra",
      "series": "575",
      "hp": 47.0,
      "valid_hp_range": [45.0, 47.0],
      "kw": 35.0,
      "typical_price_range_inr": [680000.0, 760000.0],
      "aliases": ["MAHINDRA 575", "M&M 575 DI", "575 DI SP PLUS", "575DI"]
    },
    {
      "canonical_name": "Swaraj 744 FE",
      "brand": "Swaraj",
      "series": "744",
      "hp": 48.0,
      "valid_hp_range": [45.0, 48.0],
      "kw": 35.79,
      "typical_price_range_inr": [690000.0, 780000.0],
      "aliases": ["SWARAJ 744", "744 FE", "SWARAJ 744FE", "SWARAJ744"]
    },
    {
      "canonical_name": "John Deere 5050 D",
      "brand": "John Deere",
      "series": "5050",
      "hp": 50.0,
      "valid_hp_range": [50.0, 50.0],
      "kw": 37.28,
      "typical_price_range_inr": [790000.0, 890000.0],
      "aliases": ["JD 5050 D", "JOHN DEERE 5050D", "5050D 2WD", "5050 D"]
    }
  ]
}
```

#### 2. Dealer Master Registry (`dealer_master.json`)
Compiled from agricultural dealer licensing directories across Gujarat, Maharashtra, MP, Punjab, and UP:
```json
{
  "dealers": [
    {
      "id": "DLR_GUJ_0089",
      "canonical_name": "Patel Tractors & Farm Equipment",
      "state": "Gujarat",
      "district": "Mehsana",
      "brands_authorized": ["Mahindra", "Swaraj"],
      "aliases": ["PATEL TRACTORS", "M/S PATEL TRACTORS MEHSANA", "PATEL AGRO SERVICES"]
    }
  ]
}
```

---

## 3. Part II: The Complete 6-Field Extraction Pipeline

At inference time, each document executes through the deterministic spatial extraction pipeline:

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                           EXTRACTION CONTRACT MATRIX                                                  |
+---------------------+-------------------+---------------------+-------------------------+-----------------------------+
| Field Name          | Target Data Type  | Canonical Format    | Extraction Method       | Validation & Fallback       |
+---------------------+-------------------+---------------------+-------------------------+-----------------------------+
| 1. Dealer Name      | String            | Title Case Text     | Header Spatial + Fuzzy  | Dealer Master Registry      |
| 2. Model Name       | String            | Canonical OEM Model | Item Table + N-Gram     | OEM Tractor Catalogue (450+)|
| 3. Horse Power      | Float             | Numeric (e.g. 48.0) | Regex + kW Conversion   | Range [15-120] + OEM Impute |
| 4. Asset Cost       | Float             | INR (e.g. 745000.0) | Comma Parser + Anchor   | Price Envelope + Tax Math   |
| 5. Dealer Signature | Boolean + Box     | [ymin, xmin...]     | YOLOv8 Object Detection | Stroke Width Variance Check |
| 6. Dealer Stamp     | Boolean + Box     | [ymin, xmin...]     | YOLOv8 + HSV Chromatic  | Saturation Density Filter   |
+---------------------+-------------------+---------------------+-------------------------+-----------------------------+
```

---

### Detailed Extraction Algorithm: Field 1 — Dealer Name

```
[ Page 1 Raster ] ──> [ Crop Top 35% Header ] ──> [ OCR / Text Tokens ] ──> [ Filter Blocklist ] ──> [ RapidFuzz Match ]
```

1. **Spatial Zone Restriction:** Crop tokens from Page 1 with normalized $y_{\max} \le 350$.
2. **Negative Filtering & Customer Stripping:**
   * Remove lines containing buyer indicators:
     ```python
     BUYER_PATTERNS = [
         r'(?i)\b(?:customer|buyer|farmer|purchaser|shri|smt|m/s\s+[a-z]+\s+(?:kumar|singh|patel))\b',
         r'(?i)\b(?:नाम|ग्राहक|किसान|श्री|पता|ग्राम|तहसील)\b',
         r'(?i)\b(?:to\s*:\s*|sold\s*to\s*:)\b'
     ]
     ```
   * Remove financier blocklist names: `"IDFC FIRST"`, `"HDFC Bank"`, `"Mahindra & Mahindra Financial Services"`, `"State Bank of India"`, `"Kotak"`, `"Cholamandalam"`.
3. **Typographic Prominence Scoring:**
   For each remaining text block in the header, compute letterhead score:
   $$\text{Score}(B) = \text{FontSize}(B) \times 1.5 + \text{IsBold}(B) \times 20.0 + \text{IsCentered}(B) \times 15.0 - \text{TopOffset}(B) \times 0.1$$
4. **Catalogue Alignment via RapidFuzz:**
   Run multi-token fuzzy alignment against `dealer_master.json`:
   ```python
   from rapidfuzz import process, fuzz

   best_match = process.extractOne(
       candidate_text, 
       dealer_names_list, 
       scorer=fuzz.token_set_ratio, 
       score_cutoff=85.0
   )
   ```
   * If a match with $\text{score} \ge 85.0$ is found, normalize to canonical legal name.
   * If no catalogue match is found, preserve the highest-scoring candidate text, title-cased and stripped of leading `"M/s"`, `"Messrs"`, or `"M/S."`.
5. **Stamp Text Fallback:** If the header yields no confident candidate (e.g., invoice on plain paper), OCR the detected `dealer_stamp` crop to extract the dealership name etched on the outer ring of the rubber seal.

---

### Detailed Extraction Algorithm: Field 2 — Model Name

```
[ Line Item Table ] ──> [ Scan OEM Brand Anchor ] ──> [ N-Gram Window ] ──> [ Jaro-Winkler Align ] ──> [ Filter Implements ]
```

1. **Table Region Detection:** Identify the line-item bounding box by finding column headers: `"Description"`, `"Particulars"`, `"Item Name"`, `"Model"`, `"विवरण"`, `"સાધનોની વિગત"`.
2. **Brand Anchor Scanning:** Scan line-item tokens for OEM brand anchors:
   ```python
   OEM_BRANDS = [
       "mahindra", "swaraj", "john deere", "jd", "sonalika", "eicher",
       "tafe", "massey ferguson", "mf", "farmtrac", "powertrac", "kubota",
       "new holland", "preet", "captain", "indo farm"
   ]
   ```
3. **N-Gram Window Mining:**
   From the brand anchor token index $i$, extract windows of tokens:
   $$W_k = [T_i, T_{i+1}, \dots, T_{i+k}], \quad k \in \{1, 2, 3, 4, 5\}$$
4. **Catalogue Distance Scoring:**
   Score candidate strings against all model variants in `oem_tractor_models.json`:
   ```python
   def match_model(candidate: str, brand: str) -> Optional[dict]:
       candidates_in_brand = [m for m in OEM_CATALOGUE if m["brand"].lower() == brand.lower()]
       best_score = 0.0
       best_model = None
       for m in candidates_in_brand:
           # Match against canonical name and aliases
           score = max([fuzz.token_sort_ratio(candidate, alias) for alias in m["aliases"] + [m["canonical_name"]]])
           if score > best_score:
               best_score = score
               best_model = m
       if best_score >= 80.0:
           return best_model
       return None
   ```
5. **Agricultural Implement Exclusion Filter:**
   Ensure the matched item is an actual tractor and not an accessory. Reject lines containing:
   `"Rotavator"`, `"Trolley"`, `"Cultivator"`, `"MB Plough"`, `"Harrow"`, `"Trailer"`, `"Threshing Machine"`.

---

### Detailed Extraction Algorithm: Field 3 — Horse Power (HP)

```
[ Regex Pattern Scan ] ──> [ Metric kW Detection ] ──> [ Proximity to Model ] ──> [ [15, 120] Bounds ] ──> [ OEM Impute ]
```

1. **Regex Extraction Engines:**
   Scan all document text with case-insensitive patterns:
   ```python
   import re

   HP_PATTERNS = [
       r'(?i)\b(\d{2}(?:\.\d{1,2})?)\s*(?:hp|h\.p\.|horse\s*power|एचपी|हॉर्सपावर|હોર્સપાવર)\b',
       r'(?i)\b(?:hp|h\.p\.|एचपी)\s*[:\-\.]?\s*(\d{2}(?:\.\d{1,2})?)\b'
   ]
   KW_PATTERNS = [
       r'(?i)\b(\d{2}(?:\.\d{1,2})?)\s*(?:kw|k\.w\.|किलोवाट)\b'
   ]
   ```
2. **Metric Conversion:**
   If a kilowatt rating is matched (common in official tractor CMVR invoices):
   $$\text{HP}_{\text{converted}} = \text{round}(\text{kW} \times 1.34102, 1)$$
3. **Spatial Proximity Weighting:**
   When multiple HP candidates exist, compute distance from the candidate box $B_C$ to the extracted `model_name` box $B_M$:
   $$\Delta_{\text{dist}} = |y_{C} - y_{M}| + 0.5 \cdot |x_{C} - x_{M}|$$
   Select the candidate minimizing $\Delta_{\text{dist}}$.
4. **Physical Plausibility Bounds:**
   Assert:
   $$15.0 \le \text{HP} \le 120.0$$
   * Values $< 15.0$ or $> 120.0$ are rejected. (Eliminates confusing model codes like `744`, displacement like `2979 CC`, or RPM like `2000` with HP).
5. **Catalogue Imputation Fallback:**
   If no valid HP text token exists anywhere on the quotation:
   * Look up the extracted `model_name` in `oem_tractor_models.json`.
   * Return the certified `hp` value with metadata tag: `"provenance": "imputed_from_catalogue"`.

---

### Detailed Extraction Algorithm: Field 4 — Asset Cost

```
[ Currency Regex ] ──> [ Indian Comma Normalizer ] ──> [ Bottom-Right Totals Anchors ] ──> [ Arithmetic Verification ]
```

1. **Candidate Currency Pattern Matching:**
   Extract numbers matching standard Indian currency formatting:
   ```python
   COST_PATTERNS = [
       r'(?:Rs\.?|INR|₹|रु\.?|રૂ\.)?\s*([1-9]\d{0,2}(?:,\d{2})*,\d{3}(?:\.\d{2})?)', # Formatted e.g. 7,45,000.00
       r'(?:Rs\.?|INR|₹|रु\.?|રૂ\.)?\s*([2-9]\d{5}(?:\.\d{2})?)'                      # Unformatted e.g. 745000
   ]
   ```
2. **Indian Number Formatting Normalization:**
   ```python
   def parse_indian_currency(raw_str: str) -> float:
       # Strip currency symbols, spaces, trailing '/-'
       cleaned = re.sub(r'[^\d\.]', '', raw_str.replace('/-', '').strip())
       return float(cleaned)
   ```
3. **Spatial Totals Quadrant Anchoring:**
   * Filter candidates located in the bottom 40% of the document ($y \ge 0.60$) and right half ($x \ge 0.40$).
   * Calculate proximity score to total anchor keywords:
     `"Grand Total"`, `"Total Amount"`, `"Net Amount"`, `"Total Asset Cost"`, `"Final Payable"`, `"कुल राशि"`, `"કુલ રકમ"`.
4. **Disambiguation Against Other Amounts:**
   * Penalize values located next to negative keywords:
     `"Advance"`, `"Margin Money"`, `"Down Payment"`, `"Token Amount"`, `"Balance Due"`, `"Subsidy"`, `"Discount"`.
5. **Mathematical Consistency Reconciliation:**
   When breakdown rows exist:
   $$\text{Reconciled Total} = \text{ExShowroom} + \text{CGST} + \text{SGST} + \text{Insurance} + \text{RTO}$$
   If arithmetic check matches within $\pm ₹5.00$, lock candidate with $1.0$ confidence.
6. **Words-to-Numbers Cross-Validation:**
   Run Indian words-to-number parser on the `"Amount in Words"` / `"शब्दों में राशि"` line (e.g., *"Seven Lakh Forty Five Thousand Only"* $\to 750000.0$). If numeric OCR read `"745000"`, the words verify that no zero was omitted.
7. **Plausibility Envelope:**
   Assert:
   $$₹2,00,000.00 \le \text{Asset Cost} \le ₹25,00,000.00$$

---

### Detailed Extraction Algorithm: Field 5 — Dealer Signature

```
[ Page Image ] ──> [ YOLOv8x Signature Inference ] ──> [ Stroke Width Filter ] ──> [ Bounding Box Format ]
```

1. **Model Inference:** Run `yolo_stamp_sig_v1.pt` on the preprocessed page image at standard $1024 \times 1024$ resolution with confidence threshold $0.50$.
2. **Spatial Location Filtering:**
   Agricultural invoice signatures reside strictly in the bottom 30% of the page ($y_{\min} \ge 700$ in normalized coordinates). Any proposal in the top half is rejected as a stray pen mark or customer signature.
3. **Anchor Word Proximity Check:**
   Reward proposals located within vertical distance $\Delta y \le 120\text{px}$ of anchor phrases:
   `"Authorized Signatory"`, `"For <Dealer Name>"`, `"Signature of Dealer"`, `"हस्ताक्षर"`, `"સહી"`.
4. **Handwriting Stroke Verification (CCA):**
   * Crop the detected proposal.
   * Run Connected Component Analysis (CCA) and stroke width variance test:
     $$\sigma_{\text{stroke}}^2 = \text{variance}(\{\text{width}(C_i) \mid C_i \in \text{contours}\})$$
   * Handwritten cursive strokes exhibit high variance ($\sigma_{\text{stroke}}^2 > 3.5$); printed text borders or solid underline bars exhibit near-zero variance ($\sigma_{\text{stroke}}^2 < 0.8$); reject low-variance boxes.
5. **Output Serialization:**
   * If detected: `present = True`, `bounding_box = [ymin, xmin, ymax, xmax]`, `confidence = float(conf)`.
   * If not detected: `present = False`, `bounding_box = None`, `confidence = 0.0`.

---

### Detailed Extraction Algorithm: Field 6 — Dealer Stamp

```
[ Page Image ] ──> [ YOLOv8x Stamp Inference ] ──> [ HSV Ink Saturation Filter ] ──> [ Overlap Resolving ]
```

1. **Model Inference:** Run `yolo_stamp_sig_v1.pt` for class `stamp` with confidence threshold $0.50$.
2. **HSV Inked Seal Chromatic Verification:**
   Rubber stamps in India utilize violet/purple, royal blue, or red ink. They are rarely printed in true monochrome black toner.
   * Extract image crop for proposal $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$.
   * Convert crop from BGR to HSV color space.
   * Calculate non-black, high-saturation pixel ratio:
     $$S_{\text{ratio}} = \frac{1}{H \times W} \sum_{x,y} \mathbb{I}\left(S(x,y) > 0.25 \land 0.20 \le V(x,y) \le 0.95 \land (H \in [100, 160] \lor H \in [0, 20])\right)$$
   * If $S_{\text{ratio}} \ge 0.06$, confirm as authentic physical ink stamp.
   * If $S_{\text{ratio}} < 0.02$, proposal is a printed circular company logo or square table box; reject.
3. **Stamp & Signature Overlap Resolution:**
   Signatures are frequently signed directly across or inside the rubber stamp.
   * If $\text{IoU}(\text{Box}_{\text{stamp}}, \text{Box}_{\text{signature}}) > 0.30$, both entities are preserved independently.
   * The stamp detector keeps its circular/oval convex hull; the signature detector keeps the cursive stroke bounds.
4. **Output Serialization:**
   Emit `present: bool`, `bounding_box: [ymin, xmin, ymax, xmax]`, `confidence: float`, `page_index: int`.

---

## 4. Part III: Cross-Field Consistency & Fallback Engine

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                        MULTI-FACTOR CONFIDENCE CALCULATION                                            |
+-----------------------------------------------------------------------------------------------------------------------+
|                                                                                                                       |
|   C_final = 0.30 * C_ocr + 0.30 * C_spatial + 0.25 * C_catalogue + 0.15 * C_cross_field                              |
|                                                                                                                       |
|   Where:                                                                                                              |
|   - C_ocr: Average recognition probability of characters in the value                                                 |
|   - C_spatial: Distance score relative to expected header / table / footer anchors                                   |
|   - C_catalogue: RapidFuzz / Jaro-Winkler similarity against master registry (0.0 to 1.0)                             |
|   - C_cross_field: Binary or graded score from the Cross-Field Consistency Matrix below                               |
|                                                                                                                       |
+-----------------------------------------------------------------------------------------------------------------------+
```

### The Cross-Field Consistency Matrix
Every extraction candidate must pass cross-field integrity checks before serialization:

| Check Name | Governing Logic | Error Action |
| :--- | :--- | :--- |
| **Model ↔ HP Consistency** | Verified `model_name` has rated HP range in catalogue. Extracted `horse_power` must fall within $[ \text{HP}_{\min} - 2.0, \text{HP}_{\max} + 2.0 ]$. | If HP conflicts, impute certified catalogue HP; flag warning. |
| **Model ↔ Cost Plausibility** | Extracted `asset_cost` must fall within OEM model price envelope $[ \text{Price}_{\min} \times 0.85, \text{Price}_{\max} \times 1.25 ]$. | If Cost is off by factor of 10 (e.g. ₹75,000 instead of ₹7,50,000), repair missing zero. |
| **Dealer ↔ Stamp Verification** | If both `dealer_name` and `dealer_stamp` exist, stamp OCR text must have fuzzy match $\ge 60.0$ with dealer name. | If match $< 30.0$, flag potential dealer proxy / sub-agent quotation. |

---

### Targeted Multimodal VLM Fallback Routing
To maintain operating costs $< \$0.01$ and latency $< 30\text{s}$, a Vision-Language Model is **never** invoked for the whole document on every turn. It is used as a **surgical fallback** only when internal confidence $C_{\text{final}} < 0.70$:

```python
def targeted_vlm_fallback(image_crop: Image.Image, field_name: str) -> dict:
    """
    Invoked strictly on low-confidence local regions (<5% of documents).
    Costs < $0.0003 per call via Gemini 1.5 Flash.
    """
    prompt = f"""
    You are an expert Indian invoice auditor. Extract the exact '{field_name}' from this quotation crop.
    Return strictly valid JSON conforming to this schema:
    {{"value": <extracted_value>, "confidence": <float_0_to_1>, "explanation": "<brief>"}}
    """
    response = vlm_client.generate_content([prompt, image_crop])
    return json.loads(response.text)
```

---

## 5. Part IV: Step-by-Step Implementation & Verification Guide

Follow this sequence to build, verify, and package the pipeline:

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                             IMPLEMENTATION SPRINT ROADMAP                                             |
+-----------------------------------------------------------------------------------------------------------------------+
|                                                                                                                       |
|  [ SPRINT 1: DATA INFRASTRUCTURE & ANNOTATION (Days 1–3) ]                                                            |
|  1. Ingest ~500 raw invoices into datasets/raw/                                                                       |
|  2. Run deduplication script (imagehash) to eliminate exact duplicate scans                                           |
|  3. Export 350 training files to CVAT; annotate stamps, signatures, and 4 text fields                                 |
|  4. Generate leakage-free GroupKFold splits grouped by dealership                                                     |
|                                                                                                                       |
|  [ SPRINT 2: VISION DETECTOR TRAINING (Days 4–6) ]                                                                    |
|  1. Format CVAT bounding boxes into YOLO Darknet format                                                               |
|  2. Fine-tune YOLOv8x on stamp and signature classes for 120 epochs                                                   |
|  3. Verify validation mAP@0.50 >= 0.88; export to weights/yolo_stamp_sig_v1.pt                                        |
|                                                                                                                       |
|  [ SPRINT 3: EXTRACTOR LOGIC & REGISTRIES (Days 7–10) ]                                                               |
|  1. Compile oem_tractor_models.json (450 models) and dealer_master.json                                               |
|  2. Implement dual-track text extractor (PyMuPDF vector + PaddleOCR v4 raster)                                        |
|  3. Implement field extractors (Dealer, Model, HP, Cost) with RapidFuzz scoring                                       |
|  4. Implement cross-field validation rules and Pydantic v2 serializers                                                |
|                                                                                                                       |
|  [ SPRINT 4: BENCHMARKING & REPRODUCIBLE PACKAGING (Days 11–13) ]                                                     |
|  1. Execute frozen benchmark on ~150 validation documents                                                             |
|  2. Validate Document Accuracy >= 95.0%, P95 latency <= 15.0s, unit cost < $0.005                                     |
|  3. Freeze CLI entrypoint executable.py and verify clean run in fresh virtualenv                                      |
|                                                                                                                       |
+-----------------------------------------------------------------------------------------------------------------------+
```

### Verification Commands & Health Checks
Run these commands in terminal to verify each milestone:

```bash
# 1. Verify environment dependencies and CUDA/GPU availability
python -c "import torch, fitz, paddleocr, ultralytics; print('CUDA Available:', torch.cuda.is_available())"

# 2. Run unit tests on individual field extractors and normalizers
pytest tests/unit/ -v

# 3. Test YOLO stamp and signature detector inference on sample image
python -m core.vision.detector --weights weights/yolo_stamp_sig_v1.pt --source tests/fixtures/sample_invoice.jpg

# 4. Run end-to-end extraction CLI on single document
python executable.py --input tests/fixtures/sample_invoice.pdf --output output_sample.json

# 5. Run full frozen validation benchmark
python evaluation/benchmark.py --dataset-manifest datasets/val_manifest.json --strict-eval
```
