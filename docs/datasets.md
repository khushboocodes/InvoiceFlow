# INVOICEFLOW DATASET SPECIFICATION & AUDIT MANIFEST
## Forensic Data Profiling, Quality Taxonomies, Partitioning Protocol & Metadata Schemas

---

| **Document Information** | **Specification Details** |
| :--- | :--- |
| **System Name** | **InvoiceFlow Document Intelligence** |
| **Document Classification** | Dataset Engineering & Audit Specification |
| **Document Version** | **1.0.0-PROD** |
| **Primary Dataset** | Tractor Loan Quotations & Proforma Invoices (~500 documents) |
| **Evaluation Set** | Unseen Blind Benchmark (~100 documents) |
| **Data Custody** | Financial PII Compliant / Restricted Sandbox |

---

## 1. Dataset Overview & Census

The InvoiceFlow training and development corpus consists of $\approx 500$ authentic agricultural machinery quotations and proforma invoices issued by tractor dealerships across India.

```
+----------------------------------------------------------------------------------------------------+
|                                      DATASET PHYSICAL COMPOSITION                                  |
+---------------------+-------------------+---------------------+------------------------------------+
| File Category       | Document Count    | Percentage of Total | Physical Artifact Notes            |
+---------------------+-------------------+---------------------+------------------------------------+
| **Digital Vector**  | ~75 documents     | 15.0%               | Born-digital PDFs, TrueType fonts  |
| **Clean Scans**     | ~175 documents    | 35.0%               | 200–300 DPI flatbed scans, clean   |
| **Mobile Photos**   | ~200 documents    | 40.0%               | Camera perspective warp, shadows   |
| **Degraded Scans**  | ~50 documents     | 10.0%               | Low DPI (<100), carbon copy, torn  |
+---------------------+-------------------+---------------------+------------------------------------+
| **TOTAL**           | **~500 documents**| **100.0%**          | 92% 1-page; 8% 2–3 pages           |
+---------------------+-------------------+---------------------+------------------------------------+
```

---

## 2. Linguistic & Regional Distribution

```
+----------------------------------------------------------------------------------------------------+
|                                    LINGUISTIC SCRIPT BREAKDOWN                                     |
+---------------------+-------------------+---------------------+------------------------------------+
| Language / Script   | Representation %  | Primary Regions     | Structural Characteristic          |
+---------------------+-------------------+---------------------+------------------------------------+
| **English (Latin)** | 60.0%             | Pan-India           | Standard English commercial forms  |
| **Hindi (Devanagari)** 25.0%            | UP, MP, Rajasthan   | Hindi header, terms, handwriting   |
| **Gujarati Script** | 15.0%             | Gujarat             | Gujarati letterhead, numerals      |
+---------------------+-------------------+---------------------+------------------------------------+
```

---

## 3. Quality Degradation Taxonomy

The dataset exhibits six primary noise dimensions that the preprocessing and OCR pipelines must neutralize:

```
+----------------------------------------------------------------------------------------------------+
|                                    DEGRADATION TAXONOMY & DEFENSES                                 |
+---------------------+-------------------+---------------------+------------------------------------+
| Artifact Type       | Severity Level    | Prevalence (% docs) | Pipeline Defense / Mitigation      |
+---------------------+-------------------+---------------------+------------------------------------+
| **Perspective Warp**| High              | ~32%                | Convex quad detection + Homography |
| **Rotational Skew** | Medium (+/-15 deg)| ~48%                | Hough Transform baseline deskew    |
| **Carbon Copy Bleed** High             | ~24%                | CLAHE L-channel equalization       |
| **Dot-Matrix Faint**| High              | ~18%                | Morphological dilation + Otsu      |
| **Shadow / Vignette** Medium            | ~28%                | Background division normalization  |
| **Stamp Overlap**   | Critical          | ~38%                | Multi-layer HSV color segmentation |
+---------------------+-------------------+---------------------+------------------------------------+
```

---

## 4. Leakage-Free Dataset Partitioning Protocol

### 4.1 The Core Problem: Layout Memorization Leakage
When training document intelligence models, a standard random train/test split creates **severe data leakage**. Invoices originating from the same dealership ("Patel Tractors") share identical letterhead fonts, printed logos, table column widths, and physical stamp placements. If 4 documents from Patel Tractors are in the training set and 1 is in the validation set, the model will achieve 99% accuracy by memorizing the dealer's layout rather than learning generalized document semantics.

### 4.2 Grouped Stratification Protocol
To ensure valid generalization, the dataset is partitioned using **Dealership-Grouped Stratified K-Fold (`GroupKFold`)**:

```
+----------------------------------------------------------------------------------------------------+
|                                   DATASET SPLIT SPECIFICATION                                      |
+---------------------+-------------------+---------------------+------------------------------------+
| Split Name          | Document Share    | Approximate Count   | Governing Rules                    |
+---------------------+-------------------+---------------------+------------------------------------+
| **Training Split**  | 70%               | ~350 documents      | Full data augmentation permitted   |
| **Validation Split**| 30%               | ~150 documents      | Frozen benchmark; zero leak        |
| **Evaluation Set**  | Unseen Holdout    | ~100 documents      | Blind test; air-gapped quarantine |
+---------------------+-------------------+---------------------+------------------------------------+
```

* **Grouping Key:** Canonical `dealer_name`.
* **Zero Leak Guarantee:** No dealership appearing in the Training Split will ever appear in the Validation Split.
* **Stratification Balance:** Both splits maintain balanced proportions of language scripts (English, Hindi, Gujarati) and image quality tiers (Digital, Clean Scan, Mobile Photo, Degraded).

---

## 5. Dataset Manifest Schema (`metadata_manifest.json`)

Every file in the repository is catalogued in a versioned machine-readable manifest:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "InvoiceFlowDatasetManifest",
  "type": "object",
  "properties": {
    "manifest_version": { "type": "string" },
    "documents": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "doc_id",
          "file_name",
          "file_hash_sha256",
          "phash",
          "page_count",
          "quality_tier",
          "primary_language",
          "dealer_group",
          "split"
        ],
        "properties": {
          "doc_id": { "type": "string" },
          "file_name": { "type": "string" },
          "file_hash_sha256": { "type": "string" },
          "phash": { "type": "string" },
          "page_count": { "type": "integer" },
          "quality_tier": { "enum": ["DigitalVector", "CleanScan", "MobilePhoto", "Degraded"] },
          "primary_language": { "enum": ["en", "hi", "gu", "mixed"] },
          "dealer_group": { "type": "string" },
          "split": { "enum": ["train", "val", "holdout_test"] }
        }
      }
    }
  }
}
```
