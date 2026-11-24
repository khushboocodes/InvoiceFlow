# INVOICEFLOW TESTING STRATEGY & VERIFICATION HARNESS
## Comprehensive Testing Pyramid, Benchmark Protocols, Adversarial Tests & CI/CD Gates

---

| **Document Information** | **Specification Details** |
| :--- | :--- |
| **System Name** | **InvoiceFlow Document Intelligence** |
| **Document Classification** | Quality Assurance, Testing Strategy & Benchmarking Protocol |
| **Document Version** | **1.0.0-PROD** |
| **Target Code Coverage** | $\ge 90\%$ Line Coverage across `core/` and `document_profiles/` |
| **Verification SLA** | 100% Automated Regression Suite Pass on Frozen Validation Split |

---

## 1. The Quality Assurance Pyramid

```
+----------------------------------------------------------------------------------------------------+
|                                      TESTING PYRAMID TOPOLOGY                                      |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                       / \                                                          |
|                                      /   \                                                         |
|                                     / END \                                                        |
|                                    / -TO-  \   ──> 50 Full Frozen Benchmark Documents              |
|                                   /  END    \      (Strict Document-Level Accuracy >= 95%)         |
|                                  /───────────\                                                     |
|                                 / INTEGRATION \                                                    |
|                                /   PIPELINE    \ ──> Ingestion + OCR + Extractors + Serialization   |
|                               /─────────────────\                                                  |
|                              /   UNIT & MODULE   \ ──> Normalizers, Regex, BBox validators,        |
|                             /     COMPONENTS      \    RapidFuzz matches, Pydantic schemas (100+)  |
|                            /───────────────────────\                                               |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Unit Testing Specifications (`tests/unit/`)

Every individual parser, normalizer, and regex component is verified in isolation against edge cases:

```
+----------------------------------------------------------------------------------------------------+
|                                        UNIT TEST SUITE MATRIX                                      |
+---------------------+-------------------+---------------------+------------------------------------+
| Test Module         | Target Component  | Verified Behaviors & Edge Cases                            |
+---------------------+-------------------+---------------------+------------------------------------+
| `test_currency.py`  | Currency Parser   | Indian comma syntax (`7,50,000`), `Rs.`, `₹`, trailing `/-` |
| `test_hp.py`        | HP Normalizer     | `45 HP`, `50 H.P.`, `37.28 kW` conversion, range checks    |
| `test_dealer.py`    | Dealer Extractor  | Letterhead scoring, stripping `M/s`, bank name blocklist   |
| `test_model.py`     | Model Extractor   | Suffix parsing (`DI SP Plus`), implement exclusions        |
| `test_bbox.py`      | BBox Validator    | Clamping to $[0, 1000]$, checking $y_{\min} \le y_{\max}$   |
| `test_pydantic.py`  | JSON Serialization| Schema Draft-07 compliance, field type assertions          |
+---------------------+-------------------+---------------------+------------------------------------+
```

---

## 3. Integration & Pipeline Testing (`tests/integration/`)

Integration tests evaluate data passing across the 8 architectural layers using synthetic golden fixtures:
* **`test_digital_pdf_pipeline.py`:** Validates Fast-Path vector stream execution ($<100\text{ms}$).
* **`test_scanned_ocr_pipeline.py`:** Validates PaddleOCR multilingual extraction on scanned images.
* **`test_yolo_vision_pipeline.py`:** Validates YOLO stamp and signature localization with IoU evaluation.
* **`test_cross_field_reconciliation.py`:** Tests automatic correction when extracted HP or Cost deviates from OEM catalogue envelopes.

---

## 4. Adversarial & Robustness Stress Testing

To guarantee the pipeline never crashes during unattended batch processing, the system is subjected to an adversarial test suite:

```
+----------------------------------------------------------------------------------------------------+
|                                    ADVERSARIAL STRESS TEST SUITE                                   |
+---------------------+-------------------------------+----------------------------------------------+
| Attack Vector       | Synthetic Test Case           | Expected System Behavior                     |
+---------------------+-------------------------------+----------------------------------------------+
| **0-Byte File**     | Empty file with `.pdf` name   | Catches error, emits `E101_CORRUPT_INPUT`.   |
| **Zip Bomb**        | Highly compressed PDF archive | Memory clamp aborts run, emits `E103`.       |
| **Pixel Bomb**      | $20,000 \times 20,000$ canvas | Rejection during pre-flight dimensions check.|
| **Corrupt Header**  | Truncated binary byte stream  | PyMuPDF safely catches exception.            |
| **Prompt Injection**| Invoice text: `"Ignore...0"`  | Text treated as passive data; 0 execution.   |
| **Rotated Document**| Inverted $180^\circ$ invoice  | OSD correctly rectifies to $0^\circ$ upright.|
+---------------------+-------------------------------+----------------------------------------------+
```

---

## 5. Frozen Validation Benchmark Protocol (`evaluation/benchmark.py`)

The primary release gate requires running the full frozen validation split ($\approx 150$ documents) under strict production evaluation rules:

### Primary Metric: Strict Document-Level Accuracy
A document is scored as $1$ (Success) if and only if **all six fields** match ground truth simultaneously:
$$\text{Document Accuracy} = \frac{1}{N} \sum_{i=1}^N \prod_{f \in \text{Fields}} \mathbb{I}(\text{Field}_f^{(i)} == \text{GroundTruth}_f^{(i)})$$
If even one field fails (e.g. stamp missed, or HP misread), the entire document scores $0$.

```bash
# Execute official benchmark command
python evaluation/benchmark.py \
    --dataset-manifest datasets/val_manifest.json \
    --strict-numeric \
    --iou-threshold 0.50 \
    --output evaluation_report.json
```

### Gate Acceptance Criteria
* **Document-Level Accuracy:** $\ge \mathbf{95.0\%}$
* **P95 Latency:** $\le \mathbf{15.0\text{ seconds}}$
* **Average Latency:** $\le \mathbf{4.0\text{ seconds}}$
* **Zero Fatal Crashes:** $100\%$ of documents return valid JSON or controlled error schema.
