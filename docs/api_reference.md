# INVOICEFLOW API & CLI REFERENCE SPECIFICATION
## Comprehensive Command-Line Interface, REST OpenAPI 3.1 & Schema Contracts

---

| **Document Information** | **Specification Details** |
| :--- | :--- |
| **System Name** | **InvoiceFlow Document Intelligence** |
| **Document Classification** | API Specification & CLI Manual |
| **Document Version** | **1.0.0-PROD** |
| **Standards Compliance** | OpenAPI 3.1.0, JSON Schema Draft-07, POSIX CLI |

---

## 1. Standalone CLI Specification (`executable.py`)

The primary execution entrypoint is `executable.py`, designed for standalone offline batch and single-file processing:

```bash
python executable.py --input <path_to_file_or_dir> --output <output_path> [OPTIONS]
```

### 1.1 CLI Argument Flags
```
OPTIONS:
  -i, --input PATH        Path to input PDF or image file, or directory of files. [Required]
  -o, --output PATH       Path where output JSON file or directory will be written. [Required]
  --device [cpu|cuda]     Compute device to use for inference. [Default: cpu]
  --batch-size INTEGER    Number of parallel worker processes. [Default: 4]
  --strict-eval           Enforces strict evaluation assertions (no null values).
  --verbose               Enables detailed stage-by-stage timing logs.
  --help                  Show this message and exit.
```

### 1.2 CLI Exit Codes
* `0`: Success. All files processed and valid JSON produced.
* `1`: Input Error. File not found, unreadable, or invalid argument.
* `2`: Processing Error. Unrecoverable document failure.
* `3`: Schema Validation Failure. Output failed Pydantic schema check.

---

## 2. REST API Specification (FastAPI / OpenAPI 3.1)

### `POST /api/v1/extract`
Extracts structured fields from an uploaded document.

#### Request (Multipart Form Data)
* `file`: Binary file upload (`application/pdf`, `image/jpeg`, `image/png`).
* `document_type`: Optional string (`"tractor_quotation"`, default: `"tractor_quotation"`).

#### Response (`200 OK`)
```json
{
  "document_id": "sample_invoice.pdf",
  "dealer_name": "Patel Tractors & Farm Equipment",
  "model_name": "Mahindra 575 DI SP Plus",
  "horse_power": 47.0,
  "asset_cost": 745000.0,
  "dealer_signature": {
    "present": true,
    "bounding_box": [820, 680, 910, 890],
    "confidence": 0.94,
    "page_index": 0
  },
  "dealer_stamp": {
    "present": true,
    "bounding_box": [800, 650, 930, 850],
    "confidence": 0.92,
    "page_index": 0
  },
  "metadata": {
    "execution_time_seconds": 1.84,
    "pipeline_version": "1.0.0",
    "page_count": 1
  }
}
```
