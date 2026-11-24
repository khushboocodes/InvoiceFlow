# INVOICEFLOW DATA DICTIONARY & FIELD SPECIFICATIONS
## Formal Attribute Definitions, Validation Bounds, Normalization Transforms & Null Conventions

---

| **Document Information** | **Specification Details** |
| :--- | :--- |
| **System Name** | **InvoiceFlow Document Intelligence** |
| **Document Classification** | Enterprise Data Dictionary & Attribute Taxonomy |
| **Document Version** | **1.0.0-PROD** |
| **Target Document Family** | Tractor Loan Quotations & Proforma Invoices |

---

## 1. Master Attribute Directory

```
+-----------------------------------------------------------------------------------------------------------------------------------------+
|                                                          MASTER FIELD DICTIONARY                                                        |
+---------------------+-------------------+---------------------+-------------------------+-----------------------+-----------------------+
| Field Identifier    | Target Type       | Canonical Example   | Coordinate Localization | Valid Range / Bounds  | Null / Abstention Rule|
+---------------------+-------------------+---------------------+-------------------------+-----------------------+-----------------------+
| `document_id`       | String            | `"IDFC_TRACT_0042"` | No                      | Non-empty string      | Mandatory (No null)   |
| `dealer_name`       | String            | `"Patel Tractors"`  | Yes (Source Region)     | 2 to 200 characters   | Mandatory (No null)   |
| `model_name`        | String            | `"Swaraj 744 FE"`   | Yes (Source Region)     | 2 to 100 characters   | Mandatory (No null)   |
| `horse_power`       | Float             | `48.0`              | Yes (Context Box)       | 15.0 to 120.0 HP      | Mandatory (No null)   |
| `asset_cost`        | Float             | `745000.0`          | Yes (Context Box)       | ₹2,00,000 to ₹25,00,000| Mandatory (No null)  |
| `dealer_signature`  | Object (Detection)| `{"present": true}` | Yes ([ymin, xmin...])   | Coordinates [0, 1000] | Mandatory object      |
| `dealer_stamp`      | Object (Detection)| `{"present": true}` | Yes ([ymin, xmin...])   | Coordinates [0, 1000] | Mandatory object      |
| `metadata`          | Object            | Telemetry metrics   | No                      | Standard telemetry    | Mandatory object      |
+---------------------+-------------------+---------------------+-------------------------+-----------------------+-----------------------+
```

---

## 2. Deep Field Specifications

### 2.1 `dealer_name`
* **Definition:** Official commercial name of the tractor dealership issuing the quotation.
* **Transformations Applied:**
  * Removal of honorifics and prefixes: `"M/s"`, `"Messrs"`, `"M/S."`, leading punctuation.
  * Collapsing duplicate whitespace characters.
  * Title-casing for Latin strings (`"PATEL TRACTORS"` $\to$ `"Patel Tractors"`).
  * RapidFuzz token matching against `dealer_master.json` ($\text{cutoff} \ge 85.0$).
* **Null Policy:** If unreadable, emit `"UNKNOWN_DEALER"` with zero confidence.

---

### 2.2 `model_name`
* **Definition:** Commercial make, brand, series, and variant designation of the tractor.
* **Transformations Applied:**
  * Brand abbreviation resolution (`"JD"` $\to$ `"John Deere"`, `"MF"` $\to$ `"Massey Ferguson"`).
  * Spacing normalization (`"575DI"` $\to$ `"575 DI"`).
  * Canonical lookup against `oem_tractor_models.json`.
* **Null Policy:** If no model recognized, emit `"UNKNOWN_TRACTOR_MODEL"`.

---

### 2.3 `horse_power`
* **Definition:** Rated engine power capacity in Horse Power units.
* **Transformations Applied:**
  * Metric kilowatt to horsepower conversion: $\text{HP} = \text{round}(\text{kW} \times 1.34102, 1)$.
  * Stripping unit suffixes (`"HP"`, `"H.P."`, `"Horse Power"`).
  * Conversion of Devanagari/Gujarati numeral glyphs to ASCII.
* **Plausibility Enforcement:** Strict assertion: $15.0 \le \text{HP} \le 120.0$.
* **Imputation Rule:** If missing on document, look up canonical model from `oem_tractor_models.json` and impute rated HP.

---

### 2.4 `asset_cost`
* **Definition:** Total payable invoice or quotation value in Indian Rupees (INR).
* **Transformations Applied:**
  * Stripping currency symbols (`₹`, `Rs.`, `INR`, `रु.`), slashes (`/-`), and whitespace.
  * Parsing Indian grouping commas (`7,50,000.00` $\to$ `750000.0`).
  * Arithmetic tax summation: $\text{Cost} = \text{ExShowroom} + \text{CGST} + \text{SGST} + \text{Insurance}$.
* **Plausibility Enforcement:** Strict assertion: $₹2,00,000.0 \le \text{Asset Cost} \le ₹25,00,000.0$.

---

### 2.5 `dealer_signature` & `dealer_stamp`
* **Object Structure:**
  ```json
  {
    "present": true,
    "bounding_box": [820, 680, 910, 890],
    "confidence": 0.94,
    "page_index": 0
  }
  ```
* **Coordinate Rules:**
  * All coordinates are integers normalized to $[0, 1000]$ relative to image height and width.
  * Format: `[ymin, xmin, ymax, xmax]`.
  * Strict constraint: $0 \le y_{\min} \le y_{\max} \le 1000$ and $0 \le x_{\min} \le x_{\max} \le 1000$.
  * If `present == false`, `bounding_box` must be `null` and `confidence` must be $0.0$.
