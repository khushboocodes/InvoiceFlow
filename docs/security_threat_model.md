# INVOICEFLOW SECURITY THREAT MODEL & PRIVACY SPECIFICATION
## Threat Modeling (STRIDE), Attack Surface Controls & Regulatory Data Protection

---

| **Document Information** | **Specification Details** |
| :--- | :--- |
| **System Name** | **InvoiceFlow Document Intelligence** |
| **Document Classification** | Enterprise Security & Privacy Threat Model |
| **Document Version** | **1.0.0-PROD** |
| **Regulatory Frameworks** | Indian Digital Personal Data Protection (DPDP) Act 2023, GDPR |

---

## 1. System Threat Model (STRIDE Analysis)

```
+----------------------------------------------------------------------------------------------------+
|                                    STRIDE THREAT MITIGATION MATRIX                                 |
+---------------------+-------------------------------+----------------------------------------------+
| Threat Category     | Potential Attack Vector       | InvoiceFlow Countermeasure / Control         |
+---------------------+-------------------------------+----------------------------------------------+
| **Spoofing**        | Forged stamp/signature images | HSV ink saturation and CCA stroke analysis.  |
| **Tampering**       | Altered prices or model text  | Mathematical reconciliation & price bounds.  |
| **Repudiation**     | Undetected pipeline mutations | Cryptographic SHA-256 provenance logging.    |
| **Information Leak**| Customer Aadhaar/PAN exposed  | Redaction filters; zero external network ops.|
| **Denial of Service** Decompression / Zip bomb      | $25\text{ MB}$ limit; $10\text{K px}$ clamp. |
| **Elevation of Priv** Malicious PDF parser exploit   | Isolated container; disabled JS parsing.     |
+---------------------+-------------------------------+----------------------------------------------+
```

---

## 2. Ingestion Defense & Malicious Document Quarantine

1. **PDF Parser Sandboxing:**
   * Disable all active content in PyMuPDF:
     ```python
     fitz.TOOLS.mupdf_display_errors(False) # Prevent noisy stdout injection
     ```
   * Enforce hard execution timeouts: Any document processing exceeding $30.0\text{ seconds}$ is terminated by a signal watchdog.
2. **Decompression & Memory Limits:**
   * Maximum image dimensions: $10,000 \times 10,000\text{ pixels}$.
   * Maximum memory allocation: $2.0\text{ GB}$ per worker process.
3. **Indirect Prompt Injection Defense:**
   * Any OCR text passed to fallback language models is strictly encapsulated within XML isolation blocks:
     ```xml
     <untrusted_document_context>
     {{extracted_ocr_text}}
     </untrusted_document_context>
     ```
   * Output format is rigidly enforced via Pydantic v2 JSON Schema; free-form text output is rejected.

---

## 3. Data Privacy & DPDP Act 2023 Compliance

Agricultural invoices frequently contain customer names, phone numbers, Aadhaar numbers, and bank account numbers. InvoiceFlow enforces strict data privacy:
* **Zero Customer PII Retention:** The pipeline outputs only the commercial dealership and asset fields. Customer details are never saved to disk or serialized into output JSON.
* **Log Redaction:** All application logs pass through a regex scrubber that replaces 12-digit numbers (Aadhaar), 10-digit alphanumerics (PAN), and 10-digit mobile numbers with `[REDACTED]`.
