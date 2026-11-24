# INVOICEFLOW GROUND-TRUTH ANNOTATION GUIDELINES
## Standard Operating Procedure (SOP) for Labeling Bounding Boxes & Text Entities

---

| **Document Information** | **Specification Details** |
| :--- | :--- |
| **System Name** | **InvoiceFlow Document Intelligence** |
| **Document Classification** | Annotator Standard Operating Procedure (SOP) |
| **Document Version** | **1.0.0-PROD** |
| **Tooling Standard** | CVAT (Computer Vision Annotation Tool) / Label Studio |

---

## 1. General Labeling Rules & Principles

1. **Precision First:** Tight bounding boxes with zero unnecessary padding. Do not crop out edges of stamp seals or signature flourishes.
2. **Normalized Coordinates:** All outputs export in normalized $[0, 1000]$ integer coordinate space.
3. **Double Review Adjudication:** Annotations with annotator confidence $<1.0$ require sign-off by the lead reviewer.

---

## 2. Visual Entity Bounding Box Rules

### 2.1 Dealer Stamp (`dealer_stamp`)
* **What to Annotate:** Any inked rubber stamp impression, circular seal, oval stamp, or official rectangular dealer mark.
* **Bounding Standard:** Draw the tightest bounding rectangle enclosing the outer rim of the seal.
* **Faint / Smudged Stamps:** Even if the stamp is partially smudged, if $>50\%$ of the impression is discernible, annotate the entire enclosing rectangle.
* **Overlap with Signature:** If a signature is written directly over the stamp, annotate the entire stamp as `dealer_stamp`, and annotate the signature as a separate overlapping `dealer_signature` box.

```
+---------------------------------------------------------------------------------+
|                                STAMP & SIGNATURE OVERLAP                        |
+---------------------------------------------------------------------------------+
|                                                                                 |
|       +---------------------------------------------+                           |
|       | [dealer_stamp]                              |                           |
|       |       /~~~~~\                               |                           |
|       |      | STAMP |      +-----------------------+-----+                     |
|       |       \_____/       | [dealer_signature]    |     |                     |
|       |                     |     /~~\__            |     |                     |
|       +---------------------+----(______\-----------+     |                     |
|                             |                             |                     |
|                             +-----------------------------+                     |
|                                                                                 |
+---------------------------------------------------------------------------------+
```

### 2.2 Dealer Signature (`dealer_signature`)
* **What to Annotate:** Any handwritten signature, initials, or authorized representative pen mark.
* **What NOT to Annotate:** Do NOT annotate printed company logos, decorative fonts, or printed names.
* **Signatory Anchor Text:** Do NOT include anchor text like `"Authorized Signatory"` or `"For Patel Tractors"` inside the signature box; capture strictly the pen strokes.

---

## 3. Text Entity Ground-Truth Normalization

```
+----------------------------------------------------------------------------------------------------+
|                                    TEXT NORMALIZATION GUIDELINES                                   |
+---------------------+-------------------------------+----------------------------------------------+
| Field Name          | Raw Invoice Text Example      | Ground-Truth Canonical Value                 |
+---------------------+-------------------------------+----------------------------------------------+
| `dealer_name`       | `"M/s PATEL TRACTORS & CO."`  | `"Patel Tractors & Co."`                     |
| `model_name`        | `"SWARAJ 744FE TRACTOR"`      | `"Swaraj 744 FE"`                            |
| `horse_power`       | `"50 H.P. Engine"`            | `50.0`                                       |
| `horse_power`       | `"37.28 kW"`                  | `50.0` (Converted via $kW \times 1.34102$)   |
| `asset_cost`        | `"Rs. 7,45,000/- only"`       | `745000.0`                                   |
+---------------------+-------------------------------+----------------------------------------------+
```
