# Architecture Diagram

The InvoiceFlow extraction pipeline is a six-stage system: an input document
flows through ingestion, parallel OCR + vision detection, two-tier text
extraction, normalization, confidence scoring, and JSON serialization.

## Mermaid

```mermaid
graph TD
    Input[PDF / PNG / JPG / JPEG]

    subgraph Stage1[Stage 1: Ingestion]
        Ing[PyMuPDF + Pillow + OpenCV<br/>deskew · denoise · CLAHE<br/>multi-page relevance scoring]
    end

    subgraph Stage2A[Stage 2A: OCR]
        OCR[PaddleOCR PP-OCRv4<br/>en + devanagari<br/>token streams + bbox]
    end

    subgraph Stage2B[Stage 2B: Vision]
        VIS[YOLOv8n fine-tuned<br/>signature · stamp<br/>mAP@50 = 0.88]
    end

    subgraph Stage3[Stage 3: Field Extraction]
        T1[Tier-1: anchored regex rules<br/>HP · cost · dealer · model]
        T2[Tier-2: Qwen2.5-1.5B SLM<br/>only when Tier-1 conf < 0.55<br/>+ anti-hallucination guard]
        T1 -->|low conf| T2
    end

    subgraph Stage4[Stage 4: Normalization]
        NORM[fuzzy match dealer master<br/>exact match asset master<br/>numeric range gating<br/>cross-field consistency]
    end

    subgraph Stage5[Stage 5: Confidence]
        CONF[per-field weighted score<br/>doc-level aggregation<br/>review-flag threshold]
    end

    subgraph Stage6[Stage 6: Output]
        OUT[Pydantic v2 schema<br/>doc_id · 6 fields · confidence<br/>processing_time · cost]
    end

    Input --> Ing
    Ing --> OCR
    Ing --> VIS
    OCR --> T1
    T2 --> NORM
    T1 --> NORM
    NORM --> CONF
    VIS --> CONF
    CONF --> OUT
    OUT --> JSON[result.json]
```

## Stage responsibilities

| Stage | Module | Inputs | Outputs |
|---|---|---|---|
| 1. Ingestion | `utils/ingestion.py` | file path | preprocessed page image + embedded text |
| 2A. OCR | `utils/ocr.py` | page image | list of `OcrToken` (text, bbox, conf, script) |
| 2B. Vision | `utils/detection.py` | page image | list of `Detection` (class, bbox, conf) |
| 3. Tier-1 | `utils/extraction.py` | OCR tokens | `dict[str, FieldExtraction]` with conf |
| 3. Tier-2 | `utils/slm.py` | OCR text + missing fields | refined `FieldExtraction` for low-conf fields |
| 4. Normalize | `utils/normalization.py` | Tier-1/2 output + masters | canonical values + dampened conf |
| 5. Confidence | `utils/confidence.py` | normalized fields + visuals | per-field + doc confidence |
| 6. Output | `utils/schema.py` | all stage outputs | Pydantic-validated JSON |

## Hardware adaptation

A single auto-detected `DeviceInfo` flows through every stage:

* **CUDA available** — PaddleOCR uses GPU, YOLOv8n uses GPU, Qwen-1.5B
  loads at FP16 on GPU.
* **CPU only** — every component falls back to CPU. Same artifact, no
  rebuild. Output JSON is identical except for `processing_time_sec`.

## Stage isolation

Each stage runs inside its own try/except in `utils/pipeline.py`. A
failure in any single stage does NOT crash the document — the pipeline
substitutes nulls / empty results for that stage's outputs and continues.
The full document has a 60-second hard timeout as a final safety net.

## Network policy

Every model and tokenizer ships in `backend/models/` and is loaded with
`local_files_only=True` (transformers) or via filesystem paths
(PaddleOCR, Ultralytics). The optional `--offline` CLI flag activates the
network kill-switch in `utils/offline_guard.py` which monkey-patches
`socket.socket.connect` to refuse any non-loopback address. The pipeline
is designed to pass `python executable.py path --offline` inside a
`--network=none` Docker container.
