# Requirements Document

## Introduction

The Invoice Extraction Pipeline is a fully offline, $0-marginal-cost Document AI system that ingests invoice-type documents (running use case: tractor loan quotations, generalized to retail and industrial invoices) and emits a structured JSON containing six target fields plus per-field confidence and processing time. The system is built for the IDFC GenAI Convolve 4.0 hackathon and must satisfy three non-negotiable evaluation criteria: ≥95% Document-Level Accuracy on the held-out 100-document evaluation set, ≤30 seconds average latency per document, and 100% offline operation in a network-isolated environment using only open-source models bundled with the submission.

The pipeline is composed of six sequential stages: Ingestion, OCR, Vision, Field Extraction, Normalization & Validation, and Output. The submission artifact is `submission.zip` (~1.2 GB) containing `executable.py`, bundled model weights (PaddleOCR PP-OCRv4 mobile en+hi+gu, YOLOv8n fine-tuned for signature/stamp, Qwen2.5-1.5B-Instruct Q4_K_M GGUF), a Python virtual-env-installable `requirements.txt`, mined dealer and asset masters, a sample output, and a README with architecture diagrams. An optional FastAPI bridge in a separate `demo/` folder wires the pipeline to the existing React/InvoiceFlow frontend in `src/` for the live demo round but is not part of the graded submission.

Only the YOLOv8n signature/stamp detector is fine-tuned. All other components (PaddleOCR, Qwen2.5-1.5B SLM) are used zero-shot. Dealer and asset masters are mined from the unlabeled training set itself.

## Glossary

- **Pipeline**: The end-to-end Invoice Extraction Pipeline composed of six stages.
- **Ingestion_Module**: Stage 1 component responsible for loading PDF/PNG/JPG/JPEG inputs, rasterizing scanned PDFs at 300 DPI, applying OpenCV preprocessing (deskew, denoise, contrast normalization), and selecting the most relevant page in multi-page documents.
- **OCR_Module**: Stage 2 component wrapping PaddleOCR PP-OCRv4 mobile (English, Hindi, Gujarati detection + recognition + classification models) that returns words, bounding boxes, and per-word confidences.
- **Vision_Module**: Stage 3 component wrapping the fine-tuned YOLOv8n detector that returns bounding boxes for the `signature` and `stamp` classes.
- **Extraction_Module**: Stage 4 component that produces the four text fields (`dealer_name`, `model_name`, `horse_power`, `asset_cost`) using a two-tier strategy.
- **Tier_1_Extractor**: Deterministic regex and anchor-based rule extractor that runs first.
- **Tier_2_Extractor**: Local SLM fallback (Qwen2.5-1.5B-Instruct Q4_K_M via `llama-cpp-python`) that receives OCR text only and is invoked when Tier_1 confidence is below threshold or fails.
- **Validation_Module**: Stage 5 component that normalizes values, fuzzy-matches against the dealer master, exact-matches against the asset master, performs numeric range checks, and runs cross-field consistency checks.
- **Output_Module**: Stage 6 component that produces the Pydantic-validated JSON result.
- **Confidence_Engine**: Sub-component of Validation_Module that computes per-field and document-level confidence.
- **Dealer_Master**: A JSON file at `data/dealer_master.json` containing canonical dealer names mined from the training set.
- **Asset_Master**: A JSON file at `data/asset_master.json` containing canonical tractor brand and model names mined from the training set.
- **DLA**: Document-Level Accuracy. A document scores 1 if and only if all six fields are correct per the match rules in Requirement 11; the metric is the mean across the evaluation set.
- **Match_Rule**: The per-field correctness criterion: dealer name fuzzy match ≥90%, model name exact match, horse_power and asset_cost numeric within ±5% tolerance, signature and stamp presence correct and IoU ≥ 0.5 when present.
- **Submission_Package**: The graded artifact `submission.zip` containing `executable.py`, `requirements.txt`, `README.md`, `utils/`, `models/`, `data/`, and `sample_output/result.json`.
- **Demo_Bridge**: An optional, non-graded FastAPI server in the `demo/` folder that exposes the Pipeline over localhost so the React frontend in `src/` can call it.
- **Frontend**: The pre-built React/TypeScript InvoiceFlow application in `src/` with routes for landing, dashboard, documents, upload, analytics, and `/results/:id`, backed by a Zustand store whose schema matches the six target fields.
- **Offline_Mode**: The runtime state in which `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set, no outbound network calls are made, and all model weights and tokenizers are loaded from the bundled `models/` directory.
- **Eval_Hardware**: The unknown evaluator machine. The PS specifies "CPU or low-tier GPU"; the Pipeline must treat CPU as the floor.

## Requirements

### Requirement 1: Document Ingestion

**User Story:** As a credit officer, I want the Pipeline to accept the document formats banks actually receive, so that I do not have to convert files before processing.

#### Acceptance Criteria

1. THE Ingestion_Module SHALL accept PDF, PNG, JPG, and JPEG files as input.
2. WHEN a digital PDF is provided, THE Ingestion_Module SHALL extract the embedded text layer and per-page rendered images using PyMuPDF.
3. WHEN a scanned PDF is provided, THE Ingestion_Module SHALL rasterize each page to a PNG at 300 DPI using `pdf2image` with the bundled poppler binary.
4. WHEN a PNG, JPG, or JPEG file is provided, THE Ingestion_Module SHALL load the image directly into memory using Pillow without rasterization.
5. THE Ingestion_Module SHALL apply OpenCV preprocessing consisting of deskew, denoise, and contrast normalization to every page image before passing it to downstream stages.
6. WHEN a multi-page document is provided, THE Ingestion_Module SHALL score each page for quotation relevance and select the highest-scoring page as the primary page passed to OCR_Module and Vision_Module.
7. IF a file cannot be opened or is corrupted, THEN THE Ingestion_Module SHALL return a structured error result with the field `error` populated and processing_time_sec recorded.

### Requirement 2: OCR

**User Story:** As a system integrator, I want multilingual OCR that handles English, Hindi, and Gujarati without internet access, so that documents from any Indian state can be processed offline.

#### Acceptance Criteria

1. THE OCR_Module SHALL use PaddleOCR PP-OCRv4 mobile detection, recognition, and classification models for English, Hindi, and Gujarati.
2. THE OCR_Module SHALL load all PaddleOCR weights from `models/paddleocr/` on the local filesystem.
3. THE OCR_Module SHALL return, for each detected token, the recognized text, the four-corner bounding box in pixel coordinates of the preprocessed page, and the recognition confidence in the range [0.0, 1.0].
4. WHEN CUDA is available on Eval_Hardware, THE OCR_Module SHALL initialize PaddleOCR with `use_gpu=True`.
5. IF CUDA is not available on Eval_Hardware, THEN THE OCR_Module SHALL initialize PaddleOCR with `use_gpu=False` and run on CPU using the same bundled weights.
6. THE OCR_Module SHALL make no network calls during initialization or inference.

### Requirement 3: Signature and Stamp Detection

**User Story:** As an evaluator, I want the Pipeline to detect signatures and stamps with bounding boxes that satisfy the IoU ≥ 0.5 threshold, so that the visual fields score correctly.

#### Acceptance Criteria

1. THE Vision_Module SHALL use a YOLOv8n model fine-tuned to detect exactly two classes: `signature` and `stamp`.
2. THE Vision_Module SHALL load weights from `models/yolov8n_sig_stamp.pt` on the local filesystem.
3. THE Vision_Module SHALL return, for each detected instance, the class label, the axis-aligned bounding box `[x1, y1, x2, y2]` in pixel coordinates of the preprocessed page, and the detection confidence in the range [0.0, 1.0].
4. WHEN CUDA is available on Eval_Hardware, THE Vision_Module SHALL run inference on GPU.
5. IF CUDA is not available on Eval_Hardware, THEN THE Vision_Module SHALL run inference on CPU using the same `.pt` weights.
6. THE Vision_Module SHALL achieve mAP@50 ≥ 0.85 on the internal held-out test split during model evaluation.

### Requirement 4: Dealer Name Extraction

**User Story:** As a credit officer, I want the dealer name to match the master file with high tolerance for OCR noise, so that vendor reconciliation succeeds despite small character errors.

#### Acceptance Criteria

1. THE Tier_1_Extractor SHALL search OCR output for dealer-name anchors including but not limited to "Dealer", "M/s", "Authorized Dealer", and the letterhead region of the page.
2. WHEN a dealer-name candidate is found by Tier_1_Extractor with confidence ≥ 0.7, THE Extraction_Module SHALL emit that candidate as the raw dealer_name value.
3. IF Tier_1_Extractor confidence is below 0.7 or no candidate is found, THEN THE Tier_2_Extractor SHALL be invoked with the OCR text and SHALL return a dealer_name string.
4. THE Validation_Module SHALL fuzzy-match the raw dealer_name against Dealer_Master using RapidFuzz token-set ratio.
5. WHEN the highest fuzzy-match score against Dealer_Master is ≥ 90, THE Validation_Module SHALL replace the raw dealer_name with the matching canonical entry from Dealer_Master.
6. IF the highest fuzzy-match score against Dealer_Master is below 90, THEN THE Validation_Module SHALL retain the raw dealer_name and SHALL set the dealer_name confidence component to reflect the reduced match strength.

### Requirement 5: Model Name Extraction

**User Story:** As a credit officer, I want the tractor model identified exactly so that asset records and loan-to-value computations stay consistent.

#### Acceptance Criteria

1. THE Tier_1_Extractor SHALL search OCR output for model-name anchors including "Model", "Tractor Model", and known brand keywords mined into Asset_Master (for example "Mahindra", "Sonalika", "John Deere", "Massey Ferguson", "Swaraj").
2. WHEN a model-name candidate is found by Tier_1_Extractor with confidence ≥ 0.7, THE Extraction_Module SHALL emit that candidate as the raw model_name value.
3. IF Tier_1_Extractor confidence is below 0.7 or no candidate is found, THEN THE Tier_2_Extractor SHALL be invoked and SHALL return a model_name string.
4. THE Validation_Module SHALL exact-match the raw model_name against Asset_Master using case-insensitive comparison and whitespace normalization.
5. WHEN an exact match against Asset_Master is found, THE Validation_Module SHALL replace the raw model_name with the canonical entry from Asset_Master.
6. IF no exact match against Asset_Master is found, THEN THE Validation_Module SHALL retain the raw model_name and SHALL set the model_name confidence component to reflect the missing master match.

### Requirement 6: Horse Power Extraction

**User Story:** As a credit officer, I want the horse power captured as an integer so that it can be programmatically compared against asset specifications.

#### Acceptance Criteria

1. THE Tier_1_Extractor SHALL match horse-power patterns against OCR text including the regex `\d+\s*(HP|H\.P\.|hp|बल|एचपी)` and Gujarati equivalents.
2. THE Tier_1_Extractor SHALL emit horse_power as an integer in the inclusive range [15, 150].
3. WHEN a Tier_1_Extractor horse-power candidate falls within [15, 150], THE Extraction_Module SHALL emit that candidate as the horse_power value.
4. IF no Tier_1_Extractor candidate falls within [15, 150], THEN THE Tier_2_Extractor SHALL be invoked and SHALL return an integer horse_power value.
5. THE Validation_Module SHALL reject horse_power values outside the inclusive range [15, 150] and SHALL set horse_power to null with a corresponding confidence of 0.0.

### Requirement 7: Asset Cost Extraction

**User Story:** As a credit officer, I want the asset cost as a clean integer (no currency symbols, no commas) so that it feeds directly into the loan computation engine.

#### Acceptance Criteria

1. THE Tier_1_Extractor SHALL search for asset-cost candidates anchored to "Total", "Grand Total", "Net Amount", "₹", "Rs.", "INR", and equivalent Hindi and Gujarati anchors.
2. THE Tier_1_Extractor SHALL strip currency symbols, commas, and trailing decimal-zero suffixes from candidate values and SHALL emit asset_cost as an integer.
3. WHEN a Tier_1_Extractor asset-cost candidate falls within the inclusive range [100000, 5000000], THE Extraction_Module SHALL emit that candidate as the asset_cost value.
4. IF no Tier_1_Extractor candidate falls within [100000, 5000000], THEN THE Tier_2_Extractor SHALL be invoked and SHALL return an integer asset_cost value.
5. THE Validation_Module SHALL reject asset_cost values outside [100000, 5000000] and SHALL set asset_cost to null with a corresponding confidence of 0.0.
6. THE Validation_Module SHALL run a cross-field consistency check that flags asset_cost confidence as reduced when horse_power and asset_cost fall outside an empirically calibrated correlation band.

### Requirement 8: Signature Field Output

**User Story:** As an evaluator, I want a `signature` object that always contains a `present` boolean and, when present, a bounding box satisfying IoU ≥ 0.5, so that the field is unambiguous.

#### Acceptance Criteria

1. WHEN Vision_Module returns at least one `signature` detection above the configured detection threshold, THE Output_Module SHALL emit `signature.present = true` and SHALL set `signature.bbox` to the highest-confidence `signature` detection bounding box.
2. WHEN Vision_Module returns no `signature` detection above threshold, THE Output_Module SHALL emit `signature.present = false` and `signature.bbox = null`.
3. THE Output_Module SHALL emit `signature.bbox` as a four-element integer array `[x1, y1, x2, y2]` in pixel coordinates of the page selected by Ingestion_Module.

### Requirement 9: Stamp Field Output

**User Story:** As an evaluator, I want a `stamp` object that always contains a `present` boolean and, when present, a bounding box satisfying IoU ≥ 0.5.

#### Acceptance Criteria

1. WHEN Vision_Module returns at least one `stamp` detection above the configured detection threshold, THE Output_Module SHALL emit `stamp.present = true` and SHALL set `stamp.bbox` to the highest-confidence `stamp` detection bounding box.
2. WHEN Vision_Module returns no `stamp` detection above threshold, THE Output_Module SHALL emit `stamp.present = false` and `stamp.bbox = null`.
3. THE Output_Module SHALL emit `stamp.bbox` as a four-element integer array `[x1, y1, x2, y2]` in pixel coordinates of the page selected by Ingestion_Module.

### Requirement 10: Tier-2 SLM Fallback

**User Story:** As a developer, I want a local SLM that fills in fields the deterministic rules miss, without taking the cost or latency hit of a vision model or cloud API.

#### Acceptance Criteria

1. THE Tier_2_Extractor SHALL load Qwen2.5-1.5B-Instruct in Q4_K_M GGUF format from `models/qwen2.5-1.5b-q4_k_m.gguf` using `llama-cpp-python`.
2. THE Tier_2_Extractor SHALL receive only OCR text (no images) as model input.
3. WHEN CUDA is available on Eval_Hardware, THE Tier_2_Extractor SHALL initialize `llama_cpp.Llama` with `n_gpu_layers=-1`.
4. IF CUDA is not available on Eval_Hardware, THEN THE Tier_2_Extractor SHALL initialize `llama_cpp.Llama` with `n_gpu_layers=0`.
5. THE Tier_2_Extractor SHALL be invoked only for fields whose Tier_1_Extractor confidence is below the per-field threshold defined in Requirements 4, 5, 6, and 7.
6. THE Tier_2_Extractor SHALL constrain SLM output to a JSON schema covering exactly the four text fields and SHALL reject any output that does not parse as valid JSON.
7. WHEN Tier_2_Extractor returns a string field, THE Validation_Module SHALL reject the SLM output unless it is a substring (after whitespace and case normalization) of the OCR text.
8. THE Tier_2_Extractor SHALL make no network calls during initialization or inference.

### Requirement 11: Field-Level Match Rules

**User Story:** As an evaluator, I want each of the six fields scored against the published match rules, so that DLA can be computed deterministically.

#### Acceptance Criteria

1. THE Validation_Module SHALL score `dealer_name` as correct WHEN the RapidFuzz token-set ratio against the ground-truth dealer name is ≥ 90.
2. THE Validation_Module SHALL score `model_name` as correct WHEN the case-insensitive whitespace-normalized string equals the ground-truth model name.
3. THE Validation_Module SHALL score `horse_power` as correct WHEN the absolute relative difference from the ground truth is ≤ 0.05.
4. THE Validation_Module SHALL score `asset_cost` as correct WHEN the absolute relative difference from the ground truth is ≤ 0.05.
5. THE Validation_Module SHALL score `signature` as correct WHEN `signature.present` matches the ground truth and, when both ground truth and prediction are present, the bounding-box IoU is ≥ 0.5.
6. THE Validation_Module SHALL score `stamp` as correct WHEN `stamp.present` matches the ground truth and, when both ground truth and prediction are present, the bounding-box IoU is ≥ 0.5.
7. THE Validation_Module SHALL compute DLA as the fraction of evaluation documents for which all six fields are scored correct.

### Requirement 12: Dealer Master Mining

**User Story:** As a developer, I want the dealer master built from the dataset itself, so that we are not blocked on an external master file.

#### Acceptance Criteria

1. THE Pipeline SHALL provide an offline script that mines candidate dealer names from the OCR output of all images in `train_data_idfc/train/`.
2. THE mining script SHALL cluster candidate strings using RapidFuzz similarity ≥ 85 to merge OCR-noise duplicates.
3. THE mining script SHALL emit `data/dealer_master.json` containing a list of canonical dealer-name records, each with fields `canonical`, `aliases`, and `frequency`.
4. THE mining script SHALL run end-to-end without network access.

### Requirement 13: Asset Master Mining

**User Story:** As a developer, I want the asset master built from the dataset, augmented with public open-source tractor model lists already cached locally, so that exact-match comparison has reasonable coverage.

#### Acceptance Criteria

1. THE Pipeline SHALL provide an offline script that mines candidate `(brand, model)` pairs from OCR output of all images in `train_data_idfc/train/`.
2. THE mining script SHALL emit `data/asset_master.json` containing a list of `(brand, model, full_name)` records.
3. THE mining script SHALL run end-to-end without network access.
4. WHERE a curated seed list of tractor brand and model names is bundled in the repository, THE mining script SHALL union that seed list with the mined candidates before writing `data/asset_master.json`.

### Requirement 14: Confidence Scoring

**User Story:** As an operator, I want a per-field confidence and a document-level confidence so that low-quality extractions can be queued for manual review.

#### Acceptance Criteria

1. THE Confidence_Engine SHALL compute a per-field confidence in the range [0.0, 1.0] for each of the six target fields.
2. THE Confidence_Engine SHALL compute the per-field confidence as a function of OCR token confidence, Tier_1_Extractor rule-match strength, fuzzy-match score against masters where applicable, and agreement between Tier_1_Extractor and Tier_2_Extractor when both are invoked.
3. THE Confidence_Engine SHALL compute a document-level confidence as a weighted aggregate of the six per-field confidences.
4. THE Output_Module SHALL include the document-level confidence in the output JSON under the key `confidence`.
5. THE Output_Module SHALL include each per-field confidence in the output JSON under `fields.<field>.confidence` for the four text fields and under `signature.confidence` and `stamp.confidence` for the visual fields.

### Requirement 15: Output JSON Schema

**User Story:** As an evaluator, I want a single JSON file per document conforming to the published schema, so that scoring scripts can consume it without per-team adapters.

#### Acceptance Criteria

1. THE Output_Module SHALL produce a Pydantic-validated JSON object containing the keys `doc_id`, `fields`, `confidence`, `processing_time_sec`, and `cost_estimate_usd`.
2. THE Output_Module SHALL populate `fields.dealer_name` as a string, `fields.model_name` as a string, `fields.horse_power` as an integer or null, `fields.asset_cost` as an integer or null, `fields.signature` as an object `{present: bool, bbox: [int, int, int, int] | null}`, and `fields.stamp` as an object `{present: bool, bbox: [int, int, int, int] | null}`.
3. THE Output_Module SHALL populate `processing_time_sec` with the wall-clock seconds elapsed from input load to JSON emission, rounded to two decimal places.
4. THE Output_Module SHALL populate `cost_estimate_usd` with the wall-clock seconds multiplied by a documented commodity-CPU rate, with a value of `0.0` representing the actual marginal cost.
5. IF Pydantic validation of the output object fails, THEN THE Output_Module SHALL return a structured error JSON containing the validation error message and the partial fields that were extracted.
6. THE Output_Module SHALL write the result to `sample_output/result.json` when invoked via the CLI on a single input document.

### Requirement 16: Offline Operation

**User Story:** As an evaluator, I want to run the Pipeline in a network-isolated container and have it produce identical results, so that we can verify the no-cloud claim.

#### Acceptance Criteria

1. THE Pipeline SHALL set environment variables `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before any model is loaded.
2. THE Pipeline SHALL load every model weight, tokenizer, and language asset from a path under `models/` or `data/` inside the Submission_Package.
3. THE Pipeline SHALL make no outbound network calls during ingestion, OCR, vision, extraction, validation, or output.
4. WHEN the Pipeline is executed inside a container started with `--network=none`, THE Pipeline SHALL produce a result JSON for every supported input file without raising network-related errors.
5. THE Pipeline SHALL provide an `--offline` CLI flag that, when set, monkey-patches `urllib` and `requests` at process start to refuse outbound calls and SHALL fail loudly if any code path attempts a network call.

### Requirement 17: Hardware Auto-Detection

**User Story:** As a developer, I want the same submission to run on CPU-only or low-tier GPU evaluator hardware without rebuilds, so that we ship one artifact and let it adapt.

#### Acceptance Criteria

1. THE Pipeline SHALL probe CUDA availability at startup using `torch.cuda.is_available()`.
2. WHEN CUDA is available, THE Pipeline SHALL configure OCR_Module, Vision_Module, and Tier_2_Extractor to use the GPU as specified in Requirements 2, 3, and 10.
3. IF CUDA is not available, THEN THE Pipeline SHALL configure OCR_Module, Vision_Module, and Tier_2_Extractor to use CPU using the same bundled weight files.
4. THE Pipeline SHALL log the detected device and the configuration choice to stderr at startup.

### Requirement 18: Performance Targets

**User Story:** As an evaluator, I want the Pipeline to meet the published latency and accuracy bars on CPU-only hardware, so that we can grade it on the floor configuration.

#### Acceptance Criteria

1. THE Pipeline SHALL achieve a Document-Level Accuracy of ≥ 95% on the internal validation set of at least 30 hand-labeled documents stratified by language and document quality.
2. THE Pipeline SHALL achieve an average wall-clock processing time of ≤ 30 seconds per document on a CPU-only configuration matching the PS-stated "low-tier" specification.
3. THE Pipeline SHALL report `cost_estimate_usd` ≤ $0.01 per document.
4. THE Pipeline SHALL produce a per-document p95 wall-clock processing time of ≤ 30 seconds on the internal validation set.

### Requirement 19: YOLOv8n Fine-Tuning

**User Story:** As a developer, I want a reproducible YOLOv8n training pipeline that runs on the dev RTX 3050 6GB in under 30 minutes offline, so that the signature/stamp detector can be retrained from the bundled annotations.

#### Acceptance Criteria

1. THE Pipeline SHALL include a training script that fine-tunes YOLOv8n from COCO-pretrained weights for exactly two classes: `signature` and `stamp`.
2. THE training script SHALL consume YOLO-format annotations stored under `train_data_idfc/labels/` produced from the manual annotation workflow described in Requirement 20.
3. THE training script SHALL apply data augmentation including rotation ±5°, brightness jitter, gaussian noise, and JPEG compression artifacts.
4. THE training script SHALL run for at least 50 and at most 100 epochs at image size 640.
5. THE training script SHALL emit final weights to `models/yolov8n_sig_stamp.pt`.
6. THE training script SHALL run end-to-end without network access on the RTX 3050 6GB dev machine in ≤ 30 minutes.

### Requirement 20: Annotation Workflow

**User Story:** As a developer, I want a documented annotation workflow that produces enough labeled data for both YOLO training and the validation set, given that the dataset ships with no ground truth.

#### Acceptance Criteria

1. THE Pipeline SHALL document a manual annotation workflow that produces between 50 and 80 hand-labeled images for the `signature` and `stamp` classes using LabelImg in offline desktop mode.
2. THE Pipeline SHALL document a manual annotation workflow that produces between 30 and 50 hand-labeled documents for the four text fields (`dealer_name`, `model_name`, `horse_power`, `asset_cost`) used as the validation set referenced in Requirement 18.
3. THE annotation workflow documentation SHALL specify a stratified split across English, Hindi, Gujarati, and mixed-language documents, and across digital, scanned, and mobile-camera document qualities.
4. THE Pipeline SHALL provide a pseudo-labeling script that runs Tier_1_Extractor and Tier_2_Extractor on the unlabeled images and SHALL emit confidence-weighted pseudo-labels for unsupervised augmentation of the training distribution.

### Requirement 21: Submission Package Structure

**User Story:** As an evaluator, I want a single `submission.zip` with a fixed layout and an obvious entry point, so that I can run the pipeline with one command.

#### Acceptance Criteria

1. THE Submission_Package SHALL be a single archive named `submission.zip`.
2. THE Submission_Package SHALL contain `executable.py` at the archive root as the single CLI entry point.
3. THE Submission_Package SHALL contain `requirements.txt` at the archive root with pinned versions of every Python dependency, all installable from PyPI.
4. THE Submission_Package SHALL contain `README.md` at the archive root with sections for architecture, pipeline stages, cost analysis, error analysis, and run instructions.
5. THE Submission_Package SHALL contain a `utils/` Python package with at least the modules `ingestion`, `ocr`, `detection`, `extraction`, `slm`, `normalization`, `masters`, `confidence`, and `schema`.
6. THE Submission_Package SHALL contain `models/paddleocr/`, `models/yolov8n_sig_stamp.pt`, and `models/qwen2.5-1.5b-q4_k_m.gguf`.
7. THE Submission_Package SHALL contain `data/dealer_master.json` and `data/asset_master.json`.
8. THE Submission_Package SHALL contain `sample_output/result.json`.
9. THE Submission_Package SHALL have a total uncompressed size of ≤ 2 GB and a hard cap of ≤ 5 GB.
10. WHEN the evaluator runs `python executable.py <input_path>` after `pip install -r requirements.txt` on a clean machine with no internet access, THE Submission_Package SHALL produce a valid result JSON without raising network or missing-asset errors.

### Requirement 22: CLI Entry Point

**User Story:** As an evaluator, I want a single, predictable CLI signature so that batch grading scripts can invoke the Pipeline uniformly.

#### Acceptance Criteria

1. THE `executable.py` SHALL accept a single positional argument that is the path to a PDF, PNG, JPG, or JPEG file.
2. WHEN invoked with a single positional argument, THE `executable.py` SHALL emit the result JSON to stdout and SHALL also write it to `sample_output/result.json`.
3. THE `executable.py` SHALL accept an optional `--output <path>` flag that overrides the output JSON path.
4. THE `executable.py` SHALL accept an optional `--offline` flag that triggers the network-blocking behavior described in Requirement 16.
5. IF the input path does not exist or has an unsupported extension, THEN THE `executable.py` SHALL exit with a non-zero status code and SHALL print a structured error JSON to stderr.

### Requirement 23: FastAPI Demo Bridge (Optional)

**User Story:** As a presenter, I want the existing React InvoiceFlow Frontend in `src/` to call the real Pipeline over localhost during the live demo, so that judges see the production UI driving the actual model.

#### Acceptance Criteria

1. THE Demo_Bridge SHALL live entirely under a top-level `demo/` folder and SHALL NOT be included in the Submission_Package.
2. THE Demo_Bridge SHALL expose a FastAPI server bound to `127.0.0.1` on a configurable port that defaults to 8000.
3. THE Demo_Bridge SHALL expose a `POST /api/extract` endpoint that accepts a multipart file upload and returns the same JSON schema defined in Requirement 15.
4. THE Demo_Bridge SHALL expose a `GET /api/health` endpoint that returns `{"status": "ok", "device": "<cpu|cuda>"}`.
5. THE Demo_Bridge SHALL make no outbound network calls.
6. THE Demo_Bridge SHALL include a README under `demo/README.md` documenting how to start the FastAPI server and how to point the React Frontend dev server at it.
7. THE Frontend SHALL be modified to call the Demo_Bridge endpoints in place of its current simulated processing path while preserving the existing Zustand store schema (`dealerName`, `modelName`, `horsePower`, `assetCost`, `signatureDetected`, `stampDetected`).
8. WHERE the Demo_Bridge is unreachable, THE Frontend SHALL display a non-blocking error toast and SHALL retain the simulated-processing fallback so that the UI remains demoable without the backend running.

### Requirement 24: EDA Notebook (Bonus)

**User Story:** As an evaluator awarding bonus points, I want a Jupyter notebook that characterizes the dataset, so that I can see the team understood the data.

#### Acceptance Criteria

1. THE Pipeline SHALL include a Jupyter notebook at `notebooks/eda.ipynb`.
2. THE EDA notebook SHALL include a state-distribution analysis derived from address tokens in OCR output.
3. THE EDA notebook SHALL include a language-distribution analysis derived from PaddleOCR script detection.
4. THE EDA notebook SHALL include a digital-versus-scanned split analysis derived from PDF text-layer presence and image entropy heuristics.
5. THE EDA notebook SHALL include layout clustering of pages using image-feature embeddings.
6. THE EDA notebook SHALL include a processing-time analysis showing per-stage latency on a sample of documents.
7. THE EDA notebook SHALL run end-to-end without network access.

### Requirement 25: Error Analysis Report (Bonus)

**User Story:** As an evaluator awarding bonus points, I want an error analysis report showing where the Pipeline fails.

#### Acceptance Criteria

1. THE Pipeline SHALL include an error analysis report at `docs/error_analysis.md` or as a notebook at `notebooks/error_analysis.ipynb`.
2. THE error analysis report SHALL include a per-field confusion matrix on the internal validation set.
3. THE error analysis report SHALL include a failure-category breakdown classified by at least the categories OCR error, rule miss, SLM hallucination, master miss, and detection miss.
4. THE error analysis report SHALL include at least three concrete examples per failure category with the input page, OCR output, and predicted versus ground-truth values.

### Requirement 26: Architecture Diagram (Bonus)

**User Story:** As an evaluator, I want a single picture that shows the six pipeline stages, so that the README is scannable.

#### Acceptance Criteria

1. THE Pipeline SHALL include a Mermaid architecture diagram in the README.
2. THE Pipeline SHALL include a PNG export of the architecture diagram at `docs/architecture.png`.
3. THE architecture diagram SHALL depict all six pipeline stages, the two parallel branches at Stage 2 (text and vision), and the Tier_1 / Tier_2 extraction split.
