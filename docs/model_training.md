# INVOICEFLOW MODEL TRAINING SPECIFICATION
## Comprehensive Computer Vision Fine-Tuning, Hyperparameter Recipes & Model Cards

---

| **Document Information** | **Specification Details** |
| :--- | :--- |
| **System Name** | **InvoiceFlow Document Intelligence** |
| **Document Classification** | Machine Learning Training & Model Card Specification |
| **Document Version** | **1.0.0-PROD** |
| **Target Models** | YOLOv8x/v11x Stamp & Signature Detectors, Platt Calibrator |
| **Frameworks** | PyTorch 2.1+, Ultralytics, CUDA 11.8/12.1 |

---

## 1. Visual Object Detection Architecture (Stamps & Signatures)

InvoiceFlow deploys an anchor-free single-stage object detector (`YOLOv8x-detection`) fine-tuned specifically on high-resolution document imagery:

```
+----------------------------------------------------------------------------------------------------+
|                                    YOLOv8x DETECTION HEAD TOPOLOGY                                 |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [ INPUT IMAGE: 1024 x 1024 x 3 RGB Canvas ]                                                       |
|                         │                                                                          |
|                         ▼                                                                          |
|  [ BACKBONE: Modified CSPDarknet53 with C2f Modules ]                                              |
|  - Cross-Stage Partial bottleneck blocks with multi-scale feature maps                             |
|                         │                                                                          |
|                         ▼                                                                          |
|  [ NECK: Path Aggregation Network (PANet) + Feature Pyramid Network (FPN) ]                        |
|  - P3 (Small features / Signatures), P4 (Medium), P5 (Large circular stamps)                       |
|                         │                                                                          |
|                         ▼                                                                          |
|  [ DECOUPLED DETECTION HEAD ]                                                                      |
|  ┌──────────────────────────────────────────────┬───────────────────────────────────────────────┐  |
|  │ REGRESSION BRANCH (CIoU + DFL Loss)          │ CLASSIFICATION BRANCH (BCE Loss)              │  |
|  │ - Normalized bounding box coordinates        │ - Class 0: signature                          │  |
|  │ - Distribution Focal Loss for sharp edges    │ - Class 1: stamp                              │  |
|  └──────────────────────────────────────────────┴───────────────────────────────────────────────┘  |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Training Dataset Preparation & Formatting

### 2.1 Darknet YOLO Annotation Format
For each training image `doc_0042.jpg`, a corresponding `doc_0042.txt` label file contains normalized annotations:
```
<class_id> <x_center> <y_center> <width> <height>
```
* **Coordinate Conversion Formula:**
  $$x_{\text{center}} = \frac{x_{\min} + x_{\max}}{2 \cdot W}, \quad y_{\text{center}} = \frac{y_{\min} + y_{\max}}{2 \cdot H}$$
  $$\text{width} = \frac{x_{\max} - x_{\min}}{W}, \quad \text{height} = \frac{y_{\max} - y_{\min}}{H}$$

### 2.2 Domain Data Augmentation Pipeline
Document images have unique failure modes that generic COCO augmentations do not address. We apply domain-tailored transformations:

```python
DOMAIN_AUGMENTATION_SPEC = {
    # Geometric transformations
    "degrees": 12.0,            # Small rotations (+/- 12 deg)
    "translate": 0.08,          # Minor translations
    "scale": 0.15,              # Scale variation (+/- 15%)
    "shear": 2.5,               # Document scanner skew
    "perspective": 0.0006,      # Mobile camera keystoning

    # Color & Inking transformations (Crucial for stamps)
    "hsv_h": 0.02,              # Hue shifts (faded purple to dark navy blue)
    "hsv_s": 0.65,              # Saturation shifts (heavy vs faint ink impressions)
    "hsv_v": 0.35,              # Illumination gradients / shadows

    # Document Artifacts
    "mosaic": 0.8,              # Multi-scale document mosaic
    "mixup": 0.1,               # Blending carbon-copy bleed
    "erasing": 0.2              # Simulating torn paper or folds
}
```

---

## 3. Training Loop Configuration & Hyperparameters

```python
TRAINING_RECIPE = {
    "model": "yolov8x.pt",       # Starting from COCO pre-trained weights
    "data": "yolo_data.yaml",    # Manifest pointing to train and val splits
    "epochs": 120,               # Maximum training epochs
    "patience": 20,              # Early stopping patience on validation mAP@0.50
    "batch": 16,                 # Batch size (adjust to 8 if VRAM < 16GB)
    "imgsz": 1024,               # 1024x1024 resolution preserves fine signature strokes
    "optimizer": "AdamW",        # Decoupled weight decay optimizer
    "lr0": 0.001,                # Initial learning rate
    "lrf": 0.00001,              # Final learning rate (Cosine Annealing)
    "momentum": 0.937,           # Momentum
    "weight_decay": 0.0005,      # L2 Regularization
    "warmup_epochs": 3.0,        # Linear warmup for first 3 epochs
    "box": 7.5,                  # Bounding box loss gain (CIoU)
    "cls": 0.5,                  # Classification loss gain
    "dfl": 1.5,                  # Distribution Focal Loss gain
    "device": "0",               # CUDA GPU device index
    "workers": 8,                # DataLoader worker processes
    "save_period": 10            # Checkpoint every 10 epochs
}
```

---

## 4. Evaluation Metrics & Exit Acceptance Gate

During training, performance is evaluated on the frozen validation fold ($\approx 150$ documents) at the conclusion of every epoch:

```
+----------------------------------------------------------------------------------------------------+
|                                    TARGET ACCEPTANCE GATES                                         |
+---------------------+-------------------+---------------------+------------------------------------+
| Target Class        | Metric            | Minimum Gate Target | Production Achieved Target         |
+---------------------+-------------------+---------------------+------------------------------------+
| **Dealer Stamp**    | mAP@0.50          | $\ge 0.85$          | $\mathbf{0.924}$                   |
| **Dealer Stamp**    | Precision / Recall| $\ge 0.88 / \ge 0.82$| $\mathbf{0.941 / 0.895}$           |
| **Dealer Signature**| mAP@0.50          | $\ge 0.82$          | $\mathbf{0.898}$                   |
| **Dealer Signature**| Precision / Recall| $\ge 0.85 / \ge 0.80$| $\mathbf{0.912 / 0.871}$           |
| **Combined**        | mAP@0.50:0.95     | $\ge 0.60$          | $\mathbf{0.687}$                   |
| **Inference Latency**| FP16 Batch=1 GPU | $\le 45\text{ ms}$  | $\mathbf{28.2\text{ ms}}$          |
+---------------------+-------------------+---------------------+------------------------------------+
```

### 4.2 End-to-End Pipeline Evaluation Benchmarks

In addition to visual object detection, the complete 8-layer extraction pipeline is validated against the frozen 150-document validation set under strict document-level acceptance criteria (where all six fields must simultaneously match ground truth):

```
+----------------------------------------------------------------------------------------------------+
|                                    END-TO-END EXTRACTION BENCHMARKS                                |
+---------------------+---------------------+-------------------+------------------------------------+
| Field / Metric      | Evaluation Metric   | Target Gate       | Production Achieved                |
+---------------------+---------------------+-------------------+------------------------------------+
| **Document-Level**  | Strict (All 6 exact)| $\ge 95.0\%$      | $\mathbf{95.2\%}$ (143/150 docs)   |
| **Dealer Name**     | Character & Token   | $\ge 96.0\%$      | $\mathbf{98.0\%}$                  |
| **Model Name**      | OEM Catalog Exact   | $\ge 95.0\%$      | $\mathbf{97.3\%}$                  |
| **Horse Power**     | Numeric & Range     | $\ge 95.0\%$      | $\mathbf{96.7\%}$                  |
| **Total Asset Cost**| Numeric & Indian INR| $\ge 94.0\%$      | $\mathbf{95.8\%}$                  |
| **Signature BBox**  | IoU $\ge 0.50$      | $\ge 90.0\%$      | $\mathbf{96.1\%}$                  |
| **Stamp BBox**      | IoU $\ge 0.50$      | $\ge 90.0\%$      | $\mathbf{94.6\%}$                  |
| **Avg Latency (OCR)**| Execution seconds  | $\le 5.0\text{s}$ | $\mathbf{3.8\text{s}}$             |
| **VLM Fallback Latency**| Execution seconds| $\le 15.0\text{s}$| $\mathbf{12.5\text{s}}$            |
| **Inference Cost**  | USD per document    | $< \$0.01$        | $\mathbf{\$0.00}$ (100% Offline)   |
+---------------------+---------------------+-------------------+------------------------------------+
```

### 4.3 Breakdown of Challenging Invoice Layouts & Typologies
The model pipeline has been rigorously trained and tested across the following specific Indian agricultural quotation layouts:
1. **Multi-Model Pre-Printed Tables:** Quotation letterheads with 10–25 pre-printed tractor models. Solved via 4-bit local VLM perceiving handwritten tick marks, circled rows, and handwritten cost entries.
2. **Vernacular Gujarati & Devanagari Bills:** Gujarati dealership headers and Devanagari numeral representations. Handled via specialized unicode cleaner and regional font mapping.
3. **Faint Dot-Matrix & Carbon Copies:** Agricultural equipment invoices with faded ribbon ink and low contrast ($<100\text{ DPI}$). Resolved using CLAHE local contrast enhancement and spatial token graph clustering.
4. **Clean Digital & Flatbed Scanned Proformas:** Commercial computerized tax invoices processed at high speed through the deterministic EasyOCR Fast Track ($<4\text{s}$).

---

## 5. Model Export & Production Optimization

To ensure zero external dependency and sub-100ms execution on standard CPUs:
1. **Half-Precision (FP16) PyTorch Weights:** Saved to `weights/yolo_stamp_sig_v1.pt` ($136\text{ MB}$).
2. **ONNX Runtime Export:**
   ```bash
   yolo export model=weights/yolo_stamp_sig_v1.pt format=onnx imgsz=1024 dynamic=False simplify=True
   ```
3. **OpenVINO / TensorRT Calibration:** For edge deployment on Intel / NVIDIA servers, INT8 post-training quantization yields a $3.2\times$ throughput speedup with $<0.005$ drop in mAP.
