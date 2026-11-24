"""
InvoiceFlow YOLOv8 Training Script (Phase 4 / Tasks 4.2 - 4.8)
Trains YOLOv8 nano detector for rubber stamps and authorized signatures on NVIDIA RTX 3050 GPU.
Exports final checkpoint and ONNX weights to core/vision/weights/.
"""
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = PROJECT_ROOT / "datasets" / "yolo_data" / "dataset.yaml"
WEIGHTS_DIR = PROJECT_ROOT / "core" / "vision" / "weights"

def train():
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Starting YOLOv8 training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})...")
    
    # Initialize lightweight nano model pre-trained on COCO
    model = YOLO("yolov8n.pt")

    # Train model
    results = model.train(
        data=str(YAML_PATH),
        epochs=15,
        imgsz=640,
        batch=8,
        device=device,
        workers=2,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        patience=10,
        save=True,
        project=str(PROJECT_ROOT / "runs" / "train"),
        name="stamp_sig_exp",
        exist_ok=True,
        verbose=True
    )

    # Copy best weights to core/vision/weights
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    best_weights = PROJECT_ROOT / "runs" / "train" / "stamp_sig_exp" / "weights" / "best.pt"
    target_weights = WEIGHTS_DIR / "yolo_stamp_sig.pt"
    
    if best_weights.is_file():
        shutil.copy2(best_weights, target_weights)
        print(f"Copied best weights to: {target_weights}")

        # Task 4.8: Export ONNX runtime model
        print("Exporting model to ONNX format...")
        try:
            trained_model = YOLO(str(target_weights))
            onnx_path = trained_model.export(format="onnx", imgsz=640, dynamic=True)
            print(f"ONNX export successful: {onnx_path}")
        except Exception as e:
            print(f"ONNX export warning: {e}")
    else:
        print("Warning: best.pt not found.")

if __name__ == "__main__":
    train()
