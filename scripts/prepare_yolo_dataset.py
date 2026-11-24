"""
Prepare YOLOv8 Dataset for Stamp and Signature Detection (Phase 4 / Task 4.1)
Generates normalized Darknet annotations [class_id x_center y_center width height]
and dataset.yaml.
"""
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
YOLO_DIR = PROJECT_ROOT / "datasets" / "yolo_data"
MANIFEST_DIR = PROJECT_ROOT / "datasets" / "manifests"

def extract_proposals(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    proposals = []

    # Focus on footer region (bottom 40%)
    y_start = int(h * 0.60)
    footer = img[y_start:h, 0:w]
    fh, fw = footer.shape[:2]

    # --- 1. Stamp detection (Class 1) via HSV chromatic segmentation ---
    hsv = cv2.cvtColor(footer, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, np.array([100, 40, 40]), np.array([165, 255, 255]))
    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([15, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([165, 50, 40]), np.array([180, 255, 255]))
    stamp_mask = cv2.bitwise_or(mask_blue, cv2.bitwise_or(mask_red1, mask_red2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    stamp_closed = cv2.morphologyEx(stamp_mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(stamp_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if 600 < area < (fh * fw * 0.30):
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = float(cw) / max(1, ch)
            if 0.4 <= aspect <= 2.5:
                # Convert to absolute coords
                abs_x1, abs_y1 = x, y_start + y
                abs_x2, abs_y2 = x + cw, y_start + y + ch
                # YOLO format: class x_c y_c w h in [0, 1]
                xc = (abs_x1 + abs_x2) / (2.0 * w)
                yc = (abs_y1 + abs_y2) / (2.0 * h)
                nw = cw / float(w)
                nh = ch / float(h)
                proposals.append((1, xc, yc, nw, nh))
                break # Take primary prominent stamp

    # --- 2. Signature detection (Class 0) via stroke density ---
    gray_footer = cv2.cvtColor(footer, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_footer, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    sig_cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_sig = None
    best_sig_score = 0
    for cnt in sig_cnts:
        area = cv2.contourArea(cnt)
        if 400 < area < 35000:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw > 40 and ch > 15 and (cw / ch) > 0.8:
                score = area * (1.2 if x > (fw * 0.5) else 0.7)
                if score > best_sig_score:
                    best_sig_score = score
                    abs_x1, abs_y1 = x, y_start + y
                    abs_x2, abs_y2 = x + cw, y_start + y + ch
                    xc = (abs_x1 + abs_x2) / (2.0 * w)
                    yc = (abs_y1 + abs_y2) / (2.0 * h)
                    nw = cw / float(w)
                    nh = ch / float(h)
                    best_sig = (0, xc, yc, nw, nh)

    if best_sig:
        proposals.append(best_sig)

    return proposals

def prepare_split(split_name: str, manifest_path: Path, max_samples: int = 100):
    print(f"Preparing {split_name} split from {manifest_path} (limit {max_samples})...")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = data.get("documents", [])[:max_samples]

    img_out_dir = YOLO_DIR / "images" / split_name
    lbl_out_dir = YOLO_DIR / "labels" / split_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for doc in docs:
        fname = doc["file_name"]
        src_path = RAW_DIR / fname
        if not src_path.is_file():
            continue

        proposals = extract_proposals(src_path)
        if not proposals:
            continue

        # Copy image
        dst_img = img_out_dir / fname
        if not dst_img.is_file():
            shutil.copy2(src_path, dst_img)

        # Write label file
        label_stem = src_path.stem
        dst_lbl = lbl_out_dir / f"{label_stem}.txt"
        with open(dst_lbl, "w", encoding="utf-8") as lf:
            for cls_id, xc, yc, nw, nh in proposals:
                lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
        count += 1

    print(f"Prepared {count} images with labels for {split_name}.")

def main():
    train_manifest = MANIFEST_DIR / "train_manifest.json"
    val_manifest = MANIFEST_DIR / "val_manifest.json"

    prepare_split("train", train_manifest, max_samples=80)
    prepare_split("val", val_manifest, max_samples=20)

    yaml_content = f"""# InvoiceFlow YOLOv8 Dataset Configuration
path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: signature
  1: stamp
"""
    yaml_path = YOLO_DIR / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"Wrote dataset config to: {yaml_path}")

if __name__ == "__main__":
    main()
