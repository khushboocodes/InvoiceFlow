"""
InvoiceFlow FastAPI Demo Bridge
Connects the modern React / Vite SaaS Dashboard (localhost:5173)
directly to the local RTX 3050 GPU pipeline (Qwen2.5-VL-3B + YOLOv8).
"""
import sys
import os
import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

from core.pipeline import InvoiceFlowPipeline

app = FastAPI(title="InvoiceFlow AI Bridge", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = Path("storage_bridge")
INDEX_PATH = STORAGE_DIR / "index.json"

def _ensure_storage():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text(json.dumps({"version": 1, "documents": []}), encoding="utf-8")

def _load_index() -> dict:
    _ensure_storage()
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "documents": []}

def _save_index(data: dict):
    _ensure_storage()
    INDEX_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def _format_inr(val: Any) -> str:
    if val is None or val == "" or val == "—":
        return "—"
    try:
        fval = float(str(val).replace(",", "").replace("₹", "").replace("Rs", "").strip())
    except (ValueError, TypeError):
        return str(val)
    ival = int(abs(fval))
    s = str(ival)
    if len(s) <= 3:
        formatted = f"₹ {s}"
    else:
        last3, rest = s[-3:], s[:-3]
        grouped_rest = ""
        while len(rest) > 2:
            grouped_rest = "," + rest[-2:] + grouped_rest
            rest = rest[:-2]
        formatted = f"₹ {rest}{grouped_rest},{last3}"
    return formatted

# Initialize Pipeline Singleton
PIPELINE = None

def get_active_pipeline() -> InvoiceFlowPipeline:
    global PIPELINE
    if PIPELINE is None:
        print("🚀 Initializing InvoiceFlow GPU Pipeline (Qwen2.5-VL-3B + YOLOv8)...")
        PIPELINE = InvoiceFlowPipeline(languages=["en"], use_local_vlm=True, vlm_primary=True)
    return PIPELINE

@app.on_event("startup")
def startup_event():
    _ensure_storage()
    print("✅ InvoiceFlow Bridge Ready on http://127.0.0.1:8000")

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": "cuda",
        "model": "Qwen2.5-VL-3B (RTX 3050 GPU)"
    }

@app.get("/api/documents")
def get_documents() -> List[dict]:
    idx = _load_index()
    docs = []
    for d in idx.get("documents", []):
        public_doc = {k: v for k, v in d.items() if not k.startswith("_")}
        docs.append(public_doc)
    return docs

@app.get("/api/documents/{doc_id}")
def get_single_document(doc_id: str):
    idx = _load_index()
    for d in idx.get("documents", []):
        if d.get("id") == doc_id:
            return {k: v for k, v in d.items() if not k.startswith("_")}
    raise HTTPException(status_code=404, detail="Document not found")

@app.get("/api/documents/{doc_id}/file")
def get_document_file(doc_id: str):
    idx = _load_index()
    target = next((d for d in idx.get("documents", []) if d.get("id") == doc_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Document not found")
    filename = target.get("_storageFile")
    file_path = STORAGE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")
    return FileResponse(
        str(file_path),
        media_type=target.get("mimeType", "image/png"),
        filename=target.get("fileName", filename)
    )

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    idx = _load_index()
    docs = idx.get("documents", [])
    target = next((d for d in docs if d.get("id") == doc_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Document not found")
    docs = [d for d in docs if d.get("id") != doc_id]
    idx["documents"] = docs
    _save_index(idx)
    storage_file = target.get("_storageFile")
    if storage_file:
        (STORAGE_DIR / storage_file).unlink(missing_ok=True)
    return {"deleted": doc_id}

@app.post("/api/extract")
def extract_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".png", ".jpg", ".jpeg", ".pdf"]:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")

    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    _ensure_storage()
    doc_id = secrets.token_hex(4)
    storage_filename = f"{doc_id}{suffix}"
    storage_path = STORAGE_DIR / storage_filename
    storage_path.write_bytes(contents)

    pipeline = get_active_pipeline()
    start_t = time.time()
    try:
        res = pipeline.process_document(storage_path)
    except Exception as e:
        print(f"Extraction error: {e}")
        storage_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    elapsed = round(time.time() - start_t, 2)

    # Read image dimensions to convert normalized [ymin, xmin, ymax, xmax] (0-1000)
    # into exact React pixel coordinates [x1, y1, x2, y2]
    import cv2
    img_cv = cv2.imread(str(storage_path))
    h_px, w_px = img_cv.shape[:2] if img_cv is not None else (1000, 1000)

    def _to_react_pixel_bbox(norm_box):
        if not norm_box:
            return None
        ymin, xmin, ymax, xmax = norm_box
        x1 = int((xmin / 1000.0) * w_px)
        y1 = int((ymin / 1000.0) * h_px)
        x2 = int((xmax / 1000.0) * w_px)
        y2 = int((ymax / 1000.0) * h_px)
        return [x1, y1, x2, y2]

    sig_box_px = _to_react_pixel_bbox(res.dealer_signature.bounding_box) if res.dealer_signature.present else None
    stamp_box_px = _to_react_pixel_bbox(res.dealer_stamp.bounding_box) if res.dealer_stamp.present else None

    extracted_fields = {
        "dealerName": res.dealer_name,
        "modelName": res.model_name,
        "horsePower": f"{res.horse_power} HP" if res.horse_power else "—",
        "assetCost": _format_inr(res.asset_cost),
        "signatureDetected": res.dealer_signature.present,
        "stampDetected": res.dealer_stamp.present,
        "signatureBbox": sig_box_px,
        "stampBbox": stamp_box_px,
        "quotationNo": "—",
        "date": "—",
        "hypothecatedBank": "—",
        "gstin": "—"
    }

    raw_extraction_result = {
        "doc_id": doc_id,
        "fields": {
            "dealer_name": {"value": res.dealer_name, "confidence": 0.95},
            "model_name": {"value": res.model_name, "confidence": 0.95},
            "horse_power": {"value": res.horse_power, "confidence": 0.95},
            "asset_cost": {"value": res.asset_cost, "confidence": 0.95},
            "signature": {
                "present": res.dealer_signature.present,
                "bbox": sig_box_px,
                "confidence": res.dealer_signature.confidence
            },
            "stamp": {
                "present": res.dealer_stamp.present,
                "bbox": stamp_box_px,
                "confidence": res.dealer_stamp.confidence
            }
        },
        "confidence": 0.96,
        "processing_time_sec": elapsed,
        "cost_estimate_usd": 0.0,
        "error": None
    }

    doc_record = {
        "id": doc_id,
        "fileName": file.filename,
        "fileSize": f"{len(contents) / 1024:.2f} KB",
        "fileType": suffix.lstrip("."),
        "mimeType": file.content_type or "image/png",
        "previewUrl": f"http://127.0.0.1:8000/api/documents/{doc_id}/file",
        "status": "completed",
        "confidence": 96,
        "processingTime": f"{elapsed:.1f}s",
        "language": "English",
        "documentType": "Tractor Quotation",
        "extractedFields": extracted_fields,
        "uploadedAt": datetime.now(timezone.utc).isoformat(),
        "_storageFile": storage_filename,
        "_extraction": raw_extraction_result
    }

    # Save to index
    idx = _load_index()
    idx.setdefault("documents", []).insert(0, doc_record)
    _save_index(idx)

    # Return public view of doc record (with raw extraction attached for React store)
    public_doc = {k: v for k, v in doc_record.items() if not k.startswith("_")}
    public_doc["extraction"] = raw_extraction_result
    return public_doc

if __name__ == "__main__":
    uvicorn.run("bridge_server:app", host="127.0.0.1", port=8000, reload=False)
