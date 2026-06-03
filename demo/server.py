"""FastAPI demo bridge.

Exposes the offline extraction pipeline over localhost so the React /
InvoiceFlow frontend in ``src/`` can call it during the live demo round.
This file is INTENTIONALLY OUTSIDE the ``backend/`` submission package —
it must NOT be included in ``submission.zip``.

Endpoints
---------
* ``GET  /api/health``                 → ``{"status": "ok", "device": "<cpu|cuda>"}``
* ``POST /api/extract``                → multipart file upload, runs the pipeline,
                                         persists the document, and returns
                                         a Document JSON (extraction + metadata).
* ``GET  /api/documents``              → array of every persisted Document.
* ``GET  /api/documents/{id}``         → a single Document by id.
* ``DELETE /api/documents/{id}``       → permanently delete a document.
* ``GET  /api/documents/{id}/file``    → original uploaded file bytes (for preview).

The persistence layer is a flat ``demo/storage/`` directory. Each document's
metadata lives in ``demo/storage/index.json`` and the original file bytes are
stored alongside as ``<id><suffix>``. This is a single-process demo bridge
on localhost — file locking, multi-tenant isolation, and concurrent writers
are intentionally out of scope.

Run::

    cd demo
    python server.py
    # then in another terminal, start the React dev server in the repo root:
    pnpm dev

Validates Requirements: 23.1-23.6
"""

from __future__ import annotations

import json
import logging
import secrets
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Add backend to sys.path so we can import the existing utils + executable.
HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent / "backend"
sys.path.insert(0, str(BACKEND))

from fastapi import FastAPI, File, HTTPException, UploadFile  # type: ignore[import-not-found]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-not-found]
from fastapi.responses import FileResponse  # type: ignore[import-not-found]

from executable import build_pipeline
from utils.ingestion import SUPPORTED_EXTENSIONS

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("demo.server")


app = FastAPI(title="InvoiceFlow Demo Bridge", version="0.2.0")

# Allow the local Vite dev server (and a few other dev-server defaults) to
# call us. ``*`` would be simpler but blocks credentials in some browsers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Persistence layer
# --------------------------------------------------------------------------- #

STORAGE_DIR = HERE / "storage"
INDEX_PATH = STORAGE_DIR / "index.json"
_index_lock = threading.RLock()


def _ensure_storage() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text(json.dumps({"version": 1, "documents": []}), encoding="utf-8")


def _load_index() -> dict:
    _ensure_storage()
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("storage index corrupt (%s); starting fresh", exc)
        return {"version": 1, "documents": []}


def _save_index(idx: dict) -> None:
    _ensure_storage()
    tmp = INDEX_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(INDEX_PATH)


def _new_doc_id() -> str:
    """Short, URL-safe id similar to what the React store generates."""
    return secrets.token_hex(4)


def _format_size(num_bytes: int) -> str:
    return f"{num_bytes / 1024:.2f} KB"


def _bbox_to_list(bbox: Any) -> Optional[list]:
    if bbox is None:
        return None
    try:
        return [int(round(float(c))) for c in bbox]
    except (TypeError, ValueError):
        return None


def _format_inr(amount: Optional[int]) -> str:
    if amount is None:
        return "—"
    s = str(abs(amount))
    if len(s) <= 3:
        return f"₹ {s}"
    last3, rest = s[-3:], s[:-3]
    grouped_rest = ""
    while len(rest) > 2:
        grouped_rest = "," + rest[-2:] + grouped_rest
        rest = rest[:-2]
    return f"₹ {rest}{grouped_rest},{last3}"


def _build_extracted_fields(extraction: dict) -> dict:
    """Project the backend ExtractionResult into the React ``extractedFields``
    shape so the UI can render without a second mapping pass."""
    fields = extraction.get("fields", {})
    dealer = fields.get("dealer_name", {}) or {}
    model = fields.get("model_name", {}) or {}
    hp = fields.get("horse_power", {}) or {}
    cost = fields.get("asset_cost", {}) or {}
    sig = fields.get("signature", {}) or {}
    stamp = fields.get("stamp", {}) or {}
    return {
        "dealerName": dealer.get("value") or "—",
        "modelName": model.get("value") or "—",
        "horsePower": (
            f"{hp.get('value')} HP" if hp.get("value") is not None else "—"
        ),
        "assetCost": _format_inr(cost.get("value")),
        "signatureDetected": bool(sig.get("present", False)),
        "stampDetected": bool(stamp.get("present", False)),
        "signatureBbox": _bbox_to_list(sig.get("bbox")),
        "stampBbox": _bbox_to_list(stamp.get("bbox")),
    }


def _document_record(
    *,
    doc_id: str,
    file_name: str,
    file_size_bytes: int,
    mime_type: str,
    extraction: dict,
    storage_filename: str,
    uploaded_at: str,
) -> dict:
    """Shape that the React store and pages expect."""
    suffix = Path(file_name).suffix.lstrip(".").lower() or "unknown"
    confidence_pct = round(float(extraction.get("confidence") or 0.0) * 100)
    proc_time = float(extraction.get("processing_time_sec") or 0.0)
    return {
        "id": doc_id,
        "fileName": file_name,
        "fileSize": _format_size(file_size_bytes),
        "fileType": suffix,
        "mimeType": mime_type or "application/octet-stream",
        # Stable backend URL — the frontend swaps this in for the data URL.
        "previewUrl": f"http://127.0.0.1:8000/api/documents/{doc_id}/file",
        "status": "completed",
        "confidence": confidence_pct,
        "processingTime": f"{proc_time:.1f}s",
        "language": "English",
        "documentType": "Tractor Quotation",
        "extractedFields": _build_extracted_fields(extraction),
        "uploadedAt": uploaded_at,
        # Server-only metadata — the React store ignores unknown fields.
        "_storageFile": storage_filename,
        "_extraction": extraction,  # full ExtractionResult JSON, for debugging
    }


def _save_document(record: dict) -> None:
    with _index_lock:
        idx = _load_index()
        idx.setdefault("documents", [])
        # Newest first, just like the React store does.
        idx["documents"].insert(0, record)
        _save_index(idx)


def _delete_document(doc_id: str) -> bool:
    with _index_lock:
        idx = _load_index()
        docs = idx.get("documents", [])
        before = len(docs)
        target = next((d for d in docs if d.get("id") == doc_id), None)
        if target is None:
            return False
        docs = [d for d in docs if d.get("id") != doc_id]
        idx["documents"] = docs
        _save_index(idx)

        storage_file = target.get("_storageFile")
        if storage_file:
            try:
                (STORAGE_DIR / storage_file).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("failed to remove %s: %s", storage_file, exc)
        return before != len(docs)


def _public_documents() -> list[dict]:
    """Return documents with internal-only fields stripped."""
    idx = _load_index()
    out: list[dict] = []
    for d in idx.get("documents", []):
        public = {k: v for k, v in d.items() if not k.startswith("_")}
        out.append(public)
    return out


# --------------------------------------------------------------------------- #
# Pipeline (lazy)
# --------------------------------------------------------------------------- #

PIPELINE = None


def _get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        logger.info("Building pipeline (one-time, ~15-30s)")
        PIPELINE = build_pipeline()
    return PIPELINE


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #


@app.get("/api/health")
def health() -> dict:
    """Liveness probe used by the React frontend on mount."""
    if PIPELINE is None:
        return {"status": "warming", "device": "unknown"}
    return {"status": "ok", "device": PIPELINE.device.kind.value}


@app.get("/api/documents")
def list_documents() -> list[dict]:
    """All persisted documents, newest first."""
    return _public_documents()


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str) -> dict:
    docs = _public_documents()
    for d in docs:
        if d.get("id") == doc_id:
            return d
    raise HTTPException(status_code=404, detail="document not found")


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    removed = _delete_document(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail="document not found")
    return {"deleted": doc_id}


@app.get("/api/documents/{doc_id}/file")
def get_document_file(doc_id: str):
    """Return the original uploaded file bytes for preview/download."""
    idx = _load_index()
    target = next((d for d in idx.get("documents", []) if d.get("id") == doc_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="document not found")
    storage_file = target.get("_storageFile")
    if not storage_file:
        raise HTTPException(status_code=404, detail="file missing on disk")
    path = STORAGE_DIR / storage_file
    if not path.exists():
        raise HTTPException(status_code=404, detail="file missing on disk")
    return FileResponse(
        str(path),
        media_type=target.get("mimeType") or "application/octet-stream",
        filename=target.get("fileName") or storage_file,
    )


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)) -> dict:
    """Run the pipeline on an uploaded file, persist the result, and return
    the Document record the React store needs."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported extension {suffix!r}; expected {sorted(SUPPORTED_EXTENSIONS)}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="empty file")

    pipeline = _get_pipeline()

    # Persist the original file under a stable, opaque id so the frontend
    # can fetch it back via /api/documents/{id}/file.
    _ensure_storage()
    doc_id = _new_doc_id()
    storage_filename = f"{doc_id}{suffix}"
    storage_path = STORAGE_DIR / storage_filename
    storage_path.write_bytes(contents)

    try:
        result = pipeline.process_one(storage_path)
    except Exception as exc:
        logger.exception("pipeline failed for %s", file.filename)
        # Best-effort cleanup so a failed run doesn't leak orphaned files.
        try:
            storage_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"pipeline failed: {exc}")

    extraction = result.model_dump()
    record = _document_record(
        doc_id=doc_id,
        file_name=file.filename,
        file_size_bytes=len(contents),
        mime_type=file.content_type or "application/octet-stream",
        extraction=extraction,
        storage_filename=storage_filename,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
    )
    _save_document(record)

    # Return the public projection (no internal _* fields).
    public = {k: v for k, v in record.items() if not k.startswith("_")}
    public["extraction"] = extraction  # exposed for clients that want raw JSON
    return public


if __name__ == "__main__":
    import uvicorn  # type: ignore[import-not-found]

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
