# InvoiceFlow Demo Bridge

FastAPI server that exposes the offline pipeline over localhost so the React
frontend in `../src/` can call it during the live demo.

**This folder is NOT part of `submission.zip`.** It exists for the demo round
only.

## Setup

```powershell
# From the demo/ folder
cd demo

# Reuse the backend's venv — it already has torch, paddle, ultralytics, transformers.
..\backend\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
# Start the bridge
..\backend\.venv\Scripts\python.exe server.py

# In a second terminal, start the React dev server from the repo root
cd ..
pnpm dev
```

The frontend on `http://localhost:5173` will call `http://127.0.0.1:8000/api/*`.

## Endpoints

- `GET  /api/health` — liveness probe; returns `{status, device}`
- `POST /api/extract` — multipart file upload, returns the full Pydantic
  `ExtractionResult` JSON

## Notes

- The pipeline is built lazily on the first `/api/extract` call (~15-30s
  warm-up). Subsequent calls reuse the loaded models.
- Same offline guarantees apply — set `--offline` in the future via env var if
  you want to enforce it during the demo.
- CORS is open to ports 5173 and 3000 (Vite + Next.js defaults).
