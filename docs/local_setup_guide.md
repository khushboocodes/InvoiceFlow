# InvoiceFlow: Local Setup & Developer Guide

This guide walks you through setting up and running **InvoiceFlow** locally on your workstation or server, whether you are running on a Windows PC with an NVIDIA GPU, a Linux server, or a CPU-only environment.

---

## 1. System Prerequisites

Before starting, ensure you have the following installed on your machine:

| Requirement | Minimum | Recommended |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11, Ubuntu 20.04+, macOS | Windows 11 or Ubuntu 22.04 LTS |
| **Python** | Python 3.10 | **Python 3.11** (strongly recommended) |
| **Node.js & npm** | Node.js v18.x | Node.js v20.x LTS + npm 10.x |
| **Git** | Git 2.30+ | Latest Git CLI |
| **Compute / GPU** | 4-core CPU, 8 GB System RAM | **NVIDIA GPU** (RTX 3050/3060/4060, T4, A10G) with 4GB+ VRAM & CUDA 12.x |

---

## 2. Step-by-Step Backend Setup

### Step 2.1: Clone the Repository
Open your terminal (PowerShell on Windows, Bash on Linux/macOS) and clone the repository:

```bash
git clone <YOUR-REPOSITORY-URL>
cd "InvoiceFlow- Document Intelligent System"
```

### Step 2.2: Create and Activate Python Virtual Environment

* **On Windows (PowerShell):**
  ```powershell
  # If running scripts is blocked, run once:
  # Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

* **On Linux / macOS (Bash):**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

---

### Step 2.3: Install PyTorch with Hardware Acceleration

#### Option A: NVIDIA GPU Users (Recommended — Fast OCR + 4-bit VLM)
If you have an NVIDIA graphics card, install PyTorch with CUDA 12.1+ support:

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Option B: CPU-Only Users (No dedicated GPU)
If you are running on a CPU-only machine or macOS:

```bash
pip install --upgrade pip
pip install torch torchvision
```

---

### Step 2.4: Install Core Dependencies

Install all required computer vision, OCR, API, and validation libraries:

```bash
pip install -r requirements.txt
```

#### Verify Environment Readiness
Run this quick one-liner to verify your GPU and library installation:

```bash
python -c "import torch, cv2, easyocr, pydantic; print('✅ CUDA Available:', torch.cuda.is_available()); print('✅ Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 3. Step-by-Step Frontend Setup (Modern React UI)

The modern dashboard runs on React, Vite, and Tailwind CSS and connects to the Python AI engine via the FastAPI bridge.

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```

---

## 4. Running the Application (3 Operational Modes)

You can run InvoiceFlow in three different ways depending on your use case:

```
+----------------------------------------------------------------------------------------------------+
|                                    OPERATIONAL RUN MODES                                           |
+---------------------+---------------------------------+--------------------------------------------+
| Run Mode            | Entrypoint                      | Description                                |
+---------------------+---------------------------------+--------------------------------------------+
| Mode A: Full Stack  | bridge_server.py + React Vite   | Complete SaaS web UI on localhost:5173     |
| Mode B: Rapid Review| app.py (Streamlit)              | Instant visual canvas on localhost:8501    |
| Mode C: Headless CLI| executable.py                   | Automated terminal batch script            |
+---------------------+---------------------------------+--------------------------------------------+
```

---

### Mode A: Modern React SaaS Web App + AI FastAPI Bridge (Recommended)

This mode runs the modern React SaaS UI connected to the local offline AI engine.

#### Terminal 1 — Start the AI Bridge Server:
In `InvoiceFlow- Document Intelligent System`:
```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn bridge_server:app --host 127.0.0.1 --port 8000
```
*Health Check:* Open `http://127.0.0.1:8000/api/health` in your browser. You should see:
```json
{"status": "ok", "device": "cuda", "model": "Qwen2.5-VL-3B (RTX 3050 GPU)"}
```

#### Terminal 2 — Start the React Web Application:
In the `frontend` folder:
```powershell
cd frontend
npm run dev
```
Open **`http://localhost:5173`** in your browser.
* You can now drag-and-drop invoices, observe real-time pipeline extraction progress, and inspect pixel-accurate bounding boxes for signatures and stamps.

---

### Mode B: Standalone Streamlit Rapid Review Canvas

For instant single-document analysis without needing Node.js:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py --server.port 8501
```
Open **`http://localhost:8501`** in your browser.
* Allows instant upload, image contrast toggles, side-by-side extracted JSON inspection, and telemetry tracking.

---

### Mode C: Headless Production CLI (Batch & Scripts)

For processing documents headlessly from scripts, cron jobs, or Loan Origination Systems (LOS):

```powershell
# Extract a single document
python executable.py --input "path/to/invoice.png" --output "output.json" --verbose

# Process an entire folder of invoices (batch processing)
python executable.py --input "path/to/invoices/" --output "batch_output/" --limit 20
```

---

## 5. Running the Automated Test Suite

To verify system integrity and ensure all business rules, bounds, and schema validations are functioning:

```powershell
# Run unit tests
python -m pytest tests/unit/ -v

# Run integration tests
python -m pytest tests/integration/ -v
```

All 17 unit tests should return `PASSED` in $< 1.0\text{s}$.

---

## 6. How the Local AI Models Work

* **EasyOCR (GPU / CPU):** Automatically caches detection and recognition models into `~/.EasyOCR/` on the first run.
* **YOLOv8 Signature & Stamp Detector:** Automatically loads pretrained weights and runs morphological contour refinement for rubber ink seals and cursive signatures.
* **Local Vision-Language Model (`Qwen2.5-VL-3B-Instruct`):**
  * Automatically downloaded from Hugging Face on the first inference and cached in `~/.cache/huggingface/`.
  * Quantized in **4-bit NF4 (`bitsandbytes`)**, allocating only **~2.25 GB VRAM** so it runs smoothly even on entry-level 4GB / 6GB laptop GPUs without Out-Of-Memory (OOM) errors.
  * **100% Offline:** Once downloaded, it operates completely offline with zero network connectivity required.

---

## 7. Troubleshooting & FAQ

### Q1: `Activate.ps1 cannot be loaded because running scripts is disabled on this system` (Windows)
* **Fix:** Open PowerShell as Administrator and run:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```

### Q2: Port Conflict: `Address already in use` (Port 8000, 5173, or 8501)
* **Fix (Windows):** Check and terminate the process holding the port:
  ```powershell
  # Find PID using port 8000
  netstat -ano | findstr :8000
  # Stop the process
  Stop-Process -Id <PID> -Force
  ```

### Q3: Running on a machine with less than 4GB GPU VRAM or CPU-only
* **Fix:** The pipeline includes automatic fallback. If CUDA is not detected, it automatically falls back to CPU execution. You can also explicitly run the fast EasyOCR engine:
  ```python
  from core.pipeline import InvoiceFlowPipeline
  pipeline = InvoiceFlowPipeline(use_local_vlm=False) # Pure OCR mode
  ```

### Q4: Bounding boxes are not rendering or misaligned
* **Fix:** Ensure both the React frontend and FastAPI bridge are running the latest versions. The bridge server automatically converts normalized $[0, 1000]$ bounds to image pixel space $[x1, y1, x2, y2]$, ensuring responsive alignment on all browser screen sizes.
