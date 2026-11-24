# INVOICEFLOW DEPLOYMENT & CONTAINERIZATION GUIDE
## Standalone Installation, Docker Runtime Environments, GPU Acceleration & Air-Gapped Setup

---

| **Document Information** | **Specification Details** |
| :--- | :--- |
| **System Name** | **InvoiceFlow Document Intelligence** |
| **Document Classification** | Operations & Deployment Manual |
| **Document Version** | **1.0.0-PROD** |
| **Target OS** | Ubuntu 22.04 LTS / Debian 12 / Windows 11 / WSL2 |

---

## 1. Quickstart: Clean Local Setup

### 1.1 Prerequisites
* Python 3.10 or 3.11 installed.
* Optional: NVIDIA Driver 535+ and CUDA 11.8/12.1 for GPU acceleration.

### 1.2 Virtual Environment Installation
```bash
# 1. Create isolated virtual environment
python -m venv venv

# 2. Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip and wheel
pip install --upgrade pip setuptools wheel

# 4. Install pinned production dependencies
pip install -r requirements.txt

# 5. Verify installation
python executable.py --help
```

---

## 2. Docker Containerization Specification

### 2.1 Standalone Dockerfile (`docker/Dockerfile`)
```dockerfile
# Stage 1: Base CUDA runtime image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-guj \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application source code and weights
COPY . .

# Set default entrypoint
ENTRYPOINT ["python3", "executable.py"]
CMD ["--help"]
```

### 2.2 Docker Build & Run Commands
```bash
# Build container image
docker build -t invoiceflow:latest -f docker/Dockerfile .

# Run single file inference
docker run --rm -v $(pwd)/data:/data invoiceflow:latest \
    --input /data/sample_invoice.pdf \
    --output /data/output.json
```

---

## 3. Air-Gapped / Offline Enterprise Deployment

To run in completely offline enterprise environments with zero external network access:
1. **Model Weights Bundle:** Ensure `weights/yolo_stamp_sig_v1.pt` and OCR/VLM model caches are pre-downloaded and packaged into the directory:
   ```bash
   export PADDLE_HOME=/app/weights/paddle_cache
   ```
2. **Wheel Archive:** Pre-download all PyPI dependencies into a local wheelhouse:
   ```bash
   pip download -r requirements.txt -d ./wheels
   pip install --no-index --find-links=./wheels -r requirements.txt
   ```
3. **Execution Command:** Execute using the local CPU/CUDA device:
   ```bash
   python executable.py --input ./input_documents/ --output ./output.json --device cpu
   ```
