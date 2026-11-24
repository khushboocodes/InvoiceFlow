# InvoiceFlow Production Dockerfile (Task 11.3 & 11.4)
# Multi-stage air-gapped CUDA-enabled container for high-throughput document intelligence.

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache EasyOCR models into container for 100% offline air-gapped execution
RUN mkdir -p /root/.EasyOCR/model && \
    curl -L -o /root/.EasyOCR/model/craft_mlt_25k.pth https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/craft_mlt_25k.pth && \
    curl -L -o /root/.EasyOCR/model/english_g2.pth https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.pth

# Copy application source code
COPY core/ /app/core/
COPY document_profiles/ /app/document_profiles/
COPY executable.py /app/executable.py
COPY README.md /app/README.md

# Volumes for input invoices and output JSONs
VOLUME ["/data/input", "/data/output"]

# Default entrypoint
ENTRYPOINT ["python", "executable.py"]
CMD ["--help"]
