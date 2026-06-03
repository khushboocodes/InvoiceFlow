"""InvoiceFlow backend utilities.

This package contains the offline Document AI pipeline modules. Public API
surface is intentionally small — modules expose typed dataclasses and pure
functions that the CLI orchestrator (``executable.py``) wires together.

Module map:

* :mod:`utils.device`         — CUDA detection, single source of truth.
* :mod:`utils.offline_guard`  — Network kill-switch.
* :mod:`utils.schema`         — Pydantic output contract.
* :mod:`utils.ingestion`      — Stage 1: PDF/image loading + preprocessing.
* :mod:`utils.ocr`            — Stage 2A: PaddleOCR wrapper.
* :mod:`utils.detection`      — Stage 2B: YOLOv8n wrapper.
* :mod:`utils.extraction`     — Stage 3 Tier-1: regex / anchor rules.
* :mod:`utils.slm`            — Stage 3 Tier-2: Qwen2.5 GGUF fallback.
* :mod:`utils.normalization`  — Stage 4: fuzzy match, range checks.
* :mod:`utils.masters`        — Dealer / asset master loading + mining.
* :mod:`utils.confidence`     — Stage 5: per-field + doc-level scoring.

WINDOWS DLL ORDER NOTE
----------------------
On Windows, ``torch`` and ``paddle`` ship conflicting MKL / OpenMP DLLs.
Importing paddle before torch causes ``WinError 127`` inside
``torch.__init__``. We pre-import torch here so any subsequent ``import
utils.<anything>`` is safe regardless of which submodule the caller
touches first.
"""

# Pre-import torch to lock its DLLs first; failure is non-fatal because
# torch is only required for GPU paths, and CPU-only environments may
# legitimately not have it.
try:
    import torch as _torch  # noqa: F401
except ImportError:
    pass

__version__ = "0.1.0"
