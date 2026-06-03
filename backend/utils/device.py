"""Hardware detection — single source of truth for GPU/CPU choice.

The Pipeline must run on either a CPU-only machine or a low-tier GPU without
rebuilding the artifact. ``detect()`` is called once at process startup and
the resulting :class:`DeviceInfo` is threaded through every model-loading
module so every component agrees on the device.

Importing :mod:`torch` is unavoidable because ``torch.cuda.is_available()``
is the standard probe; we keep the import inside :func:`detect` so module
import is cheap when the function is not called.

Validates Requirements: 17.1, 17.2, 17.3, 17.4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Device(str, Enum):
    """Supported execution devices."""

    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved device metadata.

    Attributes:
        kind:         Either :attr:`Device.CPU` or :attr:`Device.CUDA`.
        cuda_index:   GPU ordinal (0 by default) when CUDA is selected, else None.
        description:  Human-readable label, useful for startup logs and the
                      ``GET /api/health`` response in the optional demo bridge.
    """

    kind: Device
    cuda_index: int | None
    description: str

    @property
    def is_gpu(self) -> bool:
        return self.kind is Device.CUDA

    def torch_device_string(self) -> str:
        """Return the canonical PyTorch device string (e.g. ``"cuda:0"`` or ``"cpu"``)."""
        if self.kind is Device.CUDA and self.cuda_index is not None:
            return f"cuda:{self.cuda_index}"
        return "cpu"


def detect() -> DeviceInfo:
    """Probe the host once for CUDA availability.

    Returns:
        A frozen :class:`DeviceInfo` describing the chosen device. The
        function never raises — if anything goes wrong probing CUDA, we fall
        back to CPU and log the failure.
    """
    try:
        # Localized import keeps the module cheap to import in test contexts
        # that mock out CUDA detection entirely.
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            index = 0
            description = torch.cuda.get_device_name(index)
            info = DeviceInfo(kind=Device.CUDA, cuda_index=index, description=description)
            logger.info("CUDA detected: %s (cuda:%d)", description, index)
            return info
    except Exception as exc:  # pragma: no cover - defensive only
        logger.warning("CUDA probe failed (%s); falling back to CPU", exc)

    info = DeviceInfo(kind=Device.CPU, cuda_index=None, description="CPU")
    logger.info("Running on CPU")
    return info
