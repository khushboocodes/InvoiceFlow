"""Unit tests for utils.device — CUDA detection branches.

Validates Requirements: 17.1, 17.2, 17.3, 17.4
"""

from __future__ import annotations

from unittest.mock import patch

from utils.device import Device, DeviceInfo, detect


class _FakeCuda:
    def __init__(self, available: bool, name: str = "Mock GPU"):
        self._available = available
        self._name = name

    def is_available(self) -> bool:
        return self._available

    def get_device_name(self, index: int) -> str:
        return self._name


class _FakeTorch:
    def __init__(self, available: bool, name: str = "Mock GPU"):
        self.cuda = _FakeCuda(available, name)


def test_detect_returns_cpu_when_cuda_unavailable():
    fake_torch = _FakeTorch(available=False)
    with patch.dict("sys.modules", {"torch": fake_torch}):
        info = detect()

    assert info.kind is Device.CPU
    assert info.cuda_index is None
    assert info.is_gpu is False
    assert info.torch_device_string() == "cpu"


def test_detect_returns_cuda_when_available():
    fake_torch = _FakeTorch(available=True, name="NVIDIA GeForce RTX 3050")
    with patch.dict("sys.modules", {"torch": fake_torch}):
        info = detect()

    assert info.kind is Device.CUDA
    assert info.cuda_index == 0
    assert info.is_gpu is True
    assert info.torch_device_string() == "cuda:0"
    assert "RTX 3050" in info.description


def test_detect_falls_back_to_cpu_when_torch_import_fails():
    """If torch raises during import, we must not crash — we fall back to CPU."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def boom(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch is not installed")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=boom):
        info = detect()

    assert info.kind is Device.CPU


def test_device_info_is_frozen():
    info = DeviceInfo(kind=Device.CPU, cuda_index=None, description="CPU")
    try:
        info.kind = Device.CUDA  # type: ignore[misc]
    except Exception:
        return  # frozen as expected
    raise AssertionError("DeviceInfo should be frozen")
