"""Unit tests for utils.offline_guard — network kill-switch.

Validates Requirements: 16.1, 16.2, 16.3, 16.5
"""

from __future__ import annotations

import os
import socket

import pytest

from utils import offline_guard
from utils.offline_guard import OfflineViolation, _is_loopback_address, enable_offline_mode


@pytest.fixture(autouse=True)
def _reset_offline_state(monkeypatch):
    """Each test starts with the guard disabled and a clean socket.connect."""
    # Restore original socket.connect, since enable_offline_mode patches the class.
    monkeypatch.setattr(socket.socket, "connect", offline_guard._original_connect, raising=True)
    monkeypatch.setattr(offline_guard, "_offline_enabled", False)
    yield
    # Belt + suspenders cleanup
    monkeypatch.setattr(socket.socket, "connect", offline_guard._original_connect, raising=True)
    monkeypatch.setattr(offline_guard, "_offline_enabled", False)


def test_loopback_address_detection():
    assert _is_loopback_address(("127.0.0.1", 8000)) is True
    assert _is_loopback_address(("localhost", 80)) is True
    assert _is_loopback_address(("::1", 443, 0, 0)) is True
    assert _is_loopback_address(("example.com", 80)) is False
    assert _is_loopback_address(("8.8.8.8", 53)) is False


def test_enable_sets_huggingface_offline_env_vars():
    enable_offline_mode()
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_enable_blocks_non_loopback_socket_connect():
    enable_offline_mode()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.1)
    try:
        with pytest.raises(OfflineViolation):
            # 192.0.2.1 is documentation-reserved; it should not actually be reachable.
            s.connect(("192.0.2.1", 80))
    finally:
        s.close()


def test_enable_allows_loopback_socket_connect_attempts():
    """Loopback connections should NOT raise OfflineViolation.

    They may still raise ConnectionRefusedError because nothing is listening
    at this port, but that comes from the OS, not from our guard.
    """
    enable_offline_mode()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.1)
    try:
        with pytest.raises(Exception) as excinfo:
            s.connect(("127.0.0.1", 1))  # port 1 — always closed
        # The error MUST NOT be our guard refusing the call.
        assert not isinstance(excinfo.value, OfflineViolation)
    finally:
        s.close()


def test_enable_is_idempotent():
    enable_offline_mode()
    first = socket.socket.connect
    enable_offline_mode()
    second = socket.socket.connect
    assert first is second
