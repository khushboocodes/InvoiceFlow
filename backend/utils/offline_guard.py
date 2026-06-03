"""Network kill-switch for offline operation.

The pipeline must be runnable inside a ``--network=none`` container. This
module enforces that contract at runtime by:

1. Setting Hugging Face / Transformers offline environment variables before
   any model library is imported.
2. Monkey-patching ``socket.socket.connect`` to raise on any non-loopback
   address, so even a transitive dependency that tries to phone home will
   fail loudly.
3. Wrapping ``urllib.request.urlopen`` and (if installed) ``requests``'
   transport adapter to raise the same error.

We allow loopback (127.0.0.1, ::1, localhost) so the optional FastAPI demo
bridge can still talk to the React frontend on the same machine.

Validates Requirements: 16.1, 16.2, 16.3, 16.5
"""

from __future__ import annotations

import logging
import os
import socket
from typing import Any

logger = logging.getLogger(__name__)


class OfflineViolation(RuntimeError):
    """Raised when offline mode is engaged and a network call is attempted."""


_LOOPBACK_HOSTS = frozenset(
    {
        "127.0.0.1",
        "::1",
        "localhost",
        "0.0.0.0",  # bind-to-any, also treated as loopback for our purposes
    }
)

_offline_enabled = False
_original_connect = socket.socket.connect


def _is_loopback_address(address: Any) -> bool:
    """Return True when ``address`` is considered safe in offline mode.

    Accepts both IPv4 ``(host, port)`` tuples and IPv6
    ``(host, port, flow, scope)`` tuples; falls back to a ``str`` check for
    Unix sockets and other oddities.
    """
    if isinstance(address, tuple) and len(address) >= 1:
        host = address[0]
        if isinstance(host, (bytes, bytearray)):
            host = host.decode("ascii", errors="replace")
        return host in _LOOPBACK_HOSTS
    # Unix sockets, abstract namespaces, etc. — never network, always allowed.
    return True


def _guarded_connect(self: socket.socket, address: Any) -> None:
    """Replacement for ``socket.socket.connect`` that blocks remote calls."""
    if _offline_enabled and not _is_loopback_address(address):
        raise OfflineViolation(
            f"Outbound network call blocked in offline mode: connect({address!r})"
        )
    return _original_connect(self, address)


def enable_offline_mode() -> None:
    """Engage the kill-switch.

    Idempotent. Safe to call multiple times. Must be called BEFORE any model
    library (paddleocr, ultralytics, llama_cpp, transformers, huggingface_hub)
    is imported, otherwise those libraries may have already opened sockets or
    cached HTTP sessions.
    """
    global _offline_enabled

    # Always set env vars first — these affect lazy imports.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if _offline_enabled:
        return

    # Patch socket.connect for every socket instance going forward.
    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]

    # If urllib has already been imported, wrap urlopen too.
    try:
        import urllib.request

        original_urlopen = urllib.request.urlopen

        def _guarded_urlopen(*args, **kwargs):  # type: ignore[no-untyped-def]
            url = args[0] if args else kwargs.get("url", "")
            url_str = url.full_url if hasattr(url, "full_url") else str(url)
            if _is_offline_safe_url(url_str):
                return original_urlopen(*args, **kwargs)
            raise OfflineViolation(f"Outbound HTTP call blocked: {url_str!r}")

        urllib.request.urlopen = _guarded_urlopen  # type: ignore[assignment]
    except Exception as exc:  # pragma: no cover - defensive only
        logger.warning("Could not wrap urllib.request.urlopen: %s", exc)

    # If requests is installed, wrap its transport adapter too.
    try:
        import requests.adapters  # type: ignore[import-not-found]

        original_send = requests.adapters.HTTPAdapter.send

        def _guarded_send(self, request, **kwargs):  # type: ignore[no-untyped-def]
            if _is_offline_safe_url(request.url):
                return original_send(self, request, **kwargs)
            raise OfflineViolation(f"Outbound requests call blocked: {request.url!r}")

        requests.adapters.HTTPAdapter.send = _guarded_send  # type: ignore[assignment]
    except ImportError:
        # requests is not a dependency; if it isn't installed, no patching needed.
        pass

    _offline_enabled = True
    logger.info("Offline mode engaged: outbound network calls will be blocked")


def _is_offline_safe_url(url: str) -> bool:
    """True when the URL targets a loopback host."""
    if not url:
        return True
    lower = url.lower()
    for host in _LOOPBACK_HOSTS:
        if f"://{host}" in lower or f"://{host}:" in lower or lower.startswith(f"{host}/"):
            return True
    return False


def is_offline_mode() -> bool:
    """Return whether offline mode is currently engaged."""
    return _offline_enabled
