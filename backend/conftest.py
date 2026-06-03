"""Pytest bootstrap.

Adds the backend root to ``sys.path`` so tests can import ``utils.*`` without
an editable install. Runs once per test session.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
