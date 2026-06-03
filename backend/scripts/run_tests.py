"""Run the full unit test suite and exit cleanly.

Wrapping pytest in a child process and force-exiting after collecting
results works around a known issue on Windows where pytest hangs at
shutdown when ``torch`` + ``paddle`` + ``ultralytics`` are all imported in
the same process tree (DLL teardown deadlock).

Run::

    python -m scripts.run_tests
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit",
        "-q",
        "--tb=short",
        "-p",
        "no:cacheprovider",
    ]
    proc = subprocess.run(cmd, cwd=BACKEND_ROOT, capture_output=True, text=True, timeout=600)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if "failed" in proc.stdout.lower() or "error" in proc.stdout.lower():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
