#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run unit tests, then the full experiment pipeline (same shell, fail-fast).

Usage:
  python run_tests_and_experiments.py
  python run_tests_and_experiments.py --build-data

Any arguments after the script name are forwarded to run_experiments.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    extra = sys.argv[1:]
    pytest_cmd = [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"]
    print("[INFO] Step 1/2: pytest", " ".join(pytest_cmd))
    r1 = subprocess.run(pytest_cmd, cwd=str(ROOT))
    if r1.returncode != 0:
        print("[ERROR] Tests failed; skipping run_experiments.py", file=sys.stderr)
        return r1.returncode

    exp_cmd = [sys.executable, str(ROOT / "scripts" / "run_experiments.py"), *extra]
    print("[INFO] Step 2/2:", " ".join(exp_cmd))
    r2 = subprocess.run(exp_cmd, cwd=str(ROOT))
    return r2.returncode


if __name__ == "__main__":
    sys.exit(main())
