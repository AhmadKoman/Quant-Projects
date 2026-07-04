#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch VIX → daily implied vol (decimal) for the IV_baseline evaluation column.
The project may also append log(VIX) as an optional HAR-X regressor in features (see features.build_feature_sets).
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Install yfinance: pip install yfinance")
    sys.exit(1)

try:
    import config as _proj_config
    _vp = _proj_config.config["data"].get("vix_path")
    OUTPUT_PATH = Path(_vp) if _vp else SCRIPT_DIR / "vix_daily_vol.csv"
    START_DATE = str(_proj_config.config["data"]["start_date"])
    END_DATE = str(_proj_config.config["data"]["end_date"])
except Exception:
    OUTPUT_PATH = SCRIPT_DIR / "vix_daily_vol.csv"
    START_DATE = "2000-01-15"
    END_DATE = "2024-12-31"


def fetch_vix_levels(start: str, end: str) -> pd.Series:
    data = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    if data.empty or "Close" not in data.columns:
        raise RuntimeError("No VIX data downloaded.")
    return data["Close"].squeeze()


def vix_to_daily_vol(vix_levels: pd.Series) -> pd.Series:
    return vix_levels.div(100).div(np.sqrt(252))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write-manifest", action="store_true")
    p.add_argument("--verify-manifest", action="store_true")
    args = p.parse_args()
    manifest_path = SCRIPT_DIR / "fetch_manifest_vix.json"
    if args.verify_manifest:
        sys.path.insert(0, str(SCRIPT_DIR))
        import fetch_manifest as fm  # noqa: E402
        fm.verify_manifest(manifest_path, root=ROOT)
        print("Manifest verification OK.")
        sys.exit(0)

    print(f"Downloading ^VIX from {START_DATE} to {END_DATE}...")
    vix_levels = fetch_vix_levels(START_DATE, END_DATE)
    vix_daily = vix_to_daily_vol(vix_levels).dropna()
    out = pd.DataFrame({"VIX_daily_vol": vix_daily})
    out.index.name = "Date"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, float_format="%.10g", lineterminator="\n")
    print(f"Saved VIX daily vol to {OUTPUT_PATH}, shape {out.shape}")

    if args.write_manifest:
        sys.path.insert(0, str(SCRIPT_DIR))
        import fetch_manifest as fm  # noqa: E402
        fm.write_manifest(
            [OUTPUT_PATH],
            params={"start_date": START_DATE, "end_date": END_DATE, "series": "VIX"},
            out_path=manifest_path,
            project_root=ROOT,
        )
        print(f"Wrote manifest {manifest_path}")


if __name__ == "__main__":
    main()
