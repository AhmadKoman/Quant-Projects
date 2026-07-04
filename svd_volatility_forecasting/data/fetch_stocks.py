#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch 100-stock panel and compute daily log-returns.
Saves returns_100.csv (balanced panel: T x N, no NaN).

Options:
  --write-manifest   Write data/fetch_manifest.json with SHA256 of outputs.
  --verify-manifest  Exit non-zero if hashes do not match manifest (no download).

Dates and paths default from project config.py when available.
"""

from pathlib import Path
import argparse
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print("Missing dependency:", e)
    sys.exit(1)

try:
    import yfinance as yf
except ImportError:
    print("Install yfinance: pip install yfinance")
    sys.exit(1)

try:
    import config as _proj_config
    _dc = _proj_config.config["data"]
    TICKERS_PATH = Path(_dc["tickers_path"])
    OUTPUT_PATH = Path(_dc["returns_cache_path"])
    START_DATE = str(_dc["start_date"])
    END_DATE = str(_dc["end_date"])
    MAX_MISSING_PCT = float(_dc["max_missing_pct"])
except Exception:
    TICKERS_PATH = SCRIPT_DIR / "tickers_100.csv"
    OUTPUT_PATH = SCRIPT_DIR / "returns_100.csv"
    START_DATE = "2000-01-15"
    END_DATE = "2024-12-31"
    MAX_MISSING_PCT = 0.05


def load_tickers(path: Path) -> list:
    df = pd.read_csv(path)
    tickers = df["ticker"].dropna().astype(str).str.strip()
    tickers = tickers[tickers.str.len() > 0].unique().tolist()
    return tickers


def fetch_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")
    data = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data.empty:
        raise RuntimeError("No data downloaded.")
    if len(tickers) == 1:
        out = data[["Close"]].copy()
        out.columns = [tickers[0]]
        out.index.name = "Date"
        return out
    closes = {}
    if isinstance(data.columns, pd.MultiIndex):
        for sym in tickers:
            if (sym, "Close") in data.columns:
                closes[sym] = data[(sym, "Close")]
            elif (sym, "Adj Close") in data.columns:
                closes[sym] = data[(sym, "Adj Close")]
    else:
        if "Close" in data.columns:
            return data[["Close"]].rename(columns={"Close": tickers[0]})
        return data
    out = pd.DataFrame(closes)
    out.index.name = "Date"
    return out


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def main() -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Fetch stock panel → returns_100.csv")
    parser.add_argument("--write-manifest", action="store_true", help="Write data/fetch_manifest.json")
    parser.add_argument(
        "--verify-manifest",
        action="store_true",
        help="Verify outputs vs data/fetch_manifest.json (exit 1 on mismatch)",
    )
    args = parser.parse_args()

    manifest_path = SCRIPT_DIR / "fetch_manifest_stocks.json"
    if args.verify_manifest:
        sys.path.insert(0, str(SCRIPT_DIR))
        import fetch_manifest as fm  # noqa: E402
        fm.verify_manifest(manifest_path, root=ROOT)
        print("Manifest verification OK.")
        sys.exit(0)

    tickers = load_tickers(TICKERS_PATH)
    if len(tickers) < 2:
        raise ValueError("Need at least 2 tickers in tickers_100.csv")
    try:
        import importlib.metadata as imd
        yfv = imd.version("yfinance")
    except Exception:
        yfv = "unknown"
    print(f"yfinance={yfv}  pandas={pd.__version__}")
    prices = fetch_prices(tickers, START_DATE, END_DATE)
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices.sort_index()
    prices = prices.dropna(how="all", axis=0)
    missing_pct = prices.isna().mean()
    keep = missing_pct <= MAX_MISSING_PCT
    dropped = missing_pct[~keep]
    if len(dropped) > 0:
        print(
            f"Dropping {len(dropped)} tickers with >{MAX_MISSING_PCT*100:.0f}% missing: "
            f"{dropped.index.tolist()}"
        )
    prices = prices.loc[:, keep]
    if prices.shape[1] < 2:
        raise ValueError("After dropping bad tickers, fewer than 2 columns remain.")
    # Forward-fill at most 1 consecutive missing day per revision guide.
    # Propagating more than 1 stale price would introduce spurious zero-variance
    # days that inflate the covariance matrix and distort SVD features.
    prices = prices.ffill(limit=1)
    returns = to_log_returns(prices)
    returns = returns.dropna(how="any")
    returns = returns.astype(np.float64)
    if np.any(~np.isfinite(returns.values)):
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    returns.index.name = "Date"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(OUTPUT_PATH, float_format="%.10g", lineterminator="\n")
    print(f"Saved log-returns to {OUTPUT_PATH}, shape {returns.shape}")

    if args.write_manifest:
        sys.path.insert(0, str(SCRIPT_DIR))
        import fetch_manifest as fm  # noqa: E402
        tick_hash = fm.sha256_file(TICKERS_PATH)
        fm.write_manifest(
            [OUTPUT_PATH],
            params={
                "start_date": START_DATE,
                "end_date": END_DATE,
                "max_missing_pct": MAX_MISSING_PCT,
                "tickers_sha256": tick_hash,
            },
            out_path=manifest_path,
            project_root=ROOT,
            extra={"yfinance": yfv},
        )
        print(f"Wrote manifest {manifest_path}")

    return returns


if __name__ == "__main__":
    main()
