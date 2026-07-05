# Data

Place `Data4Fin.csv` in this directory before running the training scripts.

The dataset is a multi-asset daily panel with OHLCV prices and pre-computed technical indicators for 11 US equities:

| Symbol | Company |
|--------|---------|
| AAPL | Apple |
| MSFT | Microsoft |
| NVDA | NVIDIA |
| JNJ | Johnson & Johnson |
| BAC | Bank of America |
| AXP | American Express |
| CVX | Chevron |
| OXY | Occidental Petroleum |
| MRO | Marathon Oil |
| CCL | Carnival |
| RCL | Royal Caribbean |

Expected columns include `symbol`, `date`, `open`, `high`, `low`, `close`, `volume`, and indicator fields used by the QTMRT variant (RSI, EMA, MACD, Ichimoku, Bollinger bands, etc.).

The file is excluded from git because of size (~32 MB). Keep a local copy at `data/Data4Fin.csv`.
