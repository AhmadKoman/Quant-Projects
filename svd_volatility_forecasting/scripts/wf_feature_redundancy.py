#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature redundancy / correlation diagnostics for SVD vs HAR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
RESULTS_DIR = ROOT / "results"
for path in (str(ROOT), str(SRC), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import config
import features as fe
from run_experiments import load_data, load_vix_series


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="expanding")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument(
        "--root",
        type=Path,
        default=RESULTS_DIR / "walk_forward",
    )
    args = ap.parse_args()

    pdir = Path(args.root) / str(args.protocol) / f"h{int(args.horizon)}"
    pred_path = pdir / "predictions_log.csv"
    if not pred_path.is_file():
        raise SystemExit(f"Missing {pred_path}")

    dfp = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    dates = pd.DatetimeIndex(dfp.index)

    returns, asset_columns = load_data(build_data_if_missing=False)
    returns = returns * 100.0
    portfolio_returns = returns[asset_columns].mean(axis=1)
    cfg = config.config
    eps = float(cfg["training"].get("eps", 1e-8))
    feat = cfg["features"]
    h = int(args.horizon)

    rv_df = fe.build_har_rv(portfolio_returns, feat["rv_windows"])
    semi_df = fe.build_semivariance_features(portfolio_returns, feat["rv_windows"])
    svd_df, _ = fe.build_svd_features_panel_with_cov(
        returns,
        asset_columns,
        int(feat["svd_window"]),
        int(feat["K"]),
        list(feat["crisis_thresholds"]),
        eps,
        cov_estimator=str(feat.get("cov_estimator", "qis")),
    )
    common = rv_df.index.intersection(svd_df.index)
    svd_df_h = fe.smooth_svd_features(svd_df.reindex(common), h)
    X_har, har_names = fe.build_feature_sets(
        rv_df.reindex(common),
        None,
        svd_tier=0,
        default_threshold=float(feat["default_crisis_threshold"]),
        eps=eps,
        semi_df=semi_df.reindex(common),
    )
    X_t2, svd_names = fe.build_feature_sets(
        rv_df.reindex(common),
        svd_df_h,
        svd_tier=2,
        default_threshold=float(feat["default_crisis_threshold"]),
        eps=eps,
        semi_df=semi_df.reindex(common),
        interaction_smooth_h=h,
    )
    svd_only = [c for c in svd_names if c not in har_names]
    X = pd.concat([X_har, X_t2[svd_only]], axis=1)
    X = X.reindex(dates).dropna(how="any")

    vix = load_vix_series()
    if vix is not None:
        X["IV_vol"] = vix.reindex(X.index).astype(float)

    har_cols = list(dict.fromkeys(c for c in har_names if c in X.columns))
    spec_cols = [c for c in svd_only if c in X.columns]
    rows = []
    for sc in spec_cols:
        s = X[sc].astype(np.float64)
        row = {"feature": sc, "n": int(s.notna().sum())}
        for hc in har_cols:
            row[f"corr_{hc}"] = float(s.corr(X[hc].astype(np.float64)))
        if "IV_vol" in X.columns:
            row["corr_IV_vol"] = float(s.corr(X["IV_vol"].astype(np.float64)))
        har_corrs = [abs(row.get(f"corr_{hc}", np.nan)) for hc in har_cols]
        row["mean_abs_corr_har"] = float(np.nanmean(har_corrs))
        rows.append(row)
    corr_df = pd.DataFrame(rows).sort_values("mean_abs_corr_har", ascending=False)

    out_dir = pdir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"wf_feature_redundancy_{args.protocol}_h{h}.csv"
    corr_df.to_csv(out_path, index=False)

    if "gap12_norm" in X.columns:
        g = X["gap12_norm"].astype(np.float64)
        q = g.quantile([0.25, 0.5, 0.75])
        stab_rows = []
        for col in ["angle", "subspace_dist_K", "delta_AR_z"]:
            if col not in X.columns:
                continue
            cser = X[col].astype(np.float64)
            low = cser.loc[g <= q.iloc[0]].std()
            high = cser.loc[g >= q.iloc[2]].std()
            stab_rows.append(
                {
                    "feature": col,
                    "std_low_gap": float(low),
                    "std_high_gap": float(high),
                    "ratio_high_low": float(high / low) if low and low > 0 else np.nan,
                }
            )
        pd.DataFrame(stab_rows).to_csv(
            out_dir / f"wf_eigen_stability_by_gap_{args.protocol}_h{h}.csv",
            index=False,
        )

    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
