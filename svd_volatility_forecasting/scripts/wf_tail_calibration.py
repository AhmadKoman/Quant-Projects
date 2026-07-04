#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leakage-safe, online tail calibration for walk-forward variance forecasts."""

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
from evaluation import economic as econ
from evaluation import tail_calibration as tc
from run_experiments import load_data, load_vix_series


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="expanding", choices=["expanding", "rolling"])
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 5, 22])
    ap.add_argument("--model", default="HAR+SVD", help="Model name in predictions_log.csv.")
    ap.add_argument("--gate", default="AR", help="Gate variable in svd_df (e.g. AR, AR_eff).")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--dist", default="gaussian", choices=["gaussian", "student_t"])
    ap.add_argument("--df", type=float, default=8.0)
    ap.add_argument("--min_train", type=int, default=600)
    ap.add_argument("--refit_every", type=int, default=21)
    ap.add_argument(
        "--root",
        type=Path,
        default=RESULTS_DIR / "walk_forward",
    )
    args = ap.parse_args()

    protocol = str(args.protocol)
    h = int(args.horizon)
    model = str(args.model)
    gate_name = str(args.gate)

    pdir = Path(args.root) / protocol / f"h{h}"
    pred_path = pdir / "predictions_log.csv"
    if not pred_path.is_file():
        raise SystemExit(f"Missing predictions: {pred_path}")
    dfp = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    col = f"pred_var_{model}"
    if col not in dfp.columns:
        raise SystemExit(f"Missing column {col} in {pred_path.name}")

    returns, asset_columns = load_data(build_data_if_missing=False)
    returns = returns * 100.0
    portfolio_returns = returns[asset_columns].mean(axis=1)

    cfg_feat = config.config["features"]
    eps = float(config.config["training"].get("eps", 1e-8))
    svd_window = int(cfg_feat["svd_window"])
    K = int(cfg_feat["K"])
    crisis_thresholds = list(cfg_feat["crisis_thresholds"])
    cov_estimator_primary = str(cfg_feat.get("cov_estimator", "qis"))

    _ = load_vix_series()
    rv_df = fe.build_har_rv(portfolio_returns, cfg_feat["rv_windows"])
    svd_df, _cov_series = fe.build_svd_features_panel_with_cov(
        returns, asset_columns, svd_window, K, crisis_thresholds, eps, cov_estimator=cov_estimator_primary
    )
    common = rv_df.index.intersection(svd_df.index).dropna()
    svd_df = svd_df.reindex(common)
    svd_df_h = fe.smooth_svd_features(svd_df, h)

    test_dates = pd.DatetimeIndex(dfp.index)
    gate = svd_df_h.reindex(test_dates)[gate_name].astype(float).values
    pred_var = dfp[col].astype(float).values
    r_next = portfolio_returns.reindex(test_dates).shift(-1).values.astype(np.float64)

    ok = np.isfinite(gate) & np.isfinite(pred_var) & np.isfinite(r_next)
    test_dates = test_dates[ok]
    gate = gate[ok]
    pred_var = pred_var[ok]
    r_next = r_next[ok]

    min_train = int(args.min_train)
    refit_every = max(1, int(args.refit_every))
    alpha = float(args.alpha)
    dist = str(args.dist)
    df = float(args.df) if dist == "student_t" else None

    out = np.full_like(pred_var, np.nan, dtype=np.float64)
    params = None
    last_refit_i = -10**9
    for i in range(len(pred_var)):
        if i < min_train:
            continue
        if params is None or (i - last_refit_i) >= refit_every:
            params = tc.fit_two_regime_scale_fz0(
                r_next[:i],
                pred_var[:i],
                gate=gate[:i],
                gate_name=gate_name,
                alpha=alpha,
                q_high=0.8,
                dist=dist,  # type: ignore[arg-type]
                df=df,
            )
            last_refit_i = i
        out[i] = tc.apply_two_regime_scale(pred_var[i : i + 1], gate=gate[i : i + 1], params=params)[0]

    out_col = f"pred_var_{model}_TAILCAL_{gate_name}"
    out_df = pd.DataFrame({out_col: out}, index=test_dates)
    out_path = pdir / f"predictions_tailcal_{model}_{gate_name}.csv"
    out_df.to_csv(out_path, index=True)

    mask = np.isfinite(out)
    bt, _ser = econ.backtest_var_es(
        r_next[mask],
        out[mask],
        alpha=alpha,
        dist=dist,  # type: ignore[arg-type]
        df=df,
    )
    summ = {
        "protocol": protocol,
        "horizon": h,
        "model": model,
        "gate": gate_name,
        "n": int(mask.sum()),
        "phat": bt.phat,
        "kupiec_p": bt.kupiec_p,
        "christoffersen_p": bt.christoffersen_p,
        "mean_fz0": bt.mean_fz0,
    }
    (pdir / f"tailcal_summary_{model}_{gate_name}.json").write_text(
        pd.Series(summ).to_json(), encoding="utf-8"
    )
    print(f"[OK] wrote {out_path} and tailcal_summary_{model}_{gate_name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
