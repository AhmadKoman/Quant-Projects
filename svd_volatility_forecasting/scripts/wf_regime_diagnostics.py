#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regime-sliced diagnostics for walk-forward outputs."""

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
import evaluation as ev
import features as fe
from evaluation import economic as econ
from run_experiments import build_target_rv, load_data, load_vix_series


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _quantile_bins(x: pd.Series, q: int) -> pd.Series:
    xx = x.astype(float)
    if xx.notna().sum() < max(q * 20, 200):
        return pd.Series(index=xx.index, data=np.nan)
    try:
        return pd.qcut(xx, q=q, labels=False, duplicates="drop").astype(float)
    except Exception:
        return pd.Series(index=xx.index, data=np.nan)


def _compute_point_metrics(y_true_var: np.ndarray, pred_var: np.ndarray, *, eps: float) -> dict:
    rmse, r2, male, qlike = ev.compute_metrics(y_true_var, pred_var, eps=eps)
    return {"RMSE": float(rmse), "R2": float(r2), "MALE": float(male), "QLIKE": float(qlike)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="expanding", choices=["expanding", "rolling"])
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 5, 22])
    ap.add_argument(
        "--root",
        type=Path,
        default=RESULTS_DIR / "walk_forward",
        help="Walk-forward output root directory.",
    )
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument(
        "--regime-vars",
        type=str,
        default="AR,delta_AR_z,subspace_dist_K,gap12_norm",
    )
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--dist", type=str, default="gaussian", choices=["gaussian", "student_t"])
    ap.add_argument("--df", type=float, default=8.0)
    args = ap.parse_args()

    protocol = str(args.protocol)
    h = int(args.horizon)
    bins = int(args.bins)
    if bins < 2:
        raise SystemExit("--bins must be >= 2.")

    pdir = Path(args.root) / protocol / f"h{h}"
    pred_path = pdir / "predictions_log.csv"
    if not pred_path.is_file():
        raise SystemExit(f"predictions_log.csv not found: {pred_path}")
    out_dir = pdir / "diagnostics"
    _ensure_dir(out_dir)

    dfp = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    if "y_true_var" not in dfp.columns:
        raise SystemExit("predictions_log.csv missing y_true_var.")
    y_true = dfp["y_true_var"].values.astype(np.float64)

    pred_var_cols = sorted(c for c in dfp.columns if c.startswith("pred_var_"))
    model_names = [c.replace("pred_var_", "") for c in pred_var_cols]
    if not model_names:
        raise SystemExit("No pred_var_* columns found.")

    returns, asset_columns = load_data(build_data_if_missing=False)
    returns = returns * 100.0
    portfolio_returns = returns[asset_columns].mean(axis=1)

    cfg_feat = config.config["features"]
    cfg_train = config.config["training"]
    eps = float(cfg_train.get("eps", 1e-8))
    svd_window = int(cfg_feat["svd_window"])
    K = int(cfg_feat["K"])
    crisis_thresholds = list(cfg_feat["crisis_thresholds"])
    cov_estimator_primary = str(cfg_feat.get("cov_estimator", "qis"))

    _ = load_vix_series()
    rv_df = fe.build_har_rv(portfolio_returns, cfg_feat["rv_windows"])
    svd_df, _cov_series = fe.build_svd_features_panel_with_cov(
        returns,
        asset_columns,
        svd_window,
        K,
        crisis_thresholds,
        eps,
        cov_estimator=cov_estimator_primary,
    )
    common = rv_df.index.intersection(svd_df.index).dropna()
    rv_df = rv_df.reindex(common).ffill()
    svd_df = svd_df.reindex(common)
    yv = build_target_rv(portfolio_returns.reindex(common), h).reindex(common)
    svd_df_h = fe.smooth_svd_features(svd_df, h)

    test_dates = pd.DatetimeIndex(dfp.index)
    state = svd_df_h.reindex(test_dates)
    yv_oos = yv.reindex(test_dates).values.astype(np.float64)
    if not np.isfinite(yv_oos).all():
        m = np.isfinite(yv_oos)
        dfp = dfp.loc[test_dates[m]]
        test_dates = pd.DatetimeIndex(dfp.index)
        y_true = dfp["y_true_var"].values.astype(np.float64)
        state = state.reindex(test_dates)

    r_next = portfolio_returns.reindex(test_dates).shift(-1).values.astype(np.float64)
    ok_r = np.isfinite(r_next) & np.isfinite(y_true)
    dfp = dfp.loc[test_dates[ok_r]]
    test_dates = pd.DatetimeIndex(dfp.index)
    y_true = dfp["y_true_var"].values.astype(np.float64)
    r_next = portfolio_returns.reindex(test_dates).shift(-1).values.astype(np.float64)
    state = state.reindex(test_dates)

    regime_vars = [s.strip() for s in str(args.regime_vars).split(",") if s.strip()]
    if not regime_vars:
        raise SystemExit("--regime-vars must be non-empty.")
    regime_bins: dict[str, pd.Series] = {}
    for rv in regime_vars:
        if rv not in state.columns:
            raise SystemExit(f"Regime var {rv!r} not found in SVD state columns.")
        regime_bins[rv] = _quantile_bins(state[rv], bins)

    rows: list[dict] = []
    for rv, b in regime_bins.items():
        for bi in range(int(np.nanmax(b.values)) + 1 if np.isfinite(np.nanmax(b.values)) else 0):
            mask = (b.values == bi) & np.isfinite(y_true) & np.isfinite(r_next)
            if int(mask.sum()) < 200:
                continue

            y_bin = y_true[mask]
            r_bin = r_next[mask]
            for m in model_names:
                pv = dfp.get(f"pred_var_{m}")
                if pv is None:
                    continue
                pred_bin = np.asarray(pv.values, dtype=np.float64)[mask]
                met = _compute_point_metrics(y_bin, pred_bin, eps=eps)
                bt, _series = econ.backtest_var_es(
                    r_bin,
                    pred_bin,
                    alpha=float(args.alpha),
                    dist=str(args.dist),  # type: ignore[arg-type]
                    df=float(args.df) if str(args.dist) == "student_t" else None,
                    mu=0.0,
                    eps=1e-12,
                )
                rows.append(
                    {
                        "protocol": protocol,
                        "horizon": h,
                        "regime_var": rv,
                        "bin": int(bi),
                        "n": int(mask.sum()),
                        "model": m,
                        **met,
                        "alpha": bt.alpha,
                        "dist": bt.dist,
                        "df": bt.df,
                        "phat": bt.phat,
                        "kupiec_p": bt.kupiec_p,
                        "christoffersen_p": bt.christoffersen_p,
                        "passes_kupiec": bt.passes_kupiec,
                        "passes_christoffersen": bt.passes_christoffersen,
                        "mean_fz0": bt.mean_fz0,
                    }
                )

    if not rows:
        raise SystemExit("No regime rows computed (bins too small / too many NaNs).")

    out = pd.DataFrame(rows).sort_values(["regime_var", "bin", "mean_fz0", "QLIKE"])
    out_path = out_dir / f"wf_regime_diagnostics_{protocol}_h{h}.csv"
    out.to_csv(out_path, index=False)

    summary: dict = {}
    for (rv, bi), g in out.groupby(["regime_var", "bin"]):
        best_qlike = g.sort_values("QLIKE").iloc[0].to_dict()
        best_fz0 = g.sort_values("mean_fz0").iloc[0].to_dict()
        summary[f"{rv}/bin{int(bi)}"] = {
            "n": int(g["n"].iloc[0]),
            "best_qlike": {k: best_qlike[k] for k in ["model", "QLIKE", "RMSE", "R2", "MALE"]},
            "best_fz0": {k: best_fz0[k] for k in ["model", "mean_fz0", "phat", "kupiec_p", "christoffersen_p"]},
        }
    (out_dir / f"wf_regime_diagnostics_{protocol}_h{h}.json").write_text(
        pd.Series(summary).to_json(), encoding="utf-8"
    )

    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
