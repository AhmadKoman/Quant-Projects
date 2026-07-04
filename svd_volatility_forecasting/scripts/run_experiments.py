#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Revised pipeline for the SVD volatility forecasting paper — SOTA Fix edition.

Implements the full 8-model experimental design:
    M1  HAR (OLS, 5-6 features: log_RSV_d_minus, log_RSV_d_plus, log_RV_w,
             log_RV_10d, log_RV_m [+ log_vix if available])
    M2  HAR+SVD (ElasticNetCV with TimeSeriesSplit, 12-14 features)
    M3  DNN (HAR-only features, [64,32,16] architecture, QLIKE loss)
    M4  DNN+SVD (HAR+SVD features, wider [128,64,32] + 5-seed ensemble,
                 QLIKE loss + HAR initialization)
    M5  LSTM (HAR-only features, StandardScaler inputs + QLIKE loss)
    M6  LSTM+SVD (HAR+SVD features, 5-seed ensemble, StandardScaler + QLIKE loss)
    M7  HARNet/TCN (HAR+SVD features, QLIKE loss)
    M8  GNN (per-asset node features + covariance graph)

Plus:
    GARCH(1,1) benchmark (h=1 via rolling 1-step; h=5,22 via multi-step formula)
    IV baseline (VIX daily vol)

SOTA Fix improvements (addressing systematic under-prediction, MZ beta 1.4–2.1):
    TARGET: Variance in %-squared units (r^2 where r = 100*log_return).
        - Eliminates near-zero log issues with eps=1e-8 floor
        - Enables direct comparison with BPQ 2016 (in-sample R² ~0.52)
        - log(variance) range [-5, 2] vs [-12, -9] for decimal volatility
    FEATURES:
        - Returns scaled to % at load time (returns * 100)
        - HAR features: variance (r^2) not volatility (|r|)
        - Semivariance (RSV-, RSV+) replaces single RV_d feature
        - Cross-sectional dispersion (CSD) added to SVD feature set
        - Turbulence index + correlation surprise added to SVD feature set
        - delta_f1 (change in PC1 variance share) added to SVD features
        - Eigenvector orientation fixed (sign-flip enforced for consistency)
        - VIX as HAR-X regressor (log_vix appended to HAR features)
    LOSS: QLIKE (Patton 2011) replaces Huber — learns conditional mean not median
    INITIALIZATION: DNN warm-started from HAR predictions (HAR init, 5 MSE epochs)
    LINEAR: Optional Orthogonalized Spectral Increment (OSI) two-stage ElasticNet (config).
    BIAS CORRECTION: Duan (1983) smearing on the train-minus-validation slice (no val leakage
        into smearing factor); same index for neural nets as for M1/M2.

Multi-horizon: h in {1, 5, 22} (direct forecasting; separate model per horizon).
Bayesian / empirical uncertainty: MC Dropout (M4, M6); conjugate Bayesian OLS on the
    M1 training sample (HAR); block-bootstrap bands for M2 ElasticNet (F1 HAR+SVD).
All models trained on log(variance) target; metrics computed in variance space.
Optional log(VIX) as HAR-X when VIX data exist; IV_baseline is a separate implied-variance benchmark.
Strict no-leakage: chronological split, scalers on train only; no backward bfill on the time index.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT
SRC = ROOT / "src"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

import config
import evaluation as ev
import features as fe
from evaluation import feature_ablation as feat_abl
from models import dnn as dnn_models
from models import garch as garch_models
from models import harnet as harnet_models
from models import gnn as gnn_models
from models import linear as linear_models
from models import lstm as lstm_models
from evaluation import bayesian as bayes
from walk_forward import engine as wfe
from walk_forward import splits as wf
from walk_forward.profile import resolve_walk_forward_config
import experiment_checkpoint as eckpt
from evaluation import tail_calibration as tailcal
from evaluation import economic as econ
from transforms import pred_var_from_log_raw, smearing_corrected_pred

_MIN = config.config["training"]
MIN_ASSETS = _MIN.get("min_assets", 50)
MIN_VALID_ROWS = _MIN.get("min_valid_rows", 500)
MIN_ALIGNED_PER_HORIZON = _MIN.get("min_aligned_per_horizon", 300)

# Deterministic sub-seeds for ensemble (derived from master seed 42)
ENSEMBLE_SUB_SEEDS = [42, 1337, 2024, 314, 999]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _run_fetch_stocks() -> None:
    """Run data/fetch_stocks.py to create returns_100.csv. No fallback."""
    path = config.config["data"]["returns_cache_path"]
    if path.exists():
        return
    script = REPO_ROOT / "data" / "fetch_stocks.py"
    if not script.exists():
        raise FileNotFoundError(
            "data/returns_100.csv not found and data/fetch_stocks.py not found. "
            "Cannot run without 100-stock data."
        )
    print("[INFO] data/returns_100.csv not found. Running data/fetch_stocks.py ...")
    import subprocess
    out = subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT))
    if out.returncode != 0:
        raise RuntimeError("data/fetch_stocks.py failed.")
    if not path.exists():
        raise FileNotFoundError("data/fetch_stocks.py did not create data/returns_100.csv.")


def _run_fetch_vix() -> None:
    """Run data/fetch_vix.py to create vix_daily_vol.csv for IV baseline."""
    vix_path = config.config["data"].get("vix_path")
    if vix_path is None:
        return
    vix_path = Path(vix_path)
    if vix_path.exists():
        return
    script = REPO_ROOT / "data" / "fetch_vix.py"
    if not script.exists():
        print("[WARN] data/fetch_vix.py not found; IV baseline will be skipped.")
        return
    print("[INFO] VIX data not found. Running data/fetch_vix.py ...")
    import subprocess
    out = subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT))
    if out.returncode != 0:
        print("[WARN] data/fetch_vix.py failed; IV baseline will be skipped.")


def load_data(*, build_data_if_missing: bool = False):
    """Load 100-stock log-returns from data/returns_100.csv only (no fallback)."""
    path = config.config["data"]["returns_cache_path"]
    if not path.exists():
        if build_data_if_missing:
            _run_fetch_stocks()
            _run_fetch_vix()
        else:
            raise FileNotFoundError(
                "data/returns_100.csv not found. Run: python data/fetch_stocks.py\n"
                "Or: python run_experiments.py --build-data"
            )
    if not path.exists():
        raise FileNotFoundError("data/returns_100.csv still missing after build step.")
    returns = pd.read_csv(path, index_col=0, parse_dates=True)
    if returns.shape[1] < MIN_ASSETS:
        raise ValueError(
            f"Returns panel has only {returns.shape[1]} assets; need >= {MIN_ASSETS}."
        )
    if len(returns) < MIN_VALID_ROWS:
        raise ValueError(
            f"Returns panel has only {len(returns)} rows; need >= {MIN_VALID_ROWS}."
        )
    return returns, returns.columns.tolist()


def build_target_rv(portfolio_returns: pd.Series, horizon: int) -> pd.Series:
    """
    Forward-looking realized VARIANCE over the next `horizon` days (%-squared).

    RV_t(h) = (1/h) * sum_{i=1}^{h} r_{t+i}^2

    Returns are in %-units (already scaled by 100 at load time), so the
    target is in %-squared units (e.g., 0.01 to 5.0 range typical for equities).

    Using variance (not volatility/sqrt) eliminates the Jensen's inequality bias:
        E[sqrt(X)] < sqrt(E[X])  (concavity)
    and allows direct exp(pred_log) to recover predicted variance.

    Shift(-h) ensures y[t] uses returns t+1...t+h (strict no-look-ahead).
    Consistent with the build_har_rv() feature construction.
    """
    r = portfolio_returns
    if horizon == 1:
        return (r ** 2).shift(-1)
    return (r ** 2).rolling(window=horizon).mean().shift(-horizon)


def is_crisis_date(date, crisis_windows: list) -> bool:
    """Return True if `date` falls inside any (start, end) window."""
    if hasattr(date, "date"):
        d = date.date()
    else:
        d = pd.Timestamp(date).date()
    for start, end in crisis_windows:
        if pd.Timestamp(start).date() <= d <= pd.Timestamp(end).date():
            return True
    return False


def load_vix_series():
    """Load VIX daily vol series. Return None if not configured or file missing."""
    vix_path = config.config["data"].get("vix_path")
    if vix_path is None:
        return None
    path = Path(vix_path)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty or df.shape[1] < 1:
        return None
    return df.iloc[:, 0].squeeze()

# ---------------------------------------------------------------------------
# GNN data assembly helpers
# ---------------------------------------------------------------------------

def _build_gnn_data(
    align_idx: pd.Index,
    cov_series: list,
    returns: pd.DataFrame,
    asset_columns: list,
    svd_df: pd.DataFrame,
    adj_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-timestep adjacency and node feature arrays for GNN.
    Returns node_features (T, N, 4) and adjacencies (T, N, N), both float32.
    """
    N = len(asset_columns)
    T = len(align_idx)
    cov_dict = {date: C for date, C in cov_series}
    fallback_adj = np.eye(N, dtype=np.float32)

    node_features = gnn_models.build_node_features(
        returns=returns,
        asset_columns=asset_columns,
        dates=align_idx,
        svd_features=svd_df,
    )  # (T, N, 4)

    adjacencies = np.empty((T, N, N), dtype=np.float32)
    for i, dt in enumerate(align_idx):
        if dt in cov_dict:
            adjacencies[i] = gnn_models.build_adjacency(
                cov_dict[dt], threshold=adj_threshold
            )
        else:
            adjacencies[i] = fallback_adj

    return node_features, adjacencies


# ---------------------------------------------------------------------------
# Ensemble helper
# ---------------------------------------------------------------------------

def _train_dnn_ensemble(
    sub_seeds: list,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    input_dim: int,
    cfg_dnn: dict,
    hidden_dims: list,
    har_init_preds: np.ndarray | None = None,
    use_gate: bool = False,
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Train DNN with multiple seeds and return ensemble-mean test and train predictions.

    har_init_preds : HAR log-variance predictions on X_tr for HAR initialization.
        If provided, each ensemble member is pre-trained to match HAR before
        switching to QLIKE. This stabilises training significantly.
    use_gate : If True, builds DNN with a learned FeatureGate layer at the input.
        Gate weights can then be extracted via dnn_models.get_gate_weights().

    Returns
    -------
    mean_test_pred_log  : ensemble-mean log-variance predictions on X_test
    mean_train_pred_log : ensemble-mean log-variance predictions on X_tr
        Used for Duan smearing correction — must use the same ensemble mean as
        test predictions so residuals are consistent (avoids single-seed bias).
    last_model          : last trained model, kept for MC Dropout.
    """
    preds_test_list = []
    preds_train_list = []
    last_model = None
    master_seed = config.config["seed"]
    for s in sub_seeds:
        tf.random.set_seed(s)
        np.random.seed(s)
        model, _ = dnn_models.train_dnn(
            X_tr, y_tr, X_val, y_val,
            input_dim=input_dim,
            epochs=cfg_dnn["epochs"],
            batch_size=cfg_dnn["batch_size"],
            patience=cfg_dnn["patience"],
            hidden_dims=hidden_dims,
            dropout=cfg_dnn["dropout"],
            l2_reg=cfg_dnn["l2_reg"],
            lr=cfg_dnn["learning_rate"],
            init_har_preds=har_init_preds,
            use_gate=use_gate,
        )
        preds_test_list.append(model.predict(X_test, verbose=0).ravel())
        # Predict on full training set (X_tr) for smearing correction
        preds_train_list.append(model.predict(X_tr, verbose=0).ravel())
        last_model = model
    # Restore master seed
    set_seeds(master_seed)
    mean_test = np.stack(preds_test_list).mean(axis=0)
    mean_train = np.stack(preds_train_list).mean(axis=0)
    return mean_test, mean_train, last_model


def _train_lstm_ensemble(
    sub_seeds: list,
    X_tr_2d: np.ndarray,
    y_tr_2d: np.ndarray,
    X_val_2d: np.ndarray,
    y_val_2d: np.ndarray,
    X_test_seq: np.ndarray,
    n_features: int,
    cfg_lstm: dict,
    har_init_preds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Train LSTM with multiple seeds; return ensemble-mean test and train-sequence
    log predictions (for Duan smearing parity with _train_dnn_ensemble / M4).

    har_init_preds : HAR log-variance predictions on X_tr_2d rows (non-sequenced).
        If provided, each ensemble member is pre-initialized with MSE warm-start
        against HAR predictions before entering the composite-loss training phase.
    """
    seq_len = cfg_lstm["seq_len"]
    X_tr_seq, y_tr_seq = lstm_models.build_sequences(X_tr_2d, y_tr_2d, seq_len)
    if len(X_tr_seq) == 0:
        raise ValueError("LSTM ensemble: empty train sequences for smearing path.")
    preds_test_list = []
    preds_train_seq_list = []
    last_model = None
    master_seed = config.config["seed"]
    for s in sub_seeds:
        tf.random.set_seed(s)
        np.random.seed(s)
        model, _ = lstm_models.train_lstm(
            X_tr_2d, y_tr_2d, X_val_2d, y_val_2d,
            seq_len=seq_len,
            n_features=n_features,
            epochs=cfg_lstm["epochs"],
            batch_size=cfg_lstm["batch_size"],
            patience=cfg_lstm["patience"],
            hidden_size=cfg_lstm["hidden_size"],
            dropout=cfg_lstm["dropout"],
            recurrent_dropout=cfg_lstm["recurrent_dropout"],
            l2_reg=cfg_lstm["l2_reg"],
            lr=cfg_lstm["learning_rate"],
            init_har_preds=har_init_preds,
        )
        preds_test_list.append(model.predict(X_test_seq, verbose=0).ravel())
        preds_train_seq_list.append(model.predict(X_tr_seq, verbose=0).ravel())
        last_model = model
    set_seeds(master_seed)
    mean_test = np.stack(preds_test_list).mean(axis=0)
    mean_train_seq = np.stack(preds_train_seq_list).mean(axis=0)
    return mean_test, mean_train_seq, last_model


# ---------------------------------------------------------------------------
# DM test helpers
# ---------------------------------------------------------------------------

def _run_dm_tests(
    preds_dict: dict,
    y_true: np.ndarray,
    horizon: int,
    use_detailed: bool = False,
) -> pd.DataFrame:
    """
    Run full pairwise DM tests between all models in ``preds_dict``.

    Computes BOTH MSE-based and QLIKE-based DM statistics with the
    pre-registered horizon-aware Newey-West HAC bandwidth
    ``nlags(h, T) = max(h - 1, floor(1.5 T^{1/3}))`` (Andrews 1991 +
    West 1996 overlap floor; see ``methods/preregistration.md``).
    QLIKE is the proxy-robust primary loss (Patton 2011, JoE).

    Each row also carries the Harvey-Leybourne-Newbold (1997) finite-sample
    t-stat / p-value (``t_HLN`` / ``p_HLN``) and the Coroneo-Iacone (2020) /
    Kiefer-Vogelsang (2005) fixed-``b`` p-value (``p_fixed_b``) for
    small-sample inference.

    y_true : realized variance (variance, not vol — same units as pred^2)
    preds_dict : {model_name: predicted_variance_array}

    Returns DataFrame with columns:
        Horizon, Model1, Model2,
        DM_MSE, p_MSE,         <- squared-error loss differential
        DM_QLIKE, p_QLIKE      <- QLIKE loss differential (primary)
        and (when ``use_detailed``) HLN / fixed-b decompositions.
    """
    dm_rows = []
    dm_keys = list(preds_dict.keys())
    for i, m1_name in enumerate(dm_keys):
        for j, m2_name in enumerate(dm_keys):
            if i >= j:
                continue
            p1 = preds_dict[m1_name]
            p2 = preds_dict[m2_name]
            if p1 is None or p2 is None:
                continue
            n_min = min(len(y_true), len(p1), len(p2))
            if n_min < 10:
                continue
            yt = y_true[:n_min]
            p1a = p1[:n_min]
            p2a = p2[:n_min]

            # MSE loss differential (horizon-aware HAC bandwidth)
            e1 = (yt - p1a) ** 2
            e2 = (yt - p2a) ** 2
            valid_mse = np.isfinite(e1) & np.isfinite(e2)
            if valid_mse.sum() >= 10:
                dm_mse, p_mse = ev.dmw_test(
                    (e1 - e2)[valid_mse], nlags=None, horizon=int(horizon),
                )
                mse_det = ev.dmw_test_detailed(
                    (e1 - e2)[valid_mse], nlags=None, horizon=int(horizon),
                )
            else:
                dm_mse, p_mse = float("nan"), float("nan")
                mse_det = {
                    "d_bar": float("nan"), "V_NW": float("nan"), "T": 0,
                    "nlags": 0, "p_value_normal": float("nan"),
                    "t_HLN": float("nan"), "p_HLN": float("nan"),
                    "p_fixed_b": float("nan"), "b_fixed": float("nan"),
                }

            # QLIKE loss differential (proxy-robust; primary for variance forecasting)
            dm_qlike, p_qlike = ev.dmw_test_qlike(
                yt, p1a, p2a, nlags=None, horizon=int(horizon),
            )
            q_det = ev.dmw_test_qlike_detailed(
                yt, p1a, p2a, nlags=None, horizon=int(horizon),
            )

            row_dm = {
                "Horizon": horizon,
                "Model1": m1_name,
                "Model2": m2_name,
                "DM_MSE": dm_mse,
                "p_MSE": p_mse,
                "DM_QLIKE": dm_qlike,
                "p_QLIKE": p_qlike,
            }
            row_dm.update({
                "nlags_MSE": mse_det.get("nlags"),
                "nlags_QLIKE": q_det.get("nlags"),
                "MSE_t_HLN": mse_det.get("t_HLN"),
                "MSE_p_HLN": mse_det.get("p_HLN"),
                "MSE_p_fixed_b": mse_det.get("p_fixed_b"),
                "MSE_b_fixed": mse_det.get("b_fixed"),
                "QLIKE_t_HLN": q_det.get("t_HLN"),
                "QLIKE_p_HLN": q_det.get("p_HLN"),
                "QLIKE_p_fixed_b": q_det.get("p_fixed_b"),
                "QLIKE_b_fixed": q_det.get("b_fixed"),
            })
            if use_detailed:
                row_dm.update({
                    "MSE_d_bar": mse_det.get("d_bar"),
                    "MSE_V_NW": mse_det.get("V_NW"),
                    "MSE_T": mse_det.get("T"),
                    "MSE_p_normal": mse_det.get("p_value_normal"),
                    "QLIKE_mean_L1": q_det.get("mean_L1"),
                    "QLIKE_mean_L2": q_det.get("mean_L2"),
                    "QLIKE_mean_d": q_det.get("mean_d"),
                    "QLIKE_d_bar": q_det.get("d_bar"),
                    "QLIKE_V_NW": q_det.get("V_NW"),
                    "QLIKE_T": q_det.get("T"),
                    "QLIKE_p_normal": q_det.get("p_value_normal"),
                })
            dm_rows.append(row_dm)
    return pd.DataFrame(dm_rows)


# Keys in h_preds that are not variance forecasts (GW instruments, dates, etc.)
_PRED_DICT_SKIP_KEYS = frozenset({
    "true_vol", "test_dates", "cos_theta", "angle_test", "spectral_gap_test",
    "_h1_train_dates_for_gw",  # metadata for GW z-score moments, not forecasts
})


# ---------------------------------------------------------------------------
# Main experiment pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="SVD volatility pipeline (8 models, multi-horizon, Bayesian uncertainty)"
    )
    parser.add_argument(
        "--build-data",
        action="store_true",
        help="Run data/fetch_stocks.py if returns_100.csv is missing",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run strict walk-forward OOS evaluation and exit (default profile: headline).",
    )
    parser.add_argument(
        "--walk-forward-profile",
        type=str,
        default=None,
        choices=["headline", "full", "svd_fix"],
        help="Walk-forward model/protocol profile (default: config walk_forward.profile).",
    )
    parser.add_argument(
        "--walk-forward-enable-xai",
        action="store_true",
        help="Force in-loop XAI on refit steps (e.g. post-hoc interpretability run).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore on-disk checkpoints and re-run completed jobs.",
    )
    args = parser.parse_args()
    force_rerun = bool(getattr(args, "force_rerun", False))

    set_seeds(config.config["seed"])
    print(f"[INFO] Seeds set to {config.config['seed']}")

    def to_serializable(obj):
        """Recursive JSON helper used by walk-forward early-exit and full pipeline."""
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # -----------------------------------------------------------------------
    # 1. Data loading
    # -----------------------------------------------------------------------
    returns, asset_columns = load_data(build_data_if_missing=args.build_data)
    _run_fetch_vix()
    vix_series = load_vix_series()
    if vix_series is not None:
        print(f"[INFO] VIX daily vol loaded, length {len(vix_series)}")
    else:
        print("[INFO] No VIX data; IV baseline will be skipped.")

    # Scale returns to percentage units (100 * log-return).
    # This switches the target from decimal volatility (~0.001-0.02) to %-squared
    # variance (~0.01-5.0), consistent with BPQ 2016 / ABDL 2003 convention.
    # Benefits: eliminates near-zero log issues with eps=1e-8; log(variance) in
    # range [-5, 2] rather than [-12, -9]; enables direct R² comparison with
    # published HAR results (~0.52 in-sample for S&P 500).
    returns = returns * 100.0
    print("[INFO] Returns scaled to % units (returns * 100); target = variance in %-squared.")

    crisis_windows = config.config.get("regime", {}).get("crisis_windows", [])
    N = len(asset_columns)
    portfolio_returns = returns[asset_columns].mean(axis=1)
    print(
        f"[INFO] Returns shape {returns.shape}, N={N}, "
        f"N/M={N}/{config.config['features']['svd_window']}="
        f"{N/config.config['features']['svd_window']:.2f}"
    )

    # -----------------------------------------------------------------------
    # 2. Feature engineering
    # -----------------------------------------------------------------------
    cfg_feat = config.config["features"]
    cfg_train = config.config["training"]
    cfg_dnn = config.config["models"]["dnn"]
    cfg_lstm = config.config["models"]["lstm"]
    cfg_harnet = config.config["models"]["harnet"]
    cfg_gnn = config.config["models"]["gnn"]
    cfg_garch_bm = config.config.get("models", {}).get("garch_benchmark", {})
    mc_T = config.config["models"]["mc_dropout_samples"]

    svd_window = cfg_feat["svd_window"]
    K = cfg_feat["K"]
    crisis_thresholds = cfg_feat["crisis_thresholds"]
    default_threshold = cfg_feat["default_crisis_threshold"]
    rv_windows = cfg_feat["rv_windows"]  # now includes "biweekly": 10
    horizons = cfg_train["horizons"]
    eps = cfg_train["eps"]

    rv_df = fe.build_har_rv(portfolio_returns, rv_windows)

    # Use the extended builder that returns covariance series for GNN.
    # ``cov_estimator`` defaults to QIS (LW-2020) per pre-registered spec.
    cov_estimator_primary = str(cfg_feat.get("cov_estimator", "qis"))
    svd_df, cov_series = fe.build_svd_features_panel_with_cov(
        returns, asset_columns, svd_window, K, crisis_thresholds, eps,
        cov_estimator=cov_estimator_primary,
    )
    print(
        f"[INFO] Covariance series built: {len(cov_series)} entries "
        f"(estimator='{cov_estimator_primary}', t >= {svd_window})"
    )

    # Build cross-sectional features (all use already-%-scaled returns)
    print("[INFO] Building semivariance, CSD, and turbulence features ...")
    semi_df = fe.build_semivariance_features(portfolio_returns, rv_windows)
    csd_df = fe.build_csd_features(returns[asset_columns], rv_windows)
    turb_df = fe.build_turbulence_index(returns[asset_columns], cov_series)
    n_turb_valid = int(turb_df["turb_lag1"].notna().sum())
    print(
        f"[INFO] Semivariance: {semi_df.shape[1]} cols; "
        f"CSD: {csd_df.shape[1]} cols; "
        f"Turbulence: {n_turb_valid} valid rows."
    )

    # Align indices: HAR features, SVD features, portfolio returns
    common = rv_df.index.intersection(svd_df.index).dropna()
    rv_df = rv_df.reindex(common).ffill()
    svd_df = svd_df.reindex(common)
    rv_df = rv_df.dropna(how="all")
    common = rv_df.index.intersection(svd_df.index)
    rv_df = rv_df.loc[common]
    svd_df = svd_df.loc[common]

    # Align cross-sectional features to common index
    semi_df = semi_df.reindex(common)
    csd_df = csd_df.reindex(common)
    turb_df = turb_df.reindex(common)

    # Use the 4 core HAR columns for validity check (all must be non-NaN)
    har_core_cols = [c for c in ["RV_d", "RV_w", "RV_10d", "RV_m"] if c in rv_df.columns]
    valid = ~(rv_df[har_core_cols].isna().any(axis=1) | svd_df["f1"].isna())
    n_valid = int(valid.sum())
    print(f"[INFO] Valid rows for training: {n_valid}")
    if n_valid < MIN_VALID_ROWS:
        raise ValueError(
            f"Insufficient valid rows: {n_valid} < {MIN_VALID_ROWS}. "
            "Check data range and feature construction."
        )
    rv_df = rv_df.loc[valid]
    svd_df = svd_df.loc[valid]
    semi_df = semi_df.loc[valid]
    csd_df = csd_df.loc[valid]
    turb_df = turb_df.loc[valid]
    common = rv_df.index

    feat_cfg_global = config.config.get("features", {})
    inc_xs_primary = bool(feat_cfg_global.get("include_cross_section_primary", False))
    osi_enabled = bool(feat_cfg_global.get("osi_elastic_net_enabled", True))

    # -----------------------------------------------------------------------
    # Build tiered feature sets (VIX excluded from ALL model inputs).
    # VIX with coefficient ~0.86 would crowd out every SVD feature in a linear
    # model. Per Patton (2011), IV is retained as a standalone IV_baseline benchmark
    # only — never as a model input (see ASSUMPTIONS_FOR_MANUSCRIPT.md).
    #
    # Tier 0: HAR baseline  (semivariance + weekly/biweekly/monthly RV)
    # Tier 1: HAR + f1, log_sigma1       (eigenvalue-level features)
    # Tier 2: HAR + full SVD set         (+ AR, angle, delta_f1, crisis flag)
    # Tier 3: HAR + SVD + cross-section  (+ log_CSD_d, log_turb, corr_surprise)
    #
    # SVD tiers (1-3) are rebuilt per horizon using smooth_svd_features(svd_df, h).
    # This matches the HAR principle: at h=5, SVD features are 5-day averages;
    # at h=22, they are 22-day averages — same temporal scale as the target.
    # Tier 0 (HAR-only) is not horizon-specific and is built once here.
    # -----------------------------------------------------------------------
    X_har, har_names = fe.build_feature_sets(
        rv_df, svd_df, svd_tier=0, default_threshold=default_threshold, eps=eps,
        semi_df=semi_df,
    )
    print(f"[INFO] Tier-0 HAR set:       {len(har_names)} features -> {har_names}")
    print("[INFO] VIX excluded from all model feature vectors (IV_baseline benchmark only).")
    print("[INFO] SVD tiers will be rebuilt per horizon with horizon-matched smoothing.")

    n = len(common)
    primary_split = float(cfg_train.get("train_split", 0.7))
    _sens = cfg_train.get("train_split_sensitivity")
    split_list = [primary_split] if not _sens else [float(x) for x in _sens]
    cache_root = RESULTS_DIR / "checkpoints"
    split_sensitivity_metrics: dict[str, dict] = {}
    results_by_horizon: dict = {}
    all_predictions_by_horizon: dict[int, dict] = {}
    feature_names_m2_h1 = None  # Tier-2 names for h=1 primary (split-sample HAC)
    all_predictions: dict = {}
    raw_smeared_metrics_h1: dict | None = None

    # -----------------------------------------------------------------------
    # Walk-forward mode (journal-grade OOS). Runs and exits early.
    # -----------------------------------------------------------------------
    if bool(getattr(args, "walk_forward", False)):
        out_dir = RESULTS_DIR / "walk_forward"
        out_dir.mkdir(parents=True, exist_ok=True)

        walk_cfg = resolve_walk_forward_config(
            config.config.get("walk_forward", {}),
            profile_name=getattr(args, "walk_forward_profile", None),
        )
        if bool(getattr(args, "walk_forward_enable_xai", False)):
            walk_cfg["xai"] = {**dict(walk_cfg.get("xai", {})), "enabled": True}
        wf_fingerprint = eckpt.config_fingerprint(
            seed=config.config["seed"],
            horizons=list(horizons),
            returns_path=config.config["data"]["returns_cache_path"],
            extra={
                "mode": "walk_forward",
                "profile": str(walk_cfg.get("profile_active", "headline")),
                "osi": bool(config.config.get("features", {}).get("osi_elastic_net_enabled", True)),
            },
        )
        walk_cfg["experiment_fingerprint"] = wf_fingerprint
        protocols = list(walk_cfg.get("protocols", ["expanding", "rolling"]))
        if not protocols:
            raise ValueError("config.walk_forward.protocols must be non-empty.")
        wf_models = list(walk_cfg.get("models", []))
        need_gnn = "GNN" in wf_models

        wf_global_t0 = time.perf_counter()
        print(
            "[WF] ========== walk-forward ==========",
            flush=True,
        )
        print(
            f"[WF] profile={walk_cfg.get('profile_active')!r} | models={wf_models} | "
            f"horizons={list(horizons)} | protocols={protocols} | "
            f"tuning={walk_cfg.get('tuning_policy')} retune_every={walk_cfg.get('retune_every')} | "
            f"progress_log_every={walk_cfg.get('progress_log_every', 25)} | "
            f"ElasticNetCV n_jobs={walk_cfg.get('elastic_net_cv_n_jobs')!r}",
            flush=True,
        )
        xai_cfg = walk_cfg.get("xai", {})
        print(
            f"[WF] XAI enabled={xai_cfg.get('enabled', True)} "
            f"run_every_n_steps={xai_cfg.get('run_every_n_steps', 1)} "
            "(post-hoc: scripts/walk_forward_xai.py)",
            flush=True,
        )

        # Targets for each horizon (variance space and log space)
        y_var_by_h: dict[int, pd.Series] = {}
        y_log_by_h: dict[int, pd.Series] = {}
        for h in horizons:
            yv = build_target_rv(portfolio_returns.reindex(common), h).reindex(common)
            # Strict: no imputation for targets; drop NaNs later per-horizon.
            y_var_by_h[int(h)] = yv
            y_log_by_h[int(h)] = np.log(np.maximum(yv.values.astype(np.float64), eps))
            y_log_by_h[int(h)] = pd.Series(y_log_by_h[int(h)], index=common)

        # OSI policy for ElasticNet (Tier-2)
        use_osi = bool(config.config.get("features", {}).get("osi_elastic_net_enabled", True))
        # HAR column count in Tier-2 feature sets: semivariance(2) + RV_w + RV_10d + RV_m = 5
        # If RV_d is used instead (no semivariance), this must be updated accordingly.
        osi_har_col_count = 5

        # Build full-horizon feature matrices and run walk-forward per (protocol, horizon)
        for h in horizons:
            h = int(h)
            print(f"[WF] ----- horizon h={h} (of {list(horizons)}) -----", flush=True)
            svd_df_h = fe.smooth_svd_features(svd_df, h)

            X_t1, t1_names = fe.build_feature_sets(
                rv_df, svd_df_h, svd_tier=1, default_threshold=default_threshold, eps=eps,
                semi_df=semi_df, interaction_smooth_h=h,
            )
            X_t2, t2_names = fe.build_feature_sets(
                rv_df, svd_df_h, svd_tier=2, default_threshold=default_threshold, eps=eps,
                semi_df=semi_df, interaction_smooth_h=h,
            )
            # Dynamic spectral HAR block: lagged multi-scale spectral states.
            X_svdhar, svdhar_names = fe.build_svd_har_dynamics_block(svd_df_h)
            X_t2_dyn = pd.concat([X_har, X_svdhar], axis=1)
            t2_dyn_names = list(har_names) + list(svdhar_names)
            X_t3, t3_names = fe.build_feature_sets(
                rv_df, svd_df_h, svd_tier=3, default_threshold=default_threshold, eps=eps,
                semi_df=semi_df, csd_df=csd_df, turb_df=turb_df,
                interaction_smooth_h=h, include_cross_section=True,
            )

            # Strict index alignment
            if not (X_har.index.equals(common) and X_t1.index.equals(common) and X_t2.index.equals(common) and X_t3.index.equals(common)):
                raise ValueError("Walk-forward: feature indices must match common index.")

            # Drop rows where the target OR any model-input feature is NaN for
            # this horizon.  The global validity mask above only requires
            # ``f1`` and the HAR core columns to be finite, but the RMT-aware
            # spectral features added in features.extract_svd_features
            # (``delta_AR_z`` with a 252-day trailing window + shift(1),
            # ``subspace_dist_K`` whose first observation is NaN by
            # construction, and ``gap12_norm`` when ``lambda_1 = 0``) carry
            # additional warm-up NaN; smooth_svd_features further widens the
            # warm-up at h=5, 22 by rolling-mean + shift(1).  sklearn refuses
            # NaN in fit() (see e.g. orthogonalize_spectral_on_har ->
            # LinearRegression.fit), so we require finiteness across every
            # active feature tier here rather than relying on per-call
            # try/except suppression.  ``Z_fit``/``S_fit`` therefore stay
            # row-aligned without any leakage from inside-window imputation.
            yv = y_var_by_h[h]
            yl = y_log_by_h[h]
            feat_finite = (
                np.isfinite(np.asarray(X_har.values, dtype=np.float64)).all(axis=1)
                & np.isfinite(np.asarray(X_t1.values, dtype=np.float64)).all(axis=1)
                & np.isfinite(np.asarray(X_t2.values, dtype=np.float64)).all(axis=1)
                & np.isfinite(np.asarray(X_t3.values, dtype=np.float64)).all(axis=1)
                & np.isfinite(np.asarray(X_t2_dyn.values, dtype=np.float64)).all(axis=1)
            )
            valid_h = np.isfinite(yv.values) & np.isfinite(yl.values) & feat_finite
            idx_h = common[valid_h]
            if len(idx_h) < (walk_cfg.get("initial_train_len", 0) + walk_cfg.get("val_len", 0) + 50):
                raise ValueError(f"Walk-forward h={h}: insufficient valid rows after target alignment.")
            n_target_only = int((np.isfinite(yv.values) & np.isfinite(yl.values)).sum())
            n_dropped_for_feats = n_target_only - int(valid_h.sum())
            if n_dropped_for_feats > 0:
                print(
                    f"[INFO] Walk-forward h={h}: dropped {n_dropped_for_feats} leading rows "
                    f"with feature warm-up NaN (delta_AR_z / subspace_dist_K / horizon smoothing); "
                    f"{len(idx_h)} usable rows remain."
                )

            Xhar_h = X_har.reindex(idx_h)
            Xt1_h = X_t1.reindex(idx_h)
            Xt2_h = X_t2.reindex(idx_h)
            Xt2_dyn_h = X_t2_dyn.reindex(idx_h)
            Xt3_h = X_t3.reindex(idx_h)
            yv_h = yv.reindex(idx_h)
            yl_h = yl.reindex(idx_h)

            node_features_full, adj_full = None, None
            if need_gnn:
                node_features_full, adj_full = _build_gnn_data(
                    align_idx=idx_h,
                    cov_series=cov_series,
                    returns=returns,
                    asset_columns=asset_columns,
                    svd_df=svd_df_h.reindex(idx_h),
                    adj_threshold=float(cfg_gnn["adj_threshold"]),
                )

            feature_names = {
                "HAR": list(har_names),
                "T1": list(t1_names),
                "T2": list(t2_names),
                "T3": list(t3_names),
            }

            X_by_model: dict[str, pd.DataFrame] = {}
            if "AR" in svd_df_h.columns:
                # Gate HAR+SVD on absorption ratio regimes (fragility state).
                har_like = [c for c in ["log_RSV_d_minus", "log_RSV_d_plus", "RV_w", "RV_10d", "RV_m", "RV_d"] if c in X_t2.columns]
                X_gated, gated_names = fe.add_regime_gating_interactions(
                    X_t2,
                    gate=svd_df_h["AR"].reindex(X_t2.index),
                    gate_name="AR",
                    base_cols=har_like,
                    q_high=0.8,
                )
                X_by_model["HAR+SVD_GATED_AR"] = X_gated
                feature_names["HAR+SVD_GATED_AR"] = gated_names
            X_by_model["HAR+SVD_DYN"] = Xt2_dyn_h
            feature_names["HAR+SVD_DYN"] = t2_dyn_names

            for protocol in protocols:
                protocol = str(protocol)
                if protocol not in ("expanding", "rolling"):
                    raise ValueError(f"Unknown walk-forward protocol: {protocol}")

                pdir = out_dir / protocol / f"h{h}"
                pdir.mkdir(parents=True, exist_ok=True)
                wf_profile = str(walk_cfg.get("profile_active", "headline"))
                n_expected = eckpt.expected_wf_oos_rows(
                    len(idx_h),
                    initial_train_len=int(walk_cfg["initial_train_len"]),
                    val_len=int(walk_cfg["val_len"]),
                    step=int(walk_cfg["step"]),
                )
                if eckpt.is_walk_forward_job_complete(
                    pdir,
                    protocol=protocol,
                    horizon=h,
                    profile=wf_profile,
                    fingerprint=wf_fingerprint,
                    expected_rows=n_expected,
                ) and not force_rerun:
                    print(
                        f"[SKIP] walk-forward {protocol} h={h} profile={wf_profile!r} "
                        f"({n_expected} OOS rows on disk)",
                        flush=True,
                    )
                    continue

                wf_out = wfe.run_walk_forward_for_horizon(
                    protocol=protocol,
                    horizon=h,
                    idx=idx_h,
                    y_true_var=yv_h,
                    y_true_log=yl_h,
                    X_har=Xhar_h,
                    X_t1=Xt1_h,
                    X_t2=Xt2_h,
                    X_t3=Xt3_h,
                    X_by_model={k: v.reindex(idx_h) for k, v in X_by_model.items()} if X_by_model else None,
                    node_features_full=node_features_full,
                    adj_full=adj_full,
                    cfg_models=config.config["models"],
                    walk_cfg=walk_cfg,
                    feature_names=feature_names,
                    use_osi=use_osi,
                    osi_har_col_count=osi_har_col_count,
                    checkpoint_dir=pdir,
                )
                print(
                    f"[WF] {protocol} h={h}: OOS tensor built (n={len(wf_out.y_true_var)} dates) — "
                    "GARCH/GJR + metrics + exports next (still this protocol/horizon)...",
                    flush=True,
                )

                # Save predictions in a strict schema
                dfp = pd.DataFrame({"y_true_var": wf_out.y_true_var, "y_true_log": wf_out.y_true_log})
                for m, s in wf_out.preds_log.items():
                    dfp[f"pred_log_{m}"] = s.reindex(dfp.index)
                for m, s in wf_out.preds_var.items():
                    dfp[f"pred_var_{m}"] = s.reindex(dfp.index)

                # IV baseline (VIX) in variance space (%-squared), if available
                if vix_series is not None:
                    vix_aligned = vix_series.reindex(dfp.index).ffill()
                    if np.isfinite(vix_aligned.values).sum() < 50:
                        raise ValueError("Walk-forward: VIX series present but insufficient finite values after alignment.")
                    iv_daily_vol_decimal = np.asarray(vix_aligned.values, dtype=float)
                    iv_var = (iv_daily_vol_decimal * 100.0) ** 2
                    dfp["pred_var_IV_baseline"] = iv_var
                    dfp["pred_log_IV_baseline"] = np.log(np.maximum(iv_var.astype(np.float64), eps))

                # GARCH-family benchmarks (variance forecasts in %-squared)
                g_refit = int(config.config.get("walk_forward", {}).get("refit_cadence", {}).get("GARCH", 1))
                gcfg = config.config.get("models", {}).get("garch_benchmark", {})
                innovations = str(gcfg.get("innovations", "gaussian"))
                t_df = float(gcfg.get("t_df", 8.0))
                # Match horizon using analytical multi-step formulas for h>1
                garch_var = garch_models.build_garch_multistep(
                    portfolio_returns.reindex(common).dropna(),
                    test_dates=dfp.index,
                    horizon=h,
                    innovations=innovations,
                    t_df=t_df,
                    refit_every=g_refit,
                )
                dfp["pred_var_GARCH"] = garch_var
                dfp["pred_log_GARCH"] = np.log(np.maximum(garch_var.astype(np.float64), eps))
                gjr_var = garch_models.build_gjr_garch_multistep(
                    portfolio_returns.reindex(common).dropna(),
                    test_dates=dfp.index,
                    horizon=h,
                    innovations="t",
                    t_df=t_df,
                    refit_every=g_refit,
                )
                dfp["pred_var_GJR-GARCH-t"] = gjr_var
                dfp["pred_log_GJR-GARCH-t"] = np.log(np.maximum(gjr_var.astype(np.float64), eps))
                dfp.to_csv(pdir / "predictions_log.csv", index=True)
                with open(pdir / "metadata.json", "w") as f:
                    json.dump(to_serializable(wf_out.metadata), f, indent=2)
                if wf_out.xai and "records" in wf_out.xai:
                    with open(pdir / "xai_permutation_importance.json", "w") as f:
                        json.dump(to_serializable(wf_out.xai["records"]), f, indent=2)

                # ------------------------------------------------------------
                # Walk-forward evaluation exports (metrics, DM, MCS)
                # ------------------------------------------------------------
                y_true = wf_out.y_true_var.values.astype(np.float64)
                pred_var_cols = sorted([c for c in dfp.columns if c.startswith("pred_var_")])
                model_names = [c.replace("pred_var_", "") for c in pred_var_cols]

                # Metrics table
                met_rows = []
                for m in model_names:
                    y_pred = dfp[f"pred_var_{m}"].values.astype(np.float64)
                    rmse, r2, male, qlike = ev.compute_metrics(y_true, y_pred, eps=eps)
                    met_rows.append({"model": m, "RMSE": rmse, "R2": r2, "MALE": male, "QLIKE": qlike})
                df_met = pd.DataFrame(met_rows).sort_values("QLIKE")
                df_met.to_csv(pdir / "metrics.csv", index=False)
                with open(pdir / "metrics.json", "w") as f:
                    json.dump(to_serializable({r["model"]: {k: r[k] for k in ["RMSE", "R2", "MALE", "QLIKE"]} for r in met_rows}), f, indent=2)

                # Loss matrices for MCS
                eps_q = ev.EPS_DEFAULT
                Y = np.maximum(y_true, eps_q)
                loss_qlike = []
                loss_mse = []
                for m in model_names:
                    P = np.maximum(dfp[f"pred_var_{m}"].values.astype(np.float64), eps_q)
                    ratio = Y / P
                    Lq = ratio - np.log(ratio) - 1.0
                    loss_qlike.append(Lq)
                    loss_mse.append((Y - P) ** 2)
                Lq_mat = np.vstack(loss_qlike).T  # (T, K)
                Lm_mat = np.vstack(loss_mse).T

                # MCS on QLIKE losses (primary)
                mcs = ev.compute_mcs(
                    loss_matrix=Lq_mat,
                    model_names=model_names,
                    alpha=0.10,
                    n_boot=int(config.config.get("inference", {}).get("optional_inference", {}).get("bootstrap_n_boot", 1999)),
                    block_size=int(wfe.compute_dm_nlags_for_horizon(h)),
                    seed=int(config.config.get("seed", 42)),
                )
                pd.DataFrame(
                    [{"model": mn, "included": bool(mcs["included"][mn]), "p_value": float(mcs["p_values"][mn])} for mn in model_names]
                ).to_csv(pdir / "mcs_qlike.csv", index=False)
                with open(pdir / "mcs_qlike.json", "w") as f:
                    json.dump(to_serializable(mcs), f, indent=2)

                # Pairwise DM tests (QLIKE + MSE) under the pre-registered
                # horizon-aware HAC bandwidth.
                T_dm = int(Lq_mat.shape[0])
                nlags_dm = int(wfe.compute_dm_nlags_for_horizon(h, T=T_dm))
                dm_rows_q = []
                dm_rows_m = []
                for i in range(len(model_names)):
                    for j in range(i + 1, len(model_names)):
                        m1, m2 = model_names[i], model_names[j]
                        p1 = dfp[f"pred_var_{m1}"].values.astype(np.float64)
                        p2 = dfp[f"pred_var_{m2}"].values.astype(np.float64)
                        det_q = ev.dmw_test_qlike_detailed(
                            Y, p1, p2, nlags=nlags_dm, eps=eps_q, horizon=int(h),
                        )
                        dm_rows_q.append({"m1": m1, "m2": m2, "nlags": nlags_dm, **det_q})
                        d_mse = (Y - p1) ** 2 - (Y - p2) ** 2
                        det_m = ev.dmw_test_detailed(
                            d_mse, nlags=nlags_dm,
                            alternative="two-sided", horizon=int(h),
                        )
                        dm_rows_m.append({"m1": m1, "m2": m2, "nlags": nlags_dm, **det_m})
                ev.dm_add_fdr_columns(pd.DataFrame(dm_rows_q)).to_csv(pdir / "dm_tests_qlike.csv", index=False)
                ev.dm_add_fdr_columns(pd.DataFrame(dm_rows_m)).to_csv(pdir / "dm_tests_mse.csv", index=False)

                # ------------------------------------------------------------
                # Economic evaluation (primary at h=1): VaR/ES + Vol Targeting
                # ------------------------------------------------------------
                if int(h) == 1:
                    # Align next-day realized return to forecast origin dates.
                    # Predictions at date t correspond to RV over (t+1); apply to r_{t+1}.
                    r_next_pct = portfolio_returns.reindex(common).shift(-1).reindex(dfp.index).values.astype(np.float64)
                    if not np.isfinite(r_next_pct).all():
                        raise ValueError("Walk-forward economic eval: non-finite next-day returns after alignment.")
                    # Work in DECIMAL returns for economic metrics; convert variance accordingly.
                    r_next_dec = r_next_pct / 100.0

                    econ_rows = []
                    econ_series = {}
                    for alpha_var in (0.01, 0.05):
                        for dist in ("gaussian", "student_t"):
                            df_t = 8.0 if dist == "student_t" else None
                            for m in model_names:
                                var_pct2 = dfp[f"pred_var_{m}"].values.astype(np.float64)
                                var_dec = var_pct2 / (100.0 ** 2)
                                res, series = econ.backtest_var_es(
                                    r_next_dec,
                                    var_dec,
                                    alpha=float(alpha_var),
                                    dist=dist,
                                    df=df_t,
                                    mu=0.0,
                                )
                                econ_rows.append({
                                    "model": m,
                                    "alpha": res.alpha,
                                    "dist": res.dist,
                                    "df": res.df,
                                    "n": res.n,
                                    "phat": res.phat,
                                    "kupiec_p": res.kupiec_p,
                                    "christoffersen_p": res.christoffersen_p,
                                    "passes_kupiec": res.passes_kupiec,
                                    "passes_christoffersen": res.passes_christoffersen,
                                    "mean_fz0": res.mean_fz0,
                                })
                                econ_series[f"{m}_a{alpha_var}_{dist}"] = {
                                    "VaR": series["VaR"],
                                    "ES": series["ES"],
                                    "violations": series["violations"],
                                    "FZ0": series["FZ0"],
                                }
                    pd.DataFrame(econ_rows).to_csv(pdir / "var_es_backtests.csv", index=False)
                    with open(pdir / "var_es_backtests.json", "w") as f:
                        json.dump(to_serializable(econ_rows), f, indent=2)
                    # Save only violation series to keep file size bounded and deterministic.
                    # Full series can be reconstructed from VaR/ES formulas if needed.
                    with open(pdir / "var_es_series_violations.json", "w") as f:
                        json.dump(to_serializable({k: v["violations"] for k, v in econ_series.items()}), f, indent=2)

                    # --- VaR/ES tournament (multi-alpha, multi-model, regime strata;
                    # Acerbi-Szekely Z2, DQ, Du-Escanciano, CP duration, FZ0-DM)
                    pred_var_dec_tourn = {
                        m: (dfp[f"pred_var_{m}"].values.astype(np.float64) / (100.0 ** 2))
                        for m in model_names
                    }
                    regime_lab = np.empty(len(dfp.index), dtype=object)
                    for ii, ts in enumerate(dfp.index):
                        ts_pd = pd.Timestamp(ts)
                        lbl = "Calm"
                        for wi, (ws, we) in enumerate(crisis_windows):
                            if pd.Timestamp(ws) <= ts_pd <= pd.Timestamp(we):
                                lbl = f"Crisis_{wi}"
                                break
                        regime_lab[ii] = lbl
                    z2_boot = int(
                        config.config.get("inference", {})
                        .get("optional_inference", {})
                        .get("var_es_z2_n_boot", 1999)
                    )
                    tour = econ.var_es_tournament(
                        returns_dec=r_next_dec,
                        pred_var_dec=pred_var_dec_tourn,
                        alphas=(0.005, 0.01, 0.025, 0.05),
                        distributions=(("gaussian", None), ("student_t", 8.0)),
                        regime_labels=regime_lab,
                        horizon=int(h),
                        fz0_baseline="HAR" if "HAR" in model_names else None,
                        z2_n_boot=z2_boot,
                        z2_seed=int(config.config["seed"]),
                    )
                    with open(pdir / "var_es_tournament.json", "w") as f:
                        json.dump(to_serializable(tour), f, indent=2)
                    # Flat summary for spreadsheet readers
                    flat_rows = []
                    for mname, by_dist in tour.get("per_model", {}).items():
                        for dkey, by_reg in by_dist.items():
                            for reg_name, stats_d in by_reg.items():
                                flat_rows.append({"model": mname, "dist_alpha_key": dkey,
                                                  "regime": reg_name, **stats_d})
                    if flat_rows:
                        pd.DataFrame(flat_rows).to_csv(pdir / "var_es_tournament_flat.csv", index=False)

                    # Volatility targeting (sensitivity over transaction costs)
                    vt_rows = []
                    vt_series = {}
                    for tc_bps in (0.0, 5.0, 10.0):
                        for m in model_names:
                            var_pct2 = dfp[f"pred_var_{m}"].values.astype(np.float64)
                            var_dec = var_pct2 / (100.0 ** 2)
                            vt_res, vt_ser = econ.volatility_targeting(
                                r_next_dec,
                                var_dec,
                                target_vol_annual=0.10,
                                max_leverage=3.0,
                                tc_bps=float(tc_bps),
                            )
                            vt_rows.append({
                                "model": m,
                                "tc_bps": float(tc_bps),
                                "n": vt_res.n,
                                "ann_return": vt_res.ann_return,
                                "ann_vol": vt_res.ann_vol,
                                "sharpe": vt_res.sharpe,
                                "max_drawdown": vt_res.max_drawdown,
                                "turnover": vt_res.turnover,
                            })
                            vt_series[f"{m}_tc{tc_bps}"] = {
                                "w": vt_ser["w"],
                                "rp": vt_ser["rp"],
                                "eq": vt_ser["eq"],
                                "drawdown": vt_ser["drawdown"],
                            }
                    pd.DataFrame(vt_rows).to_csv(pdir / "vol_targeting.csv", index=False)
                    with open(pdir / "vol_targeting.json", "w") as f:
                        json.dump(to_serializable(vt_rows), f, indent=2)
                    # Save equity curves only (compact)
                    with open(pdir / "vol_targeting_series_eq.json", "w") as f:
                        json.dump(to_serializable({k: v["eq"] for k, v in vt_series.items()}), f, indent=2)

                eckpt.mark_walk_forward_job_complete(
                    pdir,
                    protocol=protocol,
                    horizon=h,
                    profile=wf_profile,
                    fingerprint=wf_fingerprint,
                    n_rows=len(dfp),
                )

        print(
            f"[INFO] Walk-forward horizon h={h} finished → results under {out_dir} (all protocols)",
            flush=True,
        )
        print(
            f"[WF] all walk-forward horizons done — total wall { (time.perf_counter() - wf_global_t0) / 60.0 :.1f}m",
            flush=True,
        )
        return

    # -----------------------------------------------------------------------
    # 3. Horizon loop: train all 8 models (optional multi-split sensitivity)
    # -----------------------------------------------------------------------
    for train_split in split_list:
        is_primary = abs(float(train_split) - primary_split) < 1e-9
        print(f"\n[INFO] ========== train_split = {train_split} (primary={is_primary}) ==========")

        rb: dict = {}
        ap: dict[int, dict] = {}
        hist_dnn_h1 = None
        hist_lstm_h1 = None
        model_dnn_svd_h1 = None
        model_lstm_svd_h1 = None
        X_har_svd_h1_scaled_test = None
        X_har_seq_svd_test_h1 = None
        X_har_train_h1 = None
        y_train_h1 = None
        X_har_test_h1 = None
        m2_pipeline_h1 = None
        X_m2_train_boot = None
        y_m2_train_boot = None
        X_m2_test_boot = None

        exp_fp = eckpt.config_fingerprint(
            seed=config.config["seed"],
            horizons=list(horizons),
            train_split=float(train_split),
            returns_path=config.config["data"]["returns_cache_path"],
            extra={
                "mode": "fixed_split",
                "osi": bool(config.config.get("features", {}).get("osi_elastic_net_enabled", True)),
                "inc_xs": bool(config.config.get("features", {}).get("include_cross_section_primary", True)),
            },
        )

        if eckpt.all_fixed_split_horizons_complete(
            cache_root,
            train_split=float(train_split),
            horizons=list(horizons),
            fingerprint=exp_fp,
            force=force_rerun,
        ):
            rb_loaded, ap_loaded = eckpt.load_fixed_split_horizon_results(
                cache_root,
                train_split=float(train_split),
                horizons=list(horizons),
                fingerprint=exp_fp,
            )
            rb.update(rb_loaded)
            ap.update(ap_loaded)
            print(
                f"[SKIP] fixed_split train_split={train_split}: all horizons cached",
                flush=True,
            )
            split_sensitivity_metrics[str(train_split)] = rb
            if is_primary:
                results_by_horizon = rb
                all_predictions_by_horizon = ap
                if 1 in ap:
                    all_predictions = {k: v for k, v in ap[1].items()}
            continue

        for h in horizons:
            print(f"\n[INFO] === Horizon h={h} ===")
            fsc = eckpt.FixedSplitHorizonCache(
                cache_root,
                train_split=float(train_split),
                horizon=int(h),
                fingerprint=exp_fp,
                force=force_rerun,
            )
            if fsc.is_horizon_complete():
                pack = fsc.load_horizon_pack()
                if pack is not None:
                    rb[int(h)] = pack["rb"]
                    ap[int(h)] = fsc.restore_arrays(pack)
                    print(
                        f"[SKIP] fixed_split train_split={train_split} h={h} "
                        f"(horizon cache hit)",
                        flush=True,
                    )
                    if int(h) == 1 and is_primary:
                        all_predictions_by_horizon[1] = ap[1]
                        all_predictions = {k: v for k, v in ap[1].items()}
                    continue

            # Build horizon-matched SVD features using temporal smoothing.
            # smooth_svd_features applies rolling(h).mean().shift(1) to all
            # continuous SVD columns, matching the frequency of the h-step target.
            # At h=1 this is a no-op; at h=5/h=22 it eliminates the high-frequency
            # noise that caused SVD features to degrade at longer horizons.
            svd_df_h = fe.smooth_svd_features(svd_df, h)

            X_svd_t1, svd_t1_names = fe.build_feature_sets(
                rv_df, svd_df_h, svd_tier=1, default_threshold=default_threshold, eps=eps,
                semi_df=semi_df, interaction_smooth_h=h,
            )
            X_har_svd, har_svd_names = fe.build_feature_sets(
                rv_df, svd_df_h, svd_tier=2, default_threshold=default_threshold, eps=eps,
                semi_df=semi_df, interaction_smooth_h=h,
            )
            X_har_svd_xs, har_svd_xs_names = fe.build_feature_sets(
                rv_df, svd_df_h, svd_tier=3, default_threshold=default_threshold, eps=eps,
                semi_df=semi_df, csd_df=csd_df, turb_df=turb_df,
                interaction_smooth_h=h,
                include_cross_section=True,
            )
            if h == 1:
                print(f"[INFO] Tier-1 SVD-level:     {len(svd_t1_names)} features -> {svd_t1_names}")
                print(f"[INFO] Tier-2 HAR+SVD:       {len(har_svd_names)} features -> {har_svd_names}")
                print(f"[INFO] Tier-3 HAR+SVD+XS:    {len(har_svd_xs_names)} features -> {har_svd_xs_names}")
                print(f"[INFO] Primary cross-section (M2c): {inc_xs_primary}  OSI linear: {osi_enabled}")
            else:
                print(f"[INFO] h={h}: Tier-2 uses {h}-day smoothed SVD features ({len(har_svd_names)} total)")

            # Apply per-horizon hyperparameter overrides (stronger regularisation
            # at longer horizons to prevent overfitting on the smoother h-step target).
            _dnn_base = config.config["models"]["dnn"]
            _lstm_base = config.config["models"]["lstm"]
            _dnn_overrides = _dnn_base.get("horizon_overrides", {}).get(h, {})
            _lstm_overrides = _lstm_base.get("horizon_overrides", {}).get(h, {})
            cfg_dnn = {**_dnn_base, **_dnn_overrides}
            cfg_lstm = {**_lstm_base, **_lstm_overrides}
            if _dnn_overrides:
                print(
                    f"[INFO] h={h}: DNN overrides applied -> "
                    f"dropout={cfg_dnn['dropout']}  l2={cfg_dnn['l2_reg']}  "
                    f"layers={cfg_dnn['hidden_layers']}"
                )
            if _lstm_overrides:
                print(
                    f"[INFO] h={h}: LSTM overrides applied -> "
                    f"hidden={cfg_lstm['hidden_size']}  dropout={cfg_lstm['dropout']}  "
                    f"l2={cfg_lstm['l2_reg']}"
                )

            # Build forward-looking target (RMS of next h returns)
            target_rv = build_target_rv(portfolio_returns, h)
            target_rv_aligned = target_rv.reindex(common).dropna()
            align_idx = target_rv_aligned.index
            if len(align_idx) == 0:
                raise ValueError(f"Horizon {h}: no valid target after alignment.")
    
            # Filter to rows with no NaN features across all tiers
            X_har_block = X_har.loc[align_idx]
            X_svd_block = X_har_svd.loc[align_idx]
            clean = (X_har_block.notna().all(axis=1) & X_svd_block.notna().all(axis=1)).values
            align_idx = align_idx[clean]
            if len(align_idx) < MIN_ALIGNED_PER_HORIZON:
                raise ValueError(
                    f"Horizon {h}: insufficient aligned samples: {len(align_idx)} < "
                    f"{MIN_ALIGNED_PER_HORIZON}."
                )
    
            X_har_h = np.asarray(X_har.loc[align_idx].values, dtype=np.float64)
            X_svd_t1_h = np.asarray(X_svd_t1.loc[align_idx].values, dtype=np.float64)
            X_har_svd_h = np.asarray(X_har_svd.loc[align_idx].values, dtype=np.float64)
            X_har_svd_xs_h = np.asarray(X_har_svd_xs.loc[align_idx].values, dtype=np.float64)
            # Verify feature dimensions
            assert X_har_h.shape[1] == len(har_names), (
                f"Tier-0 features: expected (n, {len(har_names)}), got {X_har_h.shape}"
            )
            assert X_har_svd_h.shape[1] == len(har_svd_names), (
                f"Tier-2 features: expected (n, {len(har_svd_names)}), got {X_har_svd_h.shape}"
            )
    
            y_h = np.log(target_rv_aligned.loc[align_idx].values.ravel() + eps)
    
            # Chronological train / test split
            n_h = len(align_idx)
            split_h = int(n_h * train_split)
            train_mask = np.zeros(n_h, dtype=bool)
            train_mask[:split_h] = True
            test_mask = ~train_mask
    
            X_har_train, X_har_test = X_har_h[train_mask], X_har_h[test_mask]
            X_svd_train, X_svd_test = X_har_svd_h[train_mask], X_har_svd_h[test_mask]
            y_train, y_test = y_h[train_mask], y_h[test_mask]
            true_vol = np.exp(y_test)
            test_dates = align_idx[test_mask]
    
            seq_len = cfg_lstm["seq_len"]
            val_size = max(seq_len, int(0.15 * split_h))
            if len(X_har_train) <= val_size:
                raise ValueError(
                    f"Horizon {h}: training length {len(X_har_train)} <= val_size {val_size}."
                )
            X_har_fit = X_har_h[: split_h - val_size]
            y_fit = y_h[: split_h - val_size]
            X_svd_t1_fit = X_svd_t1_h[: split_h - val_size]
            X_svd_fit = X_har_svd_h[: split_h - val_size]
            X_svd_xs_fit = X_har_svd_xs_h[: split_h - val_size]
    
            # Standardise features for DNN / LSTM / HARNet (fit on train only).
            clip_std = float(cfg_train.get("feature_clip_std", 5.0))
            scaler_har = StandardScaler().fit(X_har_train)
            scaler_svd = StandardScaler().fit(X_svd_train)
            X_har_h_scaled = np.clip(scaler_har.transform(X_har_h), -clip_std, clip_std)
            X_har_svd_h_scaled = np.clip(scaler_svd.transform(X_har_svd_h), -clip_std, clip_std)
            X_har_train_s = X_har_h_scaled[train_mask]
            X_har_test_s = X_har_h_scaled[test_mask]
            X_svd_train_s = X_har_svd_h_scaled[train_mask]
            X_svd_test_s = X_har_svd_h_scaled[test_mask]
    
            # -------------------------------------------------------------------
            # Unified validation split: last 15% of training data for ALL models.
            # (val_size computed above; same for early stopping and linear M1/M2 fit.)
            # -------------------------------------------------------------------
            # DNN-style validation (2D arrays)
            X_har_tr = X_har_train_s[:-val_size]
            y_har_tr = y_train[:-val_size]
            X_har_val = X_har_train_s[-val_size:]
            y_har_val = y_train[-val_size:]
            X_svd_tr = X_svd_train_s[:-val_size]
            y_svd_tr = y_train[:-val_size]
            X_svd_val = X_svd_train_s[-val_size:]
            y_svd_val = y_train[-val_size:]
    
            # LSTM/HARNet-style: needs same split boundaries in the full (non-split) scaled array
            X_har_tr_2d = X_har_h_scaled[: split_h - val_size]
            y_har_tr_2d = y_h[: split_h - val_size]
            X_har_val_2d = X_har_h_scaled[split_h - val_size : split_h]
            y_har_val_2d = y_h[split_h - val_size : split_h]
            X_svd_tr_2d = X_har_svd_h_scaled[: split_h - val_size]
            y_svd_tr_2d = y_h[: split_h - val_size]
            X_svd_val_2d = X_har_svd_h_scaled[split_h - val_size : split_h]
            y_svd_val_2d = y_h[split_h - val_size : split_h]
    
            if len(X_har_tr_2d) < seq_len or len(X_har_val_2d) < seq_len:
                raise ValueError(
                    f"Horizon {h}: insufficient rows for LSTM train/val "
                    f"(need >= seq_len={seq_len})."
                )
    
            # -------------------------------------------------------------------
            # M1 & M2 (and Tier 1 / Tier 3 ablation): ALL linear models use the
            # SAME ElasticNet pipeline so comparisons isolate feature content only.
            # Using OLS for M1 and ElasticNet for M2 was a confound: ElasticNet's
            # regularisation and CV alone can win against OLS independent of SVD.
            # -------------------------------------------------------------------
            # M1: HAR baseline (Tier 0) — ElasticNet
            har_elastic = linear_models.train_har_svd_elastic(X_har_fit, y_fit)
            har_pred_log = linear_models.predict_har_svd_elastic(har_elastic, X_har_h[test_mask])
            har_train_pred_log = linear_models.predict_har_svd_elastic(har_elastic, X_har_fit)
            har_pred = smearing_corrected_pred(har_pred_log, har_train_pred_log, y_fit)
            m1 = ev.four_metrics(true_vol, har_pred, eps)
            print(
                f"  M1 HAR(EN)   RMSE={m1['RMSE']:.4e}  R2={m1['R2']:.4f}  MALE={m1['MALE']:.1f}%"
                f"  [alpha={har_elastic.named_steps['enet'].alpha_:.4f}"
                f"  l1={har_elastic.named_steps['enet'].l1_ratio_:.2f}]"
            )
            # Tier 1: HAR + f1 + log_sigma1 (eigenvalue-level features only)
            svd_t1_elastic = linear_models.train_har_svd_elastic(X_svd_t1_fit, y_fit)
            svd_t1_pred_log = linear_models.predict_har_svd_elastic(svd_t1_elastic, X_svd_t1_h[test_mask])
            svd_t1_train_pred_log = linear_models.predict_har_svd_elastic(svd_t1_elastic, X_svd_t1_fit)
            svd_t1_pred = smearing_corrected_pred(svd_t1_pred_log, svd_t1_train_pred_log, y_fit)
            m_t1 = ev.four_metrics(true_vol, svd_t1_pred, eps)
            print(
                f"  M1b SVD-T1   RMSE={m_t1['RMSE']:.4e}  R2={m_t1['R2']:.4f}  MALE={m_t1['MALE']:.1f}%"
            )
            # M2: HAR + full SVD (Tier 2) — primary SVD ablation model
            har_svd_elastic = linear_models.train_har_svd_elastic(X_svd_fit, y_fit)
            har_svd_pred_log = linear_models.predict_har_svd_elastic(har_svd_elastic, X_har_svd_h[test_mask])
            har_svd_train_pred_log = linear_models.predict_har_svd_elastic(har_svd_elastic, X_svd_fit)
            har_svd_pred = smearing_corrected_pred(har_svd_pred_log, har_svd_train_pred_log, y_fit)
            m2 = ev.four_metrics(true_vol, har_svd_pred, eps)
            enet_step = har_svd_elastic.named_steps["enet"]
            print(
                f"  M2 HAR+SVD   RMSE={m2['RMSE']:.4e}  R2={m2['R2']:.4f}  MALE={m2['MALE']:.1f}%"
                f"  [ElasticNet alpha={enet_step.alpha_:.4f}  l1_ratio={enet_step.l1_ratio_:.2f}]"
            )
            coef_dict = linear_models.get_elastic_coefs(har_svd_elastic, har_svd_names)
            active = {k: v for k, v in coef_dict.items() if abs(v) > 1e-6}
            print(
                f"    [h={h}] HAR+SVD ElasticNet active ({len(active)}/{len(har_svd_names)}): {active}"
            )

            # Regime-gated HAR+SVD: interact HAR block with absorption ratio (AR)
            try:
                if "AR" in svd_df_h.columns:
                    X_har_svd_df = X_har_svd.loc[align_idx]
                    har_like = [c for c in ["log_RSV_d_minus", "log_RSV_d_plus", "RV_w", "RV_10d", "RV_m", "RV_d"] if c in X_har_svd_df.columns]
                    X_gated_df, gated_names = fe.add_regime_gating_interactions(
                        X_har_svd_df,
                        gate=svd_df_h["AR"].reindex(X_har_svd_df.index),
                        gate_name="AR",
                        base_cols=har_like,
                        q_high=0.8,
                    )
                    X_gated = np.asarray(X_gated_df.values, dtype=np.float64)
                    X_gated_fit = X_gated[: split_h - val_size]
                    gated_elastic = linear_models.train_har_svd_elastic(X_gated_fit, y_fit)
                    gated_pred_log = linear_models.predict_har_svd_elastic(gated_elastic, X_gated[test_mask])
                    gated_train_pred_log = linear_models.predict_har_svd_elastic(gated_elastic, X_gated_fit)
                    gated_pred = smearing_corrected_pred(gated_pred_log, gated_train_pred_log, y_fit)
                    m2_gated = ev.four_metrics(true_vol, gated_pred, eps)
                    print(
                        f"  M2g HAR+SVD_GATED_AR RMSE={m2_gated['RMSE']:.4e}  "
                        f"R2={m2_gated['R2']:.4f}  MALE={m2_gated['MALE']:.1f}%"
                    )
                else:
                    gated_pred = np.full(len(true_vol), np.nan)
                    m2_gated = {"RMSE": np.nan, "R2": np.nan, "MALE": np.nan, "QLIKE": np.nan}
            except Exception as gated_err:
                print(f"  [WARN] HAR+SVD gated model failed h={h}: {gated_err}")
                gated_pred = np.full(len(true_vol), np.nan)
                m2_gated = {"RMSE": np.nan, "R2": np.nan, "MALE": np.nan, "QLIKE": np.nan}

            # Complexity-with-shrinkage: Random Fourier Features + Ridge
            try:
                rff_pipe = linear_models.train_ridge_rff(
                    X_svd_fit,
                    y_fit,
                    cv_splits=5,
                    tune=True,
                    n_components=256,
                    gamma=1.0,
                    seed=int(config.config["seed"]) + 123,
                )
                rff_pred_log = np.asarray(rff_pipe.predict(X_har_svd_h[test_mask]), dtype=np.float64).ravel()
                rff_train_pred_log = np.asarray(rff_pipe.predict(X_svd_fit), dtype=np.float64).ravel()
                rff_pred = smearing_corrected_pred(rff_pred_log, rff_train_pred_log, y_fit)
                m2_rff = ev.four_metrics(true_vol, rff_pred, eps)
                print(
                    f"  M2r HAR+SVD_RFF_RIDGE RMSE={m2_rff['RMSE']:.4e}  "
                    f"R2={m2_rff['R2']:.4f}  MALE={m2_rff['MALE']:.1f}%"
                )
            except Exception as rff_err:
                print(f"  [WARN] HAR+SVD RFF+Ridge failed h={h}: {rff_err}")
                rff_pred = np.full(len(true_vol), np.nan)
                m2_rff = {"RMSE": np.nan, "R2": np.nan, "MALE": np.nan, "QLIKE": np.nan}

            # Orthogonalized Spectral Increment (OSI): two-stage ElasticNet on training slice.
            if osi_enabled:
                try:
                    osi_pred_log, osi_train_pred_log, _osi_predictor = linear_models.train_predict_osi_elastic(
                        np.asarray(X_svd_fit, dtype=np.float64),
                        np.asarray(X_har_svd_h[test_mask], dtype=np.float64),
                        y_fit,
                        len(har_names),
                    )
                    osi_pred = smearing_corrected_pred(osi_pred_log, osi_train_pred_log, y_fit)
                    m_osi = ev.four_metrics(true_vol, osi_pred, eps)
                    print(
                        f"  M2b HAR+SVD_OSI RMSE={m_osi['RMSE']:.4e}  R2={m_osi['R2']:.4f}"
                        f"  MALE={m_osi['MALE']:.1f}%  QLIKE={m_osi['QLIKE']:.4f}"
                    )
                except Exception as osi_err:
                    print(f"  [WARN] OSI ElasticNet failed h={h}: {osi_err}")
                    osi_pred = np.full(len(true_vol), np.nan)
                    m_osi = {"RMSE": np.nan, "R2": np.nan, "MALE": np.nan, "QLIKE": np.nan}
            else:
                osi_pred = np.full(len(true_vol), np.nan)
                m_osi = {"RMSE": np.nan, "R2": np.nan, "MALE": np.nan, "QLIKE": np.nan}

            # -------------------------------------------------------------------
            # Tail calibration (VaR/ES objective): fit 2-regime sigma scaling using AR
            # on the *fit* sample only, then apply to test forecasts.
            # This produces a variance forecast optimized for FZ0 rather than QLIKE.
            # -------------------------------------------------------------------
            tailcal_pred = np.full(len(true_vol), np.nan)
            tailcal_meta = None
            if h == 1 and "AR" in svd_df_h.columns:
                try:
                    fit_dates = align_idx[: split_h - val_size]
                    test_dates_local = test_dates
                    # Next-day returns aligned to forecast origin dates.
                    r_fit_dec = (portfolio_returns.reindex(fit_dates).shift(-1).values.astype(np.float64) / 100.0)
                    r_te_dec = (portfolio_returns.reindex(test_dates_local).shift(-1).values.astype(np.float64) / 100.0)
                    gate_fit = svd_df_h.reindex(fit_dates)["AR"].values.astype(np.float64)
                    gate_te = svd_df_h.reindex(test_dates_local)["AR"].values.astype(np.float64)
                    # Predicted variance in DECIMAL units
                    var_fit_pct2 = smearing_corrected_pred(har_svd_train_pred_log, har_svd_train_pred_log, y_fit)
                    var_fit_dec = np.asarray(var_fit_pct2, dtype=np.float64).ravel() / (100.0 ** 2)
                    var_te_dec = np.asarray(har_svd_pred, dtype=np.float64).ravel() / (100.0 ** 2)
                    ok_fit = np.isfinite(r_fit_dec) & np.isfinite(var_fit_dec) & np.isfinite(gate_fit)
                    ok_te = np.isfinite(r_te_dec) & np.isfinite(var_te_dec) & np.isfinite(gate_te)
                    if ok_fit.sum() >= 600 and ok_te.sum() >= 200:
                        params = tailcal.fit_two_regime_scale_fz0(
                            r_fit_dec[ok_fit],
                            var_fit_dec[ok_fit],
                            gate=gate_fit[ok_fit],
                            gate_name="AR",
                            alpha=0.01,
                            q_high=0.8,
                            dist="gaussian",
                            df=None,
                        )
                        var_te_dec_adj = tailcal.apply_two_regime_scale(
                            var_te_dec[ok_te], gate=gate_te[ok_te], params=params
                        )
                        # Convert back to %-squared variance for consistency with pipeline
                        tailcal_pred = np.full_like(var_te_dec, np.nan, dtype=np.float64)
                        tailcal_pred[ok_te] = var_te_dec_adj * (100.0 ** 2)
                        tailcal_meta = {
                            "threshold": params.threshold,
                            "sigma_mult_low": params.sigma_mult_low,
                            "sigma_mult_high": params.sigma_mult_high,
                        }
                        print(
                            f"  TailCal(HAR+SVD, AR): thr={params.threshold:.3f} "
                            f"m_low={params.sigma_mult_low:.3f} m_high={params.sigma_mult_high:.3f}"
                        )
                except Exception as tc_err:
                    print(f"  [WARN] Tail calibration failed h=1: {tc_err}")

            # Tier 3: HAR + SVD + cross-section (optional primary; XS block kept for ablations)
            if inc_xs_primary:
                svd_xs_elastic = linear_models.train_har_svd_elastic(X_svd_xs_fit, y_fit)
                svd_xs_pred_log = linear_models.predict_har_svd_elastic(
                    svd_xs_elastic, X_har_svd_xs_h[test_mask]
                )
                svd_xs_train_pred_log = linear_models.predict_har_svd_elastic(
                    svd_xs_elastic, X_svd_xs_fit
                )
                svd_xs_pred = smearing_corrected_pred(
                    svd_xs_pred_log, svd_xs_train_pred_log, y_fit
                )
                m_t3 = ev.four_metrics(true_vol, svd_xs_pred, eps)
                print(
                    f"  M2c SVD-T3   RMSE={m_t3['RMSE']:.4e}  R2={m_t3['R2']:.4f}"
                    f"  MALE={m_t3['MALE']:.1f}%"
                )
            else:
                print(
                    "  M2c SVD-T3   (skipped — features.include_cross_section_primary=False; "
                    "see appendix / config)"
                )
                svd_xs_pred = np.full(len(true_vol), np.nan)
                m_t3 = {"RMSE": np.nan, "R2": np.nan, "MALE": np.nan, "QLIKE": np.nan}

            # HAR (OLS) predictions on training-without-val rows for DNN warm-start.
            har_model_ols = linear_models.train_har(X_har_fit, y_fit)
            har_init_preds_tr = linear_models.predict_har(har_model_ols, X_har_fit)

            # --- M3: DNN (HAR features, QLIKE + HAR initialization) -----------
            dnn_har, _ = dnn_models.train_dnn(
                X_har_tr, y_har_tr, X_har_val, y_har_val,
                input_dim=X_har_train_s.shape[1],
                epochs=cfg_dnn["epochs"],
                batch_size=cfg_dnn["batch_size"],
                patience=cfg_dnn["patience"],
                hidden_dims=cfg_dnn["hidden_layers"],
                dropout=cfg_dnn["dropout"],
                l2_reg=cfg_dnn["l2_reg"],
                lr=cfg_dnn["learning_rate"],
                init_har_preds=har_init_preds_tr,
            )
            dnn_har_pred_log = dnn_har.predict(X_har_test_s, verbose=0).ravel()
            dnn_har_train_pred_log = dnn_har.predict(X_har_tr, verbose=0).ravel()
            dnn_har_pred = smearing_corrected_pred(
                dnn_har_pred_log, dnn_har_train_pred_log, y_har_tr
            )
            m3 = ev.four_metrics(true_vol, dnn_har_pred, eps)
            print(f"  M3 DNN_HAR   RMSE={m3['RMSE']:.4e}  R2={m3['R2']:.4f}  MALE={m3['MALE']:.1f}%")
    
            # --- M4: DNN+SVD (wider architecture + 5-seed ensemble + HAR init) -
            # _train_dnn_ensemble now returns (mean_test_log, mean_train_log, last_model)
            # so smearing uses the ensemble mean on both train and test — eliminating
            # the previous single-seed bias that corrupted the smear factor.
            dnn_svd_pred_log, dnn_svd_train_pred_log, last_dnn_svd = _train_dnn_ensemble(
                sub_seeds=ENSEMBLE_SUB_SEEDS,
                X_tr=X_svd_tr,
                y_tr=y_svd_tr,
                X_val=X_svd_val,
                y_val=y_svd_val,
                X_test=X_svd_test_s,
                input_dim=X_svd_train_s.shape[1],
                cfg_dnn=cfg_dnn,
                hidden_dims=cfg_dnn["hidden_layers_svd"],
                har_init_preds=har_init_preds_tr,
                use_gate=True,
            )
            # Note: dnn_svd_train_pred_log is already the ensemble mean on X_svd_tr
            # (train minus validation), so we align y to the same rows.
            dnn_svd_pred = smearing_corrected_pred(
                dnn_svd_pred_log, dnn_svd_train_pred_log, y_svd_tr
            )
            m4 = ev.four_metrics(true_vol, dnn_svd_pred, eps)
            print(f"  M4 DNN+SVD   RMSE={m4['RMSE']:.4e}  R2={m4['R2']:.4f}  MALE={m4['MALE']:.1f}%  (ens5)")
            if h == 1:
                model_dnn_svd_h1 = last_dnn_svd
                X_har_svd_h1_scaled_test = X_svd_test_s
    
            # --- M5: LSTM (HAR features, QLIKE + composite warm-start) --------
            lstm_har, _ = lstm_models.train_lstm(
                X_har_tr_2d, y_har_tr_2d, X_har_val_2d, y_har_val_2d,
                seq_len=seq_len, n_features=X_har_h_scaled.shape[1],
                epochs=cfg_lstm["epochs"],
                batch_size=cfg_lstm["batch_size"],
                patience=cfg_lstm["patience"],
                hidden_size=cfg_lstm["hidden_size"],
                dropout=cfg_lstm["dropout"],
                recurrent_dropout=cfg_lstm["recurrent_dropout"],
                l2_reg=cfg_lstm["l2_reg"],
                lr=cfg_lstm["learning_rate"],
                init_har_preds=har_init_preds_tr,
            )
            X_har_seq, y_har_seq = lstm_models.build_sequences(X_har_h_scaled, y_h, seq_len)
            seq_test_start = max(0, split_h - seq_len + 1)
            X_har_test_seq = X_har_seq[seq_test_start:]
            lstm_har_pred_log = lstm_har.predict(X_har_test_seq, verbose=0).ravel()
            # Smearing for LSTM: use training-portion sequences
            X_har_tr_seq_smear, y_har_tr_seq_smear = lstm_models.build_sequences(
                X_har_h_scaled[: split_h - val_size],
                y_h[: split_h - val_size],
                seq_len,
            )
            lstm_har_tr_pred_log = lstm_har.predict(X_har_tr_seq_smear, verbose=0).ravel()
            lstm_har_pred = smearing_corrected_pred(
                lstm_har_pred_log, lstm_har_tr_pred_log,
                y_har_tr_seq_smear[-len(lstm_har_tr_pred_log):]
            )
            n_align5 = min(len(true_vol), len(lstm_har_pred))
            m5 = ev.four_metrics(true_vol[:n_align5], lstm_har_pred[:n_align5], eps)
            print(f"  M5 LSTM_HAR  RMSE={m5['RMSE']:.4e}  R2={m5['R2']:.4f}  MALE={m5['MALE']:.1f}%")
    
            # --- M6: LSTM+SVD (5-seed ensemble, composite + QLIKE loss) --------
            X_svd_seq, _ = lstm_models.build_sequences(X_har_svd_h_scaled, y_h, seq_len)
            X_svd_test_seq = X_svd_seq[seq_test_start:]
            lstm_svd_pred_log, lstm_svd_train_pred_log, last_lstm_svd = _train_lstm_ensemble(
                sub_seeds=ENSEMBLE_SUB_SEEDS,
                X_tr_2d=X_svd_tr_2d,
                y_tr_2d=y_svd_tr_2d,
                X_val_2d=X_svd_val_2d,
                y_val_2d=y_svd_val_2d,
                X_test_seq=X_svd_test_seq,
                n_features=X_har_svd_h_scaled.shape[1],
                cfg_lstm=cfg_lstm,
                har_init_preds=har_init_preds_tr,
            )
            _, y_svd_tr_seq_for_smear = lstm_models.build_sequences(
                X_svd_tr_2d, y_svd_tr_2d, seq_len
            )
            lstm_svd_pred = smearing_corrected_pred(
                lstm_svd_pred_log, lstm_svd_train_pred_log, y_svd_tr_seq_for_smear
            )
            n_align6 = min(len(true_vol), len(lstm_svd_pred))
            m6 = ev.four_metrics(true_vol[:n_align6], lstm_svd_pred[:n_align6], eps)
            print(f"  M6 LSTM+SVD  RMSE={m6['RMSE']:.4e}  R2={m6['R2']:.4f}  MALE={m6['MALE']:.1f}%  (ens5)")
            if h == 1:
                model_lstm_svd_h1 = last_lstm_svd
                X_har_seq_svd_test_h1 = X_svd_test_seq
    
            # --- M7: HARNet / TCN (HAR+SVD features, QLIKE) -------------------
            harnet_svd, harnet_hist = harnet_models.train_harnet(
                X_svd_tr_2d, y_svd_tr_2d, X_svd_val_2d, y_svd_val_2d,
                seq_len=cfg_harnet["seq_len"],
                n_features=X_har_svd_h_scaled.shape[1],
                epochs=cfg_harnet["epochs"],
                batch_size=cfg_harnet["batch_size"],
                patience=cfg_harnet["patience"],
                filters=cfg_harnet["filters"],
                dilations=cfg_harnet["dilations"],
                dropout=cfg_harnet["dropout"],
                lr=cfg_harnet["learning_rate"],
                init_har_preds=har_init_preds_tr,  # warm-start HARNet to HAR predictions
            )
            X_harnet_test_seq = X_svd_seq[seq_test_start:]
            harnet_pred_log = harnet_svd.predict(X_harnet_test_seq, verbose=0).ravel()
            harnet_tr_seq_smear, y_harnet_tr_seq_smear = lstm_models.build_sequences(
                X_har_svd_h_scaled[: split_h - val_size],
                y_h[: split_h - val_size],
                cfg_harnet["seq_len"],
            )
            harnet_tr_pred_log = harnet_svd.predict(harnet_tr_seq_smear, verbose=0).ravel()
            harnet_pred = smearing_corrected_pred(
                harnet_pred_log, harnet_tr_pred_log,
                y_harnet_tr_seq_smear[-len(harnet_tr_pred_log):]
            )
            n_align7 = min(len(true_vol), len(harnet_pred))
            m7 = ev.four_metrics(true_vol[:n_align7], harnet_pred[:n_align7], eps)
            print(f"  M7 HARNet    RMSE={m7['RMSE']:.4e}  R2={m7['R2']:.4f}  MALE={m7['MALE']:.1f}%")
    
            # --- M8: GNN baseline -----------------------------------------------
            print(f"  [INFO] Building GNN data for h={h} ...")
            try:
                node_feat_all, adj_all = _build_gnn_data(
                    align_idx=align_idx,
                    cov_series=cov_series,
                    returns=returns,
                    asset_columns=asset_columns,
                    svd_df=svd_df_h,
                    adj_threshold=cfg_gnn["adj_threshold"],
                )
                nf_shape = node_feat_all.shape  # (T, N, 4)
                nf_train_flat = node_feat_all[train_mask].reshape(-1, nf_shape[-1])
                nf_scaler_gnn = StandardScaler().fit(nf_train_flat)
                node_feat_scaled = nf_scaler_gnn.transform(
                    node_feat_all.reshape(-1, nf_shape[-1])
                ).reshape(nf_shape)
    
                nf_train_gnn = node_feat_scaled[train_mask]
                adj_train_gnn = adj_all[train_mask]
                nf_test_gnn = node_feat_scaled[test_mask]
                adj_test_gnn = adj_all[test_mask]
    
                gnn_val_size = max(1, int(0.15 * nf_train_gnn.shape[0]))
                nf_tr_gnn = nf_train_gnn[:-gnn_val_size]
                adj_tr_gnn = adj_train_gnn[:-gnn_val_size]
                y_tr_gnn = y_train[:-gnn_val_size]
                nf_val_gnn = nf_train_gnn[-gnn_val_size:]
                adj_val_gnn = adj_train_gnn[-gnn_val_size:]
                y_val_gnn = y_train[-gnn_val_size:]
    
                gnn_model, _ = gnn_models.train_gnn(
                    node_features_train=nf_tr_gnn,
                    adj_train=adj_tr_gnn,
                    y_train=y_tr_gnn,
                    node_features_val=nf_val_gnn,
                    adj_val=adj_val_gnn,
                    y_val=y_val_gnn,
                    n_nodes=N,
                    node_feat_dim=nf_shape[-1],
                    hidden_dim=cfg_gnn["hidden_dim"],
                    dropout=cfg_gnn["dropout"],
                    lr=cfg_gnn["learning_rate"],
                    epochs=cfg_gnn["epochs"],
                    batch_size=cfg_gnn["batch_size"],
                    patience=cfg_gnn["patience"],
                )
                gnn_pred_log = gnn_models.gnn_predict(gnn_model, nf_test_gnn, adj_test_gnn)
                # Smearing correction for GNN (uses Huber/MSE loss → Jensen's bias)
                gnn_tr_pred_log = gnn_models.gnn_predict(gnn_model, nf_tr_gnn, adj_tr_gnn)
                gnn_pred = smearing_corrected_pred(gnn_pred_log, gnn_tr_pred_log, y_tr_gnn)
                m8 = ev.four_metrics(true_vol, gnn_pred, eps)
                print(f"  M8 GNN       RMSE={m8['RMSE']:.4e}  R2={m8['R2']:.4f}  MALE={m8['MALE']:.1f}%")
                gnn_ok = True
            except Exception as gnn_err:
                print(f"  [WARN] GNN failed for h={h}: {gnn_err}")
                gnn_pred = np.full(len(true_vol), np.nan)
                m8 = {"RMSE": np.nan, "R2": np.nan, "MALE": np.nan, "QLIKE": np.nan}
                gnn_ok = False
    
            # --- GARCH(1,1) benchmark (multi-horizon) ---------------------------
            try:
                _g_refit = cfg_garch_bm.get("refit_every")
                garch_refit = int(_g_refit) if _g_refit is not None else None
                if h == 1:
                    garch_pred_h = garch_models.build_garch(
                        portfolio_returns,
                        test_dates,
                        innovations=str(cfg_garch_bm.get("innovations", "gaussian")),
                        t_df=float(cfg_garch_bm.get("t_df", 8.0)),
                        refit_every=garch_refit,
                    )
                else:
                    garch_pred_h = garch_models.build_garch_multistep(
                        portfolio_returns,
                        test_dates,
                        horizon=h,
                        innovations=str(cfg_garch_bm.get("innovations", "gaussian")),
                        t_df=float(cfg_garch_bm.get("t_df", 8.0)),
                        refit_every=garch_refit,
                    )
                garch_ok = np.isfinite(garch_pred_h).sum() > 50
            except Exception as garch_e:
                print(f"  [WARN] GARCH h={h} failed: {garch_e}")
                garch_pred_h = np.full(len(true_vol), np.nan)
                garch_ok = False

            if garch_ok:
                m_garch = ev.four_metrics(true_vol, garch_pred_h, eps)
                print(
                    f"  GARCH(1,1)   RMSE={m_garch['RMSE']:.4e}  R2={m_garch['R2']:.4f}"
                    f"  MALE={m_garch['MALE']:.1f}%"
                )
            else:
                m_garch = {"RMSE": np.nan, "R2": np.nan, "MALE": np.nan, "QLIKE": np.nan}

            # --- GJR-GARCH(1,1)-t benchmark (asymmetric leverage + fat tails) --
            # Required by reviewers: "no serious volatility paper uses symmetric
            # GARCH as sole benchmark" (Glosten, Jagannathan & Runkle 1993).
            try:
                if h == 1:
                    gjr_pred_h = garch_models.build_gjr_garch(
                        portfolio_returns, test_dates,
                        innovations="t", t_df=float(cfg_garch_bm.get("t_df", 8.0)),
                        refit_every=garch_refit,
                    )
                else:
                    gjr_pred_h = garch_models.build_gjr_garch_multistep(
                        portfolio_returns, test_dates, horizon=h,
                        innovations="t", t_df=float(cfg_garch_bm.get("t_df", 8.0)),
                        refit_every=garch_refit,
                    )
                gjr_ok = np.isfinite(gjr_pred_h).sum() > 50
            except Exception as gjr_e:
                print(f"  [WARN] GJR-GARCH h={h} failed: {gjr_e}")
                gjr_pred_h = np.full(len(true_vol), np.nan)
                gjr_ok = False

            if gjr_ok:
                m_gjr = ev.four_metrics(true_vol, gjr_pred_h, eps)
                print(
                    f"  GJR-GARCH-t  RMSE={m_gjr['RMSE']:.4e}  R2={m_gjr['R2']:.4f}"
                    f"  MALE={m_gjr['MALE']:.1f}%"
                )
            else:
                m_gjr = {"RMSE": np.nan, "R2": np.nan, "MALE": np.nan, "QLIKE": np.nan}
                gjr_pred_h = np.full(len(true_vol), np.nan)

            # --- Forecast combinations (equal-weighted and inverse-MSE-weighted) --
            # Equal-weighted: proves SVD contains complementary information to HAR.
            # Inverse-MSE: time-series-split weights from training performance.
            combo_eq_pred = 0.5 * har_pred + 0.5 * har_svd_pred
            m_combo_eq = ev.four_metrics(true_vol, combo_eq_pred, eps)

            # Inverse-MSE weights: compute on training set (no leakage)
            mse_har_tr = float(np.mean((np.exp(y_fit) - np.exp(har_train_pred_log)) ** 2))
            mse_svd_tr = float(np.mean((np.exp(y_fit) - np.exp(har_svd_train_pred_log)) ** 2))
            w_har_inv = 1.0 / max(mse_har_tr, 1e-20)
            w_svd_inv = 1.0 / max(mse_svd_tr, 1e-20)
            w_total = w_har_inv + w_svd_inv
            combo_mse_pred = (w_har_inv * har_pred + w_svd_inv * har_svd_pred) / w_total
            m_combo_mse = ev.four_metrics(true_vol, combo_mse_pred, eps)
            print(
                f"  Combo_EqWt   RMSE={m_combo_eq['RMSE']:.4e}  R2={m_combo_eq['R2']:.4f}"
                f"  QLIKE={m_combo_eq['QLIKE']:.4f}"
            )
            print(
                f"  Combo_InvMSE RMSE={m_combo_mse['RMSE']:.4e}  R2={m_combo_mse['R2']:.4f}"
                f"  QLIKE={m_combo_mse['QLIKE']:.4f}"
            )

            # --- Bates-Granger optimal weights with stationary block-bootstrap CI -
            # Reports the minimum-variance weight on HAR vs HAR+SVD, with a 95%
            # confidence interval.  Out-of-sample combined forecast uses a rolling,
            # leak-free recursive weight (computed on the training residuals).
            try:
                bg_in_sample = ev.bates_granger_with_bootstrap_ci(
                    true_vol, har_pred, har_svd_pred,
                    block_len=22, n_boot=999, alpha=0.05, seed=42, nonneg=True,
                )
                print(
                    f"  Bates-Granger (HAR vs HAR+SVD): w_HAR={bg_in_sample['w_a']:.3f} "
                    f"[{bg_in_sample['w_a_lo']:.3f}, {bg_in_sample['w_a_hi']:.3f}]  "
                    f"mse_comb={bg_in_sample['mse_comb']:.4e}  corr={bg_in_sample['corr_ab']:.3f}"
                )
            except Exception as bg_err:
                print(f"  [WARN] Bates-Granger CI failed at h={h}: {bg_err}")
                bg_in_sample = None

            # Leak-free OOS Bates-Granger forecast: weight estimated on the
            # training-fit residuals (same data used for inverse-MSE), held
            # constant across the OOS test sample.
            try:
                e_har_tr = np.exp(y_fit) - np.exp(har_train_pred_log)
                e_svd_tr = np.exp(y_fit) - np.exp(har_svd_train_pred_log)
                s_aa_tr = float(np.mean(e_har_tr ** 2))
                s_bb_tr = float(np.mean(e_svd_tr ** 2))
                s_ab_tr = float(np.mean(e_har_tr * e_svd_tr))
                denom_tr = s_aa_tr + s_bb_tr - 2.0 * s_ab_tr
                if not np.isfinite(denom_tr) or abs(denom_tr) < 1e-20:
                    w_har_bg = 0.5
                else:
                    w_har_bg = float(np.clip((s_bb_tr - s_ab_tr) / denom_tr, 0.0, 1.0))
                combo_bg_pred = w_har_bg * har_pred + (1.0 - w_har_bg) * har_svd_pred
                m_combo_bg = ev.four_metrics(true_vol, combo_bg_pred, eps)
                print(
                    f"  Combo_BG     RMSE={m_combo_bg['RMSE']:.4e}  R2={m_combo_bg['R2']:.4f}"
                    f"  QLIKE={m_combo_bg['QLIKE']:.4f}  w_HAR={w_har_bg:.3f}"
                )
            except Exception as bg2_err:
                print(f"  [WARN] OOS Bates-Granger forecast failed at h={h}: {bg2_err}")
                combo_bg_pred = combo_eq_pred
                m_combo_bg = m_combo_eq
                w_har_bg = 0.5

            # --- M2 leave-one-group-out ablations (ElasticNet + smearing) ----------
            fab_cfg_h = config.config.get("feature_ablation", {})
            abl_prefix_h = str(fab_cfg_h.get("model_key_prefix", "HAR+SVD_minus_"))
            abl_rb: dict = {}
            abl_preds: dict = {}
            if fab_cfg_h.get("enabled", False):
                groups_run = list(fab_cfg_h.get("groups_to_drop", []))
                if fab_cfg_h.get("include_tier3_ablation", False):
                    for g_extra in fab_cfg_h.get("tier3_additional_groups", ["G4_xs"]):
                        if g_extra not in groups_run:
                            groups_run.append(g_extra)
                for g in groups_run:
                    if g == "G4_xs":
                        res_ab = feat_abl.fit_predict_m2_ablation(
                            X_svd_xs_fit,
                            np.asarray(X_har_svd_xs_h[test_mask], dtype=np.float64),
                            y_fit,
                            y_fit,
                            har_svd_xs_names,
                            g,
                            linear_models.train_har_svd_elastic,
                            linear_models.predict_har_svd_elastic,
                            smearing_corrected_pred,
                            eps,
                        )
                    else:
                        res_ab = feat_abl.fit_predict_m2_ablation(
                            X_svd_fit,
                            np.asarray(X_har_svd_h[test_mask], dtype=np.float64),
                            y_fit,
                            y_fit,
                            har_svd_names,
                            g,
                            linear_models.train_har_svd_elastic,
                            linear_models.predict_har_svd_elastic,
                            smearing_corrected_pred,
                            eps,
                        )
                    if res_ab is None:
                        print(f"  [WARN] Feature ablation skipped for {g} (too few columns).")
                        continue
                    pred_ab, _pipe_ab, _mask_ab = res_ab
                    ak = feat_abl.ablation_model_key(g, abl_prefix_h)
                    abl_rb[ak] = ev.four_metrics(true_vol, pred_ab, eps)
                    abl_preds[ak] = pred_ab
                    print(
                        f"  M2−{g:10} RMSE={abl_rb[ak]['RMSE']:.4e}  R2={abl_rb[ak]['R2']:.4f}"
                        f"  QLIKE={abl_rb[ak]['QLIKE']:.4f}"
                    )
    
            # Collect horizon results
            rb[h] = {
                "HAR": m1,
                "HAR_SVD_T1": m_t1,
                "HAR+SVD": m2,
                "HAR+SVD_GATED_AR": m2_gated,
                "HAR+SVD_TAILCAL_AR": (
                    ev.four_metrics(true_vol, tailcal_pred, eps) if np.isfinite(tailcal_pred).any() else {"RMSE": np.nan, "R2": np.nan, "MALE": np.nan, "QLIKE": np.nan}
                ),
                "HAR+SVD_RFF_RIDGE": m2_rff,
                "HAR+SVD_OSI": m_osi,
                "HAR_SVD_T3": m_t3,
                "DNN_HAR": m3,
                "DNN_HAR+SVD": m4,
                "LSTM_HAR": m5,
                "LSTM_HAR+SVD": m6,
                "HARNet": m7,
                "GNN": m8,
                "GARCH": m_garch,
                "GJR-GARCH-t": m_gjr,
                "Combo_EqWt": m_combo_eq,
                "Combo_InvMSE": m_combo_mse,
                "Combo_BG": m_combo_bg,
                "Combo_BG_meta": {
                    "w_har": float(w_har_bg),
                    "w_har_lo": float(bg_in_sample["w_a_lo"]) if bg_in_sample else float("nan"),
                    "w_har_hi": float(bg_in_sample["w_a_hi"]) if bg_in_sample else float("nan"),
                    "n_boot": int(bg_in_sample["n_boot"]) if bg_in_sample else 0,
                },
                **abl_rb,
            }

            # Store all-horizon predictions for multi-horizon DM tests
            # LSTM/HARNet predictions are shorter; we keep them as-is and align
            # in the DM test by taking min(len(true_vol), len(pred)).
            h_preds: dict = {
                "true_vol": true_vol,
                "test_dates": test_dates,
                "HAR": har_pred,
                "HAR_SVD_T1": svd_t1_pred,
                "HAR+SVD": har_svd_pred,
                "HAR+SVD_GATED_AR": gated_pred,
                "HAR+SVD_TAILCAL_AR": tailcal_pred,
                "HAR+SVD_RFF_RIDGE": rff_pred,
                "HAR+SVD_OSI": osi_pred,
                "HAR_SVD_T3": svd_xs_pred,
                "DNN_HAR": dnn_har_pred,
                "DNN_HAR+SVD": dnn_svd_pred,
                "GARCH": garch_pred_h,
                "GJR-GARCH-t": gjr_pred_h,
                "Combo_EqWt": combo_eq_pred,
                "Combo_InvMSE": combo_mse_pred,
                "Combo_BG": combo_bg_pred,
            }
            # Pad sequence-model predictions to match true_vol length with NaN
            for name, pred_arr in [
                ("LSTM_HAR", lstm_har_pred),
                ("LSTM_HAR+SVD", lstm_svd_pred),
                ("HARNet", harnet_pred),
            ]:
                arr = np.full(len(true_vol), np.nan, dtype=float)
                arr[:min(len(true_vol), len(pred_arr))] = pred_arr[:min(len(true_vol), len(pred_arr))]
                h_preds[name] = arr
            if gnn_ok:
                h_preds["GNN"] = gnn_pred
            else:
                h_preds["GNN"] = np.full(len(true_vol), np.nan)
            h_preds.update(abl_preds)
            ap[h] = h_preds

            if h == 1 and is_primary:
                # har_pred_log here is from the ElasticNet HAR (M1)
                raw_smeared_metrics_h1 = {
                    "HAR": {
                        "smeared": ev.four_metrics(true_vol, har_pred, eps),
                        "raw_exp_log_clip": ev.four_metrics(
                            true_vol, pred_var_from_log_raw(har_pred_log, y_fit), eps
                        ),
                    },
                    "HAR+SVD": {
                        "smeared": ev.four_metrics(true_vol, har_svd_pred, eps),
                        "raw_exp_log_clip": ev.four_metrics(
                            true_vol, pred_var_from_log_raw(har_svd_pred_log, y_fit), eps
                        ),
                    },
                    "DNN_HAR": {
                        "smeared": ev.four_metrics(true_vol, dnn_har_pred, eps),
                        "raw_exp_log_clip": ev.four_metrics(
                            true_vol, pred_var_from_log_raw(dnn_har_pred_log, y_har_tr), eps
                        ),
                    },
                    "DNN_HAR+SVD": {
                        "smeared": ev.four_metrics(true_vol, dnn_svd_pred, eps),
                        "raw_exp_log_clip": ev.four_metrics(
                            true_vol, pred_var_from_log_raw(dnn_svd_pred_log, y_svd_tr), eps
                        ),
                    },
                }

            # Store h=1 predictions for figures and Bayesian uncertainty (primary split only)
            if h == 1 and is_primary:
                # Training calendar rows for GW instrument z-scores (before copying h_preds).
                h_preds["_h1_train_dates_for_gw"] = [
                    str(pd.Timestamp(t)) for t in align_idx[train_mask]
                ]
                all_predictions = {k: v for k, v in h_preds.items()}
                # Conjugate Bayesian HAR uses same fit set as M1 (train minus val tail).
                X_har_train_h1 = X_har_fit
                y_train_h1 = y_fit
                X_har_test_h1 = X_har_h[test_mask]
                m2_pipeline_h1 = har_svd_elastic
                X_m2_train_boot = np.asarray(X_svd_fit, dtype=np.float64)
                y_m2_train_boot = np.asarray(y_fit, dtype=np.float64)
                X_m2_test_boot = np.asarray(X_svd_test, dtype=np.float64)
                feature_names_m2_h1 = list(har_svd_names)

                # SVD crisis flags, cos_theta, and angle for regime / GW test analysis.
                # Also store in h_preds so all_predictions_by_horizon[1] has these
                # for the GW test section (which reads from h1_preds).
                try:
                    _cos_theta_vals = svd_df.loc[test_dates, "cos_theta"].values
                    all_predictions["cos_theta"] = _cos_theta_vals
                    h_preds["cos_theta"] = _cos_theta_vals
                except Exception:
                    all_predictions["cos_theta"] = np.full(len(test_dates), np.nan)
                try:
                    _angle_vals = svd_df.loc[test_dates, "angle"].values
                    all_predictions["angle_test"] = _angle_vals
                    h_preds["angle_test"] = _angle_vals
                except Exception:
                    all_predictions["angle_test"] = np.full(len(test_dates), np.nan)
                try:
                    _spectral_gap_vals = svd_df.loc[test_dates, "spectral_gap"].values
                    all_predictions["spectral_gap_test"] = _spectral_gap_vals
                    h_preds["spectral_gap_test"] = _spectral_gap_vals
                except Exception:
                    all_predictions["spectral_gap_test"] = np.full(len(test_dates), np.nan)
                for th in crisis_thresholds:
                    col = f"crisis_{th}"
                    if col in svd_df.columns:
                        try:
                            _crisis_vals = svd_df.loc[test_dates, col].values
                            all_predictions[col] = _crisis_vals
                            h_preds[col] = _crisis_vals
                        except Exception:
                            pass

            fsc.mark_horizon_complete(rb[int(h)], ap[int(h)])

        split_sensitivity_metrics[str(train_split)] = rb
        if is_primary:
            results_by_horizon = rb
            all_predictions_by_horizon = ap

    # -----------------------------------------------------------------------
    # 4. Bayesian uncertainty quantification (h=1)
    # -----------------------------------------------------------------------
    uncertainty: dict = {}
    print("\n[INFO] Computing Bayesian uncertainty intervals (h=1) ...")

    if model_dnn_svd_h1 is not None and X_har_svd_h1_scaled_test is not None:
        try:
            dnn_mean, dnn_std, dnn_lo, dnn_hi = bayes.mc_dropout_predict(
                model_dnn_svd_h1, X_har_svd_h1_scaled_test, T=mc_T
            )
            # Heteroskedastic lognormal correction: E[var] = exp(mu + 0.5*sigma^2)
            # where mu = E[log_var|X] from MC Dropout mean, sigma^2 = Var[log_var|X].
            # This is more precise than Duan smearing when variance of log-prediction
            # is available per sample.
            dnn_svd_lognorm_corrected = np.exp(dnn_mean + 0.5 * dnn_std ** 2)
            uncertainty["DNN+SVD"] = {
                "mean_log": dnn_mean,
                "std_log": dnn_std,
                "lower_log": dnn_lo,
                "upper_log": dnn_hi,
                "lower_vol": np.exp(dnn_lo),
                "upper_vol": np.exp(dnn_hi),
                "lognorm_corrected": dnn_svd_lognorm_corrected,
            }
            print(
                f"  DNN+SVD MC Dropout: mean std={dnn_std.mean():.4f}  "
                f"lognorm correction factor={np.exp(0.5 * dnn_std.mean()**2):.4f}"
            )
        except Exception as e:
            print(f"  [WARN] MC Dropout for DNN failed: {e}")

    if model_lstm_svd_h1 is not None and X_har_seq_svd_test_h1 is not None:
        try:
            lstm_mean, lstm_std, lstm_lo, lstm_hi = bayes.mc_dropout_predict(
                model_lstm_svd_h1, X_har_seq_svd_test_h1, T=mc_T
            )
            uncertainty["LSTM+SVD"] = {
                "mean_log": lstm_mean,
                "std_log": lstm_std,
                "lower_log": lstm_lo,
                "upper_log": lstm_hi,
                "lower_vol": np.exp(lstm_lo),
                "upper_vol": np.exp(lstm_hi),
            }
            print(f"  LSTM+SVD MC Dropout: mean std={lstm_std.mean():.4f}")
        except Exception as e:
            print(f"  [WARN] MC Dropout for LSTM failed: {e}")

    # Conjugate Bayesian HAR (add constant column as done in models_linear)
    import statsmodels.api as sm
    if X_har_train_h1 is not None:
        try:
            X_tr_bayes = sm.add_constant(X_har_train_h1)
            X_te_bayes = sm.add_constant(X_har_test_h1)
            bay_mean, bay_lo, bay_hi = bayes.bayesian_har_predict(
                X_tr_bayes, y_train_h1, X_te_bayes
            )
            bay_coef_samples = bayes.bayesian_har_posterior_samples(
                X_tr_bayes, y_train_h1, n_samples=2000
            )
            uncertainty["HAR_Bayes"] = {
                "mean_log": bay_mean,
                "lower_log": bay_lo,
                "upper_log": bay_hi,
                "lower_vol": np.exp(bay_lo),
                "upper_vol": np.exp(bay_hi),
                "coef_samples": bay_coef_samples,
                # har_names labels the HAR feature columns (without constant)
                "feature_names": har_names,
            }
            print("  Bayesian HAR: posterior predictive intervals computed.")
        except Exception as e:
            print(f"  [WARN] Bayesian HAR failed: {e}")

    # Block-bootstrap predictive bands for M2 (ElasticNet) — same estimator as F1 HAR+SVD curve
    boot_cfg = config.config.get("models", {}).get("har_svd_bootstrap", {})
    if (
        boot_cfg.get("enabled", True)
        and m2_pipeline_h1 is not None
        and X_m2_train_boot is not None
        and y_m2_train_boot is not None
        and X_m2_test_boot is not None
    ):
        try:
            lo_b, hi_b = bayes.block_bootstrap_predictive_quantiles(
                m2_pipeline_h1,
                X_m2_train_boot,
                y_m2_train_boot,
                X_m2_test_boot,
                n_boot=int(boot_cfg.get("n_boot", 150)),
                block_len=int(boot_cfg.get("block_len", 22)),
                random_state=config.config["seed"],
                min_success_frac=float(boot_cfg.get("min_success_frac", 0.5)),
            )
            uncertainty["HAR_SVD_Bootstrap"] = {
                "lower_log": lo_b,
                "upper_log": hi_b,
                "lower_vol": np.exp(np.clip(lo_b, -30, 30)),
                "upper_vol": np.exp(np.clip(hi_b, -30, 30)),
            }
            print("  HAR+SVD ElasticNet: block-bootstrap 95% predictive intervals (log scale).")
        except Exception as e:
            print(f"  [WARN] HAR+SVD block bootstrap failed: {e}")

    print(
        "[INFO] Bayesian/empirical uncertainty components available: "
        f"{list(uncertainty.keys())}"
    )
    if not uncertainty:
        print(
            "[WARN] No uncertainty entries (check MC Dropout models, "
            "Bayesian HAR inputs, and M2 bootstrap config)."
        )

    out_dir = METRICS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    exp_fp_primary = eckpt.config_fingerprint(
        seed=config.config["seed"],
        horizons=list(horizons),
        train_split=float(primary_split),
        returns_path=config.config["data"]["returns_cache_path"],
        extra={
            "mode": "fixed_split",
            "osi": bool(config.config.get("features", {}).get("osi_elastic_net_enabled", True)),
            "inc_xs": bool(config.config.get("features", {}).get("include_cross_section_primary", True)),
        },
    )
    # -----------------------------------------------------------------------
    # 4b. Fractional-Kelly targeting, Moreira-Muir alpha, FKO CE fee (h=1)
    # -----------------------------------------------------------------------
    print("\n[INFO] Economic value: fractional Kelly, Moreira-Muir, FKO ...")
    fk_export: list[dict] = []
    mm_export: list[dict] = []
    fko_export: list[dict] = []
    boot_ci_export: list[dict] = []

    try:
        import statsmodels.api as sm

        test_ix = pd.to_datetime(pd.Index(all_predictions.get("test_dates", [])))
        pred_h1 = all_predictions_by_horizon.get(1, {})
        if len(test_ix) > 40 and pred_h1.get("HAR") is not None:
            r_dec = (
                portfolio_returns.reindex(test_ix).shift(-1).values.astype(np.float64)
                / 100.0
            )
            s2_by_tag: dict[str, np.ndarray] = {}
            if X_har_train_h1 is not None and X_har_test_h1 is not None:
                Xtr = sm.add_constant(X_har_train_h1)
                Xte = sm.add_constant(X_har_test_h1)
                s2_by_tag["HAR"] = bayes.bayesian_har_predictive_variance_log(
                    Xtr, y_train_h1, Xte
                )
            if "HAR_SVD_Bootstrap" in uncertainty:
                hb = uncertainty["HAR_SVD_Bootstrap"]
                s2_by_tag["HAR+SVD"] = bayes.posterior_var_log_from_log_quantiles(
                    hb["lower_log"], hb["upper_log"]
                )
            if "DNN+SVD" in uncertainty and "std_log" in uncertainty["DNN+SVD"]:
                s2_by_tag["DNN_HAR+SVD"] = bayes.posterior_var_log_from_mc_dropout(
                    uncertainty["DNN+SVD"]["std_log"]
                )
            if "LSTM+SVD" in uncertainty and "std_log" in uncertainty["LSTM+SVD"]:
                s2_by_tag["LSTM_HAR+SVD"] = bayes.posterior_var_log_from_mc_dropout(
                    uncertainty["LSTM+SVD"]["std_log"]
                )

            model_pred_tags = [
                ("HAR", "HAR"),
                ("HAR+SVD", "HAR+SVD"),
                ("HAR_SVD_T3", "HAR_SVD_T3"),
                ("DNN_HAR+SVD", "DNN_HAR+SVD"),
                ("LSTM_HAR+SVD", "LSTM_HAR+SVD"),
            ]
            rp_vt_har = None
            rp_vt_svd = None
            for disp_name, pred_key in model_pred_tags:
                pv = pred_h1.get(pred_key)
                if pv is None:
                    continue
                pv_a = np.asarray(pv, dtype=np.float64).ravel()
                n_use = min(len(r_dec), len(pv_a))
                if n_use < 40:
                    continue
                r_s = r_dec[:n_use]
                var_pct2 = pv_a[:n_use]
                h_dec = var_pct2 / (100.0 ** 2)
                s2_full = s2_by_tag.get(pred_key)
                if s2_full is not None:
                    s2_a = np.asarray(s2_full, dtype=np.float64).ravel()
                    n_s = min(n_use, len(s2_a))
                    r_s = r_s[:n_s]
                    h_dec = h_dec[:n_s]
                    s2_use = s2_a[:n_s]
                else:
                    n_s = n_use
                    s2_use = None
                for rule in ("bayes", "rkw", "kl_tilt"):
                    fk_res, _fk_ser = econ.kelly_fractional_targeting(
                        r_s,
                        h_dec,
                        s2_use,
                        rule=rule,  # type: ignore[arg-type]
                        kappa=1.0,
                        target_vol_annual=0.10,
                        max_leverage=3.0,
                        tc_bps=5.0,
                    )
                    fk_export.append({
                        "model": disp_name,
                        "rule": rule,
                        "n": fk_res.n,
                        "sharpe": fk_res.sharpe,
                        "delta_g_log": fk_res.delta_g_log,
                        "mean_fraction": fk_res.mean_fraction,
                        "max_drawdown": fk_res.max_drawdown,
                        "turnover": fk_res.turnover,
                    })
                mm_d = econ.moreira_muir_alpha(
                    r_s, h_dec, tc_bps=0.0, horizon=1,
                )
                mm_export.append({"model": disp_name, **mm_d})
                _, ser_vt = econ.volatility_targeting(
                    r_s, h_dec,
                    target_vol_annual=0.10,
                    max_leverage=3.0,
                    tc_bps=5.0,
                )
                if disp_name == "HAR":
                    rp_vt_har = ser_vt["rp"]
                elif disp_name == "HAR+SVD":
                    rp_vt_svd = ser_vt["rp"]

            if rp_vt_har is not None and rp_vt_svd is not None:
                n_f = min(len(rp_vt_har), len(rp_vt_svd))
                if n_f >= 40:
                    fko_d = econ.fko_ce_fee(rp_vt_har[:n_f], rp_vt_svd[:n_f])
                    fko_export.append({"pair": "HAR+SVD_vs_HAR", **fko_d})
                    ci_sh = econ.stationary_block_bootstrap_ci(
                        rp_vt_har[:n_f],
                        rp_vt_svd[:n_f],
                        statistic="sharpe_diff",
                        n_boot=1999,
                        seed=int(config.config["seed"]),
                    )
                    boot_ci_export.append({"metric": "sharpe_diff_HAR_SVD_minus_HAR", **ci_sh})

    except Exception as econ_misc_err:
        print(f"  [WARN] Fractional-Kelly / MM / FKO export failed: {econ_misc_err}")

    if fk_export:
        with open(out_dir / "fractional_kelly.json", "w") as f:
            json.dump(to_serializable(fk_export), f, indent=2)
        pd.DataFrame(fk_export).to_csv(out_dir / "fractional_kelly.csv", index=False)
        print(f"  [INFO] Fractional-Kelly table saved ({len(fk_export)} rows).")
    if mm_export:
        with open(out_dir / "moreira_muir_alpha.json", "w") as f:
            json.dump(to_serializable(mm_export), f, indent=2)
        pd.DataFrame(mm_export).to_csv(out_dir / "moreira_muir_alpha.csv", index=False)
    if fko_export:
        with open(out_dir / "fko_certainty_equivalent.json", "w") as f:
            json.dump(to_serializable(fko_export), f, indent=2)
    if boot_ci_export:
        with open(out_dir / "economic_bootstrap_ci.json", "w") as f:
            json.dump(to_serializable(boot_ci_export), f, indent=2)

    # -----------------------------------------------------------------------
    # 5. Compute Mincer-Zarnowitz R2 for all models at h=1
    # -----------------------------------------------------------------------
    print("\n[INFO] Computing Mincer-Zarnowitz efficiency tests (h=1) ...")
    mz_results = {}
    true_vol_h1 = all_predictions_by_horizon[1]["true_vol"]
    for model_name, pred_arr in all_predictions_by_horizon[1].items():
        if model_name in _PRED_DICT_SKIP_KEYS or model_name.startswith("crisis_"):
            continue
        if model_name.startswith("_"):
            continue
        if pred_arr is None:
            continue
        try:
            pa = np.asarray(pred_arr, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if pa.size == 0 or not np.isfinite(pa).any():
            continue
        n_min = min(len(true_vol_h1), len(pa))
        mz = ev.mincer_zarnowitz(true_vol_h1[:n_min], pa[:n_min], horizon=1)
        mz_results[model_name] = mz
        se_a = mz.get("mz_alpha_se", float("nan"))
        se_b = mz.get("mz_beta_se", float("nan"))
        ta = mz.get("mz_alpha_t", float("nan"))
        tb = mz.get("mz_beta_t", float("nan"))
        print(
            f"  {model_name}: alpha={mz['mz_alpha']:.4f} (SE={se_a:.4f}, t={ta:.3f})  "
            f"beta={mz['mz_beta']:.4f} (SE={se_b:.4f}, t={tb:.3f})  R2={mz['mz_r2']:.4f}"
        )

    # -----------------------------------------------------------------------
    # 6. Save results
    # -----------------------------------------------------------------------
    with open(out_dir / "results_by_horizon.json", "w") as f:
        json.dump(to_serializable(results_by_horizon), f, indent=2)
    print("[INFO] Results saved to", out_dir / "results_by_horizon.json")

    if len(split_list) > 1:
        with open(out_dir / "split_sensitivity_metrics.json", "w") as f:
            json.dump(to_serializable(split_sensitivity_metrics), f, indent=2)
        print("[INFO] Split sensitivity saved to", out_dir / "split_sensitivity_metrics.json")

    if raw_smeared_metrics_h1 is not None:
        with open(out_dir / "metrics_raw_vs_smeared_h1.json", "w") as f:
            json.dump(to_serializable(raw_smeared_metrics_h1), f, indent=2)
        print("[INFO] Raw vs smeared h=1 metrics saved.")

    with open(out_dir / "mz_results.json", "w") as f:
        json.dump(to_serializable(mz_results), f, indent=2)
    print("[INFO] Mincer-Zarnowitz results saved.")
    if mz_results:
        mz_cols = [
            "model",
            "mz_alpha", "mz_alpha_se", "mz_alpha_t", "mz_alpha_p",
            "mz_beta", "mz_beta_se", "mz_beta_t", "mz_beta_p",
            "mz_r2",
        ]
        mz_tab = []
        for mname, mz in sorted(mz_results.items(), key=lambda x: x[0]):
            row = {c: mz.get(c, float("nan")) for c in mz_cols if c != "model"}
            row["model"] = mname
            for k, v in mz.items():
                if k not in row:
                    row[k] = v
            mz_tab.append(row)
        all_mz_keys = set()
        for r in mz_tab:
            all_mz_keys.update(r.keys())
        extra_cols = sorted(k for k in all_mz_keys if k not in mz_cols)
        df_mz = pd.DataFrame(mz_tab).reindex(columns=mz_cols + extra_cols)
        df_mz.to_csv(out_dir / "mz_results.csv", index=False)
        print("[INFO] Mincer-Zarnowitz table saved -> mz_results.csv")

    hp = config.get_hyperparameter_table()
    with open(out_dir / "hyperparameters.json", "w") as f:
        json.dump(to_serializable(hp), f, indent=2)
    print("[INFO] Hyperparameters saved.")

    # -----------------------------------------------------------------------
    # 7. DM tests at all three horizons
    # -----------------------------------------------------------------------
    print("\n[INFO] Computing DM tests at all horizons ...")
    cfg_inf = config.config.get("inference", {})
    dm_all_horizons: dict[int, pd.DataFrame] = {}

    for h in horizons:
        h_preds_dm = all_predictions_by_horizon.get(h, {})
        y_true_h = h_preds_dm.get("true_vol")
        if y_true_h is None:
            continue
        preds_for_dm = {
            k: v for k, v in h_preds_dm.items()
            if k not in _PRED_DICT_SKIP_KEYS
            and not k.startswith("crisis_")
            and v is not None
            and np.isfinite(v).sum() > 50
        }
        use_det = bool(cfg_inf.get("dm_export_detailed_columns", True))
        dm_df_h = _run_dm_tests(preds_for_dm, y_true_h, horizon=h, use_detailed=use_det)
        # Add FDR-adjusted p-values separately for MSE and QLIKE tests
        dm_df_h = ev.dm_add_fdr_columns(dm_df_h, p_col="p_MSE").rename(
            columns={"p_fdr_bh": "p_MSE_fdr_bh", "reject_fdr_0.05": "reject_MSE_fdr_0.05"}
        )
        dm_df_h = ev.dm_add_fdr_columns(dm_df_h, p_col="p_QLIKE").rename(
            columns={"p_fdr_bh": "p_QLIKE_fdr_bh", "reject_fdr_0.05": "reject_QLIKE_fdr_0.05"}
        )
        dm_all_horizons[h] = dm_df_h
        if not dm_df_h.empty:
            fname = f"dm_tests_h{h}.csv"
            dm_df_h.to_csv(out_dir / fname, index=False)
            print(f"  [INFO] DM tests h={h} saved ({len(dm_df_h)} pairs) -> {fname}")

            # Print key SVD vs baseline comparisons (both MSE and QLIKE DM)
            key_pairs = [
                ("HAR", "HAR+SVD"),
                ("DNN_HAR", "DNN_HAR+SVD"),
                ("LSTM_HAR", "LSTM_HAR+SVD"),
            ]
            for m1_n, m2_n in key_pairs:
                row = dm_df_h[
                    (dm_df_h["Model1"] == m1_n) & (dm_df_h["Model2"] == m2_n)
                ]
                if not row.empty:
                    r = row.iloc[0]
                    sig_mse = "***" if r["p_MSE"] < 0.01 else ("**" if r["p_MSE"] < 0.05 else ("*" if r["p_MSE"] < 0.10 else ""))
                    sig_qlike = "***" if r["p_QLIKE"] < 0.01 else ("**" if r["p_QLIKE"] < 0.05 else ("*" if r["p_QLIKE"] < 0.10 else ""))
                    print(
                        f"    h={h}: {m1_n} vs {m2_n}:  "
                        f"DM_MSE={r['DM_MSE']:.3f} p={r['p_MSE']:.4f}{sig_mse}  |  "
                        f"DM_QLIKE={r['DM_QLIKE']:.3f} p={r['p_QLIKE']:.4f}{sig_qlike}"
                    )

    # Backward-compatible alias for h=1 (legacy code may reference dm_tests.csv)
    if 1 in dm_all_horizons and not dm_all_horizons[1].empty:
        dm_all_horizons[1].to_csv(out_dir / "dm_tests.csv", index=False)

    # --- 7a. Feature ablation: DM (full M2 vs M2−G) + summary ----------------
    fab_out = config.config.get("feature_ablation", {})
    abl_pf_out = str(fab_out.get("model_key_prefix", "HAR+SVD_minus_"))
    q_tol_out = float(fab_out.get("qlike_noninferiority_delta", 0.0))
    if fab_out.get("enabled", False):
        summary_payload: dict = {
            "qlike_noninferiority_delta": q_tol_out,
            "model_key_prefix": abl_pf_out,
            "horizons": {},
        }
        for h in horizons:
            hp = all_predictions_by_horizon.get(h, {})
            yt = hp.get("true_vol")
            base = hp.get("HAR+SVD")
            har_b = hp.get("HAR")
            if yt is None or base is None:
                continue
            rows_ab = []
            abl_keys = sorted(
                k for k in hp
                if k.startswith(abl_pf_out)
                and hp[k] is not None
                and np.isfinite(hp[k]).sum() > 50
            )
            for ak in abl_keys:
                pred_m = hp[ak]
                d_full = ev.dmw_test_qlike_detailed(
                    yt, base, pred_m, nlags=None, horizon=int(h),
                )
                row_ab = {
                    "horizon": h,
                    "comparison": f"HAR+SVD_vs_{ak}",
                    "ablation_key": ak,
                    **d_full,
                }
                if har_b is not None and np.isfinite(har_b).sum() > 50:
                    d_har = ev.dmw_test_qlike_detailed(
                        yt, har_b, pred_m, nlags=None, horizon=int(h),
                    )
                    row_ab["DM_QLIKE_HAR_vs_ablation"] = d_har.get("dm_stat")
                    row_ab["p_QLIKE_HAR_vs_ablation"] = d_har.get("p_value")
                rows_ab.append(row_ab)

            rb_h = results_by_horizon.get(h, {})
            h_summary: dict = {}
            for ak in abl_keys:
                qfull = float(rb_h.get("HAR+SVD", {}).get("QLIKE", float("nan")))
                qminus = float(rb_h.get(ak, {}).get("QLIKE", float("nan")))
                p_dm = float("nan")
                for r in rows_ab:
                    if r["ablation_key"] == ak:
                        p_dm = float(r.get("p_value", float("nan")))
                        break
                qdiff = qminus - qfull
                h_summary[ak] = {
                    "QLIKE_full_M2": qfull,
                    "QLIKE_ablation": qminus,
                    "QLIKE_diff_ablation_minus_full": qdiff,
                    "DM_pvalue_full_vs_ablation": p_dm,
                    "redundant_by_rule": (
                        np.isfinite(p_dm) and p_dm > 0.05
                        and np.isfinite(qdiff)
                        and qdiff <= q_tol_out
                    ),
                }
            if rows_ab:
                df_ab = pd.DataFrame(rows_ab)
                df_ab = ev.dm_add_fdr_columns(df_ab, p_col="p_value").rename(
                    columns={
                        "p_fdr_bh": "p_QLIKE_DM_fdr_bh",
                        "reject_fdr_0.05": "reject_QLIKE_DM_fdr_0.05",
                    }
                )
                df_ab.to_csv(out_dir / f"feature_ablation_dm_h{h}.csv", index=False)
                print(f"  [INFO] Feature ablation DM h={h} -> feature_ablation_dm_h{h}.csv")
            if h_summary:
                summary_payload["horizons"][str(h)] = h_summary
        with open(out_dir / "feature_ablation_summary.json", "w") as f:
            json.dump(to_serializable(summary_payload), f, indent=2)
        print("[INFO] Feature ablation summary -> feature_ablation_summary.json")

    # --- 7a2. Encompassing regression (full HAC table) per horizon --------------
    for h in horizons:
        hp = all_predictions_by_horizon.get(h, {})
        yt_e = hp.get("true_vol")
        ph = hp.get("HAR")
        ps = hp.get("HAR+SVD")
        if yt_e is None or ph is None or ps is None:
            continue
        n_e = min(len(yt_e), len(ph), len(ps))
        if n_e < 20:
            continue
        try:
            enc_h = ev.forecast_encompassing_test(
                yt_e[:n_e], ph[:n_e], ps[:n_e], horizon=int(h),
            )
            enc_row = {"horizon": h, **{k: enc_h[k] for k in sorted(enc_h.keys())}}
            enc_serial = {k: enc_row[k] for k in enc_row}
            with open(out_dir / f"encompassing_full_h{h}.json", "w") as fj:
                json.dump(to_serializable(enc_serial), fj, indent=2)
            pd.DataFrame([enc_row]).to_csv(
                out_dir / f"encompassing_full_h{h}.csv", index=False
            )
            print(
                f"  [INFO] Encompassing full table -> encompassing_full_h{h}.csv "
                f"& encompassing_full_h{h}.json"
            )
        except Exception as enc_csv_err:
            print(f"  [WARN] Encompassing CSV h={h} failed: {enc_csv_err}")

    # -----------------------------------------------------------------------
    # 7b. Model Confidence Set (MCS) — Hansen, Lunde & Nason (2011)
    #
    # MCS identifies the smallest set of models that contains the genuinely
    # best model with probability >= 90%. Unlike pairwise DM, MCS controls
    # for simultaneous comparison of all models. Computed for BOTH QLIKE
    # (primary, proxy-robust) and MSE (secondary) losses at each horizon.
    # -----------------------------------------------------------------------
    print("\n[INFO] Computing Model Confidence Sets ...")
    mcs_results: dict = {}
    eps_mcs = float(cfg_train.get("eps", 1e-8))
    for h in horizons:
        h_preds_dm = all_predictions_by_horizon.get(h, {})
        y_true_h = h_preds_dm.get("true_vol")
        if y_true_h is None:
            continue
        model_pred_pairs = [
            (k, v) for k, v in h_preds_dm.items()
            if k not in _PRED_DICT_SKIP_KEYS
            and not k.startswith("crisis_")
            and v is not None
            and np.isfinite(v).sum() > 50
        ]
        fab_mcs = config.config.get("feature_ablation", {})
        abl_pf = str(fab_mcs.get("model_key_prefix", "HAR+SVD_minus_"))
        if not fab_mcs.get("include_ablations_in_mcs", False):
            model_pred_pairs = [
                (k, v) for k, v in model_pred_pairs if not k.startswith(abl_pf)
            ]
        cfg_inf_mcs = config.config.get("inference", {})
        if cfg_inf_mcs.get("mcs_core_models_only", False):
            allow = cfg_inf_mcs.get("mcs_core_model_names") or []
            if allow:
                allowed = frozenset(str(x) for x in allow)
                model_pred_pairs = [(k, v) for k, v in model_pred_pairs if k in allowed]
        if len(model_pred_pairs) < 2:
            continue

        mcs_model_names = [name for name, _ in model_pred_pairs]
        T_mcs = len(y_true_h)

        # Build loss matrices (T x n_models)
        mse_mat = np.full((T_mcs, len(mcs_model_names)), np.nan)
        qlike_mat = np.full((T_mcs, len(mcs_model_names)), np.nan)
        h_true = np.maximum(y_true_h, eps_mcs)

        for col_idx, (mname, preds) in enumerate(model_pred_pairs):
            n_align = min(T_mcs, len(preds))
            h_hat = np.maximum(preds[:n_align], eps_mcs)
            # MSE loss
            mse_mat[:n_align, col_idx] = (y_true_h[:n_align] - preds[:n_align]) ** 2
            # QLIKE loss: h/h_hat - log(h/h_hat) - 1
            ratio = h_true[:n_align] / h_hat
            qlike_mat[:n_align, col_idx] = ratio - np.log(ratio) - 1.0

        try:
            mcs_seed = int(config.config.get("seed", 42))
            mcs_qlike = ev.compute_mcs(qlike_mat, mcs_model_names, alpha=0.10, seed=mcs_seed)
            mcs_mse = ev.compute_mcs(mse_mat, mcs_model_names, alpha=0.10, seed=mcs_seed)
            # Tighter 75% MCS (alpha=0.25) improves power when 90% set includes all models.
            mcs_qlike_75 = ev.compute_mcs(qlike_mat, mcs_model_names, alpha=0.25, seed=mcs_seed + 791)
            mcs_mse_75 = ev.compute_mcs(mse_mat, mcs_model_names, alpha=0.25, seed=mcs_seed + 792)
            # Block-length sweep so the headline MCS does not depend on a
            # single bootstrap block-size choice (Politis-White 2004
            # robustness check).
            mcs_qlike_sweep = ev.compute_mcs_block_sweep(
                qlike_mat, mcs_model_names, horizon=int(h),
                alpha=0.10, n_boot=1999, seed=mcs_seed + 800,
            )
            mcs_mse_sweep = ev.compute_mcs_block_sweep(
                mse_mat, mcs_model_names, horizon=int(h),
                alpha=0.10, n_boot=1999, seed=mcs_seed + 900,
            )
            mcs_results[h] = {
                "QLIKE": mcs_qlike, "MSE": mcs_mse,
                "QLIKE_75": mcs_qlike_75, "MSE_75": mcs_mse_75,
                "QLIKE_block_sweep": mcs_qlike_sweep,
                "MSE_block_sweep": mcs_mse_sweep,
            }

            print(f"  h={h} MCS (QLIKE, 90%): {mcs_qlike['mcs_set']}")
            print(f"  h={h} MCS (MSE,   90%): {mcs_mse['mcs_set']}")
            print(f"  h={h} MCS (QLIKE, 75%): {mcs_qlike_75['mcs_set']}")
            print(f"  h={h} MCS (MSE,   75%): {mcs_mse_75['mcs_set']}")
            sweep_keys = [
                k for k in mcs_qlike_sweep
                if k != "meta" and isinstance(mcs_qlike_sweep[k], dict)
            ]
            for k in sweep_keys:
                ent = mcs_qlike_sweep[k]
                if "mcs_set" in ent:
                    print(f"  h={h} MCS QLIKE sweep block_len={k}: {ent['mcs_set']}")

            # Save MCS results to CSV
            mcs_rows = []
            _p_note = (
                "p_values are tied for surviving models at the last non-rejection round "
                "(see evaluation.compute_mcs)"
            )
            for mname in mcs_model_names:
                mcs_rows.append({
                    "Horizon": h,
                    "Model": mname,
                    "in_MCS_QLIKE": mcs_qlike["included"].get(mname, False),
                    "p_QLIKE": mcs_qlike["p_values"].get(mname, float("nan")),
                    "in_MCS_MSE": mcs_mse["included"].get(mname, False),
                    "p_MSE": mcs_mse["p_values"].get(mname, float("nan")),
                    "in_MCS_QLIKE_75": mcs_qlike_75["included"].get(mname, False),
                    "p_QLIKE_75": mcs_qlike_75["p_values"].get(mname, float("nan")),
                    "in_MCS_MSE_75": mcs_mse_75["included"].get(mname, False),
                    "p_MSE_75": mcs_mse_75["p_values"].get(mname, float("nan")),
                    "MCS_p_value_note": _p_note,
                })
            mcs_df = pd.DataFrame(mcs_rows)
            mcs_df.to_csv(out_dir / f"mcs_h{h}.csv", index=False)
            print(f"  [INFO] MCS h={h} saved -> mcs_h{h}.csv")
        except Exception as mcs_err:
            print(f"  [WARN] MCS failed for h={h}: {mcs_err}")

    # -----------------------------------------------------------------------
    # 7b-opt. Clark–West nested MSPE, White RC, Hansen-style SPA, split HAC
    # -----------------------------------------------------------------------
    opt_inf = (cfg_inf.get("optional_inference") or {})
    if opt_inf.get("enabled", False):
        from sklearn.pipeline import Pipeline as SkPipeline

        print("\n[INFO] Optional inference (Clark–West, RC, SPA, split-sample HAC) ...")
        n_boot_o = int(opt_inf.get("bootstrap_n_boot", 1999))
        blk_o = int(opt_inf.get("bootstrap_block_len", 22))
        bench_m = str(opt_inf.get("reality_check_benchmark", "HAR"))
        opt_seed0 = int(config.config.get("seed", 42))
        optional_blob: dict = {}

        def _pred_pairs_opt(hpd: dict):
            pp = [
                (k, v) for k, v in hpd.items()
                if k not in _PRED_DICT_SKIP_KEYS and not k.startswith("crisis_")
                and v is not None and np.isfinite(v).sum() > 50
            ]
            fab_o2 = config.config.get("feature_ablation", {})
            pfx2 = str(fab_o2.get("model_key_prefix", "HAR+SVD_minus_"))
            if not fab_o2.get("include_ablations_in_mcs", False):
                pp = [(k, v) for k, v in pp if not k.startswith(pfx2)]
            return pp

        for h in horizons:
            hpd = all_predictions_by_horizon.get(h, {})
            yt = hpd.get("true_vol")
            if yt is None:
                continue
            ppairs = _pred_pairs_opt(hpd)
            cw_list = []
            for rawp in opt_inf.get("clark_west_pairs", []):
                if len(rawp) != 2:
                    continue
                rnm, unm = str(rawp[0]), str(rawp[1])
                if rnm not in hpd or unm not in hpd:
                    continue
                pr, pu = hpd[rnm], hpd[unm]
                nn = min(len(yt), len(pr), len(pu))
                if nn < 20:
                    continue
                cw = ev.clark_west_nested_mspe(
                    yt[:nn], pr[:nn], pu[:nn], nlags=None, horizon=int(h),
                )
                cw_list.append({"restricted": rnm, "unrestricted": unm, **cw})
            horizon_entry: dict = {"clark_west": cw_list}
            names_o = [k for k, _ in ppairs]
            if (
                opt_inf.get("reality_check_spa_mse", False)
                and bench_m in names_o
                and len(names_o) >= 2
            ):
                order = [bench_m] + [m for m in names_o if m != bench_m]
                Tm = len(yt)
                Lm = np.full((Tm, len(order)), np.nan)
                for j, mn in enumerate(order):
                    p = hpd[mn]
                    nn = min(Tm, len(p))
                    Lm[:nn, j] = (yt[:nn] - p[:nn]) ** 2
                fin = np.isfinite(Lm).all(axis=1)
                Lmf = Lm[fin]
                if Lmf.shape[0] >= 30:
                    rc = ev.white_reality_check_bootstrap(
                        Lmf, 0, n_boot=n_boot_o, block_len=blk_o, seed=opt_seed0 + h,
                    )
                    altn = [m for m in order if m != bench_m]
                    means_raw = rc.get("means_vs_bench") or {}
                    rc["means_vs_bench_named"] = {
                        altn[i]: list(means_raw.values())[i]
                        for i in range(min(len(altn), len(means_raw)))
                    }
                    spa = ev.hansen_spa_studentized_bootstrap(
                        Lmf, 0, nlags=None, n_boot=n_boot_o, block_len=blk_o,
                        seed=opt_seed0 + 100 + h, horizon=int(h),
                    )
                    altn2 = [m for m in order if m != bench_m]
                    tmap = spa.get("t_by_alt") or {}
                    spa["t_by_model_named"] = {
                        altn2[i]: list(tmap.values())[i]
                        for i in range(min(len(altn2), len(tmap)))
                    }
                    horizon_entry["white_reality_check_mse"] = rc
                    horizon_entry["hansen_spa_mse"] = spa

                # --- QLIKE-loss RC and SPA (primary loss; PRE-REGISTERED) ----
                eps_qlike_loss = ev.EPS_DEFAULT
                Lq = np.full((Tm, len(order)), np.nan)
                for j, mn in enumerate(order):
                    p = hpd[mn]
                    nn = min(Tm, len(p))
                    h_t = np.maximum(yt[:nn], eps_qlike_loss)
                    h_hat = np.maximum(p[:nn], eps_qlike_loss)
                    rq = h_t / h_hat
                    Lq[:nn, j] = rq - np.log(rq) - 1.0
                fin_q = np.isfinite(Lq).all(axis=1)
                Lqf = Lq[fin_q]
                if Lqf.shape[0] >= 30:
                    rc_q = ev.white_reality_check_bootstrap(
                        Lqf, 0, n_boot=n_boot_o, block_len=blk_o,
                        seed=opt_seed0 + 200 + h,
                    )
                    altq = [m for m in order if m != bench_m]
                    means_q = rc_q.get("means_vs_bench") or {}
                    rc_q["means_vs_bench_named"] = {
                        altq[i]: list(means_q.values())[i]
                        for i in range(min(len(altq), len(means_q)))
                    }
                    spa_q = ev.hansen_spa_studentized_bootstrap(
                        Lqf, 0, nlags=None, n_boot=n_boot_o, block_len=blk_o,
                        seed=opt_seed0 + 300 + h, horizon=int(h),
                    )
                    tmap_q = spa_q.get("t_by_alt") or {}
                    spa_q["t_by_model_named"] = {
                        altq[i]: list(tmap_q.values())[i]
                        for i in range(min(len(altq), len(tmap_q)))
                    }
                    # Romano-Wolf step-down on QLIKE loss differentials
                    # (HEADLINE FAMILY, 9 999 replications, FWER = 0.05).
                    rw_q = ev.romano_wolf_step_down(
                        Lqf, benchmark_col=0, model_names=order,
                        n_boot=int(opt_inf.get("rw_n_boot", 9999)),
                        block_len=None,  # Politis-White default
                        seed=opt_seed0 + 400 + h,
                        alpha=float(opt_inf.get("rw_alpha", 0.05)),
                        alternative="greater",
                        horizon=int(h),
                    )
                    horizon_entry["white_reality_check_qlike"] = rc_q
                    horizon_entry["hansen_spa_qlike"] = spa_q
                    horizon_entry["romano_wolf_qlike"] = rw_q
                    if rw_q.get("rejected_models"):
                        print(
                            f"  h={h} RW QLIKE rejected (FWER<={rw_q['alpha']:.2f}): "
                            f"{rw_q['rejected_models']}"
                        )
                    else:
                        print(f"  h={h} RW QLIKE: no rejection at FWER={rw_q.get('alpha', 0.05)}")
            optional_blob[str(h)] = horizon_entry

        if (
            opt_inf.get("split_sample_hac_m2_h1", False)
            and feature_names_m2_h1
            and m2_pipeline_h1 is not None
            and X_m2_train_boot is not None
            and y_m2_train_boot is not None
        ):
            try:
                midf = float(opt_inf.get("split_sample_mid_frac", 0.5))
                n_tr = int(X_m2_train_boot.shape[0])
                cut = max(30, min(n_tr - 30, int(n_tr * midf)))
                Xa = np.asarray(X_m2_train_boot[:cut], dtype=np.float64)
                ya = np.asarray(y_m2_train_boot[:cut], dtype=np.float64)
                Xb = np.asarray(X_m2_train_boot[cut:], dtype=np.float64)
                yb = np.asarray(y_m2_train_boot[cut:], dtype=np.float64)
                pipe_ss = linear_models.train_har_svd_elastic(Xa, ya)
                pre_steps = SkPipeline(pipe_ss.steps[:-1])
                Xb_t = pre_steps.transform(Xb)
                coef_ss = linear_models.get_elastic_coefs(pipe_ss, feature_names_m2_h1)
                active = [nm for nm in feature_names_m2_h1 if abs(coef_ss.get(nm, 0.0)) > 1e-6]
                if active:
                    idx = [feature_names_m2_h1.index(nm) for nm in active]
                    hac_ss_lags = int(wfe.compute_dm_nlags_for_horizon(1, T=int(len(yb))))
                    hac_ss = ev.hac_ols_active_columns(
                        yb, Xb_t[:, idx], active, nlags=hac_ss_lags,
                    )
                    optional_blob["split_sample_hac_m2_h1"] = {
                        "n_first_half": int(cut),
                        "n_second_half": int(len(yb)),
                        "active_features": active,
                        **hac_ss,
                    }
                else:
                    optional_blob["split_sample_hac_m2_h1"] = {
                        "note": "no active ElasticNet coefficients on first half",
                    }
            except Exception as ess:
                optional_blob["split_sample_hac_m2_h1"] = {"error": str(ess)}

        with open(out_dir / "optional_inference.json", "w") as f:
            json.dump(to_serializable(optional_blob), f, indent=2)
        print("[INFO] Optional inference saved -> optional_inference.json")

    # -----------------------------------------------------------------------
    # 7c. Advanced statistical tests: Giacomini-White, Encompassing, VaR
    # -----------------------------------------------------------------------
    print("\n[INFO] Running advanced statistical tests (GW, Encompassing, VaR) ...")

    h1_preds = all_predictions_by_horizon.get(1, {})
    y_true_h1_adv = h1_preds.get("true_vol")
    test_dates_h1_adv = h1_preds.get("test_dates")

    if y_true_h1_adv is not None and "HAR" in h1_preds and "HAR+SVD" in h1_preds:
        pred_har_adv = h1_preds["HAR"]
        pred_svd_adv = h1_preds["HAR+SVD"]
        n_adv = min(len(y_true_h1_adv), len(pred_har_adv), len(pred_svd_adv))

        # --- GW test: condition on angle and spectral_gap (continuous instruments) ---
        # We use continuous instruments only: angle (eigenvector rotation) and
        # spectral_gap (dominant factor gap). Binary crisis_0.8 was all-zero in
        # the test period, making S_hat singular → NaN. Continuous instruments
        # with non-trivial variance give a well-identified test.
        gw_results_all: dict = {}
        try:
            angle_arr = h1_preds.get("angle_test")
            spectral_gap_arr = h1_preds.get("spectral_gap_test")

            Z_parts = [np.ones(n_adv)]
            instrument_labels = ["const"]
            if angle_arr is not None and len(angle_arr) >= n_adv:
                a_col = np.nan_to_num(np.asarray(angle_arr[:n_adv], dtype=float), nan=0.0)
                if np.std(a_col) > 1e-10:
                    Z_parts.append(a_col)
                    instrument_labels.append("angle")
            if spectral_gap_arr is not None and len(spectral_gap_arr) >= n_adv:
                sg_col = np.nan_to_num(np.asarray(spectral_gap_arr[:n_adv], dtype=float), nan=0.0)
                if np.std(sg_col) > 1e-10:
                    Z_parts.append(sg_col)
                    instrument_labels.append("spectral_gap")

            Z_mat = np.column_stack(Z_parts)

            gw_std = bool(cfg_inf.get("gw_standardize_instruments", True))
            gw_mom = str(cfg_inf.get("gw_zscore_moments", "train")).lower()
            if not gw_std:
                gw_mom = "none"
            Z_train_m = None
            if gw_mom == "train":
                tr_list = h1_preds.get("_h1_train_dates_for_gw")
                if tr_list is not None and len(tr_list) > 0:
                    try:
                        svd_h1 = fe.smooth_svd_features(svd_df, 1)
                        ix_tr = pd.to_datetime(pd.Index(tr_list))
                        ztp = [np.ones(len(ix_tr), dtype=float)]
                        if angle_arr is not None and len(angle_arr) >= n_adv:
                            a_col = np.nan_to_num(np.asarray(angle_arr[:n_adv], dtype=float), nan=0.0)
                            if np.std(a_col) > 1e-10 and "angle" in svd_h1.columns:
                                ztp.append(
                                    np.nan_to_num(
                                        svd_h1.reindex(ix_tr)["angle"].values.astype(float),
                                        nan=0.0,
                                    )
                                )
                        if spectral_gap_arr is not None and len(spectral_gap_arr) >= n_adv:
                            sg_col = np.nan_to_num(
                                np.asarray(spectral_gap_arr[:n_adv], dtype=float), nan=0.0
                            )
                            if np.std(sg_col) > 1e-10 and "spectral_gap" in svd_h1.columns:
                                ztp.append(
                                    np.nan_to_num(
                                        svd_h1.reindex(ix_tr)["spectral_gap"].values.astype(float),
                                        nan=0.0,
                                    )
                                )
                        if len(ztp) == len(Z_parts):
                            Z_train_m = np.column_stack(ztp)
                    except Exception:
                        Z_train_m = None

            # HAR vs HAR+SVD
            gw_har_svd = ev.giacomini_white_test(
                y_true_h1_adv[:n_adv], pred_har_adv[:n_adv], pred_svd_adv[:n_adv],
                instruments=Z_mat, nlags=None, horizon=1,
                standardize_instruments=gw_std,
                instrument_zscore_moments=gw_mom,
                Z_train=Z_train_m,
            )
            gw_results_all["HAR_vs_HAR+SVD"] = {
                "instruments": instrument_labels,
                **{k: (v.tolist() if hasattr(v, "tolist") else v)
                   for k, v in gw_har_svd.items()},
            }
            print(
                f"  GW test (HAR vs HAR+SVD, Z=[{','.join(instrument_labels)}]):"
                f"  stat={gw_har_svd['gw_stat']:.3f}  p={gw_har_svd['p_value']:.4f}"
                f"  df={gw_har_svd['df']}"
            )

            # DNN+HAR vs DNN+HAR+SVD (if available)
            if "DNN_HAR" in h1_preds and "DNN_HAR+SVD" in h1_preds:
                p_dnn = h1_preds["DNN_HAR"]
                p_dnn_svd = h1_preds["DNN_HAR+SVD"]
                n_dnn = min(n_adv, len(p_dnn), len(p_dnn_svd))
                if n_dnn >= 20:
                    gw_dnn = ev.giacomini_white_test(
                        y_true_h1_adv[:n_dnn], p_dnn[:n_dnn], p_dnn_svd[:n_dnn],
                        instruments=Z_mat[:n_dnn], nlags=None, horizon=1,
                        standardize_instruments=gw_std,
                        instrument_zscore_moments=gw_mom,
                        Z_train=Z_train_m,
                    )
                    gw_results_all["DNN_HAR_vs_DNN+SVD"] = {
                        "instruments": instrument_labels,
                        **{k: (v.tolist() if hasattr(v, "tolist") else v)
                           for k, v in gw_dnn.items()},
                    }
                    print(
                        f"  GW test (DNN_HAR vs DNN+SVD):  "
                        f"stat={gw_dnn['gw_stat']:.3f}  p={gw_dnn['p_value']:.4f}"
                    )

            with open(out_dir / "gw_test_results.json", "w") as f:
                json.dump(to_serializable(gw_results_all), f, indent=2)
            print("  [INFO] Giacomini-White test results saved -> gw_test_results.json")
        except Exception as gw_err:
            print(f"  [WARN] Giacomini-White test failed: {gw_err}")

        # --- Harvey-Leybourne-Newbold Forecast Encompassing Test ---------------
        try:
            enc_result = ev.forecast_encompassing_test(
                y_true_h1_adv[:n_adv], pred_har_adv[:n_adv], pred_svd_adv[:n_adv],
                horizon=1,
            )
            with open(out_dir / "encompassing_test_h1.json", "w") as f:
                json.dump(to_serializable(enc_result), f, indent=2)
            print(
                f"  Encompassing (HAR vs HAR+SVD, H0: lambda2=0): "
                f"lambda1={enc_result['lambda1']:.3f}  lambda2={enc_result['lambda2']:.3f}"
                f"  t={enc_result['lambda2_t']:.3f}  p_two={enc_result['lambda2_p']:.4f}"
                f"  p_one(>0)={enc_result.get('lambda2_p_one_sided', float('nan')):.4f}"
                f"  -> {enc_result.get('encompassing_verdict', 'n/a')}"
            )
        except Exception as enc_err:
            print(f"  [WARN] Encompassing test failed: {enc_err}")

        # --- VaR Backtesting (1% and 5%) ----------------------------------------
        var_results_all: dict = {}
        portfolio_returns_test = portfolio_returns.reindex(
            pd.Index(test_dates_h1_adv) if test_dates_h1_adv is not None else pd.Index([])
        ).values if test_dates_h1_adv is not None else None

        if portfolio_returns_test is not None and len(portfolio_returns_test) >= 20:
            for alpha_var, level_label in [(0.01, "1%"), (0.05, "5%")]:
                for model_n, pred_arr_v in [
                    ("HAR", pred_har_adv), ("HAR+SVD", pred_svd_adv)
                ]:
                    n_var = min(len(portfolio_returns_test), len(pred_arr_v), n_adv)
                    if n_var < 20:
                        continue
                    try:
                        vr = ev.var_backtest(
                            portfolio_returns_test[:n_var],
                            pred_arr_v[:n_var],
                            alpha_level=alpha_var,
                        )
                        key = f"{model_n}_{level_label}"
                        var_results_all[key] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                                                for k, v in vr.items()
                                                if k != "violations"}
                        print(
                            f"  VaR {level_label} ({model_n}): "
                            f"viol_rate={vr['phat']:.3f} (exp={alpha_var:.2f})  "
                            f"Kupiec_p={vr['kupiec_p']:.4f} {'PASS' if vr['passes_kupiec'] else 'FAIL'}  "
                            f"Christoffersen_p={vr['christoffersen_p']:.4f} "
                            f"{'PASS' if vr['passes_christoffersen'] else 'FAIL'}"
                        )
                    except Exception as var_err:
                        print(f"  [WARN] VaR {level_label} {model_n} failed: {var_err}")

            if var_results_all:
                with open(out_dir / "var_backtest_results.json", "w") as f:
                    json.dump(to_serializable(var_results_all), f, indent=2)
                print("  [INFO] VaR backtest results saved -> var_backtest_results.json")

    # -----------------------------------------------------------------------
    # 8. Evaluation dataframe (h=1 test set) and figures
    # -----------------------------------------------------------------------
    if "test_dates" not in all_predictions or "true_vol" not in all_predictions:
        print("[WARN] No h=1 predictions stored; skipping figures.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    test_dates_h1 = all_predictions["test_dates"]
    true_vol_h1 = all_predictions["true_vol"]
    n_test = len(true_vol_h1)

    # Rename "true_vol" to "TrueVol" for DataFrame and downstream figure code
    build = {"TrueVol": true_vol_h1}
    pred_model_keys = [
        "HAR", "HAR+SVD", "DNN_HAR", "DNN_HAR+SVD",
        "LSTM_HAR", "LSTM_HAR+SVD", "HARNet", "GNN",
        "GARCH", "IV_baseline",
    ]
    for k in pred_model_keys:
        if k in all_predictions and len(all_predictions[k]) == n_test:
            build[k] = all_predictions[k]

    # VIX IV baseline
    # vix_series is in decimal daily vol (VIX/100/sqrt(252)).
    # Convert to variance in %-squared to match the target: (vol_decimal * 100)^2
    if vix_series is not None:
        vix_aligned = vix_series.reindex(test_dates_h1).ffill()
        if np.isfinite(vix_aligned.values).sum() > 50:
            iv_daily_vol_decimal = np.asarray(vix_aligned.values, dtype=float)
            iv_vals = (iv_daily_vol_decimal * 100.0) ** 2  # convert to %-squared variance
            build["IV_baseline"] = iv_vals
            all_predictions["IV_baseline"] = iv_vals
            iv_metrics = ev.four_metrics(true_vol_h1, iv_vals, eps)
            with open(out_dir / "iv_baseline_metrics.json", "w") as f:
                json.dump(to_serializable(iv_metrics), f, indent=2)
            print("[INFO] IV baseline metrics saved.")

    for th in crisis_thresholds:
        c = f"crisis_{th}"
        if c in all_predictions and len(all_predictions[c]) == n_test:
            build[c] = all_predictions[c]
    if "cos_theta" in all_predictions and len(all_predictions["cos_theta"]) == n_test:
        build["cos_theta"] = all_predictions["cos_theta"]

    df_eval = pd.DataFrame(build, index=test_dates_h1)

    # Date-based regime labels
    if crisis_windows:
        df_eval["Regime_date"] = [
            "Crisis" if is_crisis_date(d, crisis_windows) else "Calm"
            for d in test_dates_h1
        ]

    pred_cols = [c for c in pred_model_keys if c in df_eval.columns]
    crisis_cols = {th: f"crisis_{th}" for th in crisis_thresholds if f"crisis_{th}" in df_eval.columns}

    # Performance by regime
    if crisis_windows and "Regime_date" in df_eval.columns and pred_cols:
        regime_df = ev.performance_by_regime_table(
            df_eval, pred_cols, "TrueVol", "Regime_date", eps
        )
        if not regime_df.empty:
            regime_df.to_csv(out_dir / "performance_by_regime.csv", index=False)
            print("[INFO] Performance by regime saved.")

    # Threshold sensitivity — percentile-based rotation regimes.
    # The absolute cos_theta thresholds (0.7-0.9) are never breached in the test
    # period, so we instead use top-P% of the angle series to define a
    # "High Rotation" regime.  Three percentile levels give the same 3-panel
    # structure while producing non-empty regime subsets.
    angle_sens_cols: dict[float, str] = {}
    if "cos_theta" in df_eval.columns:
        angle_series = 1.0 - df_eval["cos_theta"].values  # angle = 1 - cos_theta
        for pct in [0.20, 0.30, 0.40]:
            col_name = f"high_rotation_{int(pct*100)}pct"
            threshold_val = np.nanpercentile(angle_series, (1 - pct) * 100)
            df_eval[col_name] = (angle_series >= threshold_val).astype(int)
            angle_sens_cols[pct] = col_name

    if pred_cols and angle_sens_cols:
        sens_df = ev.threshold_sensitivity_table(
            df_eval, pred_cols, "TrueVol", angle_sens_cols, eps
        )
        # Relabel regimes from Crisis/Calm to High-Rotation/Low-Rotation to be explicit.
        if "Regime" in sens_df.columns:
            sens_df["Regime"] = sens_df["Regime"].replace(
                {"Crisis": "High Rotation", "Calm": "Low Rotation"}
            )
        sens_df.to_csv(out_dir / "threshold_sensitivity.csv", index=False)
        print("[INFO] Threshold sensitivity saved.")
    elif pred_cols and crisis_cols:
        sens_df = ev.threshold_sensitivity_table(
            df_eval, pred_cols, "TrueVol", crisis_cols, eps
        )
        sens_df.to_csv(out_dir / "threshold_sensitivity.csv", index=False)
        print("[INFO] Threshold sensitivity saved.")
    else:
        sens_df = pd.DataFrame()

    # -----------------------------------------------------------------------
    # 9. Publication-quality figures F1-F7
    # -----------------------------------------------------------------------
    fig_dir = FIGURES_DIR

    # F1: Posterior predictive forecast
    ev.plot_posterior_forecast(
        df_eval=df_eval,
        uncertainty=uncertainty,
        crisis_windows=crisis_windows,
        out_dir=fig_dir,
    )

    # F2: Ablation heatmap (% RMSE change vs HAR baseline)
    ev.plot_ablation_heatmap(results_by_horizon=results_by_horizon, out_dir=fig_dir)

    # F3: Threshold sensitivity (4-panel) — percentile rotation or cos-\u03b8 crisis cols
    if pred_cols and not sens_df.empty:
        ev.plot_threshold_sensitivity_4panel(sens_df=sens_df, out_dir=fig_dir)

    # F4: Crisis window deep-dives
    ev.plot_crisis_deep_dives(
        df_eval=df_eval,
        crisis_windows=crisis_windows,
        uncertainty=uncertainty,
        out_dir=fig_dir,
    )

    # F5: Epistemic uncertainty + HAR posterior coefficients
    ev.plot_uncertainty_figure(
        df_eval=df_eval,
        uncertainty=uncertainty,
        crisis_windows=crisis_windows,
        out_dir=fig_dir,
    )

    # F6: Enhanced residual diagnostics
    ev.plot_residual_diagnostics(
        df_eval=df_eval,
        pred_columns=pred_cols,
        out_dir=fig_dir,
    )

    # F7: DM statistic heatmap (all horizons)
    for h in horizons:
        if h in dm_all_horizons and not dm_all_horizons[h].empty:
            ev.plot_dm_matrix(dm_df=dm_all_horizons[h], out_dir=fig_dir, horizon=h)

    # F9: Mincer–Zarnowitz (all models, HAC α/β vs ideal 0, 1)
    if mz_results:
        try:
            ev.plot_mincer_zarnowitz_summary(mz_results, fig_dir)
        except Exception as mz_fig_err:
            print(f"[WARN] F9 Mincer–Zarnowitz summary plot failed: {mz_fig_err}")

    # Legacy figures
    ev.generate_figures(
        df_eval, fig_dir, true_col="TrueVol",
        pred_columns=[c for c in pred_cols if c not in ("GARCH", "IV_baseline")],
        crisis_col="crisis_0.8",
    )
    if model_dnn_svd_h1 is not None:
        ev.plot_feature_importance(model_dnn_svd_h1, har_svd_names, fig_dir)
    if crisis_windows and pred_cols:
        ev.plot_crisis_windows(df_eval, crisis_windows, pred_cols, "TrueVol", fig_dir)

    # F8: CSLD + Rolling DM (shows WHEN SVD helps)
    if "HAR" in df_eval.columns and "HAR+SVD" in df_eval.columns:
        try:
            ev.plot_csld_and_rolling_dm(
                true_var=true_vol_h1,
                pred_har=df_eval["HAR"].values,
                pred_svd=df_eval["HAR+SVD"].values,
                index=df_eval.index,
                out_dir=fig_dir,
                crisis_windows=crisis_windows,
                rolling_window=252,
                horizon=1,
            )
        except Exception as csld_err:
            print(f"[WARN] F8 CSLD plot failed: {csld_err}")

    # DNN feature gate weights (only if use_gate=True); file F9_gate_weights_h*.pdf
    try:
        if model_dnn_svd_h1 is not None and X_har_svd_h1_scaled_test is not None:
            # Try to extract gate weights if the model has the gate_weights layer
            if any(l.name == "gate_weights" for l in model_dnn_svd_h1.layers):
                gate_w = dnn_models.get_gate_weights(model_dnn_svd_h1, X_har_svd_h1_scaled_test)
                crisis_col_arr = df_eval.get(f"crisis_{default_threshold}")
                if crisis_col_arr is None and f"crisis_{default_threshold}" in df_eval.columns:
                    crisis_col_arr = df_eval[f"crisis_{default_threshold}"].values
                ev.plot_gate_weights(
                    gate_weights=gate_w,
                    feature_names=har_svd_names,
                    index=df_eval.index,
                    crisis_col=crisis_col_arr,
                    out_dir=fig_dir,
                    horizon=1,
                )
    except Exception as gate_err:
        print(f"[WARN] F9 gate weight plot failed: {gate_err}")

    print(f"[INFO] All figures saved to {fig_dir}")

    # -----------------------------------------------------------------------
    # 10. Robustness checks (Appendix material)
    #
    # These checks do NOT re-run all models. They use ElasticNet (M1 / M2
    # equivalents) which are fast, and report how QLIKE and RMSE change:
    #   (a) SVD window sensitivity: 126 vs 252 days
    #   (b) K absorption-ratio sensitivity: K = 3, 4, 5, 6, 8, 10
    #   (c) Empirical coverage of uncertainty bands (PIT histogram stub)
    #
    # Results are saved as robustness_svd_window.csv and robustness_K.csv.
    # Full multi-split sensitivity is controlled by train_split_sensitivity.
    # -----------------------------------------------------------------------
    print("\n[INFO] Running robustness checks ...")

    # (a) SVD window sensitivity
    svd_window_list = list(cfg_feat.get("svd_window_sensitivity", [svd_window]))
    K_sens_list = list(cfg_feat.get("K_sensitivity", [K]))

    robust_rows_window: list[dict] = []
    for w in svd_window_list:
        if w == svd_window:
            # Already computed in main run — reuse h=1 primary results
            if 1 in results_by_horizon:
                for model_name, metrics in results_by_horizon[1].items():
                    robust_rows_window.append({
                        "svd_window": w, "model": model_name,
                        **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                    })
            continue
        print(f"  [Robustness] SVD window = {w} ...")
        try:
            svd_df_w, _ = fe.build_svd_features_panel_with_cov(
                returns, asset_columns, w, K, crisis_thresholds, eps,
                cov_estimator=cov_estimator_primary,
            )
            # Align to rv_df index first (rv_df is already validity-filtered to 6025 rows);
            # then keep only rows where the reindexed SVD features are non-NaN.
            svd_df_w_aligned = svd_df_w.reindex(rv_df.index)
            valid_w = ~svd_df_w_aligned["f1"].isna()
            rv_df_w = rv_df.loc[valid_w]
            svd_df_w = svd_df_w_aligned.loc[rv_df_w.index]
            semi_df_w = semi_df.reindex(rv_df_w.index)

            X_har_w, _ = fe.build_feature_sets(rv_df_w, svd_df_w, svd_tier=0, eps=eps, semi_df=semi_df_w)
            X_svd_w, _ = fe.build_feature_sets(rv_df_w, svd_df_w, svd_tier=2, eps=eps, semi_df=semi_df_w)

            target_rv_w = build_target_rv(portfolio_returns, 1)
            align_w = target_rv_w.reindex(rv_df_w.index).dropna().index
            align_w = align_w[X_har_w.loc[align_w].notna().all(axis=1).values & X_svd_w.loc[align_w].notna().all(axis=1).values]

            y_w = np.log(target_rv_w.reindex(align_w).values + eps)
            X_har_wn = X_har_w.loc[align_w].values
            X_svd_wn = X_svd_w.loc[align_w].values
            n_w = len(align_w)
            sp_w = int(n_w * primary_split)
            val_w = max(22, int(0.15 * sp_w))
            fit_slice = slice(0, sp_w - val_w)
            test_slice = slice(sp_w, None)

            for feat_name, Xall in [("HAR", X_har_wn), ("HAR+SVD", X_svd_wn)]:
                enet_w = linear_models.train_har_svd_elastic(Xall[fit_slice], y_w[fit_slice])
                pred_log_w = linear_models.predict_har_svd_elastic(enet_w, Xall[test_slice])
                pred_w = smearing_corrected_pred(pred_log_w, linear_models.predict_har_svd_elastic(enet_w, Xall[fit_slice]), y_w[fit_slice])
                true_w = np.exp(y_w[test_slice])
                m_w = ev.four_metrics(true_w, pred_w, eps)
                robust_rows_window.append({"svd_window": w, "model": feat_name, **m_w})
        except Exception as rob_err:
            print(f"    [WARN] SVD window={w} robustness check failed: {rob_err}")

    if robust_rows_window:
        pd.DataFrame(robust_rows_window).to_csv(out_dir / "robustness_svd_window.csv", index=False)
        print(f"  [INFO] Robustness SVD window check saved -> robustness_svd_window.csv")

    # (b) K absorption ratio sensitivity
    robust_rows_K: list[dict] = []
    for k_val in K_sens_list:
        print(f"  [Robustness] K = {k_val} ...")
        try:
            svd_df_k, _ = fe.build_svd_features_panel_with_cov(
                returns, asset_columns, svd_window, k_val, crisis_thresholds, eps,
                cov_estimator=cov_estimator_primary,
            )
            # Align to rv_df index before applying validity mask (avoids 6277 vs 6025 broadcast).
            svd_df_k_aligned = svd_df_k.reindex(rv_df.index)
            valid_k = ~svd_df_k_aligned["f1"].isna()
            rv_df_k = rv_df.loc[valid_k]
            svd_df_k = svd_df_k_aligned.loc[rv_df_k.index]
            semi_df_k = semi_df.reindex(rv_df_k.index)

            X_svd_k, _ = fe.build_feature_sets(rv_df_k, svd_df_k, svd_tier=2, eps=eps, semi_df=semi_df_k)
            target_rv_k = build_target_rv(portfolio_returns, 1)
            align_k = target_rv_k.reindex(rv_df_k.index).dropna().index
            align_k = align_k[X_svd_k.loc[align_k].notna().all(axis=1).values]
            y_k = np.log(target_rv_k.reindex(align_k).values + eps)
            X_svd_kn = X_svd_k.loc[align_k].values
            n_k = len(align_k)
            sp_k = int(n_k * primary_split)
            val_k = max(22, int(0.15 * sp_k))
            fit_k = slice(0, sp_k - val_k)
            test_k = slice(sp_k, None)
            enet_k = linear_models.train_har_svd_elastic(X_svd_kn[fit_k], y_k[fit_k])
            pred_log_k = linear_models.predict_har_svd_elastic(enet_k, X_svd_kn[test_k])
            pred_k = smearing_corrected_pred(pred_log_k, linear_models.predict_har_svd_elastic(enet_k, X_svd_kn[fit_k]), y_k[fit_k])
            true_k = np.exp(y_k[test_k])
            m_k = ev.four_metrics(true_k, pred_k, eps)

            # Also compute AR mean for this K (quantifies how much variance the K PCs explain)
            ar_mean = float(svd_df_k["AR"].dropna().mean())
            robust_rows_K.append({"K": k_val, "AR_mean": ar_mean, **m_k})
        except Exception as rob_err_k:
            print(f"    [WARN] K={k_val} robustness check failed: {rob_err_k}")

    if robust_rows_K:
        pd.DataFrame(robust_rows_K).to_csv(out_dir / "robustness_K.csv", index=False)
        print(f"  [INFO] K sensitivity check saved -> robustness_K.csv")

    # (b2) Covariance-estimator sensitivity (LW linear, QIS, BBP RIE).
    # Verifies the SVD features behave consistently across rotation-invariant
    # population estimators (KMZ-2024 Prop. 2 robustness).
    cov_estimator_grid = list(cfg_feat.get("cov_estimator_sensitivity", [])) or []
    robust_rows_cov: list[dict] = []
    for est_name in cov_estimator_grid:
        if not est_name:
            continue
        if est_name == cov_estimator_primary:
            tag = f"{est_name}_primary"
        else:
            tag = est_name
        print(f"  [Robustness] cov_estimator = {est_name} ...")
        try:
            svd_df_e, _ = fe.build_svd_features_panel_with_cov(
                returns, asset_columns, svd_window, K, crisis_thresholds, eps,
                cov_estimator=est_name,
            )
            svd_df_e_aligned = svd_df_e.reindex(rv_df.index)
            valid_e = ~svd_df_e_aligned["f1"].isna()
            rv_df_e = rv_df.loc[valid_e]
            svd_df_e = svd_df_e_aligned.loc[rv_df_e.index]
            semi_df_e = semi_df.reindex(rv_df_e.index)
            X_har_e, _ = fe.build_feature_sets(
                rv_df_e, svd_df_e, svd_tier=0, eps=eps, semi_df=semi_df_e,
            )
            X_svd_e, _ = fe.build_feature_sets(
                rv_df_e, svd_df_e, svd_tier=2, eps=eps, semi_df=semi_df_e,
            )
            target_rv_e = build_target_rv(portfolio_returns, 1)
            align_e = target_rv_e.reindex(rv_df_e.index).dropna().index
            mask_har_e = X_har_e.loc[align_e].notna().all(axis=1).values
            mask_svd_e = X_svd_e.loc[align_e].notna().all(axis=1).values
            align_e = align_e[mask_har_e & mask_svd_e]
            y_e = np.log(target_rv_e.reindex(align_e).values + eps)
            X_har_en = X_har_e.loc[align_e].values
            X_svd_en = X_svd_e.loc[align_e].values
            n_e = len(align_e)
            sp_e = int(n_e * primary_split)
            val_e = max(22, int(0.15 * sp_e))
            fit_e = slice(0, sp_e - val_e)
            test_e = slice(sp_e, None)
            for feat_name, Xall in [("HAR", X_har_en), ("HAR+SVD", X_svd_en)]:
                enet_e = linear_models.train_har_svd_elastic(Xall[fit_e], y_e[fit_e])
                pred_log_e = linear_models.predict_har_svd_elastic(enet_e, Xall[test_e])
                pred_e = smearing_corrected_pred(
                    pred_log_e,
                    linear_models.predict_har_svd_elastic(enet_e, Xall[fit_e]),
                    y_e[fit_e],
                )
                true_e = np.exp(y_e[test_e])
                m_e = ev.four_metrics(true_e, pred_e, eps)
                robust_rows_cov.append({
                    "cov_estimator": tag, "model": feat_name, **m_e,
                })
        except Exception as cov_err:
            print(f"    [WARN] cov_estimator={est_name} failed: {cov_err}")

    if robust_rows_cov:
        pd.DataFrame(robust_rows_cov).to_csv(
            out_dir / "robustness_cov_estimator.csv", index=False
        )
        print(
            "  [INFO] Cov-estimator sensitivity (LW / QIS / BBP-RIE) saved -> "
            "robustness_cov_estimator.csv"
        )

    # (c) Empirical PIT coverage (MC Dropout)
    # Coverage at 90% (i.e., what fraction of true values fall in [5th, 95th] predictive percentile)
    if uncertainty.get("DNN+SVD") is not None:
        try:
            unc_svd = uncertainty["DNN+SVD"]
            lo_mc = unc_svd.get("lo_95", unc_svd.get("lo_90"))
            hi_mc = unc_svd.get("hi_95", unc_svd.get("hi_90"))
            if lo_mc is not None and hi_mc is not None:
                true_in_band = np.sum((true_vol_h1 >= lo_mc) & (true_vol_h1 <= hi_mc))
                coverage_frac = true_in_band / len(true_vol_h1)
                with open(out_dir / "pit_coverage.json", "w") as f:
                    json.dump({"MC_dropout_90pct_coverage": float(coverage_frac)}, f, indent=2)
                print(f"  [INFO] MC Dropout empirical 90% coverage: {coverage_frac:.3f} (expected: 0.90)")
        except Exception as cov_err:
            print(f"  [WARN] PIT coverage computation failed: {cov_err}")

    # -----------------------------------------------------------------------
    # 11. Save h=1 models for reproducibility
    # -----------------------------------------------------------------------
    models_dir = RESULTS_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    if model_dnn_svd_h1 is not None:
        try:
            model_dnn_svd_h1.save(models_dir / "dnn_har_svd_h1.keras")
            print("[INFO] DNN HAR+SVD model saved.")
        except Exception as e:
            print(f"[WARN] Could not save DNN model: {e}")
    if model_lstm_svd_h1 is not None:
        try:
            model_lstm_svd_h1.save(models_dir / "lstm_har_svd_h1.keras")
            print("[INFO] LSTM HAR+SVD model saved.")
        except Exception as e:
            print(f"[WARN] Could not save LSTM model: {e}")

    if not force_rerun:
        eckpt.mark_exports_complete(out_dir, exp_fp_primary)
    print("\n[INFO] Pipeline complete. Results in results/metrics/ and results/figures/")


if __name__ == "__main__":
    main()
