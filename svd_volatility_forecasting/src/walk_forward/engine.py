"""
Walk-forward orchestration for the volatility forecasting pipeline.

This module consumes pre-aligned feature/target arrays and produces per-date
out-of-sample predictions under rolling and expanding protocols.

It is intentionally strict:
  - raises on insufficient history for sequences
  - requires explicit train/val/test blocks
  - requires model refit cadence to be specified in config
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import config
from models import linear as linear_models
from evaluation import xai as xai_mod
from models.runners import (
    fit_predict_dnn,
    fit_predict_dnn_ensemble,
    fit_predict_elasticnet,
    fit_predict_ridge_rff,
    fit_predict_gnn,
    fit_predict_har,
    fit_predict_harnet,
    fit_predict_lstm,
    fit_predict_lstm_ensemble,
    fit_scaler_train_only,
)
from .splits import iter_walk_forward_splits
from experiment_checkpoint import (
    load_walk_forward_checkpoint,
    save_walk_forward_checkpoint,
)
from .profile import (
    model_enabled,
    needs_refit,
    should_tune_elasticnet,
)


Protocol = Literal["expanding", "rolling"]


@dataclass
class WalkForwardOutputs:
    """
    Stores walk-forward predictions for a single (protocol, horizon).

    preds_log[model] is a Series indexed by forecast dates with log-variance predictions.
    """

    protocol: Protocol
    horizon: int
    y_true_var: pd.Series
    y_true_log: pd.Series
    preds_log: dict[str, pd.Series]
    preds_var: dict[str, pd.Series]
    xai: dict[str, object]
    metadata: dict[str, object]


def _walk_forward_run_xai_now(step_id: int, xcfg: dict) -> bool:
    """
    Gate for blocked permutation importance on this walk-forward step.

    ``run_every_n_steps > 1`` throttles only the XAI artifact density; it does
    not change ``pred_log`` / ``pred_var`` series or refit cadence.
    """
    if not bool(xcfg.get("enabled", True)):
        return False
    every = int(xcfg.get("run_every_n_steps", 1))
    if every < 1:
        every = 1
    return (int(step_id) % every) == 0


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _build_test_sequence_from_full(
    X_full_scaled: np.ndarray,
    full_idx: pd.Index,
    test_date: pd.Timestamp,
    *,
    seq_len: int,
) -> np.ndarray:
    """
    Build a single (1, seq_len, n_features) sequence ending at test_date, using
    scaled feature rows from full history (already transformed by a train-only scaler).
    """
    try:
        pos = int(full_idx.get_loc(test_date))
    except KeyError as e:
        raise ValueError("test_date not found in feature index.") from e
    start = pos - seq_len + 1
    if start < 0:
        raise ValueError(f"Insufficient history for seq_len={seq_len} at {test_date}.")
    seq = X_full_scaled[start : pos + 1]
    if seq.shape[0] != seq_len:
        raise ValueError("Sequence construction failed (unexpected length).")
    return seq.reshape(1, seq_len, seq.shape[1])


def run_walk_forward_for_horizon(
    *,
    protocol: Protocol,
    horizon: int,
    idx: pd.Index,
    y_true_var: pd.Series,
    y_true_log: pd.Series,
    # Feature blocks (all must share the same idx)
    X_har: pd.DataFrame,
    X_t1: pd.DataFrame,
    X_t2: pd.DataFrame,
    X_t3: pd.DataFrame,
    # Optional extra per-model feature blocks (must share idx)
    X_by_model: dict[str, pd.DataFrame] | None = None,
    # GNN full arrays (optional; required if GNN is included)
    node_features_full: np.ndarray | None,
    adj_full: np.ndarray | None,
    # Config blocks
    cfg_models: dict,
    walk_cfg: dict,
    feature_names: dict[str, list[str]],
    use_osi: bool,
    osi_har_col_count: int,
    checkpoint_dir: Path | None = None,
) -> WalkForwardOutputs:
    """
    Produce walk-forward predictions for all models at one horizon.

    Caller responsibilities:
      - idx and all X_* and y_* must be aligned and free of NaNs on the evaluation region
      - X_* must be constructed with no look-ahead (your pipeline uses shift(1))
      - y_true_var/y_true_log must correspond to the forward target aligned to idx
    """
    idx = pd.Index(idx)
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError("idx must be strictly increasing with no duplicates.")

    # Hard alignment checks
    for name, df in [("X_har", X_har), ("X_t1", X_t1), ("X_t2", X_t2), ("X_t3", X_t3)]:
        if not df.index.equals(idx):
            raise ValueError(f"{name} index must equal idx exactly.")
    if X_by_model is not None:
        for k, df in X_by_model.items():
            if not df.index.equals(idx):
                raise ValueError(f"X_by_model[{k!r}] index must equal idx exactly.")
    if not y_true_var.index.equals(idx) or not y_true_log.index.equals(idx):
        raise ValueError("y_true series must be indexed by idx exactly.")

    # Determine models and refit cadence (profile-driven via walk_cfg["models"])
    cadence = walk_cfg["refit_cadence"]
    required_models = list(walk_cfg.get("models", []))
    if not required_models:
        raise ValueError("walk_forward.models must be non-empty (set profile or models list).")
    for m in required_models:
        if m not in cadence:
            raise ValueError(f"walk_forward.refit_cadence missing model key: {m}")

    active = set(required_models)
    profile_name = str(walk_cfg.get("profile_active", walk_cfg.get("profile", "default")))
    wf_fingerprint = str(walk_cfg.get("experiment_fingerprint", ""))

    # Storage (log predictions; conversion to variance happens downstream with smearing policy)
    preds_log: dict[str, list[float]] = {m: [] for m in required_models}
    preds_var: dict[str, list[float]] = {m: [] for m in required_models}
    test_dates: list[pd.Timestamp] = []

    # Cached fitted models for refit cadence
    cache: dict[str, object] = {}
    cache_step_last_fit: dict[str, int] = {m: -10**9 for m in required_models}
    cache_elastic_hp: dict[str, dict[str, float]] = {}
    cache_scalers: dict[str, object] = {}
    cache_smearing: dict[str, float] = {}
    xai_records: list[dict[str, object]] = []
    resume_si = 0
    if checkpoint_dir is not None:
        ckpt = load_walk_forward_checkpoint(
            checkpoint_dir,
            protocol=protocol,
            horizon=horizon,
            profile=profile_name,
            fingerprint=wf_fingerprint,
        )
        if ckpt is not None:
            resume_si = int(ckpt.get("last_si", -1)) + 1
            for m in required_models:
                preds_log[m] = list(ckpt.get("preds_log", {}).get(m, []))
                preds_var[m] = list(ckpt.get("preds_var", {}).get(m, []))
            test_dates = [pd.Timestamp(d) for d in ckpt.get("test_dates", [])]
            print(
                f"[WF] resume {protocol} h={horizon} profile={profile_name!r} "
                f"from split index {resume_si} (step_id>{ckpt.get('last_step_id')})",
                flush=True,
            )

    horizons = [horizon]
    splits_list = list(
        iter_walk_forward_splits(
            idx,
            protocol=protocol,
            horizons=horizons,
            initial_train_len=int(walk_cfg["initial_train_len"]),
            rolling_train_len=int(walk_cfg["rolling_train_len"]),
            val_len=int(walk_cfg["val_len"]),
            step=int(walk_cfg["step"]),
        )
    )
    n_splits = len(splits_list)
    if n_splits < 1:
        raise ValueError("Walk-forward produced zero splits; check idx length vs initial_train_len+val_len.")

    cfg_dnn = cfg_models["dnn"].copy()
    cfg_lstm = cfg_models["lstm"].copy()
    cfg_harnet = cfg_models["harnet"].copy()
    cfg_gnn = cfg_models["gnn"].copy()

    # Apply horizon overrides (fixed ex ante in config)
    if "horizon_overrides" in cfg_dnn and horizon in cfg_dnn["horizon_overrides"]:
        cfg_dnn.update(cfg_dnn["horizon_overrides"][horizon])
    if "horizon_overrides" in cfg_lstm and horizon in cfg_lstm["horizon_overrides"]:
        cfg_lstm.update(cfg_lstm["horizon_overrides"][horizon])

    seq_len_lstm = int(cfg_lstm["seq_len"])
    seq_len_harnet = int(cfg_harnet["seq_len"])

    # Hidden dims for DNN are already chosen in config; for parity use cfg_dnn["hidden_layers"]
    hidden_dnn = list(cfg_dnn["hidden_layers"])
    hidden_dnn_svd = list(cfg_dnn.get("hidden_layers_svd", hidden_dnn))

    tcfg = config.config.get("training", {})
    sf_lo, sf_hi = tcfg.get("smearing_factor_bounds", [0.1, 50.0])
    clip_w = float(tcfg.get("pred_log_clip_width", 4.0))

    def _compute_smearing_factor(y_tr_log: np.ndarray, pred_tr_log: np.ndarray) -> float:
        # Align lengths (sequence models return shorter train predictions)
        y = np.asarray(y_tr_log, dtype=np.float64).ravel()
        p = np.asarray(pred_tr_log, dtype=np.float64).ravel()
        if len(p) < 10:
            raise ValueError("Smearing: insufficient train predictions.")
        if len(p) != len(y):
            if len(p) > len(y):
                raise ValueError("Smearing: pred_train longer than y_train.")
            offset = len(y) - len(p)
            y = y[offset:]
        resid = y - p
        resid = resid[np.isfinite(resid)]
        if resid.size < 10:
            raise ValueError("Smearing: insufficient finite residuals.")
        sf = float(np.exp(resid).mean())
        return float(np.clip(sf, float(sf_lo), float(sf_hi)))

    def _pred_var_from_log(pred_log: float, y_tr_log: np.ndarray, smearing_factor: float) -> float:
        y_mean = float(np.nanmean(y_tr_log))
        lo_l, hi_l = y_mean - clip_w, y_mean + clip_w
        pl = float(np.clip(pred_log, lo_l, hi_l))
        return float(np.exp(np.clip(pl, -80.0, 80.0)) * smearing_factor)

    def _pred_var_from_log_vec(pred_log: np.ndarray, y_tr_log: np.ndarray, smearing_factor: float) -> np.ndarray:
        y_mean = float(np.nanmean(y_tr_log))
        lo_l, hi_l = y_mean - clip_w, y_mean + clip_w
        pl = np.clip(np.asarray(pred_log, dtype=np.float64).ravel(), lo_l, hi_l)
        return np.exp(np.clip(pl, -80.0, 80.0)) * float(smearing_factor)

    ens_seeds = list(walk_cfg.get("ensemble_sub_seeds", [42, 1337, 2024, 314, 999]))
    if len(ens_seeds) < 2:
        raise ValueError("walk_forward.ensemble_sub_seeds must have at least 2 seeds for ensemble training.")

    enet_njobs_raw = walk_cfg.get("elastic_net_cv_n_jobs")
    enet_n_jobs = int(enet_njobs_raw) if enet_njobs_raw is not None else None
    prog_every = max(1, int(walk_cfg.get("progress_log_every", 25)))
    xai_every = int(walk_cfg.get("xai", {}).get("run_every_n_steps", 1))
    wf_t0 = time.perf_counter()
    xai_enabled = bool(walk_cfg.get("xai", {}).get("enabled", True))
    tune_policy = str(walk_cfg.get("tuning_policy", "scheduled"))
    retune_every = int(walk_cfg.get("retune_every", 63))
    print(
        f"[WF] start protocol={protocol!r} h={horizon} profile={profile_name!r} | "
        f"models={required_models} | {n_splits} OOS splits | "
        f"tuning={tune_policy} retune_every={retune_every} | "
        f"ElasticNetCV n_jobs={enet_n_jobs!r} | XAI enabled={xai_enabled} "
        f"every {max(1, xai_every)} step(s) | progress every {prog_every} split(s)",
        flush=True,
    )

    for si, split in enumerate(splits_list):
        if si < resume_si:
            continue
        assert split.horizon == horizon
        step_id = int(split.step_id)
        td = pd.Timestamp(split.test_idx[0])
        test_dates.append(td)

        if si % prog_every == 0 or si == n_splits - 1:
            elapsed = time.perf_counter() - wf_t0
            done = si + 1
            rate = done / max(elapsed, 1e-9)
            eta_s = (n_splits - done) / max(rate, 1e-9)
            print(
                f"[WF] {protocol} h={horizon} | split {done}/{n_splits} "
                f"(step_id={step_id}, test={td.date()}) | "
                f"elapsed {elapsed / 60.0:.1f}m | ETA ~{eta_s / 60.0:.1f}m",
                flush=True,
            )
        # Slice indices to integer positions for fast numpy slicing
        tr_pos = idx.get_indexer(split.train_idx)
        va_pos = idx.get_indexer(split.val_idx)
        te_pos = idx.get_indexer(split.test_idx)
        if np.any(tr_pos < 0) or np.any(va_pos < 0) or np.any(te_pos < 0):
            raise ValueError("Index mapping failed (negative positions).")

        # y slices (log target for training)
        y_tr = y_true_log.iloc[tr_pos].values.astype(np.float64)
        y_va = y_true_log.iloc[va_pos].values.astype(np.float64)
        y_va_var = y_true_var.iloc[va_pos].values.astype(np.float64)

        # Tabular feature matrices
        Xhar_tr = X_har.iloc[tr_pos].values.astype(np.float64)
        Xhar_va = X_har.iloc[va_pos].values.astype(np.float64)
        Xhar_te = X_har.iloc[te_pos].values.astype(np.float64)

        Xt1_tr = X_t1.iloc[tr_pos].values.astype(np.float64)
        Xt1_te = X_t1.iloc[te_pos].values.astype(np.float64)

        Xt2_tr = X_t2.iloc[tr_pos].values.astype(np.float64)
        Xt2_va = X_t2.iloc[va_pos].values.astype(np.float64)
        Xt2_te = X_t2.iloc[te_pos].values.astype(np.float64)

        Xt3_tr = X_t3.iloc[tr_pos].values.astype(np.float64)
        Xt3_va = X_t3.iloc[va_pos].values.astype(np.float64)
        Xt3_te = X_t3.iloc[te_pos].values.astype(np.float64)

        # Optional per-model feature blocks (for gated / expanded variants)
        Xt2_gated_ar_tr = Xt2_gated_ar_va = Xt2_gated_ar_te = None
        if X_by_model is not None and model_enabled("HAR+SVD_GATED_AR", required_models):
            Xg = X_by_model.get("HAR+SVD_GATED_AR")
            if Xg is None:
                raise ValueError("HAR+SVD_GATED_AR enabled but X_by_model missing it.")
            Xt2_gated_ar_tr = Xg.iloc[tr_pos].values.astype(np.float64)
            Xt2_gated_ar_va = Xg.iloc[va_pos].values.astype(np.float64)
            Xt2_gated_ar_te = Xg.iloc[te_pos].values.astype(np.float64)

        Xt2_dyn_tr = Xt2_dyn_va = Xt2_dyn_te = None
        if X_by_model is not None and model_enabled("HAR+SVD_DYN", required_models):
            Xd = X_by_model.get("HAR+SVD_DYN")
            if Xd is None:
                raise ValueError("HAR+SVD_DYN enabled but X_by_model missing it.")
            Xt2_dyn_tr = Xd.iloc[tr_pos].values.astype(np.float64)
            Xt2_dyn_va = Xd.iloc[va_pos].values.astype(np.float64)
            Xt2_dyn_te = Xd.iloc[te_pos].values.astype(np.float64)

        # -----------------------------------------------------------------
        # HAR (OLS)
        # -----------------------------------------------------------------
        if model_enabled("HAR", required_models) and (
            needs_refit("HAR", step_id, cadence, cache, cache_step_last_fit)
        ):
            cache["HAR"] = fit_predict_har(Xhar_tr, y_tr, Xhar_te)
            cache_smearing["HAR"] = _compute_smearing_factor(y_tr, cache["HAR"].pred_log_train)
            cache_step_last_fit["HAR"] = step_id
            # XAI on validation block (tabular)
            xcfg = walk_cfg.get("xai", {})
            if _walk_forward_run_xai_now(step_id, xcfg):
                fn = feature_names.get("HAR", [])
                if len(fn) == Xhar_va.shape[1]:
                    def _pred_var(Xin: np.ndarray) -> np.ndarray:
                        import statsmodels.api as sm
                        # has_constant="add" mirrors models_linear.predict_har
                        # so the design matrix always carries the intercept,
                        # even on degenerate single-row inputs.
                        pl = cache["HAR"].model.predict(
                            sm.add_constant(Xin, has_constant="add")
                        )
                        return _pred_var_from_log_vec(pl, y_tr, cache_smearing["HAR"])
                    res = xai_mod.blocked_permutation_importance_2d(
                        X=Xhar_va,
                        y_true_var=y_va_var,
                        feature_names=list(fn),
                        predict_var=_pred_var,
                        loss=str(xcfg.get("loss", "qlike")),
                        reps=int(xcfg.get("reps", 10)),
                        block_len=int(xcfg.get("block_len", 5)),
                        seed=int(xcfg.get("seed", 42)) + int(step_id),
                    )
                    xai_records.append({
                        "step_id": step_id,
                        "test_date": str(td.date()),
                        "model": "HAR",
                        "base_loss": res.base_loss,
                        "feature_names": res.feature_names,
                        "importance_mean": res.importances_mean.tolist(),
                        "importance_std": res.importances_std.tolist(),
                    })
        if model_enabled("HAR", required_models):
            pl = float(cache["HAR"].pred_log_test[0])
            preds_log["HAR"].append(pl)
            preds_var["HAR"].append(_pred_var_from_log(pl, y_tr, cache_smearing["HAR"]))

        # -----------------------------------------------------------------
        # Tier-1 linear (OLS) — for ablation parity; uses same runner as HAR
        # -----------------------------------------------------------------
        if model_enabled("HAR_SVD_T1", required_models) and (
            needs_refit("HAR_SVD_T1", step_id, cadence, cache, cache_step_last_fit)
        ):
            cache["HAR_SVD_T1"] = fit_predict_har(Xt1_tr, y_tr, Xt1_te)
            cache_smearing["HAR_SVD_T1"] = _compute_smearing_factor(y_tr, cache["HAR_SVD_T1"].pred_log_train)
            cache_step_last_fit["HAR_SVD_T1"] = step_id
        if model_enabled("HAR_SVD_T1", required_models):
            pl = float(cache["HAR_SVD_T1"].pred_log_test[0])
            preds_log["HAR_SVD_T1"].append(pl)
            preds_var["HAR_SVD_T1"].append(_pred_var_from_log(pl, y_tr, cache_smearing["HAR_SVD_T1"]))

        # -----------------------------------------------------------------
        # HAR+SVD ElasticNet (Tier-2)
        # -----------------------------------------------------------------
        if model_enabled("HAR+SVD", required_models):
            if needs_refit("HAR+SVD", step_id, cadence, cache, cache_step_last_fit):
                tune_hp = should_tune_elasticnet(step_id, walk_cfg)
                t_refit = time.perf_counter()
                cache["HAR+SVD"] = fit_predict_elasticnet(
                    Xt2_tr,
                    y_tr,
                    Xt2_te,
                    cv_splits=5,
                    use_osi=bool(use_osi),
                    har_col_count=int(osi_har_col_count),
                    enet_n_jobs=enet_n_jobs,
                    tune_hyperparams=tune_hp,
                    frozen_hparams=None if tune_hp else cache_elastic_hp.get("HAR+SVD"),
                )
                if tune_hp:
                    cache_elastic_hp["HAR+SVD"] = linear_models.elastic_hyperparams_from_model(
                        cache["HAR+SVD"].model
                    )
                cache_smearing["HAR+SVD"] = _compute_smearing_factor(
                    y_tr, cache["HAR+SVD"].pred_log_train
                )
                cache_step_last_fit["HAR+SVD"] = step_id
                print(
                    f"[WF] REFIT HAR+SVD tune={tune_hp} step_id={step_id} "
                    f"({time.perf_counter() - t_refit:.1f}s)",
                    flush=True,
                )
                xcfg = walk_cfg.get("xai", {})
                if _walk_forward_run_xai_now(step_id, xcfg):
                    fn = feature_names.get("T2", [])
                    if len(fn) == Xt2_va.shape[1]:
                        def _pred_var(Xin: np.ndarray) -> np.ndarray:
                            pl = cache["HAR+SVD"].model.predict(Xin)
                            return _pred_var_from_log_vec(pl, y_tr, cache_smearing["HAR+SVD"])
                        res = xai_mod.blocked_permutation_importance_2d(
                            X=Xt2_va,
                            y_true_var=y_va_var,
                            feature_names=list(fn),
                            predict_var=_pred_var,
                            loss=str(xcfg.get("loss", "qlike")),
                            reps=int(xcfg.get("reps", 10)),
                            block_len=int(xcfg.get("block_len", 5)),
                            seed=int(xcfg.get("seed", 42)) + int(step_id),
                        )
                        xai_records.append({
                            "step_id": step_id,
                            "test_date": str(td.date()),
                            "model": "HAR+SVD",
                            "base_loss": res.base_loss,
                            "feature_names": res.feature_names,
                            "importance_mean": res.importances_mean.tolist(),
                            "importance_std": res.importances_std.tolist(),
                        })
            else:
                pl_te = np.asarray(cache["HAR+SVD"].model.predict(Xt2_te)).ravel()
                cache["HAR+SVD"].pred_log_test = pl_te
            pl = float(cache["HAR+SVD"].pred_log_test[0])
            preds_log["HAR+SVD"].append(pl)
            preds_var["HAR+SVD"].append(_pred_var_from_log(pl, y_tr, cache_smearing["HAR+SVD"]))

        # -----------------------------------------------------------------
        # HAR+SVD (Regime-gated) ElasticNet
        # -----------------------------------------------------------------
        if model_enabled("HAR+SVD_GATED_AR", required_models):
            assert Xt2_gated_ar_tr is not None and Xt2_gated_ar_te is not None
            if needs_refit("HAR+SVD_GATED_AR", step_id, cadence, cache, cache_step_last_fit):
                tune_hp = should_tune_elasticnet(step_id, walk_cfg)
                t_refit = time.perf_counter()
                cache["HAR+SVD_GATED_AR"] = fit_predict_elasticnet(
                    Xt2_gated_ar_tr,
                    y_tr,
                    Xt2_gated_ar_te,
                    cv_splits=5,
                    use_osi=False,
                    enet_n_jobs=enet_n_jobs,
                    tune_hyperparams=tune_hp,
                    frozen_hparams=None if tune_hp else cache_elastic_hp.get("HAR+SVD_GATED_AR"),
                )
                if tune_hp:
                    cache_elastic_hp["HAR+SVD_GATED_AR"] = linear_models.elastic_hyperparams_from_model(
                        cache["HAR+SVD_GATED_AR"].model
                    )
                cache_smearing["HAR+SVD_GATED_AR"] = _compute_smearing_factor(
                    y_tr, cache["HAR+SVD_GATED_AR"].pred_log_train
                )
                cache_step_last_fit["HAR+SVD_GATED_AR"] = step_id
                print(
                    f"[WF] REFIT HAR+SVD_GATED_AR tune={tune_hp} step_id={step_id} "
                    f"({time.perf_counter() - t_refit:.1f}s)",
                    flush=True,
                )
            else:
                pl_te = np.asarray(cache["HAR+SVD_GATED_AR"].model.predict(Xt2_gated_ar_te)).ravel()
                cache["HAR+SVD_GATED_AR"].pred_log_test = pl_te
            pl = float(cache["HAR+SVD_GATED_AR"].pred_log_test[0])
            preds_log["HAR+SVD_GATED_AR"].append(pl)
            preds_var["HAR+SVD_GATED_AR"].append(
                _pred_var_from_log(pl, y_tr, cache_smearing["HAR+SVD_GATED_AR"])
            )

        # -----------------------------------------------------------------
        # HAR+SVD (Dynamic spectral HAR block) ElasticNet
        # -----------------------------------------------------------------
        if model_enabled("HAR+SVD_DYN", required_models):
            assert Xt2_dyn_tr is not None and Xt2_dyn_te is not None
            if needs_refit("HAR+SVD_DYN", step_id, cadence, cache, cache_step_last_fit):
                tune_hp = should_tune_elasticnet(step_id, walk_cfg)
                t_refit = time.perf_counter()
                cache["HAR+SVD_DYN"] = fit_predict_elasticnet(
                    Xt2_dyn_tr,
                    y_tr,
                    Xt2_dyn_te,
                    cv_splits=5,
                    use_osi=False,
                    enet_n_jobs=enet_n_jobs,
                    tune_hyperparams=tune_hp,
                    frozen_hparams=None if tune_hp else cache_elastic_hp.get("HAR+SVD_DYN"),
                )
                if tune_hp:
                    cache_elastic_hp["HAR+SVD_DYN"] = linear_models.elastic_hyperparams_from_model(
                        cache["HAR+SVD_DYN"].model
                    )
                cache_smearing["HAR+SVD_DYN"] = _compute_smearing_factor(
                    y_tr, cache["HAR+SVD_DYN"].pred_log_train
                )
                cache_step_last_fit["HAR+SVD_DYN"] = step_id
                print(
                    f"[WF] REFIT HAR+SVD_DYN tune={tune_hp} step_id={step_id} "
                    f"({time.perf_counter() - t_refit:.1f}s)",
                    flush=True,
                )
            else:
                pl_te = np.asarray(cache["HAR+SVD_DYN"].model.predict(Xt2_dyn_te)).ravel()
                cache["HAR+SVD_DYN"].pred_log_test = pl_te
            pl = float(cache["HAR+SVD_DYN"].pred_log_test[0])
            preds_log["HAR+SVD_DYN"].append(pl)
            preds_var["HAR+SVD_DYN"].append(
                _pred_var_from_log(pl, y_tr, cache_smearing["HAR+SVD_DYN"])
            )

        # -----------------------------------------------------------------
        # HAR+SVD (RFF + Ridge) — complexity with shrinkage
        # -----------------------------------------------------------------
        if model_enabled("HAR+SVD_RFF_RIDGE", required_models):
            if needs_refit("HAR+SVD_RFF_RIDGE", step_id, cadence, cache, cache_step_last_fit):
                tune_hp = should_tune_elasticnet(step_id, walk_cfg)
                t_refit = time.perf_counter()
                frozen_alpha = None
                if not tune_hp and "HAR+SVD_RFF_RIDGE" in cache_elastic_hp:
                    frozen_alpha = float(cache_elastic_hp["HAR+SVD_RFF_RIDGE"]["alpha"])
                cache["HAR+SVD_RFF_RIDGE"] = fit_predict_ridge_rff(
                    Xt2_tr,
                    y_tr,
                    Xt2_te,
                    cv_splits=5,
                    tune_hyperparams=tune_hp,
                    frozen_alpha=frozen_alpha,
                    n_components=int(walk_cfg.get("rff_n_components", 256)),
                    gamma=float(walk_cfg.get("rff_gamma", 1.0)),
                    seed=int(walk_cfg.get("seed_offset", 10_000)) + int(step_id),
                )
                if tune_hp:
                    cache_elastic_hp["HAR+SVD_RFF_RIDGE"] = {
                        "alpha": float(linear_models.ridge_alpha_from_model(cache["HAR+SVD_RFF_RIDGE"].model))
                    }
                cache_smearing["HAR+SVD_RFF_RIDGE"] = _compute_smearing_factor(
                    y_tr, cache["HAR+SVD_RFF_RIDGE"].pred_log_train
                )
                cache_step_last_fit["HAR+SVD_RFF_RIDGE"] = step_id
                print(
                    f"[WF] REFIT HAR+SVD_RFF_RIDGE tune={tune_hp} step_id={step_id} "
                    f"({time.perf_counter() - t_refit:.1f}s)",
                    flush=True,
                )
            else:
                pl_te = np.asarray(cache["HAR+SVD_RFF_RIDGE"].model.predict(Xt2_te)).ravel()
                cache["HAR+SVD_RFF_RIDGE"].pred_log_test = pl_te
            pl = float(cache["HAR+SVD_RFF_RIDGE"].pred_log_test[0])
            preds_log["HAR+SVD_RFF_RIDGE"].append(pl)
            preds_var["HAR+SVD_RFF_RIDGE"].append(
                _pred_var_from_log(pl, y_tr, cache_smearing["HAR+SVD_RFF_RIDGE"])
            )

        # -----------------------------------------------------------------
        # Tier-3 ElasticNet (HAR+SVD+XS) — fit same as Tier-2 but with Xt3
        # -----------------------------------------------------------------
        if model_enabled("HAR_SVD_T3", required_models):
            if needs_refit("HAR_SVD_T3", step_id, cadence, cache, cache_step_last_fit):
                tune_hp = should_tune_elasticnet(step_id, walk_cfg)
                t_refit = time.perf_counter()
                cache["HAR_SVD_T3"] = fit_predict_elasticnet(
                    Xt3_tr,
                    y_tr,
                    Xt3_te,
                    cv_splits=5,
                    use_osi=False,
                    enet_n_jobs=enet_n_jobs,
                    tune_hyperparams=tune_hp,
                    frozen_hparams=None if tune_hp else cache_elastic_hp.get("HAR_SVD_T3"),
                )
                if tune_hp:
                    cache_elastic_hp["HAR_SVD_T3"] = linear_models.elastic_hyperparams_from_model(
                        cache["HAR_SVD_T3"].model
                    )
                cache_smearing["HAR_SVD_T3"] = _compute_smearing_factor(
                    y_tr, cache["HAR_SVD_T3"].pred_log_train
                )
                cache_step_last_fit["HAR_SVD_T3"] = step_id
                print(
                    f"[WF] REFIT HAR_SVD_T3 tune={tune_hp} step_id={step_id} "
                    f"({time.perf_counter() - t_refit:.1f}s)",
                    flush=True,
                )
            else:
                pl_te = linear_models.predict_har_svd_elastic(
                    cache["HAR_SVD_T3"].model, Xt3_te
                ).ravel()
                cache["HAR_SVD_T3"].pred_log_test = pl_te
            pl = float(cache["HAR_SVD_T3"].pred_log_test[0])
            preds_log["HAR_SVD_T3"].append(pl)
            preds_var["HAR_SVD_T3"].append(
                _pred_var_from_log(pl, y_tr, cache_smearing["HAR_SVD_T3"])
            )

        # -----------------------------------------------------------------
        # DNN_HAR (scaled)
        # -----------------------------------------------------------------
        if model_enabled("DNN_HAR", required_models) and (
            needs_refit("DNN_HAR", step_id, cadence, cache, cache_step_last_fit)
        ):
            Xtr_s, Xva_s, Xte_s, sc = fit_scaler_train_only(Xhar_tr, Xhar_va, Xhar_te)
            cache_scalers["DNN_HAR"] = sc
            cache["DNN_HAR"] = fit_predict_dnn_ensemble(
                ens_seeds,
                Xtr_s, y_tr, Xva_s, y_va, Xte_s,
                cfg_dnn=cfg_dnn, hidden_dims=hidden_dnn, init_har_preds_train=None, use_gate=False
            )
            cache_smearing["DNN_HAR"] = _compute_smearing_factor(y_tr, cache["DNN_HAR"].pred_log_train)
            cache_step_last_fit["DNN_HAR"] = step_id
            xcfg = walk_cfg.get("xai", {})
            if _walk_forward_run_xai_now(step_id, xcfg):
                fn = feature_names.get("HAR", [])
                if len(fn) == Xhar_va.shape[1]:
                    models = cache["DNN_HAR"].aux.get("models", []) if cache["DNN_HAR"].aux else []
                    if not models:
                        models = [cache["DNN_HAR"].model]
                    sc = cache_scalers["DNN_HAR"]
                    Xva_s = sc.transform(Xhar_va)
                    def _pred_var(Xin_scaled: np.ndarray) -> np.ndarray:
                        preds = [m.predict(Xin_scaled, verbose=0).ravel() for m in models]
                        pl = np.stack(preds).mean(axis=0)
                        return _pred_var_from_log_vec(pl, y_tr, cache_smearing["DNN_HAR"])
                    res = xai_mod.blocked_permutation_importance_2d(
                        X=Xva_s,
                        y_true_var=y_va_var,
                        feature_names=list(fn),
                        predict_var=_pred_var,
                        loss=str(xcfg.get("loss", "qlike")),
                        reps=int(xcfg.get("reps", 10)),
                        block_len=int(xcfg.get("block_len", 5)),
                        seed=int(xcfg.get("seed", 42)) + int(step_id),
                    )
                    xai_records.append({
                        "step_id": step_id,
                        "test_date": str(td.date()),
                        "model": "DNN_HAR",
                        "base_loss": res.base_loss,
                        "feature_names": res.feature_names,
                        "importance_mean": res.importances_mean.tolist(),
                        "importance_std": res.importances_std.tolist(),
                    })
        elif model_enabled("DNN_HAR", required_models):
            sc = cache_scalers["DNN_HAR"]
            Xte_s = sc.transform(Xhar_te)
            cache["DNN_HAR"].pred_log_test = cache["DNN_HAR"].model.predict(Xte_s, verbose=0).ravel()
        if model_enabled("DNN_HAR", required_models):
            pl = float(cache["DNN_HAR"].pred_log_test[0])
            preds_log["DNN_HAR"].append(pl)
            preds_var["DNN_HAR"].append(_pred_var_from_log(pl, y_tr, cache_smearing["DNN_HAR"]))

        # -----------------------------------------------------------------
        # DNN_HAR+SVD (scaled Tier-2)
        # -----------------------------------------------------------------
        if model_enabled("DNN_HAR+SVD", required_models) and (
            needs_refit("DNN_HAR+SVD", step_id, cadence, cache, cache_step_last_fit)
        ):
            Xtr_s, Xva_s, Xte_s, sc = fit_scaler_train_only(Xt2_tr, Xt2_va, Xt2_te)
            cache_scalers["DNN_HAR+SVD"] = sc
            cache["DNN_HAR+SVD"] = fit_predict_dnn_ensemble(
                ens_seeds,
                Xtr_s, y_tr, Xva_s, y_va, Xte_s,
                cfg_dnn=cfg_dnn, hidden_dims=hidden_dnn_svd, init_har_preds_train=None, use_gate=False
            )
            cache_smearing["DNN_HAR+SVD"] = _compute_smearing_factor(y_tr, cache["DNN_HAR+SVD"].pred_log_train)
            cache_step_last_fit["DNN_HAR+SVD"] = step_id
            xcfg = walk_cfg.get("xai", {})
            if _walk_forward_run_xai_now(step_id, xcfg):
                fn = feature_names.get("T2", [])
                if len(fn) == Xt2_va.shape[1]:
                    models = cache["DNN_HAR+SVD"].aux.get("models", []) if cache["DNN_HAR+SVD"].aux else []
                    if not models:
                        models = [cache["DNN_HAR+SVD"].model]
                    sc = cache_scalers["DNN_HAR+SVD"]
                    Xva_s = sc.transform(Xt2_va)
                    def _pred_var(Xin_scaled: np.ndarray) -> np.ndarray:
                        preds = [m.predict(Xin_scaled, verbose=0).ravel() for m in models]
                        pl = np.stack(preds).mean(axis=0)
                        return _pred_var_from_log_vec(pl, y_tr, cache_smearing["DNN_HAR+SVD"])
                    res = xai_mod.blocked_permutation_importance_2d(
                        X=Xva_s,
                        y_true_var=y_va_var,
                        feature_names=list(fn),
                        predict_var=_pred_var,
                        loss=str(xcfg.get("loss", "qlike")),
                        reps=int(xcfg.get("reps", 10)),
                        block_len=int(xcfg.get("block_len", 5)),
                        seed=int(xcfg.get("seed", 42)) + int(step_id),
                    )
                    xai_records.append({
                        "step_id": step_id,
                        "test_date": str(td.date()),
                        "model": "DNN_HAR+SVD",
                        "base_loss": res.base_loss,
                        "feature_names": res.feature_names,
                        "importance_mean": res.importances_mean.tolist(),
                        "importance_std": res.importances_std.tolist(),
                    })
        elif model_enabled("DNN_HAR+SVD", required_models):
            sc = cache_scalers["DNN_HAR+SVD"]
            Xte_s = sc.transform(Xt2_te)
            cache["DNN_HAR+SVD"].pred_log_test = cache["DNN_HAR+SVD"].model.predict(
                Xte_s, verbose=0
            ).ravel()
        if model_enabled("DNN_HAR+SVD", required_models):
            pl = float(cache["DNN_HAR+SVD"].pred_log_test[0])
            preds_log["DNN_HAR+SVD"].append(pl)
            preds_var["DNN_HAR+SVD"].append(
                _pred_var_from_log(pl, y_tr, cache_smearing["DNN_HAR+SVD"])
            )

        # -----------------------------------------------------------------
        # LSTM models (sequence needs last seq_len rows ending at test date)
        # -----------------------------------------------------------------
        if model_enabled("LSTM_HAR", required_models) and (
            needs_refit("LSTM_HAR", step_id, cadence, cache, cache_step_last_fit)
        ):
            Xtr_s, Xva_s, _, sc = fit_scaler_train_only(Xhar_tr, Xhar_va, Xhar_te)
            # Transform full history up to test date for test sequence construction
            X_full_scaled = sc.transform(X_har.loc[:td].values.astype(np.float64))
            X_test_seq = _build_test_sequence_from_full(X_full_scaled, X_har.loc[:td].index, td, seq_len=seq_len_lstm)
            cache_scalers["LSTM_HAR"] = sc
            cache["LSTM_HAR"] = fit_predict_lstm_ensemble(
                ens_seeds,
                Xtr_s, y_tr, Xva_s, y_va, X_test_seq, cfg_lstm=cfg_lstm, init_har_preds_train=None
            )
            cache_smearing["LSTM_HAR"] = _compute_smearing_factor(y_tr, cache["LSTM_HAR"].pred_log_train)
            cache_step_last_fit["LSTM_HAR"] = step_id
            xcfg = walk_cfg.get("xai", {})
            if _walk_forward_run_xai_now(step_id, xcfg):
                fn = feature_names.get("HAR", [])
                if len(fn) == Xhar_va.shape[1]:
                    models = cache["LSTM_HAR"].aux.get("models", []) if cache["LSTM_HAR"].aux else []
                    if not models:
                        models = [cache["LSTM_HAR"].model]
                    # Build a val-sequence predictor on scaled 2D inputs
                    sc = cache_scalers["LSTM_HAR"]
                    Xva_s_2d = sc.transform(Xhar_va)
                    yva_var_2d = y_va_var
                    def _pred_var_seq(Xseq: np.ndarray) -> np.ndarray:
                        preds = [m.predict(Xseq, verbose=0).ravel() for m in models]
                        pl = np.stack(preds).mean(axis=0)
                        return _pred_var_from_log_vec(pl, y_tr, cache_smearing["LSTM_HAR"])
                    res = xai_mod.blocked_permutation_importance_sequence_from_2d(
                        X_2d=Xva_s_2d,
                        y_true_var_2d=yva_var_2d,
                        feature_names=list(fn),
                        seq_len=seq_len_lstm,
                        predict_var_from_seq=_pred_var_seq,
                        loss=str(xcfg.get("loss", "qlike")),
                        reps=int(xcfg.get("reps", 10)),
                        block_len=int(xcfg.get("block_len", 5)),
                        seed=int(xcfg.get("seed", 42)) + int(step_id),
                    )
                    xai_records.append({
                        "step_id": step_id,
                        "test_date": str(td.date()),
                        "model": "LSTM_HAR",
                        "base_loss": res.base_loss,
                        "feature_names": res.feature_names,
                        "importance_mean": res.importances_mean.tolist(),
                        "importance_std": res.importances_std.tolist(),
                    })
        elif model_enabled("LSTM_HAR", required_models):
            sc = cache_scalers["LSTM_HAR"]
            X_full_scaled = sc.transform(X_har.loc[:td].values.astype(np.float64))
            X_test_seq = _build_test_sequence_from_full(
                X_full_scaled, X_har.loc[:td].index, td, seq_len=seq_len_lstm
            )
            cache["LSTM_HAR"].pred_log_test = cache["LSTM_HAR"].model.predict(
                X_test_seq, verbose=0
            ).ravel()
        if model_enabled("LSTM_HAR", required_models):
            pl = float(cache["LSTM_HAR"].pred_log_test[0])
            preds_log["LSTM_HAR"].append(pl)
            preds_var["LSTM_HAR"].append(_pred_var_from_log(pl, y_tr, cache_smearing["LSTM_HAR"]))

        if model_enabled("LSTM_HAR+SVD", required_models) and (
            needs_refit("LSTM_HAR+SVD", step_id, cadence, cache, cache_step_last_fit)
        ):
            Xtr_s, Xva_s, _, sc = fit_scaler_train_only(Xt2_tr, Xt2_va, Xt2_te)
            X_full_scaled = sc.transform(X_t2.loc[:td].values.astype(np.float64))
            X_test_seq = _build_test_sequence_from_full(X_full_scaled, X_t2.loc[:td].index, td, seq_len=seq_len_lstm)
            cache_scalers["LSTM_HAR+SVD"] = sc
            cache["LSTM_HAR+SVD"] = fit_predict_lstm_ensemble(
                ens_seeds,
                Xtr_s, y_tr, Xva_s, y_va, X_test_seq, cfg_lstm=cfg_lstm, init_har_preds_train=None
            )
            cache_smearing["LSTM_HAR+SVD"] = _compute_smearing_factor(y_tr, cache["LSTM_HAR+SVD"].pred_log_train)
            cache_step_last_fit["LSTM_HAR+SVD"] = step_id
            xcfg = walk_cfg.get("xai", {})
            if _walk_forward_run_xai_now(step_id, xcfg):
                fn = feature_names.get("T2", [])
                if len(fn) == Xt2_va.shape[1]:
                    models = cache["LSTM_HAR+SVD"].aux.get("models", []) if cache["LSTM_HAR+SVD"].aux else []
                    if not models:
                        models = [cache["LSTM_HAR+SVD"].model]
                    sc = cache_scalers["LSTM_HAR+SVD"]
                    Xva_s_2d = sc.transform(Xt2_va)
                    yva_var_2d = y_va_var
                    def _pred_var_seq(Xseq: np.ndarray) -> np.ndarray:
                        preds = [m.predict(Xseq, verbose=0).ravel() for m in models]
                        pl = np.stack(preds).mean(axis=0)
                        return _pred_var_from_log_vec(pl, y_tr, cache_smearing["LSTM_HAR+SVD"])
                    res = xai_mod.blocked_permutation_importance_sequence_from_2d(
                        X_2d=Xva_s_2d,
                        y_true_var_2d=yva_var_2d,
                        feature_names=list(fn),
                        seq_len=seq_len_lstm,
                        predict_var_from_seq=_pred_var_seq,
                        loss=str(xcfg.get("loss", "qlike")),
                        reps=int(xcfg.get("reps", 10)),
                        block_len=int(xcfg.get("block_len", 5)),
                        seed=int(xcfg.get("seed", 42)) + int(step_id),
                    )
                    xai_records.append({
                        "step_id": step_id,
                        "test_date": str(td.date()),
                        "model": "LSTM_HAR+SVD",
                        "base_loss": res.base_loss,
                        "feature_names": res.feature_names,
                        "importance_mean": res.importances_mean.tolist(),
                        "importance_std": res.importances_std.tolist(),
                    })
        elif model_enabled("LSTM_HAR+SVD", required_models):
            sc = cache_scalers["LSTM_HAR+SVD"]
            X_full_scaled = sc.transform(X_t2.loc[:td].values.astype(np.float64))
            X_test_seq = _build_test_sequence_from_full(
                X_full_scaled, X_t2.loc[:td].index, td, seq_len=seq_len_lstm
            )
            cache["LSTM_HAR+SVD"].pred_log_test = cache["LSTM_HAR+SVD"].model.predict(
                X_test_seq, verbose=0
            ).ravel()
        if model_enabled("LSTM_HAR+SVD", required_models):
            pl = float(cache["LSTM_HAR+SVD"].pred_log_test[0])
            preds_log["LSTM_HAR+SVD"].append(pl)
            preds_var["LSTM_HAR+SVD"].append(
                _pred_var_from_log(pl, y_tr, cache_smearing["LSTM_HAR+SVD"])
            )

        # -----------------------------------------------------------------
        # HARNet (sequence; uses Tier-2 by design in the pipeline)
        # -----------------------------------------------------------------
        if model_enabled("HARNet", required_models) and (
            needs_refit("HARNet", step_id, cadence, cache, cache_step_last_fit)
        ):
            Xtr_s, Xva_s, _, sc = fit_scaler_train_only(Xt2_tr, Xt2_va, Xt2_te)
            X_full_scaled = sc.transform(X_t2.loc[:td].values.astype(np.float64))
            X_test_seq = _build_test_sequence_from_full(X_full_scaled, X_t2.loc[:td].index, td, seq_len=seq_len_harnet)
            cache_scalers["HARNet"] = sc
            cache["HARNet"] = fit_predict_harnet(
                Xtr_s, y_tr, Xva_s, y_va, X_test_seq, cfg_harnet=cfg_harnet, init_har_preds_train=None
            )
            cache_smearing["HARNet"] = _compute_smearing_factor(y_tr, cache["HARNet"].pred_log_train)
            cache_step_last_fit["HARNet"] = step_id
        elif model_enabled("HARNet", required_models):
            sc = cache_scalers["HARNet"]
            X_full_scaled = sc.transform(X_t2.loc[:td].values.astype(np.float64))
            X_test_seq = _build_test_sequence_from_full(
                X_full_scaled, X_t2.loc[:td].index, td, seq_len=seq_len_harnet
            )
            cache["HARNet"].pred_log_test = cache["HARNet"].model.predict(
                X_test_seq, verbose=0
            ).ravel()
        if model_enabled("HARNet", required_models):
            pl = float(cache["HARNet"].pred_log_test[0])
            preds_log["HARNet"].append(pl)
            preds_var["HARNet"].append(_pred_var_from_log(pl, y_tr, cache_smearing["HARNet"]))

        # -----------------------------------------------------------------
        # GNN (requires node_features_full and adj_full aligned to idx)
        # -----------------------------------------------------------------
        if model_enabled("GNN", required_models):
            if node_features_full is None or adj_full is None:
                raise ValueError(
                    "GNN is in walk_forward.models but node_features_full/adj_full "
                    "were not provided."
                )
            if node_features_full.shape[0] != len(idx) or adj_full.shape[0] != len(idx):
                raise ValueError("GNN full arrays must have first dimension == len(idx).")

        if model_enabled("GNN", required_models) and (
            needs_refit("GNN", step_id, cadence, cache, cache_step_last_fit)
        ):
            nf_tr = node_features_full[tr_pos]
            ad_tr = adj_full[tr_pos]
            nf_va = node_features_full[va_pos]
            ad_va = adj_full[va_pos]
            nf_te = node_features_full[te_pos]
            ad_te = adj_full[te_pos]
            cache["GNN"] = fit_predict_gnn(
                nf_tr, ad_tr, y_tr,
                nf_va, ad_va, y_va,
                nf_te, ad_te,
                cfg_gnn=cfg_gnn,
            )
            cache_smearing["GNN"] = _compute_smearing_factor(y_tr, cache["GNN"].pred_log_train)
            cache_step_last_fit["GNN"] = step_id
            xcfg = walk_cfg.get("xai", {})
            if _walk_forward_run_xai_now(step_id, xcfg):
                node_feat_names = ["abs_r", "rv5", "f1", "sigma1"]
                nf_va = node_features_full[va_pos]
                ad_va = adj_full[va_pos]
                def _pred_var_nf(nf_in: np.ndarray) -> np.ndarray:
                    pl = cache["GNN"].model.predict([nf_in.astype(np.float32), ad_va.astype(np.float32)], verbose=0).ravel()
                    return _pred_var_from_log_vec(pl, y_tr, cache_smearing["GNN"])
                res = xai_mod.blocked_permutation_importance_gnn_time(
                    node_features=nf_va,
                    y_true_var=y_va_var,
                    node_feat_names=node_feat_names,
                    predict_var=_pred_var_nf,
                    loss=str(xcfg.get("loss", "qlike")),
                    reps=int(xcfg.get("reps", 10)),
                    block_len=int(xcfg.get("block_len", 5)),
                    seed=int(xcfg.get("seed", 42)) + int(step_id),
                )
                xai_records.append({
                    "step_id": step_id,
                    "test_date": str(td.date()),
                    "model": "GNN",
                    "base_loss": res.base_loss,
                    "feature_names": res.feature_names,
                    "importance_mean": res.importances_mean.tolist(),
                    "importance_std": res.importances_std.tolist(),
                })
        elif model_enabled("GNN", required_models):
            nf_te = node_features_full[te_pos]
            ad_te = adj_full[te_pos]
            cache["GNN"].pred_log_test = cache["GNN"].model.predict(
                [nf_te, ad_te], verbose=0
            ).ravel()
        if model_enabled("GNN", required_models):
            pl = float(cache["GNN"].pred_log_test[0])
            preds_log["GNN"].append(pl)
            preds_var["GNN"].append(_pred_var_from_log(pl, y_tr, cache_smearing["GNN"]))

        if checkpoint_dir is not None and (si % prog_every == 0 or si == n_splits - 1):
            save_walk_forward_checkpoint(
                checkpoint_dir,
                protocol=protocol,
                horizon=horizon,
                profile=profile_name,
                fingerprint=wf_fingerprint,
                last_step_id=step_id,
                last_si=si,
                preds_log=preds_log,
                preds_var=preds_var,
                test_dates=test_dates,
            )

    # Build output series
    test_idx = pd.Index(test_dates)
    out_preds = {m: pd.Series(np.asarray(v, dtype=np.float64), index=test_idx) for m, v in preds_log.items()}
    out_preds_var = {m: pd.Series(np.asarray(v, dtype=np.float64), index=test_idx) for m, v in preds_var.items()}
    y_true_var_oos = y_true_var.reindex(test_idx)
    y_true_log_oos = y_true_log.reindex(test_idx)
    if y_true_var_oos.isna().any() or y_true_log_oos.isna().any():
        raise ValueError("OOS y_true contains NaNs; ensure target alignment before walk-forward.")

    wf_elapsed = time.perf_counter() - wf_t0
    print(
        f"[WF] finished protocol={protocol!r} h={horizon} | {n_splits} splits | "
        f"wall {wf_elapsed / 60.0:.1f}m ({wf_elapsed / max(n_splits, 1):.2f}s/split mean)",
        flush=True,
    )

    return WalkForwardOutputs(
        protocol=protocol,
        horizon=int(horizon),
        y_true_var=y_true_var_oos,
        y_true_log=y_true_log_oos,
        preds_log=out_preds,
        preds_var=out_preds_var,
        xai={"records": xai_records},
        metadata={
            "protocol": protocol,
            "horizon": int(horizon),
            "walk_cfg": dict(walk_cfg),
            "feature_names": feature_names,
            "use_osi": bool(use_osi),
            "smearing_factor_bounds": [sf_lo, sf_hi],
            "pred_log_clip_width": clip_w,
            "n_splits": int(n_splits),
            "elastic_net_cv_n_jobs": enet_n_jobs,
            "xai_run_every_n_steps": max(1, int(walk_cfg.get("xai", {}).get("run_every_n_steps", 1))),
        },
    )


def compute_dm_nlags_for_horizon(h: int, T: int | None = None) -> int:
    """
    Pre-registered horizon-aware HAC bandwidth for DM tests.

    Two policies are supported, controlled by ``config["walk_forward"]
    ["dm_hac_nlags_policy"]``:

    * ``"max_h_minus_1_andrews"`` (default): the cube-root rule
      ``max(h - 1, floor(base_factor * T^{1/3}))`` from
      ``methods/preregistration.md`` (Andrews 1991 + West 1996 floor).
      Falls back to ``base_factor = walk_forward.dm_hac_base_factor``
      (default 1.5) and uses ``T`` if supplied.
    * ``"base_plus_h_minus_1"`` (legacy): ``base + (h - 1)``.

    When ``T`` is not provided, the cube-root rule degrades gracefully to
    ``base + (h - 1)`` so that legacy MCS block-length call sites continue
    to work.
    """
    wf = config.config.get("walk_forward", {})
    base = int(wf.get("dm_hac_nlags_base", 5))
    policy = str(wf.get("dm_hac_nlags_policy", "max_h_minus_1_andrews"))
    h_int = int(h)
    if policy == "base_plus_h_minus_1":
        return int(base + max(h_int - 1, 0))
    if policy != "max_h_minus_1_andrews":
        raise ValueError(f"Unknown dm_hac_nlags_policy: {policy}")
    base_factor = float(wf.get("dm_hac_base_factor", 1.5))
    overlap = max(h_int - 1, 0)
    if T is None:
        return int(max(overlap, base + overlap, 1))
    rule = int(np.floor(base_factor * (max(int(T), 1) ** (1.0 / 3.0))))
    return int(max(overlap, rule, 1))

