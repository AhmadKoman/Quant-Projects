"""
Model training / prediction wrappers with explicit (train, val, test) arrays.

Design principles:
  - No hidden global state.
  - No leakage: callers must pass slices already respecting time order.
  - Deterministic given seeds passed by the caller.

This file exists to support the walk-forward engine, but can also be used for
fixed-split experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from . import dnn as dnn_models
from . import garch as garch_models
from . import gnn as gnn_models
from . import harnet as harnet_models
from . import linear as linear_models
from . import lstm as lstm_models


@dataclass
class FitPredictResult:
    """Container for outputs needed downstream (metrics, smearing, uncertainty, XAI)."""

    pred_log_test: np.ndarray
    pred_log_train: np.ndarray | None = None
    model: Any | None = None
    aux: dict[str, Any] | None = None


def fit_predict_har(
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    X_test: np.ndarray,
) -> FitPredictResult:
    model = linear_models.train_har(X_train, y_train_log, add_constant=True)
    pred_log_test = linear_models.predict_har(model, X_test, add_constant=True).ravel()
    pred_log_train = linear_models.predict_har(model, X_train, add_constant=True).ravel()
    return FitPredictResult(pred_log_test=pred_log_test, pred_log_train=pred_log_train, model=model)


def fit_predict_elasticnet(
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    X_test: np.ndarray,
    *,
    cv_splits: int = 5,
    use_osi: bool = False,
    har_col_count: int | None = None,
    enet_n_jobs: int | None = None,
    tune_hyperparams: bool = True,
    frozen_hparams: dict[str, float] | None = None,
) -> FitPredictResult:
    fp = dict(frozen_hparams or {})
    if use_osi:
        if har_col_count is None:
            raise ValueError("use_osi=True requires har_col_count to be specified.")
        pred_log_test, pred_log_fit, predictor = linear_models.train_predict_osi_elastic(
            X_train,
            X_test,
            y_train_log,
            har_col_count=har_col_count,
            cv_splits=cv_splits,
            enet_n_jobs=enet_n_jobs,
            tune_har=bool(tune_hyperparams),
            tune_spec=bool(tune_hyperparams),
            alpha_har=fp.get("alpha_har"),
            l1_ratio_har=fp.get("l1_ratio_har"),
            alpha_spec=fp.get("alpha_spec"),
            l1_ratio_spec=fp.get("l1_ratio_spec"),
        )
        # Store the OsiPredictor itself in ``model`` so downstream code that
        # calls ``model.predict(X)`` (e.g. the walk-forward XAI hook) gets the
        # full two-stage forecast — including the frozen Frisch–Waugh
        # orthogonalisation of the spectral block — rather than a raw tuple
        # of pipes.  Component pipes remain available via aux for any caller
        # that needs to introspect coefficients.
        return FitPredictResult(
            pred_log_test=np.asarray(pred_log_test).ravel(),
            pred_log_train=np.asarray(pred_log_fit).ravel(),
            model=predictor,
            aux={
                "osi": True,
                "pipe_har": predictor.pipe_har,
                "pipe_spec": predictor.pipe_spec,
                "har_col_count": int(har_col_count),
            },
        )
    pipe = linear_models.train_har_svd_elastic(
        X_train,
        y_train_log,
        cv_splits=cv_splits,
        enet_n_jobs=enet_n_jobs,
        tune=bool(tune_hyperparams),
        alpha=fp.get("alpha"),
        l1_ratio=fp.get("l1_ratio"),
    )
    pred_log_test = linear_models.predict_har_svd_elastic(pipe, X_test).ravel()
    pred_log_train = linear_models.predict_har_svd_elastic(pipe, X_train).ravel()
    return FitPredictResult(pred_log_test=pred_log_test, pred_log_train=pred_log_train, model=pipe)


def fit_predict_ridge_rff(
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    X_test: np.ndarray,
    *,
    cv_splits: int = 5,
    tune_hyperparams: bool = True,
    frozen_alpha: float | None = None,
    n_components: int = 256,
    gamma: float = 1.0,
    seed: int = 42,
) -> FitPredictResult:
    pipe = linear_models.train_ridge_rff(
        X_train,
        y_train_log,
        cv_splits=cv_splits,
        tune=bool(tune_hyperparams),
        alpha=frozen_alpha,
        n_components=int(n_components),
        gamma=float(gamma),
        seed=int(seed),
    )
    pred_log_test = np.asarray(pipe.predict(X_test), dtype=np.float64).ravel()
    pred_log_train = np.asarray(pipe.predict(X_train), dtype=np.float64).ravel()
    return FitPredictResult(pred_log_test=pred_log_test, pred_log_train=pred_log_train, model=pipe)


def fit_predict_dnn(
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    X_val: np.ndarray,
    y_val_log: np.ndarray,
    X_test: np.ndarray,
    *,
    cfg_dnn: dict,
    hidden_dims: list[int],
    init_har_preds_train: np.ndarray | None = None,
    use_gate: bool = False,
) -> FitPredictResult:
    # DNN expects already-scaled inputs (handled upstream for strict consistency with LSTM/HARNet).
    model, hist = dnn_models.train_dnn(
        X_train,
        y_train_log,
        X_val,
        y_val_log,
        input_dim=int(X_train.shape[1]),
        epochs=int(cfg_dnn["epochs"]),
        batch_size=int(cfg_dnn["batch_size"]),
        patience=int(cfg_dnn["patience"]),
        hidden_dims=hidden_dims,
        dropout=float(cfg_dnn["dropout"]),
        l2_reg=float(cfg_dnn["l2_reg"]),
        lr=float(cfg_dnn["learning_rate"]),
        init_har_preds=init_har_preds_train,
        use_gate=bool(use_gate),
    )
    pred_log_test = model.predict(X_test, verbose=0).ravel()
    pred_log_train = model.predict(X_train, verbose=0).ravel()
    return FitPredictResult(
        pred_log_test=pred_log_test,
        pred_log_train=pred_log_train,
        model=model,
        aux={"history": hist},
    )


def fit_predict_dnn_ensemble(
    sub_seeds: list[int],
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    X_val: np.ndarray,
    y_val_log: np.ndarray,
    X_test: np.ndarray,
    *,
    cfg_dnn: dict,
    hidden_dims: list[int],
    init_har_preds_train: np.ndarray | None = None,
    use_gate: bool = False,
) -> FitPredictResult:
    """
    Train multiple DNNs with different seeds and average predictions.

    Returns ensemble-mean predictions on test and train (train used for smearing).
    """
    preds_test = []
    preds_train = []
    models = []
    last_hist = None
    for s in sub_seeds:
        # Callers set global TF/np seeds; we still set TF seed here for safety.
        import tensorflow as tf

        tf.random.set_seed(int(s))
        np.random.seed(int(s))
        model, hist = dnn_models.train_dnn(
            X_train,
            y_train_log,
            X_val,
            y_val_log,
            input_dim=int(X_train.shape[1]),
            epochs=int(cfg_dnn["epochs"]),
            batch_size=int(cfg_dnn["batch_size"]),
            patience=int(cfg_dnn["patience"]),
            hidden_dims=hidden_dims,
            dropout=float(cfg_dnn["dropout"]),
            l2_reg=float(cfg_dnn["l2_reg"]),
            lr=float(cfg_dnn["learning_rate"]),
            init_har_preds=init_har_preds_train,
            use_gate=bool(use_gate),
        )
        preds_test.append(model.predict(X_test, verbose=0).ravel())
        preds_train.append(model.predict(X_train, verbose=0).ravel())
        models.append(model)
        last_hist = hist
    mean_test = np.stack(preds_test).mean(axis=0)
    mean_train = np.stack(preds_train).mean(axis=0)
    return FitPredictResult(
        pred_log_test=mean_test,
        pred_log_train=mean_train,
        model=models[-1] if models else None,
        aux={"history": last_hist, "ensemble_size": int(len(sub_seeds)), "models": models},
    )


def fit_predict_lstm(
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    X_val: np.ndarray,
    y_val_log: np.ndarray,
    X_test_seq: np.ndarray,
    *,
    cfg_lstm: dict,
    init_har_preds_train: np.ndarray | None = None,
) -> FitPredictResult:
    # LSTM is trained on 2D arrays; sequences are built internally for train/val,
    # while test is passed as a prebuilt sequence array (to avoid ambiguity about
    # which history is allowed in walk-forward).
    model, hist = lstm_models.train_lstm(
        X_train,
        y_train_log,
        X_val,
        y_val_log,
        seq_len=int(cfg_lstm["seq_len"]),
        n_features=int(X_train.shape[1]),
        epochs=int(cfg_lstm["epochs"]),
        batch_size=int(cfg_lstm["batch_size"]),
        patience=int(cfg_lstm["patience"]),
        hidden_size=int(cfg_lstm["hidden_size"]),
        dropout=float(cfg_lstm["dropout"]),
        recurrent_dropout=float(cfg_lstm["recurrent_dropout"]),
        l2_reg=float(cfg_lstm["l2_reg"]),
        lr=float(cfg_lstm["learning_rate"]),
        init_har_preds=init_har_preds_train,
    )
    pred_log_test = model.predict(X_test_seq, verbose=0).ravel()
    # For smearing parity we return train-sequence predictions (caller aligns length).
    X_tr_seq, _ = lstm_models.build_sequences(X_train, y_train_log, int(cfg_lstm["seq_len"]))
    pred_log_train_seq = model.predict(X_tr_seq, verbose=0).ravel()
    return FitPredictResult(
        pred_log_test=pred_log_test,
        pred_log_train=pred_log_train_seq,
        model=model,
        aux={"history": hist},
    )


def fit_predict_lstm_ensemble(
    sub_seeds: list[int],
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    X_val: np.ndarray,
    y_val_log: np.ndarray,
    X_test_seq: np.ndarray,
    *,
    cfg_lstm: dict,
    init_har_preds_train: np.ndarray | None = None,
) -> FitPredictResult:
    """
    Train multiple LSTMs and average predictions.

    Returns:
      - test predictions on X_test_seq (already sequenced)
      - train-sequence predictions on X_train sequences (for smearing)
    """
    seq_len = int(cfg_lstm["seq_len"])
    X_tr_seq, _ = lstm_models.build_sequences(X_train, y_train_log, seq_len)
    if len(X_tr_seq) == 0:
        raise ValueError("LSTM ensemble: empty train sequences.")
    preds_test = []
    preds_train_seq = []
    models = []
    last_hist = None
    for s in sub_seeds:
        import tensorflow as tf

        tf.random.set_seed(int(s))
        np.random.seed(int(s))
        model, hist = lstm_models.train_lstm(
            X_train,
            y_train_log,
            X_val,
            y_val_log,
            seq_len=seq_len,
            n_features=int(X_train.shape[1]),
            epochs=int(cfg_lstm["epochs"]),
            batch_size=int(cfg_lstm["batch_size"]),
            patience=int(cfg_lstm["patience"]),
            hidden_size=int(cfg_lstm["hidden_size"]),
            dropout=float(cfg_lstm["dropout"]),
            recurrent_dropout=float(cfg_lstm["recurrent_dropout"]),
            l2_reg=float(cfg_lstm["l2_reg"]),
            lr=float(cfg_lstm["learning_rate"]),
            init_har_preds=init_har_preds_train,
        )
        preds_test.append(model.predict(X_test_seq, verbose=0).ravel())
        preds_train_seq.append(model.predict(X_tr_seq, verbose=0).ravel())
        models.append(model)
        last_hist = hist
    mean_test = np.stack(preds_test).mean(axis=0)
    mean_train_seq = np.stack(preds_train_seq).mean(axis=0)
    return FitPredictResult(
        pred_log_test=mean_test,
        pred_log_train=mean_train_seq,
        model=models[-1] if models else None,
        aux={"history": last_hist, "ensemble_size": int(len(sub_seeds)), "models": models},
    )


def fit_predict_harnet(
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    X_val: np.ndarray,
    y_val_log: np.ndarray,
    X_test_seq: np.ndarray,
    *,
    cfg_harnet: dict,
    init_har_preds_train: np.ndarray | None = None,
) -> FitPredictResult:
    model, hist = harnet_models.train_harnet(
        X_train,
        y_train_log,
        X_val,
        y_val_log,
        seq_len=int(cfg_harnet["seq_len"]),
        n_features=int(X_train.shape[1]),
        epochs=int(cfg_harnet["epochs"]),
        batch_size=int(cfg_harnet["batch_size"]),
        patience=int(cfg_harnet["patience"]),
        filters=int(cfg_harnet["filters"]),
        dilations=list(cfg_harnet["dilations"]),
        dropout=float(cfg_harnet["dropout"]),
        lr=float(cfg_harnet["learning_rate"]),
        init_har_preds=init_har_preds_train,
    )
    pred_log_test = model.predict(X_test_seq, verbose=0).ravel()
    # Train sequence predictions for smearing parity
    from .lstm import build_sequences

    X_tr_seq, _ = build_sequences(X_train, y_train_log, int(cfg_harnet["seq_len"]))
    pred_log_train_seq = model.predict(X_tr_seq, verbose=0).ravel()
    return FitPredictResult(
        pred_log_test=pred_log_test,
        pred_log_train=pred_log_train_seq,
        model=model,
        aux={"history": hist},
    )


def fit_predict_gnn(
    node_features_train: np.ndarray,
    adj_train: np.ndarray,
    y_train_log: np.ndarray,
    node_features_val: np.ndarray,
    adj_val: np.ndarray,
    y_val_log: np.ndarray,
    node_features_test: np.ndarray,
    adj_test: np.ndarray,
    *,
    cfg_gnn: dict,
) -> FitPredictResult:
    n_nodes = int(node_features_train.shape[1])
    node_feat_dim = int(node_features_train.shape[2])
    model, hist = gnn_models.train_gnn(
        node_features_train,
        adj_train,
        y_train_log,
        node_features_val,
        adj_val,
        y_val_log,
        n_nodes=n_nodes,
        node_feat_dim=node_feat_dim,
        hidden_dim=int(cfg_gnn["hidden_dim"]),
        dropout=float(cfg_gnn["dropout"]),
        lr=float(cfg_gnn["learning_rate"]),
        epochs=int(cfg_gnn["epochs"]),
        batch_size=int(cfg_gnn["batch_size"]),
        patience=int(cfg_gnn["patience"]),
    )
    pred_log_test = gnn_models.gnn_predict(model, node_features_test, adj_test).ravel()
    pred_log_train = gnn_models.gnn_predict(model, node_features_train, adj_train).ravel()
    return FitPredictResult(
        pred_log_test=pred_log_test,
        pred_log_train=pred_log_train,
        model=model,
        aux={"history": hist},
    )


def predict_garch_benchmark(
    returns: np.ndarray,
    train_mask: np.ndarray,
    test_dates: np.ndarray,
    *,
    innovations: str,
    t_df: float,
    refit_every: int | None,
) -> np.ndarray:
    raise NotImplementedError("Use models_garch.build_garch on pandas Series; orchestration should supply series+dates.")


def fit_scaler_train_only(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Convenience helper for strict train-only scaling (used by DNN/LSTM/HARNet)."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test), scaler

