# -*- coding: utf-8 -*-
"""
LSTM: M5 (HAR only), M6 (HAR+SVD). No IV.

Loss function: QLIKE (see models_dnn.qlike_loss_logvar for details).

Normalization: The previous Lambda-based RevIN (Kim et al. 2022) contained a
critical bug — the inverse denormalization used only the first feature's
(f0) mean/std to re-scale the scalar LSTM output, discarding statistics from
all other input channels. For a model with 5-11 features, this produced
catastrophically wrong output scales (R2 ~= 0.04 at h=1 vs HAR's 0.22).

Fix: RevIN is removed entirely. The upstream StandardScaler applied in
run_experiments.py already standardizes all features to zero-mean unit-variance
before the LSTM sees them. StandardScaler is fit on the training set only
(no leakage) and is strictly more principled than per-sequence RevIN when
the inputs have already been globally normalized. The LSTM output is a
log-variance prediction in standardized space; QLIKE loss is scale-invariant
in log-space, so no additional denormalization is needed.

HAR initialization: LSTM is pre-initialized to match HAR predictions via MSE
warm-up epochs before switching to QLIKE, preventing divergence at epoch 0.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout

from .dnn import qlike_loss_logvar, composite_qlike_mse_loss


def create_lstm(
    seq_len: int,
    n_features: int,
    hidden_size: int = 64,
    dropout: float = 0.2,
    recurrent_dropout: float = 0.0,
    l2_reg: float = 1e-4,
    lr: float = 0.001,
    use_revin: bool = False,
) -> Model:
    """
    Build LSTM for log-variance forecasting.

    Architecture: Input -> LSTM(hidden_size) -> Dropout -> Dense(1)

    Notes
    -----
    - `use_revin` is retained as a parameter (default False) for backward
      compatibility, but the RevIN branch has been removed because the prior
      implementation had a fatal inverse-normalization bug (feature-0 only).
    - `recurrent_dropout` defaults to 0.0 because non-zero recurrent dropout
      disables CuDNN fast kernels and can destabilize training on short series.
    - Compiled with composite QLIKE+MSE loss for the first training phase (see
      train_lstm for the two-phase training protocol).
    """
    inp = Input(shape=(seq_len, n_features), name="seq_features")
    x = LSTM(
        hidden_size,
        activation="tanh",
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
        recurrent_regularizer=tf.keras.regularizers.L2(l2_reg),
        name="lstm",
    )(inp)
    x = Dropout(dropout, name="lstm_drop")(x)
    out = Dense(
        1,
        kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
        name="out",
    )(x)

    model = Model(inp, out, name="LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=qlike_loss_logvar,
    )
    return model


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Build sequences: X_seq[j] = X[i-seq_len+1:i+1], y_seq[j] = y[i]."""
    n = len(y)
    if n < seq_len:
        return np.empty((0, seq_len, X.shape[1])), np.empty(0)
    X_seq = np.stack([X[i - seq_len + 1 : i + 1] for i in range(seq_len - 1, n)])
    y_seq = y[seq_len - 1 :].copy()
    return X_seq, y_seq


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seq_len: int,
    n_features: int,
    epochs: int = 500,
    batch_size: int = 32,
    patience: int = 20,
    init_har_preds: np.ndarray | None = None,
    composite_epochs: int = 50,
    **lstm_kw,
) -> tuple[Model, object]:
    """
    Train LSTM with two-phase loss schedule and optional HAR initialization.

    Phase 0 (MSE warm-start, optional):
        If `init_har_preds` is provided, the model is pre-trained for 5 epochs
        with MSE loss to match HAR predictions. This prevents QLIKE divergence
        at epoch 0 when weights are random and predictions are far off scale.

    Phase 1 (composite loss, `composite_epochs` epochs):
        Train with 0.5*QLIKE + 0.5*MSE to stabilize direction of training;
        reduces systematic under-prediction (Mincer-Zarnowitz beta > 1).

    Phase 2 (pure QLIKE, remaining epochs):
        Full proxy-robust QLIKE optimization with early stopping.

    Parameters
    ----------
    init_har_preds : array of shape (n_train,) — HAR log-variance predictions
        on the training set (non-sequenced, the first seq_len-1 are skipped).
    composite_epochs : number of composite-loss epochs before switching to pure QLIKE.
    """
    X_tr_seq, y_tr_seq = build_sequences(X_train, y_train, seq_len)
    X_val_seq, y_val_seq = build_sequences(X_val, y_val, seq_len)
    if len(X_tr_seq) == 0:
        raise ValueError("Not enough samples for LSTM sequence length")

    lr = lstm_kw.get("lr", 0.001)
    model = create_lstm(seq_len=seq_len, n_features=n_features, **lstm_kw)

    # Phase 0: HAR MSE warm-start (align scale before QLIKE)
    if init_har_preds is not None:
        # Align har_preds to the sequenced target: skip first seq_len-1 rows
        har_seq = init_har_preds[seq_len - 1 :]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
        )
        model.fit(X_tr_seq, har_seq, epochs=5, batch_size=batch_size, verbose=0)

    # Phase 1: composite QLIKE + MSE to address systematic under-prediction
    if composite_epochs > 0:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
            loss=composite_qlike_mse_loss,
        )
        model.fit(
            X_tr_seq,
            y_tr_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=composite_epochs,
            batch_size=batch_size,
            verbose=0,
        )

    # Phase 2: pure QLIKE with early stopping
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=qlike_loss_logvar,
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=0,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=5e-5,
        verbose=0,
    )
    hist = model.fit(
        X_tr_seq,
        y_tr_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early, reduce_lr],
        verbose=0,
    )
    return model, hist
