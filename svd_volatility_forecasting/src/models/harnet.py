# -*- coding: utf-8 -*-
"""
M7: HARNet -- Temporal Convolutional Network with HAR-style dilated convolutions.

Architecture replicates the HARNet design where dilated causal Conv1D layers at
lag-scales matching the HAR model (1, 5, 22 days) replace the linear HAR
specification, achieving competitive performance without RNN training instability.

Three parallel dilated branches each see the full 22-step input sequence:
    Branch 1: dilation=1  (captures daily / lag-1 dynamics)
    Branch 2: dilation=5  (captures weekly dynamics)
    Branch 3: dilation=11 (dilation 11 with kernel 3 reaches back 22 steps,
                            matching the HAR monthly component)

Each branch applies:
    Conv1D(filters, kernel_size=3, dilation, padding='causal', ReLU)
    -> Dropout
    -> Conv1D(filters//2, kernel_size=1, ReLU)
    -> take last timestep output

Branches are concatenated -> Dense(16, ReLU) -> Dense(1).

Input: (batch, seq_len=22, input_dim) where input_dim = 3 (HAR) or 8 (HAR+SVD).
Target: log(RV_{t+h}).

Reference style: Temporal Convolutional Networks for HAR-style volatility
(HARNet paradigm; see related work discussion in paper Section 10).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv1D,
    Dropout,
    Dense,
    Concatenate,
    Lambda,
)

from .dnn import qlike_loss_logvar


def create_harnet(
    input_dim: int,
    seq_len: int = 22,
    filters: int = 32,
    dilations: list[int] = None,
    dropout: float = 0.1,
    lr: float = 0.001,
) -> Model:
    """
    Build HARNet / TCN model.

    Parameters
    ----------
    input_dim : int
        Number of input features (3 for HAR-only, 8 for HAR+SVD).
    seq_len : int
        Sequence length fed as input; 22 = one trading month (matches HAR convention).
    filters : int
        Number of Conv1D filters per dilated branch.
    dilations : list[int]
        Dilation rates for the three branches. Defaults to [1, 5, 11].
        With kernel_size=3 and dilation=11, the receptive field covers
        3 + (3-1)*11 = 25 steps, sufficient to capture the 22-day HAR component.
    dropout : float
        Dropout rate applied after first Conv1D in each branch.
    lr : float
        Adam learning rate.

    Returns
    -------
    Compiled Keras Model.
    """
    if dilations is None:
        dilations = [1, 5, 11]

    inp = Input(shape=(seq_len, input_dim), name="tcn_input")
    branches = []
    for i, dilation in enumerate(dilations):
        x = Conv1D(
            filters=filters,
            kernel_size=3,
            dilation_rate=dilation,
            padding="causal",
            activation="relu",
            name=f"conv_d{dilation}_a",
        )(inp)
        x = Dropout(dropout, name=f"drop_d{dilation}")(x)
        x = Conv1D(
            filters=filters // 2,
            kernel_size=1,
            activation="relu",
            name=f"conv_d{dilation}_b",
        )(x)
        # Take only the last timestep from each branch
        last = Lambda(lambda t: t[:, -1, :], name=f"last_step_d{dilation}")(x)
        branches.append(last)

    merged = Concatenate(name="concat_branches")(branches)
    merged = Dense(16, activation="relu", name="dense_merge")(merged)
    out = Dense(1, name="out")(merged)

    model = Model(inp, out, name="HARNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=qlike_loss_logvar,
    )
    return model


def train_harnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seq_len: int,
    n_features: int,
    epochs: int = 500,
    batch_size: int = 32,
    patience: int = 20,
    filters: int = 32,
    dilations: list[int] = None,
    dropout: float = 0.1,
    lr: float = 0.001,
    init_har_preds: np.ndarray | None = None,
) -> tuple[Model, object]:
    """
    Build sequences from 2-D arrays, then train HARNet.

    Reuses `models_lstm.build_sequences` for consistent sequence construction.
    X_train and X_val are 2-D (n_samples, n_features); sequences of length seq_len
    are built internally here before training.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training features (2-D) and targets (1-D).
    X_val, y_val : np.ndarray
        Validation features (2-D) and targets (1-D).
    seq_len : int
        Sequence length (22 for HAR convention).
    n_features : int
        Number of input features.
    init_har_preds : optional 1-D array
        HAR log-variance predictions aligned to X_train timesteps.
        If provided, HARNet is warm-started with MSE (5 epochs) before QLIKE
        training — prevents QLIKE divergence from random initialization.

    Returns
    -------
    (model, history) tuple.
    """
    from .lstm import build_sequences
    from .dnn import initialize_dnn_to_har

    if dilations is None:
        dilations = [1, 5, 11]

    X_tr_seq, y_tr_seq = build_sequences(X_train, y_train, seq_len)
    X_val_seq, y_val_seq = build_sequences(X_val, y_val, seq_len)

    if len(X_tr_seq) == 0:
        raise ValueError(
            f"Not enough training samples to form sequences of length {seq_len}."
        )
    if len(X_val_seq) == 0:
        raise ValueError(
            f"Not enough validation samples to form sequences of length {seq_len}."
        )

    model = create_harnet(
        input_dim=n_features,
        seq_len=seq_len,
        filters=filters,
        dilations=dilations,
        dropout=dropout,
        lr=lr,
    )

    # HAR warm-start: pre-train on HAR predictions with MSE before QLIKE
    if init_har_preds is not None:
        # Build sequences for the init targets aligned to X_tr_seq timestamps
        # The first seq_len-1 timesteps don't have sequences, so align targets
        n_seq = len(y_tr_seq)
        init_targets = init_har_preds[-n_seq:]  # last n_seq predictions align to sequences
        if len(init_targets) == n_seq:
            initialize_dnn_to_har(model, X_tr_seq, init_targets, lr=lr)

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

    history = model.fit(
        X_tr_seq,
        y_tr_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early, reduce_lr],
        verbose=0,
    )
    return model, history
