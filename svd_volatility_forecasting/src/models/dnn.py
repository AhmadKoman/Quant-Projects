# -*- coding: utf-8 -*-
"""
Feed-forward DNN: M3 (HAR only), M4 (HAR+SVD). No IV.

Loss function: QLIKE (Patton 2011, Journal of Econometrics).
    QLIKE = mean(exp(u) - u - 1)  where u = y_true_log - y_pred_log
    Gradient: d/d_y_pred = -1 + exp(y_true - y_pred)
    Zero when ŷ = y; negative (nudges up) when ŷ < y (under-prediction).
    This directly corrects Jensen's inequality bias that arises when training
    with MSE/Huber on log-variance targets (which learn the geometric mean,
    not the arithmetic mean).

HAR initialization (Reisenhofer et al. 2022 / HARNet paper):
    DNN weights are pre-initialized to match HAR predictions via 5 MSE epochs
    before switching to QLIKE. This prevents QLIKE from diverging at epoch 0
    when random-initialized predictions are far from realized variance.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Multiply, Softmax


def qlike_loss_logvar(y_true_log: tf.Tensor, y_pred_log: tf.Tensor) -> tf.Tensor:
    """
    QLIKE loss in log-variance parameterization.

    Both y_true_log and y_pred_log are log(realized_variance).
    QLIKE = mean( exp(u) - u - 1 )  where u = y_true_log - y_pred_log
          = mean( h / h_hat - log(h / h_hat) - 1 )

    Properties:
        - Minimum at ŷ = y (unbiased in level space)
        - Asymmetric: penalises under-prediction (h_hat < h) more than over
        - Proxy-robust: forecast rankings unchanged under noisy RV proxy
        - Scale-invariant: works regardless of %-squared vs decimal units

    Numerical stability: u is clipped to [-10, 10] before exp() to prevent
    overflow when predictions are far from targets during early training epochs.
    exp(10) = 22026 is large but finite; prevents NaN/Inf that kill training.
    """
    u = tf.clip_by_value(y_true_log - y_pred_log, -10.0, 10.0)
    return tf.reduce_mean(tf.exp(u) - u - 1.0)


def composite_qlike_mse_loss(y_true_log: tf.Tensor, y_pred_log: tf.Tensor) -> tf.Tensor:
    """
    Composite 0.5*QLIKE + 0.5*MSE loss for early training phases.

    Pure QLIKE's asymmetry (heavier penalty for under-prediction) can cause
    systematic over-bias (Mincer-Zarnowitz beta > 1) during early training
    when the model is learning to calibrate its output scale.  Adding MSE
    symmetrizes the gradient signal for the first N epochs, centering the
    prediction distribution before QLIKE is used alone.  See Mincer & Zarnowitz
    (1969) and the plan Section 4.2 for motivation.
    """
    qlike = qlike_loss_logvar(y_true_log, y_pred_log)
    mse = tf.reduce_mean(tf.square(y_true_log - y_pred_log))
    return 0.5 * qlike + 0.5 * mse


def create_dnn(
    input_dim: int,
    hidden_dims: list[int] = (64, 32, 16),
    dropout: float = 0.2,
    l2_reg: float = 1e-4,
    lr: float = 0.001,
) -> Model:
    """Build DNN: input -> Dense -> BN -> Dropout -> ... -> Dense(1).

    Compiled with QLIKE loss and Adam(clipnorm=1.0) for variance forecasting.
    """
    inp = Input(shape=(input_dim,), name="features")
    x = inp
    for i, h in enumerate(hidden_dims):
        x = Dense(
            h,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
            name=f"dense_{i}",
        )(x)
        x = BatchNormalization(name=f"bn_{i}")(x)
        x = Dropout(dropout, name=f"drop_{i}")(x)
    out = Dense(1, name="out")(x)
    model = Model(inp, out, name="DNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=qlike_loss_logvar,
    )
    return model


def create_dnn_with_gate(
    input_dim: int,
    hidden_dims: list[int] = (64, 32, 16),
    dropout: float = 0.2,
    l2_reg: float = 1e-4,
    lr: float = 0.001,
) -> Model:
    """
    DNN with a learned Feature-Wise Attention Gate.

    The gate produces a softmax-normalized weight vector over the `input_dim`
    features via a two-layer bottleneck network.  The input is element-wise
    multiplied by `n_features * weights` (scale-preserving because the softmax
    weights sum to 1 and are scaled up by n_features to keep unit energy).

    This design allows post-hoc inspection: extracting gate weights on the test
    set and averaging over crisis vs. calm periods shows whether SVD features
    are systematically up-weighted during market stress.

    Architecture
    ------------
    Input -> FeatureGate(Dense(32, relu) -> Dense(n, softmax) -> scale) ->
             Dense -> BN -> Dropout -> ... -> Dense(1)
    """
    inp = Input(shape=(input_dim,), name="features")

    # Gate branch: lightweight 2-layer bottleneck
    gate = Dense(32, activation="relu", use_bias=False, name="gate_hidden")(inp)
    gate = Dense(input_dim, activation="softmax", use_bias=False, name="gate_weights")(gate)
    # Scale-preserving: weights sum to 1, multiply by n to preserve mean input energy
    gate_scaled = tf.keras.layers.Lambda(
        lambda w: w * float(input_dim), name="gate_scale"
    )(gate)
    x = Multiply(name="gate_apply")([inp, gate_scaled])

    for i, h in enumerate(hidden_dims):
        x = Dense(
            h,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
            name=f"dense_{i}",
        )(x)
        x = BatchNormalization(name=f"bn_{i}")(x)
        x = Dropout(dropout, name=f"drop_{i}")(x)
    out = Dense(1, name="out")(x)

    model = Model(inp, out, name="DNN_Gate")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=qlike_loss_logvar,
    )
    return model


def get_gate_weights(model: Model, X: np.ndarray) -> np.ndarray:
    """
    Extract gate attention weights from a DNN_Gate model.

    Returns
    -------
    weights : ndarray of shape (n_samples, n_features)
        Softmax gate weights (pre-scaling), NOT multiplied by n_features.
        Values in [0, 1] summing to 1 per sample.
    """
    gate_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("gate_weights").output,
    )
    return gate_model.predict(X, verbose=0)


def initialize_dnn_to_har(
    model: Model,
    X_train_scaled: np.ndarray,
    har_preds_log: np.ndarray,
    lr: float = 0.001,
    init_epochs: int = 5,
) -> Model:
    """
    Pre-train DNN for `init_epochs` epochs with MSE to match HAR predictions,
    then re-compile with QLIKE for main training.

    Purpose: prevents QLIKE loss from diverging at epoch 0 when random-initialized
    DNN predictions are orders of magnitude away from realized variance. The HAR
    model provides a sensible starting point in the right scale.

    Parameters
    ----------
    model : Keras Model (freshly created, compiled with QLIKE)
    X_train_scaled : scaled input features (no validation split removed)
    har_preds_log : HAR model's log-variance predictions on the same rows
    lr : learning rate (same as main training)
    init_epochs : number of MSE warm-up epochs (5 is sufficient)

    Returns
    -------
    The same model, now initialized and re-compiled with QLIKE.
    """
    # Warm-up with MSE (stable, converges quickly)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
    )
    model.fit(
        X_train_scaled,
        har_preds_log,
        epochs=init_epochs,
        batch_size=32,
        verbose=0,
    )
    # Switch to QLIKE for main training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=qlike_loss_logvar,
    )
    return model


def train_dnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    epochs: int = 500,
    batch_size: int = 32,
    patience: int = 20,
    init_har_preds: np.ndarray | None = None,
    composite_epochs: int = 50,
    use_gate: bool = False,
    **dnn_kw,
) -> tuple[Model, object]:
    """
    Train DNN with two-phase loss schedule and optional HAR initialization.

    Phase 0 (MSE warm-start, optional): pre-train 5 epochs to match HAR predictions.
    Phase 1 (composite loss, `composite_epochs` epochs): 0.5*QLIKE + 0.5*MSE.
    Phase 2 (pure QLIKE, remaining epochs with early stopping).

    Parameters
    ----------
    init_har_preds : optional HAR log-variance predictions on X_train rows.
    composite_epochs : epochs with composite QLIKE+MSE before pure QLIKE.
    use_gate : if True, use the FeatureGate DNN variant (DNN_Gate).
    """
    lr = dnn_kw.get("lr", 0.001)
    create_fn = create_dnn_with_gate if use_gate else create_dnn
    model = create_fn(input_dim=input_dim, **dnn_kw)

    # Phase 0: HAR warm-start
    if init_har_preds is not None:
        model = initialize_dnn_to_har(model, X_train, init_har_preds, lr=lr)

    # Phase 1: composite loss to stabilize scale
    if composite_epochs > 0:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
            loss=composite_qlike_mse_loss,
        )
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
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
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early, reduce_lr],
        verbose=0,
    )
    return model, hist
