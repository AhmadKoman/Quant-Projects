# -*- coding: utf-8 -*-
"""
M8: Graph Neural Network (GCN) baseline for portfolio volatility forecasting.

Constructs an asset-correlation graph from the Ledoit-Wolf covariance C_t at each
time step. Edges exist between assets whose absolute correlation exceeds a threshold
(default 0.3). A 2-layer Graph Convolutional Network (GCN) propagates information
across connected assets, then global mean pooling aggregates node representations
into a single portfolio-level embedding, which a linear head maps to log(RV_{t+h}).

GCN implementation is self-contained in TensorFlow/Keras -- no spektral or PyG
dependencies. The graph convolution follows Kipf & Welling (2017):
    H^{(l+1)} = ReLU( A_hat @ H^{(l)} @ W^{(l)} )
where A_hat is the row-normalised adjacency (with self-loops) and W is a learned
weight matrix implemented as a Dense layer applied after the adjacency multiply.

Node features at time t for asset j:
    [|r_{j,t}|,  RV_{j,5d},  f1_t,  sigma1_t]
where RV_{j,5d} is asset j's individual 5-day RMS volatility, and f1/sigma1 are
SVD portfolio-level features broadcast to all nodes. This gives node_feat_dim = 4.

Loss: QLIKE on log-variance outputs, with output bias initialized to the training-set
mean of log-variance so that early QLIKE gradients are well-scaled. All other models
(DNN, LSTM, HARNet) use QLIKE — using Huber for GNN was a training-objective confound
that could bias DM test results independently of predictive content.

Reference: Jiang et al. (2022) "Graph-based Methods for Forecasting Realized
Covariance"; Kritzman et al. (2011) absorption ratio and systemic risk.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Lambda


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def build_adjacency(cov_matrix: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Convert covariance matrix C_t to a row-normalised adjacency matrix.

    1. Compute correlation matrix from C_t.
    2. Zero out the diagonal (no self-loop in the thresholding step).
    3. Keep only |corr| > threshold (sparse graph).
    4. Add self-loops: A_hat = A + I.
    5. Row-normalise: A_norm = D^{-1} A_hat (so each row sums to 1 in the
       best-case symmetric graph; simple normalisation, not symmetric).

    Returns
    -------
    A_norm : ndarray of shape (N, N), float32.
    """
    cov = np.asarray(cov_matrix, dtype=np.float64)
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 0.0)
    A = (np.abs(corr) > threshold).astype(np.float32)
    # Add self-loops
    A += np.eye(A.shape[0], dtype=np.float32)
    # Row-normalise
    row_sums = A.sum(axis=1, keepdims=True).clip(min=1.0)
    return (A / row_sums).astype(np.float32)


def build_node_features(
    returns: pd.DataFrame,
    asset_columns: list[str],
    dates: pd.Index,
    svd_features: pd.DataFrame,
    rv_window_short: int = 5,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Build per-timestep node feature matrices for GNN input.

    For each date t in `dates`, constructs a node feature matrix of shape
    (N, node_feat_dim) where:
        col 0: |r_{j,t}|                   individual asset absolute return at t
        col 1: RV_{j,t,5d}                 individual asset 5-day RMS vol (lagged)
        col 2: f1_t (broadcast)            portfolio SVD feature, same for all nodes
        col 3: sigma1_t (broadcast)        portfolio SVD feature, same for all nodes

    All features are already available from the aligned returns and SVD panel.

    Returns
    -------
    node_feat_array : ndarray of shape (T_dates, N, node_feat_dim)
    """
    N = len(asset_columns)
    T = len(dates)
    node_feat_dim = 4
    out = np.zeros((T, N, node_feat_dim), dtype=np.float32)

    # Pre-compute 5-day rolling RMS per asset; shift(1) so no look-ahead at t
    rv5_per_asset = np.sqrt(
        (returns[asset_columns] ** 2).rolling(rv_window_short).mean().shift(1)
    )

    for i, dt in enumerate(dates):
        # Absolute return at t
        if dt in returns.index:
            abs_ret = returns.loc[dt, asset_columns].values.astype(np.float32)
        else:
            abs_ret = np.zeros(N, dtype=np.float32)
        out[i, :, 0] = np.abs(abs_ret)

        # 5-day per-asset RV
        if dt in rv5_per_asset.index:
            rv5 = rv5_per_asset.loc[dt, asset_columns].values.astype(np.float32)
            rv5 = np.where(np.isfinite(rv5), rv5, 0.0)
        else:
            rv5 = np.zeros(N, dtype=np.float32)
        out[i, :, 1] = rv5

        # SVD portfolio features broadcast to all nodes
        if dt in svd_features.index:
            f1 = float(svd_features.loc[dt, "f1"]) if np.isfinite(svd_features.loc[dt, "f1"]) else 0.0
            s1 = float(svd_features.loc[dt, "sigma1"]) if np.isfinite(svd_features.loc[dt, "sigma1"]) else 0.0
        else:
            f1, s1 = 0.0, 0.0
        out[i, :, 2] = f1
        out[i, :, 3] = s1

    return out


# ---------------------------------------------------------------------------
# GCN model
# ---------------------------------------------------------------------------

class GCNLayer(tf.keras.layers.Layer):
    """
    Single Graph Convolutional Network layer (Kipf & Welling 2017).

    Computes: H_out = ReLU( A_norm @ H_in @ W )
    where A_norm is passed at call time (per-sample varying adjacency).

    Because the adjacency is per-sample (time-varying graphs), we implement
    this as a custom layer that accepts (H, A) inputs, rather than folding A
    into the weights.
    """

    def __init__(self, out_dim: int, activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.activation_name = activation
        self.dense = Dense(out_dim, use_bias=True)
        self.act = tf.keras.activations.get(activation)

    def call(self, inputs, training=False):
        """
        inputs: (H, A)
            H: (batch, N, node_feat_dim)
            A: (batch, N, N)
        Returns: (batch, N, out_dim)
        """
        H, A = inputs
        # Linear transform: (batch, N, out_dim)
        H_transformed = self.dense(H)
        # Graph aggregation: (batch, N, out_dim) = (batch, N, N) @ (batch, N, out_dim)
        aggregated = tf.matmul(A, H_transformed)
        return self.act(aggregated)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"out_dim": self.out_dim, "activation": self.activation_name})
        return cfg


def _qlike_loss_logvar(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    QLIKE loss for log-variance predictions.

    Both y_true and y_pred are in log-variance space (i.e. log(h)).
    QLIKE(h, h_hat) = h/h_hat - log(h/h_hat) - 1
                    = exp(log_h - log_h_hat) - (log_h - log_h_hat) - 1

    Setting delta = log_h - log_h_hat (ratio in log-space):
        QLIKE = exp(delta) - delta - 1  >= 0, with equality iff delta = 0.

    This is numerically stable because we never divide by h_hat directly.
    Clipping delta to [-10, 10] prevents overflow during the first epochs
    before output bias initialization has scaled the predictions.
    """
    delta = tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)
    delta = tf.clip_by_value(delta, -10.0, 10.0)
    return tf.reduce_mean(tf.exp(delta) - delta - 1.0)


def create_gnn(
    n_nodes: int,
    node_feat_dim: int,
    hidden_dim: int = 32,
    dropout: float = 0.1,
    lr: float = 0.001,
    output_bias_init: float = 0.0,
) -> Model:
    """
    Build a 2-layer GCN for portfolio volatility forecasting.

    Architecture:
        Node features (N, node_feat_dim) + Adjacency (N, N)
        -> GCNLayer(hidden_dim, relu)
        -> Dropout
        -> GCNLayer(hidden_dim // 2, relu)
        -> Global mean pool over N nodes  -> (hidden_dim // 2,)
        -> Dense(16, relu)
        -> Dense(1, bias_init=output_bias_init)  [predicts log(RV_{t+h})]

    Training objective: QLIKE loss on log-variance (proxy-robust; Patton 2011).
    The output bias is initialized to the training-mean of log-variance so that
    QLIKE gradients are well-scaled from the first epoch, avoiding the divergence
    that occurs when a randomly-initialized network predicts near-zero log-variance
    while the target is order 1.

    Inputs to the model are two tensors:
        node_input:  (batch, N, node_feat_dim)
        adj_input:   (batch, N, N)

    Parameters
    ----------
    n_nodes : int
        Number of assets N.
    node_feat_dim : int
        Number of node features per asset (4 by default).
    hidden_dim : int
        Hidden dimension of first GCN layer.
    dropout : float
        Dropout rate between GCN layers.
    output_bias_init : float
        Initial value for the output Dense(1) bias. Set to mean(log(y_train))
        so QLIKE loss starts from a reasonable point.

    Returns
    -------
    Compiled Keras Model.
    """
    node_input = tf.keras.Input(shape=(n_nodes, node_feat_dim), name="node_features")
    adj_input = tf.keras.Input(shape=(n_nodes, n_nodes), name="adjacency")

    h = GCNLayer(hidden_dim, activation="relu", name="gcn_1")([node_input, adj_input])
    h = Dropout(dropout, name="gcn_drop")(h)
    h = GCNLayer(hidden_dim // 2, activation="relu", name="gcn_2")([h, adj_input])

    # Global mean pooling over nodes: (batch, N, hidden_dim//2) -> (batch, hidden_dim//2)
    # Must use a Keras Lambda layer; tf.reduce_mean cannot be called directly on symbolic tensors
    pooled = Lambda(lambda t: tf.reduce_mean(t, axis=1), name="global_mean_pool")(h)
    pooled = Dense(16, activation="relu", name="dense_pool")(pooled)
    out = Dense(
        1,
        name="out",
        bias_initializer=tf.keras.initializers.Constant(output_bias_init),
    )(pooled)

    model = Model(inputs=[node_input, adj_input], outputs=out, name="GNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=_qlike_loss_logvar,
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_gnn(
    node_features_train: np.ndarray,
    adj_train: np.ndarray,
    y_train: np.ndarray,
    node_features_val: np.ndarray,
    adj_val: np.ndarray,
    y_val: np.ndarray,
    n_nodes: int,
    node_feat_dim: int,
    hidden_dim: int = 32,
    dropout: float = 0.1,
    lr: float = 0.001,
    epochs: int = 300,
    batch_size: int = 32,
    patience: int = 20,
) -> tuple[Model, object]:
    """
    Train the GCN model with QLIKE loss and output bias initialization.

    The output Dense(1) bias is initialized to mean(y_train) — the training-set
    mean of log-variance — so that QLIKE gradients are well-scaled from the first
    epoch. Without this, a randomly-initialized GNN predicts near-zero log-variance
    while QLIKE is highly sensitive to underprediction (h/h_hat >> 1 -> large loss).

    Parameters
    ----------
    node_features_train : ndarray (T_train, N, node_feat_dim)
    adj_train : ndarray (T_train, N, N)  -- per-timestep adjacency matrices
    y_train : ndarray (T_train,)  -- log-variance targets
    node_features_val, adj_val, y_val : validation counterparts
    n_nodes, node_feat_dim : graph dimensions

    Returns
    -------
    (model, history) tuple.
    """
    # Initialize output bias to mean(log-variance) of training targets.
    # This grounds the QLIKE loss near its minimum at t=0 before any gradient steps.
    output_bias_init = float(np.nanmean(y_train))

    model = create_gnn(
        n_nodes=n_nodes,
        node_feat_dim=node_feat_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        output_bias_init=output_bias_init,
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

    history = model.fit(
        x=[node_features_train, adj_train],
        y=y_train,
        validation_data=([node_features_val, adj_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early, reduce_lr],
        verbose=0,
    )
    return model, history


def gnn_predict(
    model: Model,
    node_features: np.ndarray,
    adj: np.ndarray,
) -> np.ndarray:
    """Run GNN inference. Returns 1-D array of predictions."""
    return model.predict([node_features, adj], verbose=0).ravel()
