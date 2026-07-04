"""
Leakage-safe explainability utilities (XAI) for time-series forecasting.

Primary method: blocked permutation importance on a contiguous evaluation block
(typically the validation block inside each walk-forward refit).

Why blocked permutation?
  - i.i.d. shuffles break time-series dependence and can create implausible
    feature paths.
  - block permutations preserve local temporal structure while destroying the
    feature-target association at the block level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

LossName = Literal["qlike", "mse"]


def qlike_loss(true_var: np.ndarray, pred_var: np.ndarray, eps: float = 1e-8) -> float:
    y = np.maximum(np.asarray(true_var, dtype=np.float64).ravel(), eps)
    p = np.maximum(np.asarray(pred_var, dtype=np.float64).ravel(), eps)
    ratio = y / p
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mse_loss(true_var: np.ndarray, pred_var: np.ndarray) -> float:
    y = np.asarray(true_var, dtype=np.float64).ravel()
    p = np.asarray(pred_var, dtype=np.float64).ravel()
    return float(np.mean((y - p) ** 2))


def compute_loss(true_var: np.ndarray, pred_var: np.ndarray, *, loss: LossName, eps: float = 1e-8) -> float:
    if loss == "qlike":
        return qlike_loss(true_var, pred_var, eps=eps)
    if loss == "mse":
        return mse_loss(true_var, pred_var)
    raise ValueError(f"Unknown loss: {loss}")


def block_permute_1d(x: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Permute a 1-D array by shuffling contiguous blocks (non-overlapping).

    This preserves within-block structure exactly and only breaks cross-block ordering.
    """
    x = np.asarray(x)
    n = x.shape[0]
    b = int(block_len)
    if b < 1:
        raise ValueError("block_len must be >= 1")
    if n < b:
        raise ValueError("block_len larger than array length")
    n_blocks = n // b
    remainder = n - n_blocks * b
    blocks = [x[i * b : (i + 1) * b].copy() for i in range(n_blocks)]
    rng.shuffle(blocks)
    out = np.concatenate(blocks, axis=0)
    if remainder > 0:
        # Keep the remainder tail in place (strict, deterministic rule).
        out = np.concatenate([out, x[-remainder:].copy()], axis=0)
    if out.shape[0] != n:
        raise ValueError("block_permute_1d produced wrong length")
    return out


@dataclass
class PermutationImportanceResult:
    feature_names: list[str]
    base_loss: float
    importances_mean: np.ndarray  # shape (d,)
    importances_std: np.ndarray   # shape (d,)
    importances_all: np.ndarray   # shape (reps, d)


def blocked_permutation_importance_2d(
    *,
    X: np.ndarray,
    y_true_var: np.ndarray,
    feature_names: list[str],
    predict_var: Callable[[np.ndarray], np.ndarray],
    loss: LossName = "qlike",
    eps: float = 1e-8,
    reps: int = 20,
    block_len: int = 5,
    seed: int = 42,
) -> PermutationImportanceResult:
    """
    Blocked permutation importance on 2-D tabular inputs (T, d).

    Procedure:
      - Compute baseline loss on (X, y)
      - For each feature j:
          repeat reps times:
            permute X[:, j] by block shuffling -> X_perm
            compute loss on X_perm
      - Importance = loss_perm - loss_base (higher = more important)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y_true_var, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2-D (T, d)")
    T, d = X.shape
    if d != len(feature_names):
        raise ValueError("feature_names length must match X.shape[1]")
    if y.shape[0] != T:
        raise ValueError("y_true_var length must match X.shape[0]")
    if reps < 5:
        raise ValueError("reps must be >= 5 for a stable estimate")
    if T < 20:
        raise ValueError("Need T >= 20 for permutation importance")

    rng = np.random.default_rng(int(seed))
    base_pred = np.asarray(predict_var(X), dtype=np.float64).ravel()
    if base_pred.shape[0] != T:
        raise ValueError("predict_var(X) must return length T")
    base_loss = compute_loss(y, base_pred, loss=loss, eps=eps)

    imps = np.zeros((reps, d), dtype=np.float64)
    for r in range(reps):
        # Use an independent RNG stream per repetition for determinism.
        rng_r = np.random.default_rng(int(seed) + 10_000 + r)
        for j in range(d):
            Xp = X.copy()
            Xp[:, j] = block_permute_1d(Xp[:, j], block_len=block_len, rng=rng_r)
            pred_p = np.asarray(predict_var(Xp), dtype=np.float64).ravel()
            l_p = compute_loss(y, pred_p, loss=loss, eps=eps)
            imps[r, j] = l_p - base_loss

    return PermutationImportanceResult(
        feature_names=list(feature_names),
        base_loss=float(base_loss),
        importances_mean=imps.mean(axis=0),
        importances_std=imps.std(axis=0, ddof=1),
        importances_all=imps,
    )


def blocked_permutation_importance_sequence_from_2d(
    *,
    X_2d: np.ndarray,
    y_true_var_2d: np.ndarray,
    feature_names: list[str],
    seq_len: int,
    predict_var_from_seq: Callable[[np.ndarray], np.ndarray],
    loss: LossName = "qlike",
    eps: float = 1e-8,
    reps: int = 20,
    block_len: int = 5,
    seed: int = 42,
) -> PermutationImportanceResult:
    """
    Blocked permutation importance for sequence models where the raw input is a
    2-D time series X (T, d) but the model consumes sequences (T-seq_len+1, seq_len, d).

    The label for sequence ending at t is y_true_var_2d[t], so the sequence labels
    are y_true_var_2d[seq_len-1:].
    """
    X = np.asarray(X_2d, dtype=np.float64)
    y = np.asarray(y_true_var_2d, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError("X_2d must be 2-D")
    T, d = X.shape
    if len(feature_names) != d:
        raise ValueError("feature_names length must match X_2d.shape[1]")
    if y.shape[0] != T:
        raise ValueError("y_true_var_2d length must match X_2d rows")
    L = int(seq_len)
    if L < 2 or T < L + 10:
        raise ValueError("Insufficient length for sequence permutation importance.")

    def _build_sequences(Xin: np.ndarray) -> np.ndarray:
        return np.stack([Xin[i - L + 1 : i + 1] for i in range(L - 1, T)], axis=0)

    y_seq = y[L - 1 :]
    X_seq = _build_sequences(X)
    base_pred = np.asarray(predict_var_from_seq(X_seq), dtype=np.float64).ravel()
    if base_pred.shape[0] != y_seq.shape[0]:
        raise ValueError("predict_var_from_seq output length mismatch.")
    base_loss = compute_loss(y_seq, base_pred, loss=loss, eps=eps)

    rng = np.random.default_rng(int(seed))
    imps = np.zeros((reps, d), dtype=np.float64)
    for r_i in range(reps):
        rng_r = np.random.default_rng(int(seed) + 10_000 + r_i)
        for j in range(d):
            Xp = X.copy()
            Xp[:, j] = block_permute_1d(Xp[:, j], block_len=block_len, rng=rng_r)
            Xp_seq = _build_sequences(Xp)
            pred_p = np.asarray(predict_var_from_seq(Xp_seq), dtype=np.float64).ravel()
            l_p = compute_loss(y_seq, pred_p, loss=loss, eps=eps)
            imps[r_i, j] = l_p - base_loss

    return PermutationImportanceResult(
        feature_names=list(feature_names),
        base_loss=float(base_loss),
        importances_mean=imps.mean(axis=0),
        importances_std=imps.std(axis=0, ddof=1),
        importances_all=imps,
    )


def blocked_permutation_importance_gnn_time(
    *,
    node_features: np.ndarray,
    y_true_var: np.ndarray,
    node_feat_names: list[str],
    predict_var: Callable[[np.ndarray], np.ndarray],
    loss: LossName = "qlike",
    eps: float = 1e-8,
    reps: int = 20,
    block_len: int = 5,
    seed: int = 42,
) -> PermutationImportanceResult:
    """
    Blocked permutation importance for GNN node features varying over time.

    We permute each node feature *jointly across all nodes* along the time axis
    using block shuffling. This preserves cross-sectional structure per day while
    destroying the temporal alignment between that feature and the target.

    node_features: (T, N, F)
    """
    nf = np.asarray(node_features, dtype=np.float64)
    if nf.ndim != 3:
        raise ValueError("node_features must be (T, N, F)")
    T, N, F = nf.shape
    if len(node_feat_names) != F:
        raise ValueError("node_feat_names must match node_features.shape[2]")
    y = np.asarray(y_true_var, dtype=np.float64).ravel()
    if y.shape[0] != T:
        raise ValueError("y_true_var length must match node_features time dimension")

    base_pred = np.asarray(predict_var(nf), dtype=np.float64).ravel()
    if base_pred.shape[0] != T:
        raise ValueError("predict_var(node_features) must return length T")
    base_loss = compute_loss(y, base_pred, loss=loss, eps=eps)

    imps = np.zeros((reps, F), dtype=np.float64)
    for r_i in range(reps):
        rng_r = np.random.default_rng(int(seed) + 10_000 + r_i)
        for j in range(F):
            nfp = nf.copy()
            # Permute time blocks for feature j, same permutation for all nodes
            xj = nfp[:, :, j]
            # Build a representative 1D index permutation by permuting 0..T-1 blocks
            perm_idx = block_permute_1d(np.arange(T), block_len=block_len, rng=rng_r).astype(int)
            nfp[:, :, j] = xj[perm_idx, :]
            pred_p = np.asarray(predict_var(nfp), dtype=np.float64).ravel()
            l_p = compute_loss(y, pred_p, loss=loss, eps=eps)
            imps[r_i, j] = l_p - base_loss

    return PermutationImportanceResult(
        feature_names=list(node_feat_names),
        base_loss=float(base_loss),
        importances_mean=imps.mean(axis=0),
        importances_std=imps.std(axis=0, ddof=1),
        importances_all=imps,
    )

