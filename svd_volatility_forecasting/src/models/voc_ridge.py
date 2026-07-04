"""
Virtue-of-complexity (VoC) ridge models with random Fourier features (RFF).

Kelly–Malamud–Zhou (2024) motivate complexity ``c = P / T`` as the relevant axis
for forecast error when expanding model capacity; here ``G_t`` collects the
pre-registered covariates (HAR + RSV + SVD + CSD + turbulence + lags).  Pure
ridge on RFF maps avoids ElasticNet sparsity-induced discontinuities at ``c = 1``.

Rahimi & Recht (2007): ``φ_{2i-1}(G)=sin(γ ω_i^T G)``, ``φ_{2i}(G)=cos(γ ω_i^T G)``.
"""

from __future__ import annotations

import numpy as np


def median_gamma_heuristic(G_train: np.ndarray) -> float:
    """Median heuristic scale ``γ`` from training rows only (no leakage)."""
    X = np.asarray(G_train, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 5:
        return 1.0
    rng = np.random.default_rng(0)
    n = X.shape[0]
    idx_i = rng.choice(n, size=min(500, n), replace=False)
    idx_j = rng.choice(n, size=min(500, n), replace=False)
    d2 = np.sum((X[idx_i] - X[idx_j]) ** 2, axis=1)
    med = float(np.median(d2[np.isfinite(d2) & (d2 > 1e-18)]))
    return 1.0 / med if med > 0 else 1.0


def draw_rff_weights(d: int, P: int, rng: np.random.Generator) -> np.ndarray:
    """Frequency matrix ``W`` of shape ``(P, d)`` with ``N(0, I/d)`` scaling."""
    P = int(max(P, 1))
    return rng.standard_normal(size=(P, d)) / np.sqrt(float(d))


def random_fourier_features(
    G: np.ndarray,
    W: np.ndarray,
    gamma: float,
    *,
    include_bias: bool = True,
) -> np.ndarray:
    """Map ``G`` ``(T,d)`` with fixed ``W`` ``(P,d)`` to ``(T, 2P)`` or ``+bias``."""
    X = np.asarray(G, dtype=np.float64)
    proj = X @ W.T * float(gamma)
    T, P = proj.shape
    Phi = np.empty((T, 2 * P), dtype=np.float64)
    Phi[:, 0::2] = np.sin(proj)
    Phi[:, 1::2] = np.cos(proj)
    if include_bias:
        Phi = np.column_stack([np.ones(T, dtype=np.float64), Phi])
    return Phi


def ridge_fit_primal_dual(
    y: np.ndarray,
    Phi: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ridge regression: returns primal coefficients ``β`` of shape ``(F,)`` and
    in-sample predictions ``ŷ``. Uses kernel dual when ``F > T``.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    Phi = np.asarray(Phi, dtype=np.float64)
    T, F = Phi.shape
    lam = float(max(lam, 1e-18))
    if F <= T:
        coef = np.linalg.solve(Phi.T @ Phi + lam * np.eye(F), Phi.T @ y)
        return coef, Phi @ coef
    K = Phi @ Phi.T
    dual = np.linalg.solve(K + lam * np.eye(T), y)
    coef = Phi.T @ dual
    return coef, Phi @ coef


def ridge_predict_out_of_sample(
    Phi_train: np.ndarray,
    y_train: np.ndarray,
    Phi_oos: np.ndarray,
    lam: float,
) -> np.ndarray:
    coef, _ = ridge_fit_primal_dual(y_train, Phi_train, lam)
    Phi_os = np.asarray(Phi_oos, dtype=np.float64)
    return Phi_os @ coef


def qlike_loss(y: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Patton QLIKE summands in variance space."""
    Y = np.maximum(np.asarray(y, dtype=np.float64), eps)
    P = np.maximum(np.asarray(pred, dtype=np.float64), eps)
    ratio = Y / P
    return ratio - np.log(ratio) - 1.0


def tune_ridge_lambda_val_qlike(
    y_train: np.ndarray,
    Phi_train: np.ndarray,
    y_val: np.ndarray,
    Phi_val: np.ndarray,
    log_lam_grid: np.ndarray,
    eps: float = 1e-12,
) -> tuple[float, float, np.ndarray]:
    """Pick ``λ = 10**log_lam`` minimizing mean QLIKE on validation."""
    best_ll = float(log_lam_grid[0])
    best_q = float("inf")
    best_pred = np.full(len(y_val), np.nan, dtype=np.float64)
    for ll in log_lam_grid:
        lam = float(10.0 ** float(ll))
        pv = ridge_predict_out_of_sample(Phi_train, y_train, Phi_val, lam)
        L = qlike_loss(y_val, pv, eps=eps)
        mq = float(np.nanmean(L))
        if mq < best_q:
            best_q = mq
            best_ll = float(ll)
            best_pred = pv
    return best_q, best_ll, best_pred


def voc_rff_oos_curve(
    G_train: np.ndarray,
    y_train: np.ndarray,
    G_val: np.ndarray,
    y_val: np.ndarray,
    G_test: np.ndarray,
    y_test: np.ndarray,
    *,
    P: int,
    gamma: float,
    log_lam_grid: np.ndarray,
    rng: np.random.Generator,
    eps: float = 1e-12,
) -> dict:
    """
    One VoC draw: tune ``λ`` on validation, report train / val / test QLIKE.

    Parameters
    ----------
    G_* : design matrices (already aligned; **train only** used for ``γ`` if
          caller passes scaled ``G``).
    """
    W = draw_rff_weights(G_train.shape[1], P, rng)
    Phi_tr = random_fourier_features(G_train, W, gamma)
    Phi_va = random_fourier_features(G_val, W, gamma)
    Phi_te = random_fourier_features(G_test, W, gamma)

    _, best_ll, pred_val = tune_ridge_lambda_val_qlike(
        y_train, Phi_tr, y_val, Phi_va, log_lam_grid, eps=eps,
    )
    lam_star = float(10.0 ** best_ll)
    pred_test = ridge_predict_out_of_sample(Phi_tr, y_train, Phi_te, lam_star)
    pred_train = ridge_predict_out_of_sample(Phi_tr, y_train, Phi_tr, lam_star)
    coef_tr, _ = ridge_fit_primal_dual(y_train, Phi_tr, lam_star)

    def _mq(y, p):
        return float(np.nanmean(qlike_loss(y, p, eps=eps)))

    return {
        "P": int(P),
        "gamma": float(gamma),
        "best_log_lam": best_ll,
        "lambda": lam_star,
        "qlike_train": _mq(y_train, pred_train),
        "qlike_val": _mq(y_val, pred_val),
        "qlike_test": _mq(y_test, pred_test),
        "beta_norm_sq": float(np.sum(coef_tr ** 2)),
    }
