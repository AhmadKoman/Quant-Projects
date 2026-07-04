# -*- coding: utf-8 -*-
"""
Feature extraction for SVD volatility forecasting.

Notation:
    S_t  : sample covariance matrix (N x N); can be rank-deficient when N >= M.
    C_t  : Ledoit-Wolf shrinkage estimator (N x N); always symmetric positive-definite.
           C_t = (1 - delta_t) S_t + delta_t * diag(S_t)
    All SVD / eigendecompositions are performed on C_t only -- never on S_t.

Target and feature convention (BPQ 2016 / ABDL 2003 standard):
    Returns are assumed to be in PERCENTAGE units (100 * log-return).
    All RV measures are VARIANCE (squared returns), NOT volatility (abs returns).
    This eliminates Jensen's inequality bias when exp(E[log_RV]) is used and
    allows direct comparison with published HAR R² values (~0.52 in-sample).

    RV_d = r_t^2 (daily variance proxy, %-squared)
    RV_w = mean(r^2, 5 days) (weekly variance, %-squared)
    etc.

HAR feature construction uses SQUARED RETURNS (variance) for ALL windows.
The log-HAR regression (Corsi 2009) is: log(RV_{t+h}) ~ log(RV_d) + log(RV_w) + ...
This is consistent with the variance-based target.

SVD feature improvements over the original implementation:
    - log_sigma1: log of volatility of leading PC (log scale consistent with target)
    - delta_f1: one-day change in variance concentration (captures dynamic trend)
    - angle: 1 - cos_theta (rotation angle; 0=stable, 2=full reversal; more
             linearly related to structural instability than raw cos_theta)
    - Eigenvector orientation fix: sign-flip of u1 enforced so that angle measures
      true directional change, not noise from eigh's arbitrary sign convention.
    - All 5 crisis flags retained for threshold sensitivity analysis

New cross-sectional features:
    - Realized semivariance (RSV+, RSV-): neg-return variance predicts vol 2x
      as much as pos-return variance (Patton & Sheppard 2015)
    - Cross-sectional dispersion (CSD): std of cross-section of stock returns (N assets)
    - Turbulence index: Mahalanobis distance + correlation surprise component

The module exposes two variants of the SVD panel builder:
    build_svd_features_panel        -- standard; returns only the feature DataFrame.
    build_svd_features_panel_with_cov -- extended; also returns the covariance series
                                         [(date, C_t), ...] needed by the GNN.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS


# ===========================================================================
# Population-covariance estimators for the SVD pipeline
# ===========================================================================
#
# Three estimators are implemented and selectable via
# ``compute_shrunk_cov(W, estimator=...)``:
#
# 1. ``"linear_shrinkage"`` — Ledoit-Wolf (2003, JEF) constant-correlation
#    linear shrinkage of the sample covariance toward a scaled identity.
#    Legacy default; always positive definite but has a *fixed* shrinkage
#    intensity that is sub-optimal for the leading eigenvalues.
#
# 2. ``"qis"`` (recommended default) — Quadratic-Inverse Shrinkage of
#    Ledoit & Wolf (2020, Annals of Statistics 48(5), 3043-3065). A
#    non-linear shrinkage estimator that replaces sample eigenvalues by
#    population estimates derived from a kernel-smoothed Stieltjes
#    transform.  Achieves the asymptotically optimal bias-variance
#    trade-off in mean-square loss for large ``N/T`` (Theorem 3.1, LW
#    2020) and *strictly dominates* linear shrinkage for the leading
#    eigenstructure.  See also Kuchibhotla, Mukherjee & Zou (2024,
#    "KMZ"), Proposition 2, for the Davis-Kahan rate ``O_p(M^{-1/2}+δ_t)``
#    on the leading eigenvector under sub-Gaussian returns.
#
# 3. ``"bbp_rie"`` — Bouchaud-Bun-Potters Rotationally-Invariant Estimator
#    eigenvalue cleaning (Bun, Bouchaud & Potters 2017, *Physics
#    Reports* 666, 1-109).  Replaces bulk sample eigenvalues by their
#    asymptotic Marchenko-Pastur (1967) population mean while keeping
#    outliers intact; the *observable shrinkage* refinement of BBP-2017
#    Eq. 8.5 is then applied to all eigenvalues that survive the
#    bulk-clip.  Cheap and explicit; useful as a second non-linear
#    estimator for robustness.
#
# All three return a symmetric positive-definite ``C_t``.  The sample
# covariance ``S_t`` is also returned for diagnostics and for callers
# that need it explicitly.


def normalized_gap_K(lambdas: np.ndarray, K: int, eps: float = 1e-12) -> float:
    """
    Normalized eigengap at rank K: (lambda_K - lambda_{K+1}) / lambda_1.

    Used for gap-screened subspace monitoring (Davis--Kahan scale).
    """
    lam = np.asarray(lambdas, dtype=np.float64).ravel()
    lam = np.maximum(lam, 0.0)
    K = int(K)
    if lam.size < 2 or K < 1 or K >= lam.size or lam[0] <= eps:
        return float("nan")
    return float((lam[K - 1] - lam[K]) / lam[0])


def select_gap_screened_K(
    lambdas: np.ndarray,
    *,
    K_max: int = 10,
    rho_ar: float = 0.70,
    eps: float = 1e-12,
) -> dict:
    """
    Gap-screened rule: K_hat = argmax_{1<=K<=K_max} gap_norm_K subject to AR_K >= rho_ar.

    Returns K_hat, gap at K_hat, AR_{K_hat}, and whether the AR constraint was binding.
    Does not claim population rank recovery without a separation assumption.
    """
    lam = np.asarray(lambdas, dtype=np.float64).ravel()
    lam = np.maximum(lam, 0.0)
    total = float(np.sum(lam))
    N = lam.size
    if total <= eps or N < 2:
        return {
            "K_hat": 1,
            "gap_norm_K": float("nan"),
            "AR_K": float("nan"),
            "ar_constraint_met": False,
        }
    K_max = int(min(max(K_max, 1), N - 1))
    best_K, best_gap = 1, -np.inf
    for K in range(1, K_max + 1):
        ar_k = float(np.sum(lam[:K]) / total)
        if ar_k < rho_ar - 1e-12:
            continue
        g = normalized_gap_K(lam, K, eps=eps)
        if np.isfinite(g) and g > best_gap:
            best_gap = g
            best_K = K
    if not np.isfinite(best_gap):
        # Fall back: smallest K with AR_K >= rho
        for K in range(1, K_max + 1):
            if float(np.sum(lam[:K]) / total) >= rho_ar - 1e-12:
                best_K = K
                best_gap = normalized_gap_K(lam, K, eps=eps)
                break
    return {
        "K_hat": int(best_K),
        "gap_norm_K": float(best_gap) if np.isfinite(best_gap) else float("nan"),
        "AR_K": float(np.sum(lam[: best_K]) / total),
        "ar_constraint_met": bool(float(np.sum(lam[: best_K]) / total) >= rho_ar - 1e-12),
    }


def empirical_gap_identifiable(
    lambdas: np.ndarray,
    *,
    epsilon: float,
    K: int,
    factor: float = 2.0,
    eps: float = 1e-12,
) -> bool:
    """
    Corollary check: empirical gap at K exceeds ``factor * epsilon / lambda_1``.

    ``epsilon`` is a perturbation proxy (e.g. ||C - S||_op on the window).
    """
    lam = np.asarray(lambdas, dtype=np.float64).ravel()
    lam = np.maximum(lam, 0.0)
    g = normalized_gap_K(lam, K, eps=eps)
    if not np.isfinite(g) or lam[0] <= eps:
        return False
    thresh = float(factor * epsilon / lam[0])
    return bool(g > thresh)


# Davis--Kahan constant in manuscript: ||P - P_hat||_F <= 2*sqrt(2K) * eta / Delta_K.
# CDK_DEFAULT is retained for backward compatibility; it equals c_K only at K=2.
# New code should use ``cdk_for_K``.
CDK_DEFAULT = 4.0


def cdk_for_K(K: int) -> float:
    """Theorem constant c_K = 2*sqrt(2K) (projector identity x Yu-Wang-Samworth)."""
    return float(2.0 * np.sqrt(2.0 * max(int(K), 1)))


def dk_projector_term(
    eta: float,
    delta_k: float,
    K: int,
    *,
    e_fro: float | None = None,
    empirical_gap: bool = False,
) -> float:
    """
    One-date projector perturbation bound, capped at the trivial bound:

        ||P_hat - P||_F <= min( sqrt(2)*2*min(sqrt(K)*eta, ||E||_F)/Delta_K, sqrt(2K) ).

    Parameters
    ----------
    eta      : operator-norm perturbation ||E||_op (population or sample-relative).
    delta_k  : eigengap at the cut K. Population gap if available; if
               ``empirical_gap=True``, ``delta_k`` is the *empirical* gap from C and the
               valid observable surrogate (Delta_hat - 2*eta)_+ is used (Weyl on gaps).
    K        : subspace dimension (rank of the projector).
    e_fro    : optional Frobenius norm ||E||_F; sharpens the numerator via
               min(sqrt(K)*eta, ||E||_F) (Yu-Wang-Samworth Thm).

    NaN inputs propagate (the band is undefined, not zero).
    """
    K = max(int(K), 1)
    cap = float(np.sqrt(2.0 * K))
    if not np.isfinite(eta) or not np.isfinite(delta_k):
        return float("nan")
    gap = float(delta_k)
    if empirical_gap:
        gap = gap - 2.0 * float(eta)
    if gap <= 0.0:
        return cap  # weak identification: only the trivial bound is available
    num = float(np.sqrt(K) * eta)
    if e_fro is not None and np.isfinite(e_fro):
        num = min(num, float(e_fro))
    dk = float(2.0 * np.sqrt(2.0) * num / gap)
    return float(min(dk, cap))


def dk_monitoring_band(
    eta_t: float,
    eta_tm1: float,
    delta_k_t: float,
    delta_k_tm1: float,
    K_t: int,
    K_tm1: int | None = None,
    *,
    e_fro_t: float | None = None,
    e_fro_tm1: float | None = None,
    empirical_gap: bool = False,
) -> float:
    """
    Capped two-date Davis--Kahan monitoring band (Theorem, capped version):

        tau*_{K,t} = min( sqrt(K_t + K_{t-1}),
                          term(t; K_t) + term(t-1; K_{t-1}) ),

    where each term is ``dk_projector_term`` (itself capped at sqrt(2 K_s)).
    Handles K_t != K_{t-1} (rank change): the reverse-triangle argument bounds
    |D_hat - D| date-by-date at each date's own cut.

    With ``empirical_gap=True`` the observable surrogate gap (Delta_hat - 2 eta)_+ is
    used at both dates, so the band is computable from data alone and remains a valid
    upper bound whenever eta dominates the true perturbation.
    """
    K_t = max(int(K_t), 1)
    K_m = K_t if K_tm1 is None else max(int(K_tm1), 1)
    term_t = dk_projector_term(
        eta_t, delta_k_t, K_t, e_fro=e_fro_t, empirical_gap=empirical_gap,
    )
    term_m = dk_projector_term(
        eta_tm1, delta_k_tm1, K_m, e_fro=e_fro_tm1, empirical_gap=empirical_gap,
    )
    if not np.isfinite(term_t) or not np.isfinite(term_m):
        return float("nan")
    cap = float(np.sqrt(K_t + K_m))
    return float(min(term_t + term_m, cap))


def top_k_projector(U_K: np.ndarray) -> np.ndarray:
    """Rank-K orthogonal projector P = U_K U_K^T."""
    U = np.asarray(U_K, dtype=np.float64)
    return U @ U.T


def projector_frobenius_movement(U_t: np.ndarray, U_tm1: np.ndarray) -> float:
    """Observed top-K subspace movement ||P_t - P_{t-1}||_F."""
    return float(
        np.linalg.norm(top_k_projector(U_t) - top_k_projector(U_tm1), ord="fro")
    )


def gap_adjusted_alarm_threshold(
    eta_t: float,
    eta_tm1: float,
    delta_k_t: float,
    delta_k_tm1: float,
    *,
    c_dk: float = CDK_DEFAULT,
    eps: float = 1e-12,
) -> float:
    """
    Davis--Kahan perturbation upper bound (Corollary alarm / tau_{K,t}).

    tau_{K,t} = c_K (eta_t/Delta_{K,t} + eta_{t-1}/Delta_{K,t-1}) with c_K = 2*sqrt(2K).
    It is NOT a projector norm; it may exceed the maximum Frobenius projector distance
    when Delta_{K,t} is tiny (weak identification).

    Inferential role (see manuscript sec:monitor-inference):
    - Simulation: eta_t = ||C_t - Sigma_t||_op  => bound on |D_hat - D| (population).
    - Panel: eta_t = ||C_t - S_t||_op  => sample-relative proxy; tau is an uncalibrated
      descriptive weak-ID gauge, not a calibrated hypothesis test.

    delta_k_t and delta_k_tm1 must be raw eigengaps (same units as eta_t).

    .. deprecated:: use ``dk_monitoring_band`` (K-dependent constant, capped at the
       trivial projector bound, optional empirical-gap correction). Retained for
       backward compatibility with legacy scripts.
    """
    if not (
        np.isfinite(eta_t)
        and np.isfinite(eta_tm1)
        and np.isfinite(delta_k_t)
        and np.isfinite(delta_k_tm1)
    ):
        return float("nan")
    term_t = float(eta_t / max(delta_k_t, eps))
    term_m = float(eta_tm1 / max(delta_k_tm1, eps))
    return float(c_dk * (term_t + term_m))


def monitoring_signal_to_band_ratio(
    d_hat: float,
    tau: float,
    *,
    eps: float = 1e-12,
) -> float:
    """Observed projector movement relative to DK band (descriptive; not a p-value)."""
    if not np.isfinite(d_hat) or not np.isfinite(tau):
        return float("nan")
    return float(d_hat / max(tau, eps))


def select_dominant_gap_K(
    lambdas: np.ndarray,
    *,
    K_max: int = 10,
    rho_ar: float | None = 0.70,
    eps: float = 1e-12,
) -> int:
    """Argmax_k (lambda_k - lambda_{k+1}); optional AR_k >= rho filter."""
    lam = np.asarray(lambdas, dtype=np.float64).ravel()
    lam = np.maximum(lam, 0.0)
    N = lam.size
    if N < 2:
        return 1
    K_max = int(min(max(K_max, 1), N - 1))
    best_k, best_gap = 1, -np.inf
    total = float(np.sum(lam))
    for k in range(1, K_max + 1):
        if rho_ar is not None and total > eps:
            if float(np.sum(lam[:k]) / total) < float(rho_ar) - 1e-12:
                continue
        g = float(lam[k - 1] - lam[k])
        if g > best_gap:
            best_gap = g
            best_k = k
    return int(best_k)


def ar_k_decomposition(
    lam_c: np.ndarray,
    lam_sigma: np.ndarray,
    K: int,
) -> dict:
    """
    Absorption-ratio first-order decomposition AR_K(C) - AR_K(Sigma).

    Returns linear part u_K/T - S_K v/T^2 and remainder R_K (exact identity).
    """
    lc = np.maximum(np.asarray(lam_c, dtype=np.float64).ravel(), 0.0)
    ls = np.maximum(np.asarray(lam_sigma, dtype=np.float64).ravel(), 0.0)
    K = int(min(K, lc.size, ls.size))
    e = lc - ls
    T = float(np.sum(ls))
    S_K = float(np.sum(ls[:K]))
    u_K = float(np.sum(e[:K]))
    v = float(np.sum(e))
    if T <= 0:
        return {"linear": np.nan, "remainder": np.nan, "ar_err": np.nan}
    ar_c = float(np.sum(lc[:K]) / max(np.sum(lc), 1e-15))
    ar_s = S_K / T
    linear = u_K / T - S_K * v / (T * T)
    remainder = (ar_c - ar_s) - linear
    return {"linear": linear, "remainder": remainder, "ar_err": ar_c - ar_s}


def matrix_vs_ar_separation(
    lam_sigma: np.ndarray,
    K: int,
    *,
    epsilon: float = 1e-4,
    b_scale: float = 10.0,
) -> dict:
    """
    Construct C_A = Sigma + eps I, C_B = (1+b) Sigma; verify matrix vs AR ranking swap.
    """
    ls = np.maximum(np.asarray(lam_sigma, dtype=np.float64).ravel(), 0.0)
    N = ls.size
    K = int(min(K, N))
    T = float(np.sum(ls))
    S_K = float(np.sum(ls[:K]))
    mean_top = S_K / max(K, 1)
    mean_all = T / max(N, 1)
    if abs(mean_top - mean_all) < 1e-12:
        return {"separation": False, "reason": "spherical_spectrum"}
    Sigma = np.diag(ls)
    C_A = Sigma + float(epsilon) * np.eye(N)
    C_B = (1.0 + float(b_scale)) * Sigma
    fro_a = float(np.linalg.norm(C_A - Sigma, ord="fro"))
    fro_b = float(np.linalg.norm(C_B - Sigma, ord="fro"))
    la, _ = np.linalg.eigh(C_A)
    lb, _ = np.linalg.eigh(C_B)
    la, lb = la[::-1], lb[::-1]
    ar_a = abs(ar_k_decomposition(la, ls, K)["ar_err"])
    ar_b = abs(ar_k_decomposition(lb, ls, K)["ar_err"])
    return {
        "separation": bool(fro_a < fro_b and ar_a > ar_b),
        "fro_a": fro_a,
        "fro_b": fro_b,
        "ar_err_a": ar_a,
        "ar_err_b": ar_b,
    }


def _ledoit_wolf_qis(W: np.ndarray) -> np.ndarray | None:
    """
    Quadratic-Inverse Shrinkage (Ledoit & Wolf 2020,
    *Quadratic Shrinkage for Large Covariance Matrices*, working paper;
    Ledoit & Wolf 2022, *Annals of Statistics*).

    Faithful Python port of the BSD-licensed reference implementation
    ``QIS.py`` by Patrick Ledoit, distributed at
    https://github.com/pald22/covShrinkage (commit 2021-09-11), itself
    a Python translation of the canonical MATLAB ``QIS.m``.

    Notation follows the reference verbatim.  Let ``S = Y' Y / n`` be the
    de-meaned sample covariance with effective sample size ``n = T - 1``,
    spectral decomposition ``S = U diag(lambda) U'``, concentration
    ratio ``c = p / n`` and bandwidth
    ``h = min(c^2, 1/c^2)^{0.35} / p^{0.35}``.  Working in inverse-
    eigenvalue coordinates ``ell_j = 1 / lambda_j`` over the
    non-degenerate top ``min(p, n)`` directions, the smoothed Stein
    shrinker ``theta_j`` and its Hilbert-conjugate ``Htheta_j`` are

    .. math::

        \\theta_j      = \\frac{1}{n_e}\\sum_i
            \\frac{\\ell_j (\\ell_j - \\ell_i)}
                  {(\\ell_j - \\ell_i)^2 + h^2 \\ell_j^2},
        \\quad
        \\widetilde{\\theta}_j = \\frac{1}{n_e}\\sum_i
            \\frac{h\\, \\ell_j^2}
                  {(\\ell_j - \\ell_i)^2 + h^2 \\ell_j^2}.

    Letting ``A_j = theta_j^2 + Htheta_j^2``, the shrunk eigenvalues are

    .. math::
        \\delta_j = \\begin{cases}
            \\big[ (1-c)^2 \\ell_j + 2c(1-c) \\ell_j \\theta_j
                 + c^2 \\ell_j A_j \\big]^{-1}, & p \\le n,\\\\
            \\big[\\ell_j A_j\\big]^{-1}, & p > n.
        \\end{cases}

    For ``p > n`` the ``p - n`` zero sample eigenvalues are filled by
    ``\\delta_0 = 1 / [(c - 1)\\,\\overline{\\ell}]``.  The full vector
    ``\\delta`` is finally rescaled to preserve the sample trace,
    ``\\delta \\leftarrow \\delta \\cdot \\sum \\lambda / \\sum \\delta``.

    Returns ``None`` on degenerate inputs.
    """
    Y = np.asarray(W, dtype=np.float64)
    T, p = Y.shape
    if T <= 1 or p < 1:
        return None
    Yc = Y - Y.mean(axis=0, keepdims=True)
    n = T - 1
    if n < 1:
        return None

    sample = (Yc.T @ Yc) / n
    sample = (sample + sample.T) / 2.0
    lam, u = np.linalg.eigh(sample)  # ascending
    lam = np.maximum(lam, 0.0)

    c = float(p) / float(n)
    if c <= 0:
        return None
    h = (min(c ** 2, 1.0 / c ** 2) ** 0.35) / (p ** 0.35)

    # Reference picks the top ``min(p, n)`` eigenvalues in MATLAB-1-indexed
    # form ``lambda(max(1,p-n+1):p)``.  The leading ``max(p - n, 0)``
    # entries are zero in the high-dimensional regime.
    n_eff = min(p, n)
    start = max(p - n, 0)
    lam_top = lam[start:]                                   # (n_eff,) ascending
    invlam = 1.0 / np.maximum(lam_top, 1e-30)               # ell_j
    invlam_mean = float(np.mean(invlam))

    Lj = np.repeat(invlam[:, None], n_eff, axis=1)          # (n_eff, n_eff); col j fixed
    Li = Lj.T                                                # row i = ell_i
    Lj_i = Lj - Li                                           # ell_j - ell_i

    denom = Lj_i ** 2 + (h * Lj) ** 2                        # (ell_j - ell_i)^2 + h^2 ell_j^2
    denom_safe = np.where(denom > 0.0, denom, 1.0)

    theta = (Lj * Lj_i / denom_safe).mean(axis=0)            # (n_eff,)  -- mean over i
    Htheta = (h * Lj * Lj / denom_safe).mean(axis=0)         # (n_eff,)
    Atheta2 = theta ** 2 + Htheta ** 2

    if p <= n:
        denom_delta = (
            (1.0 - c) ** 2 * invlam
            + 2.0 * c * (1.0 - c) * invlam * theta
            + c ** 2 * invlam * Atheta2
        )
        denom_delta = np.where(denom_delta > 0.0, denom_delta, 1e-30)
        delta = 1.0 / denom_delta
    else:
        delta_zero = 1.0 / ((c - 1.0) * invlam_mean) if c != 1.0 else 0.0
        nonnull = 1.0 / np.maximum(invlam * Atheta2, 1e-30)
        delta = np.concatenate([np.full(start, delta_zero), nonnull])

    delta = np.maximum(delta, 0.0)
    sum_lam = float(np.sum(lam))
    sum_delta = float(np.sum(delta))
    if sum_delta > 0 and sum_lam > 0:
        delta = delta * (sum_lam / sum_delta)

    pos = delta[delta > 0]
    pos_floor = float(np.maximum(pos.min(), 1e-12)) * 1e-3 if pos.size else 1e-12
    delta = np.maximum(delta, pos_floor)

    Sigma = u @ np.diag(delta) @ u.T
    return (Sigma + Sigma.T) / 2.0


def _bbp_rie_clean(W: np.ndarray) -> np.ndarray | None:
    """
    Bouchaud-Bun-Potters Rotationally-Invariant Estimator (BBP RIE)
    eigenvalue cleaning (Bun, Bouchaud & Potters 2017, *Physics Reports*
    666, 1-109).

    The method:

    1. Compute the sample covariance ``S = W'W/(T-1)`` and its
       eigendecomposition ``S = U diag(lam) U^T``.
    2. Identify the upper Marchenko-Pastur (1967) bulk edge
       ``lam_+ = sigma2 (1 + sqrt(q))^2`` with ``q = N/T`` and
       ``sigma2 = mean(lam)`` (a robust estimator of the population
       variance under the null of i.i.d. returns).
    3. **Bulk clip**: replace eigenvalues lying inside ``[lam_-, lam_+]``
       by the average ``mean(lam in bulk)``; this kills idiosyncratic
       noise while preserving the trace.
    4. **Observable shrinkage** (BBP-2017 Eq. 8.5): for the surviving
       outlier eigenvalues, apply the rotationally-invariant correction
       ``lam_i^* = lam_i (1 - q + q lam_i lam_minus^{-1})^{-1}`` capped
       below by ``lam_+`` (this is the closed-form shrinkage for a
       single-spike model with i.i.d. residuals; the general expression
       reduces to this when sub-leading correlations vanish).

    Returns the rotation-invariant cleaned covariance ``U diag(lam^*)
    U^T``, symmetrised for numerical safety.
    """
    Y = np.asarray(W, dtype=np.float64)
    T, p = Y.shape
    if T <= 1 or p < 1:
        return None
    Yc = Y - Y.mean(axis=0, keepdims=True)
    n = T - 1
    sample = (Yc.T @ Yc) / n
    sample = (sample + sample.T) / 2.0
    lam, U = np.linalg.eigh(sample)  # ascending
    lam = np.maximum(lam, 0.0)

    if not np.any(lam > 0):
        return sample

    q = float(p) / float(T)
    # Use the robust sample mean as the population variance proxy.
    sigma2 = float(np.maximum(lam.mean(), 1e-12))
    lam_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2
    lam_minus = sigma2 * max((1.0 - np.sqrt(q)) ** 2, 0.0)

    bulk_mask = (lam >= lam_minus) & (lam <= lam_plus)
    lam_clean = lam.copy()
    if bulk_mask.sum() >= 2:
        lam_clean[bulk_mask] = float(lam[bulk_mask].mean())

    # Observable BBP shrinkage on outliers (above the bulk).
    out_mask = lam > lam_plus
    if out_mask.any():
        lam_out = lam[out_mask]
        denom = np.maximum(1.0 - q + q * lam_out / max(lam_minus, 1e-12), 1e-12)
        lam_clean[out_mask] = np.maximum(lam_out / denom, lam_plus)

    # Floor near-zero eigenvalues at a small fraction of the bulk mean
    # so the matrix remains strictly positive-definite.
    pos_floor = float(np.maximum(lam_clean[lam_clean > 0].min(), 1e-12)) * 1e-3
    lam_clean = np.maximum(lam_clean, pos_floor)

    Sigma = U @ np.diag(lam_clean) @ U.T
    return (Sigma + Sigma.T) / 2.0


def compute_shrunk_cov(
    returns_window: np.ndarray,
    estimator: str = "qis",
):
    """
    Compute (S_t, C_t) for the SVD feature pipeline.

    Parameters
    ----------
    returns_window : (M, N) returns sub-window with M observations of N assets.
    estimator      : ``"qis"``         — Ledoit-Wolf (2020) quadratic-inverse
                                          shrinkage (default; KMZ Prop 2
                                          rate-optimal eigenvector).
                     ``"linear_shrinkage"`` — Ledoit-Wolf (2003) linear
                                          shrinkage; legacy default.
                     ``"bbp_rie"``     — Bouchaud-Bun-Potters (2017) RIE
                                          eigenvalue cleaning.
                     ``"sample"``      — raw sample covariance $C_t=S_t$ (no shrinkage).

    Returns
    -------
    (S_t, C_t)  with ``S_t`` the sample covariance and ``C_t`` the chosen
    population estimate; both ``(N, N)``.  ``(None, None)`` if the window
    is degenerate.
    """
    W = np.asarray(returns_window)
    if W.ndim != 2 or W.shape[0] < 2 or W.shape[1] < 1:
        return None, None
    S_t = np.cov(W, rowvar=False, ddof=1)
    if np.any(~np.isfinite(S_t)):
        return None, None

    est_key = (estimator or "qis").lower()
    if est_key == "qis":
        C_t = _ledoit_wolf_qis(W)
        if C_t is None or not np.all(np.isfinite(C_t)):
            # Fall back to LW linear shrinkage if QIS failed numerically.
            C_t = LedoitWolf().fit(W).covariance_
    elif est_key == "bbp_rie":
        C_t = _bbp_rie_clean(W)
        if C_t is None or not np.all(np.isfinite(C_t)):
            C_t = LedoitWolf().fit(W).covariance_
    elif est_key in ("linear_shrinkage", "lw", "ledoit_wolf"):
        C_t = LedoitWolf().fit(W).covariance_
    elif est_key in ("oas",):
        # Oracle Approximating Shrinkage (Chen, Wiesel, Hero, Eldar 2010);
        # rotation-equivariant linear shrinkage toward a scaled identity.
        C_t = OAS().fit(W).covariance_
        if C_t is None or not np.all(np.isfinite(C_t)):
            C_t = LedoitWolf().fit(W).covariance_
    elif est_key in ("sample", "sample_cov", "raw"):
        C_t = S_t.copy()
    else:
        raise ValueError(
            f"Unknown covariance estimator '{estimator}'; "
            "expected 'qis', 'linear_shrinkage', 'oas', 'bbp_rie', or 'sample'."
        )
    return S_t, C_t


def extract_svd_features(
    C_t: np.ndarray,
    prev_u1: np.ndarray | None,
    K: int,
    crisis_thresholds: list[float],
    eps: float = 1e-12,
    *,
    prev_U_K: np.ndarray | None = None,
    prev_AR: float | None = None,
) -> dict:
    """
    Extract SVD-based features from the shrinkage covariance ``C_t``.

    Core features
    -------------
    f1         : variance share of leading PC (lambda_1 / sum(lambda))
    sigma1     : vol of first PC = sqrt(lambda_1), daily scale
    log_sigma1 : log(sigma1); on same scale as log(RV) target
    AR         : absorption ratio = sum(lambda[:K]) / sum(lambda)
    cos_theta  : cosine similarity of consecutive leading eigenvectors
    angle      : 1 - cos_theta (rotation angle; 0=stable, 2=full reversal)
    crisis_*   : binary flags for each threshold in crisis_thresholds
    u1         : leading eigenvector (internal; popped before storing to df)

    Spectral descriptors (Phase 3.1)
    --------------------------------
    entropy    : eigenvalue entropy ``-sum(p_i log p_i)`` with
                 ``p_i = lambda_i / sum(lambda)``.
    spectral_gap : ``lambda_1 - lambda_2``.
    log_condition : ``log(lambda_1 / lambda_N)``.
    k_90       : ``min{k : sum(lambda[:k]) / sum(lambda) >= 0.9} / N``.

    RMT / Davis-Kahan-aware additions
    ----------------------------------
    subspace_dist_K  : principal-subspace distance between the top-``K``
                       eigenspaces of consecutive shrinkage covariance
                       matrices,
                       ``sqrt(max(K - ||U_K^T U_K^{prev}||_F^2, 0))``.
                       This is the chordal Frobenius distance on the
                       Grassmannian (Davis & Kahan 1970; Yu, Wang & Samworth
                       2015), which is more robust than the single-vector
                       angle when ``lambda_1 / lambda_2`` is small (the case
                       in calm regimes where eigenvector identification is
                       unstable).
    gap12_norm       : normalized top-2 gap ``(lambda_1 - lambda_2)/lambda_1``
                       on ``[0, 1]``.  Davis-Kahan bounds the subspace
                       distance by a constant times the inverse gap, so this
                       is the natural complementary feature.
    delta_AR         : raw one-step change in the absorption ratio,
                       ``AR_t - AR_{t-1}``.  Standardized to a no-look-ahead
                       z-score (``delta_AR_z``) inside
                       :func:`build_svd_features_panel`.
    """
    if C_t is None or not np.all(np.isfinite(C_t)):
        return _nan_svd_features(crisis_thresholds)
    vals, U = np.linalg.eigh(C_t)
    idx = np.argsort(vals)[::-1]
    lambdas = vals[idx]
    U = U[:, idx]
    lambdas = np.maximum(lambdas, 0.0)
    if np.sum(lambdas) < eps:
        return _nan_svd_features(crisis_thresholds)
    total = np.sum(lambdas)
    N = len(lambdas)

    f1 = float(lambdas[0] / total)
    sigma1 = float(np.sqrt(lambdas[0]))
    log_sigma1 = float(np.log(sigma1 + eps))
    k_use = min(K, N)
    AR = float(np.sum(lambdas[:k_use]) / total)

    gap_sel = select_gap_screened_K(lambdas, K_max=min(max(K, 2), N - 1), rho_ar=0.70, eps=eps)
    k_gap = int(min(max(gap_sel["K_hat"], 1), N))

    # Adaptive K (spike strength proxy): choose k maximizing eigenvalue ratio.
    # This is a lightweight Ahn–Horenstein-style heuristic that is robust to
    # scale and tends to select a small number of dominant factors in crises.
    k_max = int(min(max(2, K), max(N - 1, 1)))
    if N >= 2 and k_max >= 2:
        ratios = []
        for j in range(0, k_max - 1):
            num = float(lambdas[j])
            den = float(lambdas[j + 1]) if float(lambdas[j + 1]) > eps else eps
            ratios.append(num / den)
        j_star = int(np.nanargmax(np.asarray(ratios))) if ratios else 0
        K_eff = int(j_star + 1)
    else:
        K_eff = int(k_use)
    K_eff = int(np.clip(K_eff, 1, max(N, 1)))
    AR_eff = float(np.sum(lambdas[:K_eff]) / total)

    u1 = U[:, 0]
    if prev_u1 is not None and prev_u1.size == u1.size:
        if np.dot(u1, prev_u1) < 0:
            u1 = -u1
        cos_theta = float(np.clip(np.dot(u1, prev_u1), -1.0, 1.0))
        angle = float(1.0 - cos_theta)
    else:
        cos_theta = np.nan
        angle = np.nan

    p_i = lambdas / total
    p_clip = np.maximum(p_i, 1e-15)
    entropy = float(-np.sum(p_clip * np.log(p_clip)))

    spectral_gap = float(lambdas[0] - lambdas[1]) if N > 1 else float(lambdas[0])
    if lambdas[0] > eps:
        gap12_norm = float(spectral_gap / lambdas[0])
        gap12_norm = float(np.clip(gap12_norm, 0.0, 1.0))
    else:
        gap12_norm = float("nan")

    lambda_min = float(lambdas[-1])
    log_condition = float(np.log(lambdas[0] / (lambda_min + eps) + eps))

    cumvar = np.cumsum(lambdas) / total
    k_90_count = int(np.searchsorted(cumvar, 0.90)) + 1
    k_90 = float(min(k_90_count, N) / N)

    # --- Top-K subspace and absorption-ratio dynamics -------------------
    # Primary monitor uses gap-screened K; fixed config K retained as k_use for AR headline.
    k_monitor = k_gap
    U_K = U[:, :k_monitor].copy()
    if (
        prev_U_K is not None
        and prev_U_K.shape[0] == U_K.shape[0]
        and prev_U_K.shape[1] == U_K.shape[1]
        and np.all(np.isfinite(prev_U_K))
    ):
        # Frobenius-norm chordal Grassmannian distance:
        #   d^2 = K - || U_K^T U_K^{prev} ||_F^2
        M = U_K.T @ prev_U_K
        sq_dist = float(k_monitor) - float(np.sum(M * M))
        subspace_dist_K = float(np.sqrt(max(sq_dist, 0.0)))
    else:
        subspace_dist_K = float("nan")

    if prev_AR is not None and np.isfinite(prev_AR):
        delta_AR = float(AR - float(prev_AR))
    else:
        delta_AR = float("nan")

    out = {
        "f1": f1,
        "sigma1": sigma1,
        "log_sigma1": log_sigma1,
        "AR": AR,
        "K_gap": float(k_gap),
        "gap_norm_sel": float(gap_sel["gap_norm_K"]),
        "AR_gap": float(gap_sel["AR_K"]),
        "K_eff": float(K_eff),
        "AR_eff": AR_eff,
        "cos_theta": cos_theta,
        "angle": angle,
        "entropy": entropy,
        "spectral_gap": spectral_gap,
        "gap12_norm": gap12_norm,
        "log_condition": log_condition,
        "k_90": k_90,
        "subspace_dist_K": subspace_dist_K,
        "delta_AR": delta_AR,
        "u1": u1.copy(),
        "U_K": U_K,
    }
    for th in crisis_thresholds:
        out[f"crisis_{th}"] = 1 if (np.isfinite(cos_theta) and cos_theta < th) else 0
    return out


def _nan_svd_features(crisis_thresholds: list[float]) -> dict:
    out = {
        "f1": np.nan,
        "sigma1": np.nan,
        "log_sigma1": np.nan,
        "AR": np.nan,
        "K_gap": np.nan,
        "gap_norm_sel": np.nan,
        "AR_gap": np.nan,
        "K_eff": np.nan,
        "AR_eff": np.nan,
        "cos_theta": np.nan,
        "angle": np.nan,
        "entropy": np.nan,
        "spectral_gap": np.nan,
        "gap12_norm": np.nan,
        "log_condition": np.nan,
        "k_90": np.nan,
        "subspace_dist_K": np.nan,
        "delta_AR": np.nan,
        "u1": None,
        "U_K": None,
    }
    for th in crisis_thresholds:
        out[f"crisis_{th}"] = 0
    return out


_NAN_ROW_BASE: dict = {
    "f1": np.nan, "sigma1": np.nan, "log_sigma1": np.nan,
    "AR": np.nan, "cos_theta": np.nan, "angle": np.nan,
    "entropy": np.nan, "spectral_gap": np.nan, "gap12_norm": np.nan,
    "log_condition": np.nan, "k_90": np.nan,
    "subspace_dist_K": np.nan, "delta_AR": np.nan,
}


def _add_delta_AR_z(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Standardize ``delta_AR`` to a no-look-ahead rolling z-score.

    ``delta_AR_z_t = (delta_AR_t - rolling_mean_{t-1}) / rolling_std_{t-1}``,
    where the moments are computed on a trailing window of length
    ``window`` and shifted by one period to avoid using the current
    observation in its own normalisation.  We require ``min_periods =
    window`` so the leading rows where the trailing sample is shorter
    than ``window`` produce ``NaN`` (consistent with the rest of the
    panel's warm-up).
    """
    s = df["delta_AR"].astype(float)
    mu = s.rolling(window=window, min_periods=window).mean().shift(1)
    sd = s.rolling(window=window, min_periods=window).std(ddof=1).shift(1)
    z = (s - mu) / sd.replace(0.0, np.nan)
    df["delta_AR_z"] = z
    return df


def build_svd_features_panel(
    returns: pd.DataFrame,
    asset_columns: list[str],
    svd_window: int,
    K: int,
    crisis_thresholds: list[float],
    eps: float = 1e-12,
    *,
    cov_estimator: str = "qis",
) -> pd.DataFrame:
    """
    Build a DataFrame of SVD features aligned to returns index.
    Each row t uses returns from [t - svd_window, t) to compute C_t, then SVD features.

    Also computes delta_f1 (first difference of f1) as a separate column.
    delta_f1 captures the trend in variance concentration -- whether the market
    is becoming more or less concentrated in one dominant factor.
    """
    index = returns.index
    n = len(index)
    rows = []
    prev_u1 = None
    prev_U_K: np.ndarray | None = None
    prev_AR: float | None = None
    for t in range(n):
        if t < svd_window:
            row = dict(_NAN_ROW_BASE)
            for th in crisis_thresholds:
                row[f"crisis_{th}"] = 0
            rows.append(row)
            continue
        W = returns.iloc[t - svd_window : t][asset_columns].values
        _, C_t = compute_shrunk_cov(W, estimator=cov_estimator)
        feats = extract_svd_features(
            C_t, prev_u1, K, crisis_thresholds, eps,
            prev_U_K=prev_U_K, prev_AR=prev_AR,
        )
        u1 = feats.pop("u1", None)
        U_K = feats.pop("U_K", None)
        if u1 is not None:
            prev_u1 = u1
        if U_K is not None:
            prev_U_K = U_K
        if np.isfinite(feats.get("AR", np.nan)):
            prev_AR = float(feats["AR"])
        rows.append(feats)

    out = pd.DataFrame(rows, index=index)
    out["delta_f1"] = out["f1"].diff()
    out = _add_delta_AR_z(out)
    return out


def build_har_rv(
    portfolio_returns: pd.Series,
    rv_windows: dict[str, int],
) -> pd.DataFrame:
    """
    Build HAR-style realized VARIANCE features from portfolio log-returns (in %).

    rv_windows: e.g. {"daily": 1, "weekly": 5, "biweekly": 10, "monthly": 22}.
    Returns DataFrame with columns RV_d, RV_w, RV_10d, RV_m.

    IMPORTANT: All columns are VARIANCE (squared returns, %-squared), NOT volatility.
    Consistent with the variance-based target:
        target_h = mean(r_{t+1}^2, ..., r_{t+h}^2)   (variance in %-squared)

    The standard log-HAR regression (Corsi 2009) regresses log(RV_h) on
    log(RV_{d,w,m}), where RV = sum(r^2)/h. Using variance (not RMS) ensures
    log(target) ~ log(feature) without the 0.5-factor artifact from sqrt.

    Features are lagged by 1 day (shift(1)) so the feature at time t uses
    only returns up to t-1 (strict no-look-ahead).
    """
    r = portfolio_returns
    out = pd.DataFrame(index=r.index)
    col_map = {
        "daily": "RV_d",
        "weekly": "RV_w",
        "biweekly": "RV_10d",
        "monthly": "RV_m",
    }
    for name, w in rv_windows.items():
        col = col_map.get(name, f"RV_{name}")
        if w == 1:
            # Single-day: r^2 = daily variance proxy (%-squared)
            out[col] = (r ** 2).shift(1)
        else:
            # Multi-day: mean(r^2) over w days = variance proxy (%-squared)
            out[col] = (r ** 2).rolling(window=w).mean().shift(1)
    return out


def build_svd_features_panel_with_cov(
    returns: pd.DataFrame,
    asset_columns: list[str],
    svd_window: int,
    K: int,
    crisis_thresholds: list[float],
    eps: float = 1e-12,
    *,
    cov_estimator: str = "qis",
) -> tuple[pd.DataFrame, list[tuple]]:
    """
    Extended version of build_svd_features_panel that also returns the covariance series.

    Returns
    -------
    svd_df : pd.DataFrame
        Same output as build_svd_features_panel (including log_sigma1, angle, delta_f1).
    cov_series : list of (pd.Timestamp, np.ndarray) tuples
        Each entry is (date, C_t) for rows t >= svd_window.
        Rows t < svd_window are omitted (no valid C_t).
    """
    index = returns.index
    n = len(index)
    rows = []
    cov_series = []
    prev_u1 = None
    prev_U_K: np.ndarray | None = None
    prev_AR: float | None = None
    for t in range(n):
        if t < svd_window:
            row = dict(_NAN_ROW_BASE)
            for th in crisis_thresholds:
                row[f"crisis_{th}"] = 0
            rows.append(row)
            continue
        W = returns.iloc[t - svd_window : t][asset_columns].values
        _, C_t = compute_shrunk_cov(W, estimator=cov_estimator)
        if C_t is not None:
            cov_series.append((index[t], C_t.copy()))
        feats = extract_svd_features(
            C_t, prev_u1, K, crisis_thresholds, eps,
            prev_U_K=prev_U_K, prev_AR=prev_AR,
        )
        u1 = feats.pop("u1", None)
        U_K = feats.pop("U_K", None)
        if u1 is not None:
            prev_u1 = u1
        if U_K is not None:
            prev_U_K = U_K
        if np.isfinite(feats.get("AR", np.nan)):
            prev_AR = float(feats["AR"])
        rows.append(feats)
    out = pd.DataFrame(rows, index=index)
    out["delta_f1"] = out["f1"].diff()
    out = _add_delta_AR_z(out)
    return out, cov_series


def build_feature_sets(
    rv_df: pd.DataFrame,
    svd_df: pd.DataFrame | None,
    use_svd: bool = False,
    default_threshold: float = 0.8,
    eps: float = 1e-8,
    semi_df: pd.DataFrame | None = None,
    csd_df: pd.DataFrame | None = None,
    turb_df: pd.DataFrame | None = None,
    svd_tier: int | None = None,
    interaction_smooth_h: int | None = None,
    include_cross_section: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build feature matrix with tiered SVD feature sets.

    VIX (log_vix) is NEVER included as a model feature. VIX is retained
    exclusively as a standalone IV_baseline benchmark. Including VIX in
    model features creates circular logic (IV as input AND benchmark) and
    empirically dominates all SVD predictors with coefficient ~0.86.

    svd_tier controls which feature groups are included:
        Tier 0 (HAR-only): log_RSV_d_minus, log_RSV_d_plus, RV_w, RV_10d, RV_m
                           5 features — pure HAR baseline with semivariance.
        Tier 1 (HAR + SVD-level): +f1, log_sigma1
                           7 features — adds eigenvalue-level summaries.
        Tier 2 (HAR + SVD-full): +AR, angle, delta_f1, crisis_0.8,
                                   entropy, spectral_gap, log_condition, k_90,
                                   angle*log_RSV_d_minus (interaction)
                           16 features — adds eigenstructure, dynamics, and
                                          new spectral features (Phase 3).
        Tier 3 (HAR + SVD + cross-section): +log_CSD_d, log_turb, corr_surprise
                           19 features — adds cross-sectional & turbulence.

    include_cross_section : bool
        If False and svd_tier==3, returns the same columns as Tier 2 (G4 appendix-only).

    When svd_tier is None, falls back to use_svd bool: False=Tier 0,
    True=Tier 2 (previous default behaviour, for backward compatibility).

    Ablation interpretation:
        Tier 1 - Tier 0 : marginal value of eigenvalue-level summaries
        Tier 2 - Tier 1 : marginal value of eigenstructure & dynamics
        Tier 3 - Tier 2 : marginal value of cross-sectional dispersion / turbulence

    Each tier uses the same ElasticNet pipeline so comparisons are not
    confounded by estimator differences.

    rv_df: must have RV_d (or RSV columns), RV_w, RV_10d, RV_m columns.
    svd_df: from build_svd_features_panel, with f1, log_sigma1, AR, angle,
            delta_f1, crisis_*.
    semi_df: from build_semivariance_features (RSV_d_minus, RSV_d_plus).
    csd_df: from build_csd_features (CSD_d) — used only for Tier 3.
    turb_df: from build_turbulence_index (turb_lag1, corr_surprise_lag1) — Tier 3.
    interaction_smooth_h : if > 1, `angle_x_logRSV_minus` uses h-day rolling mean
        of log_RSV_d_minus (shift(1)) so the interaction matches the horizon-smoothed
        angle from `smooth_svd_features` (HAR-style multi-scale alignment).

    Returns (X_df, feature_names).
    """
    # Resolve tier from svd_tier or legacy use_svd flag
    if svd_tier is None:
        tier = 2 if use_svd else 0
    else:
        tier = int(svd_tier)
    assert tier in (0, 1, 2, 3), f"svd_tier must be 0-3, got {tier}"

    names: list[str] = []
    X = pd.DataFrame(index=rv_df.index)

    # --- HAR features (Tier 0+) — semivariance split + multi-horizon log-RV ---
    if semi_df is not None and "RSV_d_minus" in semi_df.columns:
        # Replace single RV_d with downside/upside semivariance components.
        # Patton & Sheppard (2015, REStat): RSV- has ~2x the predictive
        # coefficient of RSV+ for future realized variance.
        X["log_RSV_d_minus"] = np.log(
            semi_df["RSV_d_minus"].reindex(rv_df.index).values.astype(np.float64) + eps
        )
        X["log_RSV_d_plus"] = np.log(
            semi_df["RSV_d_plus"].reindex(rv_df.index).values.astype(np.float64) + eps
        )
        names += ["log_RSV_d_minus", "log_RSV_d_plus"]
    elif "RV_d" in rv_df.columns:
        X["RV_d"] = np.log(rv_df["RV_d"].values.astype(np.float64) + eps)
        names.append("RV_d")

    for col in ["RV_w", "RV_10d", "RV_m"]:
        if col in rv_df.columns:
            X[col] = np.log(rv_df[col].values.astype(np.float64) + eps)
            names.append(col)

    if tier == 0 or svd_df is None:
        return X, names

    # --- Tier 1: eigenvalue-level summaries ---
    # f1 = variance share of leading PC; log_sigma1 = log-vol of first PC.
    # These capture "size" of the dominant market factor but may be correlated
    # with realized variance level features; isolated here for clean attribution.
    for c in ["f1", "log_sigma1"]:
        if c in svd_df.columns:
            X[c] = svd_df[c].reindex(rv_df.index).values
            names.append(c)
        elif c == "log_sigma1" and "sigma1" in svd_df.columns:
            X["log_sigma1"] = np.log(svd_df["sigma1"].reindex(rv_df.index).values + eps)
            names.append("log_sigma1")

    if tier == 1:
        return X, names

    # --- Tier 2: eigenstructure, dynamics, and new spectral features ---
    # AR = absorption ratio (K eigenvalues / total variance; fragility measure).
    # angle = 1 - cos_theta (eigenvector rotation; structural instability).
    # delta_f1 = day-over-day change in PC1 share (captures transitions).
    # crisis_* = binary flag for angle-based regime (threshold sensitivity).
    # entropy, spectral_gap, log_condition, k_90: new spectral descriptors (Phase 3).
    # interaction: angle * log_RSV_d_minus ("rotation amplifies negative semivariance").
    for c in ["AR", "AR_eff", "K_eff", "angle", "delta_f1", f"crisis_{default_threshold}"]:
        if c in svd_df.columns:
            X[c] = svd_df[c].reindex(rv_df.index).values
            names.append(c)
        elif c == "angle" and "cos_theta" in svd_df.columns:
            X["angle"] = 1.0 - svd_df["cos_theta"].reindex(rv_df.index).values
            names.append("angle")

    # New spectral features: always included in Tier 2+ if available
    # ``subspace_dist_K``, ``gap12_norm`` and ``delta_AR_z`` are the
    # Davis-Kahan / RMT-aware additions from Phase 3.2.
    for c in [
        "entropy", "spectral_gap", "gap12_norm",
        "log_condition", "k_90",
        "subspace_dist_K", "delta_AR_z",
    ]:
        if c in svd_df.columns:
            X[c] = svd_df[c].reindex(rv_df.index).values
            names.append(c)

    # Interaction feature: angle * log_RSV_d_minus
    # "Eigenvector rotation amplifies negative semivariance signal"
    # ElasticNet will select/shrink this if not predictive.
    if "angle" in X.columns and "log_RSV_d_minus" in X.columns:
        angle_vals = X["angle"].values
        log_rsv_vals = X["log_RSV_d_minus"].values.astype(np.float64, copy=False)
        if interaction_smooth_h is not None and interaction_smooth_h > 1:
            log_rsv_series = pd.Series(log_rsv_vals, index=rv_df.index)
            log_rsv_vals = (
                log_rsv_series.rolling(window=interaction_smooth_h, min_periods=interaction_smooth_h)
                .mean()
                .shift(1)
                .values.astype(np.float64, copy=False)
            )
        interaction = angle_vals * log_rsv_vals
        # Only add if we have finite values (angle may be NaN at start)
        X["angle_x_logRSV_minus"] = interaction
        names.append("angle_x_logRSV_minus")

    if tier == 2:
        return X, names

    # Tier 3 without cross-section blocks (primary spec): identical to Tier 2.
    if tier == 3 and not include_cross_section:
        return X, names

    # --- Tier 3: cross-sectional dispersion and turbulence ---
    # These are NOT derived from SVD singular vectors but from the same C_t
    # covariance matrix. Isolated here so Tier 2 - Tier 0 cleanly measures
    # spectral feature contribution independent of Mahalanobis-type measures.
    if csd_df is not None and "CSD_d" in csd_df.columns:
        X["log_CSD_d"] = np.log(
            csd_df["CSD_d"].reindex(rv_df.index).values.astype(np.float64) + eps
        )
        names.append("log_CSD_d")

    if turb_df is not None and "turb_lag1" in turb_df.columns:
        X["log_turb"] = np.log(
            np.maximum(turb_df["turb_lag1"].reindex(rv_df.index).values.astype(np.float64), eps)
        )
        names.append("log_turb")
    if turb_df is not None and "corr_surprise_lag1" in turb_df.columns:
        X["corr_surprise"] = turb_df["corr_surprise_lag1"].reindex(rv_df.index).values
        names.append("corr_surprise")

    return X, names


def smooth_svd_features(svd_df: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Horizon-match SVD features by applying the same multi-scale temporal aggregation
    that makes HAR features effective at multiple horizons.

    HAR principle:
        RV_d  = r_{t-1}^2                (daily)
        RV_w  = mean(r^2, last 5 days)   (weekly)
        RV_m  = mean(r^2, last 22 days)  (monthly)

    SVD analogue:
        At h=1 : use daily  angle, delta_f1, f1, etc.  (no change)
        At h=5 : replace with 5-day  rolling mean of each feature (shift(1))
        At h=22: replace with 22-day rolling mean of each feature (shift(1))

    This is the core fix for the horizon-degradation problem: daily eigenvector
    rotation (`angle`, `delta_f1`) is informative for 1-day-ahead volatility but
    is essentially noise for a 22-day average target.  Smoothed rotation captures
    the persistent directional trend in market structure rather than day-to-day jitter.

    Columns smoothed (continuous spectral features):
        f1, sigma1, log_sigma1, AR, angle, delta_f1, cos_theta,
        entropy, spectral_gap, log_condition, k_90

    crisis_* binary flags (kept as float rate: fraction of days in window with flag=1)

    The returned DataFrame has the same index as svd_df; NaN rows introduced by
    rolling are propagated (they match the NaN leading rows from the SVD warm-up
    window and are already filtered out by the caller).
    """
    if h <= 1:
        return svd_df.copy()

    out = svd_df.copy()

    # Continuous spectral features — apply rolling mean then shift(1)
    continuous_cols = [
        "f1", "sigma1", "log_sigma1", "AR", "AR_eff", "K_eff",
        "angle", "delta_f1", "cos_theta",
        "entropy", "spectral_gap", "gap12_norm",
        "log_condition", "k_90",
        "subspace_dist_K", "delta_AR", "delta_AR_z",
    ]
    for col in continuous_cols:
        if col in out.columns:
            out[col] = (
                out[col].rolling(window=h, min_periods=h).mean().shift(1)
            )

    # Binary crisis flags — convert to fraction-in-window (still informative at
    # longer horizons: "what % of past h days were crisis days")
    crisis_cols = [c for c in out.columns if c.startswith("crisis_")]
    for col in crisis_cols:
        out[col] = out[col].rolling(window=h, min_periods=h).mean().shift(1)

    return out


def add_regime_gating_interactions(
    X: pd.DataFrame,
    *,
    gate: pd.Series,
    gate_name: str,
    base_cols: list[str] | None = None,
    q_high: float = 0.8,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create a simple regime-gated expansion of an existing feature matrix.

    Purpose
    -------
    Absorption-ratio / eigen-instability measures are often most informative
    *conditionally* (fragile vs calm regimes). A linear model can express this
    by interacting a gate variable with baseline predictors.

    Construction
    ------------
    Let g_t be the gate series aligned to X.index.
    - g_z = (g - mean) / std   (finite-only moments; NaN preserved)
    - I_high = 1[g >= quantile_q_high]
    Add columns:
      - gate_z::<gate_name>
      - high::<gate_name>
      - x::<col>::gate_z::<gate_name>      for col in base_cols
      - x::<col>::high::<gate_name>        for col in base_cols

    Returns
    -------
    (X_new, new_column_names)
    """
    if not (0.5 < float(q_high) < 0.99):
        raise ValueError("q_high must be in (0.5, 0.99).")
    if not X.index.equals(gate.index):
        gate = gate.reindex(X.index)

    g = gate.astype(float)
    g_finite = g[np.isfinite(g.values)]
    if g_finite.size < 200:
        raise ValueError("Gate series has insufficient finite observations.")

    mu = float(g_finite.mean())
    sd = float(g_finite.std(ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        sd = 1.0
    g_z = (g - mu) / sd
    thr = float(g_finite.quantile(float(q_high)))
    high = (g >= thr).astype(float)

    Xn = X.copy()
    new_names: list[str] = []
    col_gate = f"gate_z::{gate_name}"
    col_high = f"high::{gate_name}"
    Xn[col_gate] = g_z.values
    Xn[col_high] = high.values
    new_names += [col_gate, col_high]

    if base_cols is None:
        base_cols = list(X.columns)
    for c in base_cols:
        if c not in Xn.columns:
            continue
        x = np.asarray(Xn[c].values, dtype=np.float64)
        Xn[f"x::{c}::gate_z::{gate_name}"] = x * np.asarray(g_z.values, dtype=np.float64)
        Xn[f"x::{c}::high::{gate_name}"] = x * np.asarray(high.values, dtype=np.float64)
        new_names += [f"x::{c}::gate_z::{gate_name}", f"x::{c}::high::{gate_name}"]

    return Xn, list(Xn.columns)


def build_svd_har_dynamics_block(
    svd_df: pd.DataFrame,
    *,
    cols: list[str] | None = None,
    windows: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    HAR-style lag block for spectral state variables.

    Motivation: eigenvalue/eigenspace quantities are persistent and may matter
    through multi-scale averages (Corsi HAR logic). This block turns a spectral
    state series into {daily, weekly, monthly}-style lagged predictors.

    Implementation: for each column c in cols and each window w:
      - w == 1 : x_{t-1}
      - w > 1  : mean(x_{t-w:t-1}) via rolling(w).mean().shift(1)
    """
    if cols is None:
        cols = ["AR", "AR_eff", "log_sigma1", "gap12_norm", "subspace_dist_K", "delta_AR_z"]
    if windows is None:
        windows = {"d": 1, "w": 5, "m": 22}
    X = pd.DataFrame(index=svd_df.index)
    names: list[str] = []
    for c in cols:
        if c not in svd_df.columns:
            continue
        s = svd_df[c].astype(float)
        for tag, w in windows.items():
            w = int(w)
            if w <= 1:
                v = s.shift(1)
            else:
                v = s.rolling(window=w, min_periods=w).mean().shift(1)
            name = f"svdhar::{c}::{tag}"
            X[name] = v.values
            names.append(name)
    return X, names

def build_semivariance_features(
    portfolio_returns_pct: pd.Series,
    rv_windows: dict[str, int],
) -> pd.DataFrame:
    """
    Realized semivariance features from daily portfolio returns (in %-units).

    RSV-_t = r_t^2 * 1{r_t < 0}  (downside variance proxy)
    RSV+_t = r_t^2 * 1{r_t >= 0} (upside variance proxy)

    Note: RSV-_t + RSV+_t = RV_d_t (they partition daily variance).

    Patton & Sheppard (2015, REStat) show that RSV- has approximately 2x the
    predictive coefficient of RSV+ for future realized variance, motivating the
    replacement of the single RV_d feature with these two components.

    All features are lagged by 1 day (shift(1)) for no look-ahead.
    """
    r = portfolio_returns_pct
    rsv_minus = (r ** 2) * (r < 0).astype(float)
    rsv_plus = (r ** 2) * (r >= 0).astype(float)
    out = pd.DataFrame(index=r.index)
    out["RSV_d_minus"] = rsv_minus.shift(1)
    out["RSV_d_plus"] = rsv_plus.shift(1)
    return out


def build_csd_features(
    returns_pct: pd.DataFrame,
    rv_windows: dict[str, int],
) -> pd.DataFrame:
    """
    Cross-sectional dispersion (CSD) features from individual stock returns (in %).

    CSD_t = std(r_{1,t}, ..., r_{N,t}) -- standard deviation across N stocks.

    Garcia et al. (2014) show that CSD adds 1-3 pp R² over HAR alone as a
    predictor of portfolio variance, because high cross-sectional dispersion
    (individual stocks moving differently) signals elevated idiosyncratic risk
    that aggregate measures may under-represent.

    All features are lagged by 1 day (shift(1)) for no look-ahead.
    """
    csd_daily = returns_pct.std(axis=1)
    out = pd.DataFrame({"CSD_d": csd_daily.shift(1)}, index=returns_pct.index)
    return out


def build_turbulence_index(
    returns_pct: pd.DataFrame,
    cov_series: list[tuple],
) -> pd.DataFrame:
    """
    Turbulence index and correlation surprise from individual stock returns.

    turb_t = r_t' C_t^{-1} r_t   (Mahalanobis distance, dimensionless)

    Decomposes into two components (Kinlaw & Turkington 2013):
        vol_comp_t     = sum_i(r_{i,t}^2 / sigma_{i,t}^2)  (standardised variance)
        corr_surprise_t = turb_t - vol_comp_t               (correlation component)

    The correlation surprise captures days when cross-asset correlations are
    unusually high given individual stock movements — exactly the kind of systemic
    risk the SVD framework is designed to detect. Kinlaw & Turkington provide
    OOS predictive evidence for turb and corr_surprise across US equity, EU, FX.

    Uses the Ledoit-Wolf covariance matrices already computed in the SVD pipeline
    (cov_series), so no extra estimation cost.

    Returns a DataFrame with turb, vol_comp, corr_surprise, and their 1-day lags
    (the lagged versions are the actual features used in models).
    """
    cov_dict = {date: C for date, C in cov_series}
    index = returns_pct.index
    n = len(index)
    turb_vals = np.full(n, np.nan)
    vol_comp_vals = np.full(n, np.nan)
    corr_surp_vals = np.full(n, np.nan)

    for i, date in enumerate(index):
        if date not in cov_dict:
            continue
        C = cov_dict[date]
        r = returns_pct.loc[date].values
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(C)):
            continue
        try:
            # Mahalanobis: solve C @ x = r to avoid explicit inversion
            Cinv_r = np.linalg.solve(C, r)
            turb = float(r @ Cinv_r)
            # Vol component: each stock standardised by its own variance (diagonal of C)
            diag_var = np.maximum(np.diag(C), 1e-12)
            vol_comp = float(np.sum(r ** 2 / diag_var))
            turb_vals[i] = turb
            vol_comp_vals[i] = vol_comp
            corr_surp_vals[i] = turb - vol_comp
        except np.linalg.LinAlgError:
            continue

    out = pd.DataFrame(
        {"turb": turb_vals, "vol_comp": vol_comp_vals, "corr_surprise": corr_surp_vals},
        index=index,
    )
    # Lag by 1 day — these are the features used in models
    out["turb_lag1"] = out["turb"].shift(1)
    out["corr_surprise_lag1"] = out["corr_surprise"].shift(1)
    return out
