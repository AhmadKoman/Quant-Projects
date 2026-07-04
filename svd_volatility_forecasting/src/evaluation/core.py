# -*- coding: utf-8 -*-
"""
Evaluation metrics, statistical tests, and publication-quality figures.

Metrics:
    RMSE, R², MALE (Mean Absolute Log Error = log-MAPE), QLIKE (variance ratio)

    MALE replaces broken MAPE: dividing by small actual vol values produces
    MAPE > 500%. MALE = mean(|log(y_pred/y_true)|)*100 is bounded, symmetric,
    and standard in volatility forecasting evaluation.

    QLIKE uses variance ratios (y^2): QLIKE = mean(h/h_hat - log(h/h_hat) - 1)
    where h = y_true^2 (realized variance) and h_hat = y_pred^2 (forecast variance).

Statistical tests:
    Diebold-Mariano-West with Newey-West HAC (nlags=5)

Figures:
    F1  Posterior predictive forecast (6-panel, uncertainty bands, crisis shading)
    F2  Ablation heatmap (% RMSE change vs HAR baseline, 3 horizons)
    F3  Crisis threshold sensitivity (4-panel: RMSE/R2 x Crisis/Calm vs tau)
    F4  Crisis window deep-dives (COVID, Ukraine, SVB)
    F5  Epistemic uncertainty over time + HAR posterior coefficient violins
    F6  Enhanced residual diagnostics (histogram, Q-Q, resid vs fitted, ACF^2)
    F7  DM statistic heatmap (8x8 signed DM stats with significance markers)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

EPS_DEFAULT = 1e-8

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False


def qlike_loss(h: float, f: float, eps: float = EPS_DEFAULT) -> float:
    """Patton QLIKE at one observation: h/f - log(h/f) - 1."""
    h = float(max(h, eps))
    f = float(max(f, eps))
    return float(h / f - np.log(h / f) - 1.0)


def qlike_increment_exact(
    h: float,
    f: float,
    delta: float,
    eps: float = EPS_DEFAULT,
) -> float:
    """
    QLIKE difference for rival forecast f + delta vs benchmark f (Proposition QLIKE).
    """
    f = float(max(f, eps))
    fd = float(max(f + delta, eps))
    return float(np.log(1.0 + delta / f) - (h / (f * fd)) * delta)


def qlike_increment_expansion(
    h: float,
    f: float,
    delta: float,
    eps: float = EPS_DEFAULT,
) -> tuple[float, float]:
    """
    First- and second-order terms in delta for qlike_increment_exact (small delta).
    Returns (linear_term, quad_term); O(delta^3) omitted.
    """
    f = float(max(f, eps))
    lin = float((delta / (f * f)) * (f - h))
    quad = float(0.5 * (delta * delta) * (2.0 * h - f) / (f ** 3))
    return lin, quad


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = EPS_DEFAULT):
    """
    Compute RMSE, R², MALE (log-MAPE), QLIKE.

    IMPORTANT: Both arrays must be in the same scale. As of the SOTA Fix,
    the pipeline uses VARIANCE in %-squared units (not volatility), so
    y_true and y_pred here are realized variance and predicted variance.

    MALE = Mean Absolute Log Error = mean(|log(y_pred / y_true)|) * 100.
    For variance inputs this is the log-ratio of variances (2x the log-ratio
    of volatilities); interpretable as "% error in log-variance space".

    QLIKE (Patton 2011) for variance forecasting:
        h_t   = y_true  (realized variance — inputs are already variance)
        h_hat = y_pred  (forecast variance)
        QLIKE = mean(h_t / h_hat - log(h_t / h_hat) - 1)
    Note: previously y_true was volatility and was squared here; now it's
    already variance so squaring is omitted to avoid double-squaring.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = np.asarray(y_true)[mask]
    y_p = np.asarray(y_pred)[mask]
    if y_t.size == 0:
        return np.inf, -np.inf, np.inf, np.inf
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)
    # MALE (log-MAPE): symmetric, bounded, standard in volatility forecasting
    male = np.mean(np.abs(np.log((np.maximum(y_p, eps)) / (np.maximum(y_t, eps))))) * 100
    # QLIKE on variance: inputs ARE variance, so use directly (Patton 2011)
    h_t = np.maximum(y_t, eps)        # realized variance (positive)
    h_hat = np.maximum(y_p, eps)      # forecast variance (positive)
    ratio = h_t / h_hat
    qlike = np.mean(ratio - np.log(ratio) - 1)
    return rmse, r2, male, qlike


def four_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = EPS_DEFAULT) -> dict:
    rmse, r2, male, qlike = compute_metrics(y_true, y_pred, eps)
    return {"RMSE": rmse, "R2": r2, "MALE": male, "QLIKE": qlike}


def mincer_zarnowitz(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    nlags: int | None = None,
    horizon: int = 1,
) -> dict:
    """
    Mincer-Zarnowitz (1969) forecast evaluation regression:
        y_true = alpha + beta * y_pred + eps

    A perfectly calibrated forecast has alpha ≈ 0, beta ≈ 1.
    Returns mz_alpha, mz_beta, and mz_r2.

    This is the standard efficiency test for volatility forecasts.
    Reference: Mincer & Zarnowitz (1969), Pagan & Schwert (1990).
    """
    import statsmodels.api as sm_mz
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = np.asarray(y_true)[mask]
    y_p = np.asarray(y_pred)[mask]
    if y_t.size < 10:
        return {
            "mz_alpha": np.nan, "mz_beta": np.nan, "mz_r2": np.nan,
            "mz_alpha_se": np.nan, "mz_beta_se": np.nan,
            "mz_alpha_t": np.nan, "mz_beta_t": np.nan,
            "mz_alpha_p": np.nan, "mz_beta_p": np.nan,
        }
    X_mz = sm_mz.add_constant(y_p)
    if nlags is None:
        nlags = hac_lags_for_horizon(T=int(y_t.size), horizon=int(horizon))
    nlags = int(max(nlags, 1))
    try:
        res = sm_mz.OLS(y_t, X_mz).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": int(nlags)},
        )
        return {
            "mz_alpha": float(res.params[0]),
            "mz_beta": float(res.params[1]),
            "mz_r2": float(res.rsquared),
            "mz_alpha_se": float(res.bse[0]),
            "mz_beta_se": float(res.bse[1]),
            "mz_alpha_t": float(res.tvalues[0]),
            "mz_beta_t": float(res.tvalues[1]),
            "mz_alpha_p": float(res.pvalues[0]),
            "mz_beta_p": float(res.pvalues[1]),
        }
    except Exception:
        return {
            "mz_alpha": np.nan, "mz_beta": np.nan, "mz_r2": np.nan,
            "mz_alpha_se": np.nan, "mz_beta_se": np.nan,
            "mz_alpha_t": np.nan, "mz_beta_t": np.nan,
            "mz_alpha_p": np.nan, "mz_beta_p": np.nan,
        }


def hac_lags_for_horizon(
    T: int,
    horizon: int = 1,
    base_factor: float = 1.5,
    min_lag: int = 1,
) -> int:
    """
    Horizon-aware Newey-West Bartlett HAC bandwidth.

    Pre-registered rule (see ``methods/preregistration.md``):
    ::

        nlags(h, T) = max(h - 1, floor(base_factor * T^{1/3}), min_lag)

    The cube-root term is the Andrews-(1991)-style data-driven width for the
    Bartlett kernel; the ``h - 1`` floor enforces the requirement of West
    (1996) that the long-run variance estimator includes at least the
    overlap induced by ``h``-step-ahead forecasts.

    Parameters
    ----------
    T            : sample size of the loss differential.
    horizon      : forecast horizon ``h`` (1 for daily one-step).
    base_factor  : multiplier on ``T^{1/3}``; 1.5 is the paper default.
    min_lag      : safety floor; should be at least 1 for the kernel sum.
    """
    T = max(int(T), 1)
    overlap = max(int(horizon) - 1, 0)
    rule = int(np.floor(base_factor * (T ** (1.0 / 3.0))))
    return int(max(overlap, rule, int(min_lag)))


def hln_small_sample_correction(
    t_dm: float,
    T: int,
    horizon: int,
) -> tuple[float, float]:
    """
    Harvey-Leybourne-Newbold (1997) small-sample correction for DM-style
    statistics with ``h``-step overlap.

    Multiplier
    ::

        m(T, h) = sqrt[(T + 1 - 2h + h(h-1)/T) / T] ,

    and reference distribution :math:`t_{T-1}` (rather than standard normal).
    Returns ``(t_star, p_value_two_sided)``.

    Notes
    -----
    The multiplier is bounded above by 1 and below by 0; for ``h`` close to
    ``T`` the correction is severe and we return NaN.
    """
    if not np.isfinite(t_dm) or T <= 2 * int(horizon):
        return float("nan"), float("nan")
    h = int(horizon)
    Tn = int(T)
    factor_sq = (Tn + 1 - 2 * h + h * (h - 1) / max(Tn, 1)) / max(Tn, 1)
    if factor_sq <= 0 or not np.isfinite(factor_sq):
        return float("nan"), float("nan")
    factor = float(np.sqrt(factor_sq))
    t_star = factor * float(t_dm)
    df = max(Tn - 1, 1)
    p_two = float(2.0 * (1.0 - stats.t.cdf(abs(t_star), df=df)))
    return float(t_star), p_two


def coroneo_iacone_fixed_b_pvalue(
    t_dm: float,
    T: int,
    nlags: int,
    alternative: str = "two-sided",
) -> dict:
    """
    Coroneo & Iacone (2020) fixed-``b`` inference for DM under the Bartlett
    kernel.

    Implements the Kiefer-Vogelsang (2005, ET) closed-form approximation to
    the fixed-``b`` Bartlett t critical values, which Coroneo & Iacone (2020)
    show is a strong small-sample improvement over normal critical values
    when ``b = nlags / T`` is non-trivial.  For ``b -> 0`` this reduces to
    standard normal critical values.

    Two-sided 5% Bartlett critical value (KV-2005, Eq. 17):
    ::

        cv95(b) = 1.96 + 2.9694 b + 0.4160 b^2 - 0.5324 b^3 .

    We invert this approximation to obtain a fixed-``b`` p-value via a
    monotone parametrisation in ``b`` of the level set, calibrated at the
    1%, 5%, 10%, 90%, 95%, 99% quantiles tabulated by KV-2005, Tables I-II;
    the p-value is computed by linear interpolation in the ``log p``-domain.

    Returns ``{"t_dm": t, "b": b, "cv95_two_sided": cv, "p_value_fixed_b":
    p, "alternative": alternative}``.
    """
    nan = float("nan")
    if not np.isfinite(t_dm) or T <= 0 or nlags <= 0:
        return {"t_dm": float(t_dm), "b": nan, "cv95_two_sided": nan,
                "p_value_fixed_b": nan, "alternative": alternative}
    b = float(nlags) / float(T)
    b = float(np.clip(b, 0.0, 1.0))
    # KV (2005) critical-value polynomials for the Bartlett kernel
    # (two-sided level alpha):
    #   alpha     a0      a1       a2        a3
    coeffs = {
        0.10: (1.6449, 2.1859, 1.6532, -0.7180),
        0.05: (1.9600, 2.9694, 0.4160, -0.5324),
        0.01: (2.5758, 4.6026, -0.7253, -0.0117),
    }
    def cv(alpha_lvl: float, b_val: float) -> float:
        a0, a1, a2, a3 = coeffs[alpha_lvl]
        return a0 + a1 * b_val + a2 * b_val ** 2 + a3 * b_val ** 3

    cv90 = cv(0.10, b)
    cv95 = cv(0.05, b)
    cv99 = cv(0.01, b)
    # Reference grid of (|t|, two-sided alpha) used to interpolate p:
    grid_abs = np.array([0.0, cv90, cv95, cv99, max(cv99 * 2.0, cv99 + 1.0)])
    grid_p = np.array([1.0, 0.10, 0.05, 0.01, 0.001])
    abs_t = abs(float(t_dm))
    if abs_t <= grid_abs[0]:
        p_two = 1.0
    elif abs_t >= grid_abs[-1]:
        p_two = float(grid_p[-1] * np.exp(- (abs_t - grid_abs[-1])))
        p_two = max(p_two, 1e-8)
    else:
        log_p = np.interp(abs_t, grid_abs, np.log(grid_p))
        p_two = float(np.exp(log_p))
    if alternative == "two-sided":
        p_out = float(np.clip(p_two, 0.0, 1.0))
    elif alternative == "greater":
        p_out = float(p_two / 2.0) if t_dm >= 0 else float(1.0 - p_two / 2.0)
    else:
        p_out = float(p_two / 2.0) if t_dm <= 0 else float(1.0 - p_two / 2.0)
    return {
        "t_dm": float(t_dm),
        "b": float(b),
        "cv95_two_sided": float(cv95),
        "p_value_fixed_b": float(p_out),
        "alternative": str(alternative),
    }


def nw_long_run_variance(loss_diff: np.ndarray, nlags: int) -> tuple[float, int]:
    """
    Newey–West HAC long-run variance of a mean-zero scalar series d_t (Bartlett kernel).
    Returns (V_NW, T) with T = len(d).
    """
    d = np.asarray(loss_diff).ravel()
    T = int(d.size)
    if T < 2:
        return float("nan"), T
    gamma0 = float(np.var(d, ddof=1))
    gamma_sum = 0.0
    for lag in range(1, nlags + 1):
        w = 1.0 - lag / (nlags + 1)
        if lag < T:
            cov = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
            gamma_sum += 2.0 * w * cov
    long_run_var = gamma0 + gamma_sum
    if long_run_var <= 0:
        long_run_var = gamma0
    if long_run_var <= 0:
        return float("nan"), T
    return float(long_run_var), T


def dmw_test_detailed(
    loss_diff: np.ndarray,
    nlags: int | None = None,
    alternative: str = "two-sided",
    horizon: int = 1,
    base_factor: float = 1.5,
) -> dict:
    """
    Diebold-Mariano-West with full reporting (d_bar, V_NW, T, p-values),
    Harvey-Leybourne-Newbold (1997) finite-sample correction, and
    Coroneo-Iacone (2020) / Kiefer-Vogelsang (2005) fixed-``b`` p-value.

    If ``nlags`` is ``None`` (the recommended default) the HAC bandwidth is
    drawn from :func:`hac_lags_for_horizon` using ``horizon`` and
    ``base_factor`` (pre-registered rule).  Pass an explicit integer only
    when you really want to override the policy.

    Returns extra keys
    ``t_HLN``, ``p_HLN``, ``b_fixed``, ``p_fixed_b``, ``cv95_fixed_b``
    in addition to the legacy ``dm_stat``, ``p_value``, ``p_value_normal``.

    p_value uses :math:`t_{T-1}` (implementation choice; large-T normal
    limit as ``p_value_normal`` for sensitivity footnotes; fixed-``b``
    Bartlett critical values as ``p_fixed_b`` for small-sample inference).
    """
    d = np.asarray(loss_diff).ravel()
    T = int(d.size)
    if nlags is None:
        nlags = hac_lags_for_horizon(T=T, horizon=horizon, base_factor=base_factor)
    nlags = int(max(nlags, 1))
    if T < 2:
        nan = float("nan")
        return {
            "d_bar": nan, "V_NW": nan, "T": T, "nlags": int(nlags),
            "dm_stat": nan, "p_value": nan, "p_value_normal": nan,
            "t_HLN": nan, "p_HLN": nan,
            "b_fixed": nan, "p_fixed_b": nan, "cv95_fixed_b": nan,
            "horizon": int(horizon),
        }
    d_bar = float(d.mean())
    long_run_var, T_n = nw_long_run_variance(d, nlags)
    if not np.isfinite(long_run_var) or long_run_var <= 0:
        nan = float("nan")
        return {
            "d_bar": d_bar, "V_NW": nan, "T": T, "nlags": int(nlags),
            "dm_stat": nan, "p_value": nan, "p_value_normal": nan,
            "t_HLN": nan, "p_HLN": nan,
            "b_fixed": nan, "p_fixed_b": nan, "cv95_fixed_b": nan,
            "horizon": int(horizon),
        }
    dm_stat = float(d_bar / np.sqrt(long_run_var / T_n))
    df_t = max(T_n - 1, 1)
    if alternative == "two-sided":
        pval = float(2 * (1 - stats.t.cdf(abs(dm_stat), df=df_t)))
        pnorm = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
    elif alternative == "greater":
        pval = float(1 - stats.t.cdf(dm_stat, df=df_t))
        pnorm = float(1 - stats.norm.cdf(dm_stat))
    else:
        pval = float(stats.t.cdf(dm_stat, df=df_t))
        pnorm = float(stats.norm.cdf(dm_stat))
    t_hln, p_hln = hln_small_sample_correction(dm_stat, T_n, horizon=int(horizon))
    fb = coroneo_iacone_fixed_b_pvalue(
        dm_stat, T=T_n, nlags=nlags, alternative=alternative,
    )
    return {
        "d_bar": d_bar,
        "V_NW": float(long_run_var),
        "T": T_n,
        "nlags": int(nlags),
        "dm_stat": dm_stat,
        "p_value": pval,
        "p_value_normal": pnorm,
        "t_HLN": float(t_hln),
        "p_HLN": float(p_hln),
        "b_fixed": float(fb["b"]),
        "p_fixed_b": float(fb["p_value_fixed_b"]),
        "cv95_fixed_b": float(fb["cv95_two_sided"]),
        "horizon": int(horizon),
    }


def dmw_test(
    loss_diff: np.ndarray,
    nlags: int | None = None,
    alternative: str = "two-sided",
    horizon: int = 1,
) -> tuple[float, float]:
    """
    Diebold-Mariano-West test with Newey-West HAC long-run variance estimator.

    Computes the DM statistic as ``d_bar / sqrt(V_NW / T)`` where ``V_NW`` is
    the Bartlett-kernel HAC long-run variance with the pre-registered
    horizon-aware bandwidth ``nlags(h, T) = max(h-1, floor(1.5 T^{1/3}))``
    when ``nlags`` is left ``None``.

    loss_diff : 1-d array of d_t = L_1 - L_2.
    horizon   : forecast horizon (used to set the HAC bandwidth and the HLN
                small-sample correction).

    Returns ``(dm_stat, p_value)``. Negative ``dm_stat`` means model 1 has
    lower expected loss.
    """
    det = dmw_test_detailed(loss_diff, nlags=nlags, alternative=alternative, horizon=horizon)
    return det["dm_stat"], det["p_value"]


def dmw_test_qlike_detailed(
    true_var: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    nlags: int | None = None,
    eps: float = EPS_DEFAULT,
    horizon: int = 1,
    alternative: str = "two-sided",
) -> dict:
    """
    DM test on QLIKE loss differential with full decomposition (means + NW
    variance).  When ``nlags`` is ``None`` the HAC bandwidth is the
    pre-registered ``hac_lags_for_horizon(T, horizon)``.

    Returns dict including keys from :func:`dmw_test_detailed` plus
    ``mean_L1``, ``mean_L2``, ``mean_d``, the HLN-corrected ``t_HLN`` /
    ``p_HLN`` and the fixed-``b`` ``p_fixed_b``.
    """
    h = np.asarray(true_var, dtype=np.float64).ravel()
    h1 = np.clip(np.asarray(pred1, dtype=np.float64).ravel(), eps, None)
    h2 = np.clip(np.asarray(pred2, dtype=np.float64).ravel(), eps, None)
    h = np.maximum(h, eps)

    n = min(len(h), len(h1), len(h2))
    h, h1, h2 = h[:n], h1[:n], h2[:n]

    ratio1 = h / h1
    ratio2 = h / h2
    L1 = ratio1 - np.log(ratio1) - 1.0
    L2 = ratio2 - np.log(ratio2) - 1.0
    d = L1 - L2

    finite_mask = np.isfinite(d)
    L1f = L1[finite_mask]
    L2f = L2[finite_mask]
    d = d[finite_mask]
    nan = float("nan")
    if len(d) < 10:
        return {
            "mean_L1": nan, "mean_L2": nan, "mean_d": nan,
            "d_bar": nan, "V_NW": nan, "T": int(len(d)),
            "nlags": int(nlags) if nlags is not None else 0,
            "dm_stat": nan, "p_value": nan, "p_value_normal": nan,
            "t_HLN": nan, "p_HLN": nan,
            "b_fixed": nan, "p_fixed_b": nan, "cv95_fixed_b": nan,
            "horizon": int(horizon),
        }
    det = dmw_test_detailed(
        d, nlags=nlags, alternative=alternative, horizon=horizon,
    )
    det["mean_L1"] = float(np.mean(L1f))
    det["mean_L2"] = float(np.mean(L2f))
    det["mean_d"] = float(np.mean(d))
    return det


def dmw_test_qlike(
    true_var: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    nlags: int | None = None,
    eps: float = EPS_DEFAULT,
    horizon: int = 1,
) -> tuple[float, float]:
    """
    Diebold-Mariano-West test using the QLIKE loss differential.

    QLIKE is the preferred loss for variance forecasts because it is
    *proxy-robust*: rankings under QLIKE are preserved even when the true
    variance is replaced by a noisy realized-variance proxy (Patton 2011,
    JoE). MSE rankings are not proxy-robust and can reverse when the proxy
    is imprecise.

    Per-observation QLIKE loss for model k:
        L_k(t) = h_t / h_hat_k(t) - log(h_t / h_hat_k(t)) - 1
    where h_t = true_var (realized variance) and h_hat_k = pred_k^2.

    Loss differential: d_t = L_1(t) - L_2(t)
    DM statistic: d_bar / sqrt(V_NW / T) with Newey-West HAC.
    Negative DM stat => model 1 has lower QLIKE (better calibration).

    Parameters
    ----------
    true_var : realized variance series (same units as pred^2)
    pred1    : variance forecasts from model 1 (NOT log-variance)
    pred2    : variance forecasts from model 2 (NOT log-variance)
    nlags    : Newey-West lags (default 5; one trading week)
    eps      : floor to avoid log(0)

    Returns
    -------
    (dm_stat, p_value)  two-sided t-distribution p-value, df = T-1
    """
    det = dmw_test_qlike_detailed(true_var, pred1, pred2, nlags=nlags, eps=eps, horizon=horizon)
    return det["dm_stat"], det["p_value"]


def dm_add_fdr_columns(dm_df: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    """
    Benjamini–Hochberg FDR adjustment across all pairwise p-values in one DM table.
    Adds columns p_fdr_bh and reject_fdr_0.05.
    """
    if dm_df is None or dm_df.empty or p_col not in dm_df.columns:
        return dm_df
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        return dm_df
    out = dm_df.copy()
    p = np.asarray(out[p_col], dtype=float)
    mask = np.isfinite(p) & (p >= 0) & (p <= 1)
    p_fdr = np.full(len(p), np.nan, dtype=float)
    reject = np.full(len(p), False, dtype=bool)
    if mask.sum() > 0:
        rej, p_adj, _, _ = multipletests(p[mask], alpha=0.05, method="fdr_bh")
        p_fdr[mask] = p_adj
        reject[mask] = rej
    out["p_fdr_bh"] = p_fdr
    out["reject_fdr_0.05"] = reject
    return out


# ---------------------------------------------------------------------------
# Optional: Clark–West nested MSPE; White Reality Check; Hansen-style SPA
# (Clark & West 2007, JoE; White 2000, Econometrica; Hansen 2005, JBES)
# ---------------------------------------------------------------------------


def clark_west_nested_mspe(
    y_true: np.ndarray,
    pred_restricted: np.ndarray,
    pred_unrestricted: np.ndarray,
    nlags: int | None = None,
    eps: float = EPS_DEFAULT,
    horizon: int = 1,
) -> dict:
    """
    Clark–West (2007) statistic for **nested** models under **MSPE** loss.

    Restricted = parsimonious (nested) forecast; unrestricted = larger model.
    Uses CW adjustment term so the null is approximate equality of MSPE.

        f_t = e_{1t}^2 - (e_{2t}^2 - (\\hat y_{2t} - \\hat y_{1t})^2)

    with e_{jt} = y_t - \\hat y_{jt}. Asymptotically standard normal (one-sided
    upper tail: large CW rejects in favor of the unrestricted model).

    Parameters
    ----------
    y_true : realized variance (same units as predictions).
    pred_restricted, pred_unrestricted : variance forecasts (nested: restricted ⊂ unrestricted).
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    f1 = np.asarray(pred_restricted, dtype=np.float64).ravel()
    f2 = np.asarray(pred_unrestricted, dtype=np.float64).ravel()
    y = np.maximum(y, eps)
    f1 = np.maximum(f1, eps)
    f2 = np.maximum(f2, eps)
    n = min(len(y), len(f1), len(f2))
    y, f1, f2 = y[:n], f1[:n], f2[:n]
    e1 = y - f1
    e2 = y - f2
    f_cw = e1 * e1 - (e2 * e2 - (f2 - f1) ** 2)
    m = np.isfinite(f_cw)
    f_cw = f_cw[m]
    if nlags is None:
        nlags = hac_lags_for_horizon(T=int(len(f_cw)), horizon=int(horizon))
    nlags = int(max(nlags, 1))
    if len(f_cw) < 10:
        return {
            "cw_stat": float("nan"), "p_value_one_sided": float("nan"),
            "mean_f_cw": float("nan"), "T": int(len(f_cw)), "nlags": int(nlags),
        }
    lrv, t_n = nw_long_run_variance(f_cw, nlags)
    mean_f = float(np.mean(f_cw))
    if not np.isfinite(lrv) or lrv <= 0:
        return {
            "cw_stat": float("nan"), "p_value_one_sided": float("nan"),
            "mean_f_cw": mean_f, "T": int(t_n), "nlags": int(nlags),
        }
    cw = float(mean_f / np.sqrt(lrv / t_n))
    p_one = float(1.0 - stats.norm.cdf(cw)) if np.isfinite(cw) else float("nan")
    return {
        "cw_stat": cw,
        "p_value_one_sided": p_one,
        "mean_f_cw": mean_f,
        "T": int(t_n),
        "nlags": int(nlags),
    }


def _circular_block_resample_rows(X: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Joint circular block bootstrap of rows of X (T, d)."""
    T = X.shape[0]
    b = max(1, int(block_len))
    n_blocks = int(np.ceil(T / b))
    starts = rng.integers(0, T, size=n_blocks)
    parts = []
    for s in starts:
        parts.append(X[(np.arange(s, s + b) % T), :])
    out = np.vstack(parts)[:T]
    return out


def white_reality_check_bootstrap(
    loss_matrix: np.ndarray,
    benchmark_col: int = 0,
    n_boot: int = 1999,
    block_len: int = 22,
    seed: int = 42,
) -> dict:
    """
    White (2000) Reality Check style **upper-tail** block-bootstrap p-value.

    loss_matrix : (T, K) with **lower loss better**. Column `benchmark_col` is
    the benchmark; alternatives are other columns.

    Statistic: V = max_{k != b} ( mean(L_bench - L_k) ). Large V ⇒ some
    alternative beats the benchmark on average (MSPE or any loss).

    Bootstrap resamples **rows jointly** (circular blocks) to preserve temporal
    dependence; p-value = fraction of bootstrap draws with V* >= V_obs.
    """
    L = np.asarray(loss_matrix, dtype=np.float64)
    T, K = L.shape
    if K < 2 or T < 20:
        return {"V_stat": float("nan"), "p_value": float("nan"), "means_vs_bench": {}}
    rng = np.random.default_rng(seed)
    L0 = L[:, benchmark_col]
    means = []
    for k in range(K):
        if k == benchmark_col:
            continue
        means.append(float(np.mean(L0 - L[:, k])))
    V_obs = float(max(means)) if means else float("nan")
    V_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx_mat = _circular_block_resample_rows(L, block_len, rng)
        m_b = []
        for k in range(K):
            if k == benchmark_col:
                continue
            m_b.append(float(np.mean(idx_mat[:, benchmark_col] - idx_mat[:, k])))
        V_boot[b] = max(m_b) if m_b else float("nan")
    p_val = float(np.mean(V_boot >= V_obs)) if np.isfinite(V_obs) else float("nan")
    alt_ks = [j for j in range(K) if j != benchmark_col]
    names_means = {f"col_{alt_ks[i]}": means[i] for i in range(len(alt_ks))}
    return {"V_stat": V_obs, "p_value": p_val, "means_vs_bench": names_means, "T": T, "K": K}


def romano_wolf_step_down(
    loss_matrix: np.ndarray,
    *,
    benchmark_col: int = 0,
    model_names: list[str] | None = None,
    n_boot: int = 9999,
    block_len: int | None = None,
    seed: int = 42,
    alpha: float = 0.05,
    alternative: str = "greater",
    horizon: int = 1,
) -> dict:
    """
    Romano-Wolf (2005, 2016) step-down family-wise error control over the
    family of "alternative `k` beats benchmark" hypotheses,
    :math:`H_{0,k}: \\mathbb{E}[L_{\\text{bench},t} - L_{k,t}] \\le 0`,
    based on studentized loss differentials.

    For each alternative ``k`` (column index ``k != benchmark_col``) the
    studentized t-statistic is
    ::

        t_k = bar(d_k) / sqrt(V_NW(d_k) / T) ,

    with ``d_{k,t} = L_{bench,t} - L_{k,t}`` (positive => alternative
    *beats* benchmark) and Newey-West HAC ``V_NW`` at the pre-registered
    horizon-aware bandwidth.

    The step-down recursion is the classical RW iteration: at each step we
    bootstrap the joint maximum of the *centered* studentized statistics
    over the *current* set of yet-unrejected hypotheses; reject any with
    observed t exceeding the bootstrap upper quantile, and iterate.

    Parameters
    ----------
    loss_matrix    : (T, K) per-observation losses, lower=better.
    benchmark_col  : column of the benchmark (0 by default).
    n_boot         : number of bootstrap replicates (9 999 is the paper
                     default for FWER ≈ 0.05).
    block_len      : stationary-bootstrap mean block length; if ``None``
                     the Politis-White (2004) default ``1.75 * T^{1/3}``
                     (capped at ``T/3``) is used.
    seed           : RNG seed.
    alpha          : nominal FWER level.
    alternative    : ``"greater"`` (paper default; alternative beats
                     benchmark in the upper tail) or ``"two-sided"``.

    Returns
    -------
    dict with
      ``names``            list of K-1 alternative column labels,
      ``t_obs``            observed studentized t-statistics (length K-1),
      ``p_rw``             RW-adjusted FWER p-values (length K-1),
      ``reject``           bool array (length K-1) at level ``alpha``,
      ``rejected_models``  ordered list of rejected names,
      ``block_len``, ``n_boot``, ``alpha``, ``alternative``, ``T``.
    """
    L = np.asarray(loss_matrix, dtype=np.float64)
    if L.ndim != 2 or L.shape[1] < 2 or L.shape[0] < 30:
        return {"names": [], "t_obs": [], "p_rw": [], "reject": [],
                "rejected_models": [], "block_len": int(block_len or 0),
                "n_boot": int(n_boot), "alpha": float(alpha),
                "alternative": str(alternative), "T": int(L.shape[0])}
    T, K = L.shape
    L0 = L[:, benchmark_col]
    alt_cols = [k for k in range(K) if k != benchmark_col]
    if model_names is None:
        names = [f"col_{k}" for k in alt_cols]
    else:
        names = [str(model_names[k]) for k in alt_cols]
    d = np.column_stack([L0 - L[:, k] for k in alt_cols])  # (T, K-1)
    finite_mask = np.all(np.isfinite(d), axis=1)
    d = d[finite_mask]
    T_eff = int(d.shape[0])
    if T_eff < 30:
        return {"names": names, "t_obs": [float("nan")] * len(names),
                "p_rw": [float("nan")] * len(names),
                "reject": [False] * len(names), "rejected_models": [],
                "block_len": int(block_len or 0), "n_boot": int(n_boot),
                "alpha": float(alpha), "alternative": str(alternative), "T": T_eff}

    nlags = hac_lags_for_horizon(T=T_eff, horizon=int(horizon))

    def _t_studentized(arr: np.ndarray) -> np.ndarray:
        out = np.empty(arr.shape[1], dtype=np.float64)
        for j in range(arr.shape[1]):
            col = arr[:, j]
            mu = float(np.mean(col))
            lrv, tn = nw_long_run_variance(col, nlags)
            if not np.isfinite(lrv) or lrv <= 0 or tn <= 0:
                out[j] = float("nan")
            else:
                out[j] = float(mu / np.sqrt(lrv / tn))
        return out

    t_obs = _t_studentized(d)
    if alternative not in ("greater", "two-sided"):
        raise ValueError("alternative must be 'greater' or 'two-sided'")

    if block_len is None:
        block_len = int(np.clip(np.floor(1.75 * T_eff ** (1.0 / 3.0)),
                                 1, max(T_eff // 3, 1)))
    block_len = int(max(int(block_len), 1))

    rng = np.random.default_rng(int(seed))
    p_geom = 1.0 / float(block_len)
    centered = d - d.mean(axis=0, keepdims=True)
    K_alt = d.shape[1]
    boot_max = np.full((n_boot, K_alt), np.nan, dtype=np.float64)
    for b in range(n_boot):
        idx = np.empty(T_eff, dtype=np.int64)
        i = 0
        while i < T_eff:
            start = int(rng.integers(0, T_eff))
            length = int(rng.geometric(p_geom)) if p_geom < 1.0 else 1
            length = min(length, T_eff - i)
            for k_ in range(length):
                idx[i + k_] = (start + k_) % T_eff
            i += length
        d_b = centered[idx, :]
        boot_max[b, :] = _t_studentized(d_b)

    if alternative == "two-sided":
        boot_max = np.abs(boot_max)
        t_used = np.abs(t_obs)
    else:
        t_used = t_obs

    p_rw = np.full(K_alt, np.nan, dtype=np.float64)
    reject = np.zeros(K_alt, dtype=bool)
    remaining = np.arange(K_alt)
    while remaining.size > 0:
        boot_subset = boot_max[:, remaining]
        max_subset = np.nanmax(boot_subset, axis=1)
        t_sub = t_used[remaining]
        finite = np.isfinite(t_sub)
        if not finite.any():
            break
        order = np.argsort(-t_sub)
        argmax_idx = order[0]
        idx_global = int(remaining[argmax_idx])
        t_max = float(t_sub[argmax_idx])
        if not np.isfinite(t_max):
            break
        max_subset = max_subset[np.isfinite(max_subset)]
        if max_subset.size == 0:
            break
        p_val = float(np.mean(max_subset >= t_max))
        # Monotonicity adjustment: RW-adjusted p-values are non-decreasing
        # in the order of rejection.
        prev_p = float(np.nanmax(p_rw[reject])) if reject.any() else 0.0
        p_rw[idx_global] = max(prev_p, p_val)
        if p_rw[idx_global] <= alpha:
            reject[idx_global] = True
            remaining = remaining[remaining != idx_global]
        else:
            for k_ in remaining:
                if not reject[k_]:
                    p_rw[k_] = max(prev_p, p_val)
            break

    rejected_models = [names[i] for i in range(K_alt) if reject[i]]
    return {
        "names": names,
        "t_obs": [float(x) for x in t_obs],
        "p_rw": [float(x) for x in p_rw],
        "reject": [bool(x) for x in reject],
        "rejected_models": rejected_models,
        "block_len": int(block_len),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "alternative": str(alternative),
        "T": int(T_eff),
        "nlags": int(nlags),
        "horizon": int(horizon),
    }


def hansen_spa_studentized_bootstrap(
    loss_matrix: np.ndarray,
    benchmark_col: int = 0,
    nlags: int | None = None,
    n_boot: int = 1999,
    block_len: int = 22,
    seed: int = 42,
    horizon: int = 1,
) -> dict:
    """
    Hansen (2005) **studentized** SPA-style block bootstrap (simplified).

    For each alternative k, d_{t,k} = L_bench,t - L_{k,t} (positive ⇒ k better).
    Studentized t_k = mean(d_k) / sqrt( NW_LRV(d_k) / T ). Test statistic
    T_SPA = max_k t_k. Bootstrap: **joint** circular block resampling of d,
    recomputing the same studentization on each draw (dependence-aware).

    For publication-critical values compare against Hansen (2005, JBES)
    when strict control is required; this is a practicable code-path default.
    """
    L = np.asarray(loss_matrix, dtype=np.float64)
    T, K = L.shape
    if K < 2 or T < 30:
        return {"T_spa": float("nan"), "p_value": float("nan"), "t_by_alt": {}}
    L0 = L[:, benchmark_col]
    alt_cols = [k for k in range(K) if k != benchmark_col]
    d = np.column_stack([L0 - L[:, k] for k in alt_cols])  # (T, K-1)
    m = np.all(np.isfinite(d), axis=1)
    d = d[m]
    T2 = d.shape[0]
    if T2 < 30:
        return {"T_spa": float("nan"), "p_value": float("nan"), "t_by_alt": {}}
    if nlags is None:
        nlags = hac_lags_for_horizon(T=T2, horizon=int(horizon))
    nlags = int(max(nlags, 1))
    rng = np.random.default_rng(seed)
    t_obs_list = []
    for j in range(d.shape[1]):
        lrv, tn = nw_long_run_variance(d[:, j], nlags)
        se = np.sqrt(lrv / tn) if lrv > 0 else float("nan")
        t_obs_list.append(float(np.mean(d[:, j]) / se) if se > 0 else float("nan"))
    T_obs = float(np.nanmax(t_obs_list))
    T_boot = np.empty(n_boot)
    for b in range(n_boot):
        db = _circular_block_resample_rows(d, block_len, rng)
        tb = []
        for j in range(db.shape[1]):
            lrv, tn = nw_long_run_variance(db[:, j], nlags)
            se = np.sqrt(lrv / tn) if lrv > 0 else float("nan")
            tb.append(float(np.mean(db[:, j]) / se) if se > 0 else float("nan"))
        T_boot[b] = float(np.nanmax(tb))
    p_val = float(np.mean(T_boot >= T_obs)) if np.isfinite(T_obs) else float("nan")
    t_by_alt = {f"col_{alt_cols[j]}": t_obs_list[j] for j in range(len(alt_cols))}
    return {"T_spa": T_obs, "p_value": p_val, "t_by_alt": t_by_alt, "T": T2, "nlags": nlags}


def hac_ols_active_columns(
    y: np.ndarray,
    X_active: np.ndarray,
    feature_labels: list[str],
    nlags: int = 5,
) -> dict:
    """
    OLS of y on [const, X_active] with Newey–West HAC (maxlags=nlags).

    Intended for **split-sample** inference after selection on a disjoint
    training slice (post-selection HAC; not debiased high-dimensional lasso).
    """
    import statsmodels.api as sm_ols

    yv = np.asarray(y, dtype=np.float64).ravel()
    Xa = np.asarray(X_active, dtype=np.float64)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    n = min(len(yv), Xa.shape[0])
    yv, Xa = yv[:n], Xa[:n]
    m = np.isfinite(yv) & np.all(np.isfinite(Xa), axis=1)
    yv, Xa = yv[m], Xa[m]
    if len(yv) < 10 or Xa.shape[1] == 0:
        return {"params": {}, "table": []}
    Z = sm_ols.add_constant(Xa, has_constant="add")
    try:
        res = sm_ols.OLS(yv, Z).fit(cov_type="HAC", cov_kwds={"maxlags": nlags})
    except Exception:
        return {"params": {}, "table": []}
    names = ["const"] + list(feature_labels)
    rows = []
    for i, nm in enumerate(names):
        if i >= len(res.params):
            break
        rows.append({
            "name": nm,
            "coef": float(res.params[i]),
            "se_hac": float(res.bse[i]),
            "t": float(res.tvalues[i]),
            "p": float(res.pvalues[i]),
        })
    return {"params": {r["name"]: r["coef"] for r in rows}, "table": rows, "r2": float(res.rsquared)}


def compute_mcs(
    loss_matrix: np.ndarray,
    model_names: list[str],
    alpha: float = 0.10,
    n_boot: int = 1000,
    block_size: int = 5,
    seed: int = 42,
) -> dict:
    """
    Model Confidence Set (MCS) procedure — Hansen, Lunde & Nason (2011, Econometrica).

    The MCS identifies the smallest set M* of models that contains the best
    model with probability >= 1 - alpha. It eliminates models iteratively
    using a range test T_R with bootstrap critical values, making no distributional
    assumptions about forecast errors.

    Algorithm
    ---------
    1. Compute pairwise loss differentials d_{ij,t} = L_{i,t} - L_{j,t}.
    2. T_R statistic = max_{i,j in M} |t_{ij}| where t_{ij} = d_bar_{ij} / se(d_bar_{ij}).
       Standard error estimated with Newey-West HAC (nlags = block_size).
    3. Bootstrap critical value: resample T by circular block bootstrap,
       recompute T_R on each bootstrap sample. p-value = fraction of
       bootstrap statistics >= observed T_R.
    4. If p-value < alpha: eliminate the model with the highest mean loss;
       repeat from step 1 with the reduced set.
    5. Return the surviving set and associated p-values.

    Parameters
    ----------
    loss_matrix : (T, n_models) array of per-period losses for each model.
                  Lower is better. Use squared errors for MSE-MCS or QLIKE
                  values for QLIKE-MCS.
    model_names : list of model name strings, length n_models.
    alpha       : confidence level for exclusion (default 0.10 -> 90% MCS).
    n_boot      : number of bootstrap replications.
    block_size  : circular block length for bootstrap (default 5 ~ 1 week).
    seed        : random seed for reproducibility.

    Returns
    -------
    dict with keys:
        "mcs_set"     : list of model names in the MCS
        "p_values"    : {model_name: p_value_at_elimination} for all models.
                        Survivors that remain when the procedure stops because
                        p_val >= alpha all receive the **same** last_p (last
                        non-rejection round); eliminated models retain their
                        elimination-round p-values.
        "included"    : {model_name: bool} membership in final MCS
    """
    rng = np.random.default_rng(seed)
    L = np.asarray(loss_matrix, dtype=np.float64)
    T, n = L.shape
    if n != len(model_names):
        raise ValueError("loss_matrix columns must match len(model_names)")

    # Replace non-finite losses with row-mean (should be rare; prevents bootstrap collapse)
    for t in range(T):
        row_fin = np.isfinite(L[t])
        if row_fin.sum() > 0 and not row_fin.all():
            L[t, ~row_fin] = L[t, row_fin].mean()

    active = list(range(n))
    p_values: dict[int, float] = {}

    def _t_stat_nw(d_series: np.ndarray, nlags: int) -> float:
        """t-statistic for H0: E[d] = 0 with Newey-West HAC."""
        d = d_series - d_series.mean()
        n_d = len(d_series)
        gamma0 = np.var(d_series, ddof=1)
        gamma_sum = 0.0
        for lag in range(1, nlags + 1):
            w = 1.0 - lag / (nlags + 1)
            if lag < n_d:
                gamma_sum += 2.0 * w * np.cov(d_series[lag:], d_series[:-lag], ddof=1)[0, 1]
        lrv = max(gamma0 + gamma_sum, 1e-30)
        return float(d_series.mean() / np.sqrt(lrv / n_d))

    def _T_R(loss_sub: np.ndarray) -> float:
        """Range test statistic: max_{i,j} |t_{ij}| over pairwise differentials."""
        n_sub = loss_sub.shape[1]
        if n_sub <= 1:
            return 0.0
        t_max = 0.0
        for i in range(n_sub):
            for j in range(i + 1, n_sub):
                diff = loss_sub[:, i] - loss_sub[:, j]
                t_ij = abs(_t_stat_nw(diff, block_size))
                if t_ij > t_max:
                    t_max = t_ij
        return t_max

    def _circular_block_bootstrap(loss_sub: np.ndarray, B: int, b: int) -> np.ndarray:
        """
        Circular block bootstrap of the mean-demeaned loss matrix.
        Returns (B,) array of T_R statistics.
        """
        T_sub = loss_sub.shape[0]
        # Demean columnwise so bootstrap captures dependence only, not level
        demeaned = loss_sub - loss_sub.mean(axis=0, keepdims=True)
        n_blocks = int(np.ceil(T_sub / b))
        t_r_boot = np.empty(B)
        for rep in range(B):
            starts = rng.integers(0, T_sub, size=n_blocks)
            resampled_rows = []
            for s in starts:
                idx = np.arange(s, s + b) % T_sub
                resampled_rows.append(demeaned[idx])
            boot_sample = np.vstack(resampled_rows)[:T_sub]
            t_r_boot[rep] = _T_R(boot_sample)
        return t_r_boot

    last_p = 1.0
    while len(active) > 1:
        L_active = L[:, active]
        t_obs = _T_R(L_active)
        boot_stats = _circular_block_bootstrap(L_active, n_boot, block_size)
        p_val = float((boot_stats >= t_obs).mean())

        if p_val >= alpha:
            # Fail to reject; all remaining models are in the MCS
            last_p = p_val
            break

        # Eliminate the model with the highest mean loss
        mean_losses = L_active.mean(axis=0)
        worst_local = int(np.argmax(mean_losses))
        worst_global = active[worst_local]
        p_values[worst_global] = p_val
        active.remove(worst_global)
        last_p = p_val

    # Assign last p-value to survivors
    for idx in active:
        p_values[idx] = last_p

    mcs_names = [model_names[i] for i in active]
    included = {model_names[i]: (i in active) for i in range(n)}
    p_vals_named = {model_names[i]: p_values.get(i, float("nan")) for i in range(n)}

    return {
        "mcs_set": mcs_names,
        "p_values": p_vals_named,
        "included": included,
    }


def compute_mcs_block_sweep(
    loss_matrix: np.ndarray,
    model_names: list[str],
    block_lengths: list[int] | None = None,
    *,
    horizon: int = 1,
    alpha: float = 0.10,
    n_boot: int = 1999,
    seed: int = 42,
) -> dict:
    """
    Robustness sweep of :func:`compute_mcs` over a grid of bootstrap block
    lengths.

    Reports the surviving MCS at each block length so the headline MCS does
    not silently depend on a single ``block_size`` choice.  Defaults to
    ``[h, 2h, 4h, max(22, h), max(63, h)]`` (daily, fortnight, ~quarter)
    deduplicated and ordered.

    Returns a dict keyed by block length (int) and a ``"meta"`` entry with
    horizon, alpha, n_boot, seed.
    """
    if block_lengths is None:
        h = int(max(horizon, 1))
        candidates = sorted({h, 2 * h, 4 * h, max(22, h), max(63, h)})
        block_lengths = candidates
    out: dict = {"meta": {"horizon": int(horizon), "alpha": float(alpha),
                          "n_boot": int(n_boot), "seed": int(seed)}}
    for bl in block_lengths:
        bl_int = int(max(int(bl), 1))
        try:
            mcs = compute_mcs(
                loss_matrix=np.asarray(loss_matrix),
                model_names=list(model_names),
                alpha=float(alpha),
                n_boot=int(n_boot),
                block_size=bl_int,
                seed=int(seed),
            )
            out[str(bl_int)] = {
                "block_size": bl_int,
                "mcs_set": list(mcs["mcs_set"]),
                "p_values": dict(mcs["p_values"]),
                "included": dict(mcs["included"]),
            }
        except Exception as exc:
            out[str(bl_int)] = {"block_size": bl_int, "error": str(exc)}
    return out


def performance_by_regime_table(
    df_eval: pd.DataFrame,
    pred_columns: list[str],
    true_col: str,
    regime_col: str,
    eps: float = EPS_DEFAULT,
) -> pd.DataFrame:
    """
    For each (model, regime), compute RMSE, R2, MAPE, QLIKE.
    regime_col: column with values 'Crisis' or 'Calm' (date-based or other definition).
    Used for Table: Performance by Regime (Crisis vs Calm).
    """
    rows = []
    for model in pred_columns:
        if model not in df_eval.columns:
            continue
        y_true = df_eval[true_col].values
        y_pred = df_eval[model].values
        for regime_name in df_eval[regime_col].dropna().unique():
            mask = (df_eval[regime_col] == regime_name).values
            if mask.sum() < 2:
                continue
            rmse, r2, male, qlike = compute_metrics(
                y_true[mask], y_pred[mask], eps
            )
            rows.append({
                "Model": model,
                "Regime": regime_name,
                "RMSE": rmse,
                "R2": r2,
                "MALE": male,
                "QLIKE": qlike,
            })
    return pd.DataFrame(rows)


def threshold_sensitivity_table(
    df_eval: pd.DataFrame,
    pred_columns: list[str],
    true_col: str,
    crisis_cols: dict[float, str],
    eps: float = EPS_DEFAULT,
) -> pd.DataFrame:
    """
    For each (model, threshold, regime), compute RMSE, R2, MAPE, QLIKE.
    df_eval: index=date, columns include true_col, pred_columns, and for each threshold
             a column crisis_cols[th] with 0/1 (1 = crisis).
    crisis_cols: e.g. {0.8: "crisis_0.8"}.
    """
    rows = []
    for model in pred_columns:
        if model not in df_eval.columns:
            continue
        y_true = df_eval[true_col].values
        y_pred = df_eval[model].values
        for th, col in crisis_cols.items():
            if col not in df_eval.columns:
                continue
            crisis_mask = (df_eval[col] == 1).values
            calm_mask = ~crisis_mask
            for regime_name, mask in [("Crisis", crisis_mask), ("Calm", calm_mask)]:
                if mask.sum() < 2:
                    continue
                rmse, r2, male, qlike = compute_metrics(
                    y_true[mask], y_pred[mask], eps
                )
                rows.append({
                    "Model": model,
                    "Threshold": th,
                    "Regime": regime_name,
                    "RMSE": rmse,
                    "R2": r2,
                    "MALE": male,
                    "QLIKE": qlike,
                })
    return pd.DataFrame(rows)


def generate_figures(
    df_eval: pd.DataFrame,
    out_dir: Path,
    true_col: str = "TrueVol",
    pred_columns: list[str] | None = None,
    crisis_col: str = "crisis_0.8",
    crisis_window_pct: float = 0.15,
    calm_window_pct: float = 0.15,
) -> None:
    """
    Generate publication figures: full sample, crisis window, calm window,
    scatter actual vs predicted, residuals, threshold sensitivity (if sens_df provided).
    Saves to out_dir (e.g. results_revised/figures/).
    """
    if not _HAS_MPL:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if pred_columns is None:
        pred_columns = [c for c in ["HAR", "HAR+SVD", "DNN_HAR", "DNN_HAR+SVD", "LSTM_HAR", "LSTM_HAR+SVD"] if c in df_eval.columns]
    pred_columns = [c for c in pred_columns if c in df_eval.columns]
    if true_col not in df_eval.columns or not pred_columns:
        return
    y_true = df_eval[true_col].values
    index = df_eval.index
    n = len(y_true)

    # 1. Full sample: actual vs predicted (time series)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(n), y_true, label="Actual", color="black", alpha=0.8, linewidth=0.8)
    for i, col in enumerate(pred_columns[:4]):
        y_pred = df_eval[col].values
        valid = np.isfinite(y_pred)
        if valid.sum() > 0:
            ax.plot(np.where(valid)[0], y_pred[valid], label=col, alpha=0.7, linewidth=0.6)
    ax.set_xlabel("Test observation")
    ax.set_ylabel("Realized variance (%²)")
    ax.set_title("Full sample: Actual vs predicted variance")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "full_sample_actual_vs_predicted.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Crisis window (last crisis_pct of test set where crisis=1, or a contiguous crisis block)
    if crisis_col in df_eval.columns:
        crisis_mask = (df_eval[crisis_col] == 1).values
        if crisis_mask.sum() > 20:
            crisis_idx = np.where(crisis_mask)[0]
            start = max(0, crisis_idx[0] - 5)
            end = min(n, crisis_idx[-1] + 6)
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(range(start, end), y_true[start:end], label="Actual", color="black", linewidth=0.9)
            for col in pred_columns[:4]:
                y_p = df_eval[col].values[start:end]
                valid = np.isfinite(y_p)
                if valid.sum() > 0:
                    ax.plot(np.arange(start, end)[valid], y_p[valid], label=col, alpha=0.7)
            ax.set_xlabel("Test observation")
            ax.set_ylabel("Realized variance (%²)")
            ax.set_title("Crisis window: Actual vs predicted variance")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "crisis_window_actual_vs_predicted.pdf", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # 3. Calm window (first calm_pct where crisis=0)
    if crisis_col in df_eval.columns:
        calm_mask = (df_eval[crisis_col] == 0).values
        if calm_mask.sum() > 20:
            calm_idx = np.where(calm_mask)[0]
            start = calm_idx[0]
            end = min(n, start + int(n * calm_window_pct) + 50)
            end = min(end, calm_idx[-1] + 1)
            if end - start > 20:
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.plot(range(start, end), y_true[start:end], label="Actual", color="black", linewidth=0.9)
                for col in pred_columns[:4]:
                    y_p = df_eval[col].values[start:end]
                    valid = np.isfinite(y_p)
                    if valid.sum() > 0:
                        ax.plot(np.arange(start, end)[valid], y_p[valid], label=col, alpha=0.7)
                ax.set_xlabel("Test observation")
                ax.set_ylabel("Realized variance (%²)")
                ax.set_title("Calm window: Actual vs predicted variance")
                ax.legend(loc="best", fontsize=8)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(out_dir / "calm_window_actual_vs_predicted.pdf", dpi=150, bbox_inches="tight")
                plt.close(fig)

    # 4. Scatter: actual vs predicted (one panel per model or combined)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()
    for idx, col in enumerate(pred_columns[:4]):
        ax = axes[idx] if idx < 4 else axes[-1]
        y_p = df_eval[col].values
        valid = np.isfinite(y_p) & np.isfinite(y_true)
        if valid.sum() > 2:
            ax.scatter(y_true[valid], y_p[valid], alpha=0.3, s=5)
            lims = [min(y_true[valid].min(), y_p[valid].min()), max(y_true[valid].max(), y_p[valid].max())]
            ax.plot(lims, lims, "k--", alpha=0.5, label="45°")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(col)
            ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    for j in range(len(pred_columns[:4]), 4):
        axes[j].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_actual_vs_predicted.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Residuals for best model (first non-HAR that has valid preds, or HAR+SVD)
    for col in ["DNN_HAR+SVD", "LSTM_HAR+SVD", "HAR+SVD", "HAR"]:
        if col not in df_eval.columns:
            continue
        y_p = df_eval[col].values
        valid = np.isfinite(y_p) & np.isfinite(y_true)
        if valid.sum() < 10:
            continue
        res = y_true[valid] - y_p[valid]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].hist(res, bins=50, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Residual")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Residuals histogram ({col})")
        axes[1].plot(res, alpha=0.7)
        axes[1].set_xlabel("Test observation")
        axes[1].set_ylabel("Residual")
        axes[1].set_title(f"Residuals over time ({col})")
        axes[1].axhline(0, color="gray", linestyle="--")
        fig.tight_layout()
        fig.savefig(out_dir / f"residuals_{col.replace('+', '_')}.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)
        break


def plot_threshold_sensitivity(
    sens_df: pd.DataFrame,
    out_dir: Path,
    metric: str = "R2",
    models: list[str] | None = None,
) -> None:
    """Plot metric (e.g. R2 or RMSE) vs crisis threshold, by regime (Crisis/Calm)."""
    if not _HAS_MPL or sens_df.empty:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "Threshold" not in sens_df.columns or "Regime" not in sens_df.columns or metric not in sens_df.columns:
        return
    if models is None:
        models = sens_df["Model"].unique().tolist()
    fig, ax = plt.subplots(figsize=(7, 4))
    for regime in ["Crisis", "Calm"]:
        sub = sens_df[sens_df["Regime"] == regime]
        if sub.empty:
            continue
        for model in models:
            msub = sub[sub["Model"] == model]
            if msub.empty:
                continue
            ax.plot(msub["Threshold"], msub[metric], "o-", label=f"{model} ({regime})", alpha=0.8, markersize=4)
    ax.set_xlabel("Crisis threshold (cos θ)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs crisis threshold by regime")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"threshold_sensitivity_{metric}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_loss_curves(
    hist_dnn: object,
    hist_lstm: object,
    out_dir: Path,
) -> None:
    """Plot train/val loss vs epoch (log scale) for DNN and LSTM (horizon 1)."""
    if not _HAS_MPL:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if hist_dnn is not None and hasattr(hist_dnn, "history"):
        h = hist_dnn.history
        fig, ax = plt.subplots(figsize=(8, 4))
        epochs = range(1, len(h.get("loss", [])) + 1)
        if h.get("loss"):
            ax.semilogy(epochs, h["loss"], label="Train (DNN HAR+SVD)", alpha=0.8)
        if h.get("val_loss"):
            ax.semilogy(epochs, h["val_loss"], label="Val (DNN HAR+SVD)", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training loss curves (DNN HAR+SVD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "training_loss_dnn.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)
    if hist_lstm is not None and hasattr(hist_lstm, "history"):
        h = hist_lstm.history
        fig, ax = plt.subplots(figsize=(8, 4))
        epochs = range(1, len(h.get("loss", [])) + 1)
        if h.get("loss"):
            ax.semilogy(epochs, h["loss"], label="Train (LSTM HAR+SVD)", alpha=0.8)
        if h.get("val_loss"):
            ax.semilogy(epochs, h["val_loss"], label="Val (LSTM HAR+SVD)", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training loss curves (LSTM HAR+SVD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "training_loss_lstm.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_feature_importance(
    model,
    feature_names: list[str],
    out_dir: Path,
    title: str = "Feature importance (DNN HAR+SVD, first layer)",
) -> None:
    """Bar plot of mean absolute first-layer weights; optionally check SVD features in top 3."""
    if not _HAS_MPL or model is None or not feature_names:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    first_layer = None
    for layer in model.layers:
        if "dense" in layer.name.lower() and "out" not in layer.name:
            w = layer.get_weights()
            if w and len(w[0].shape) == 2 and w[0].shape[0] == len(feature_names):
                first_layer = w[0]
                break
    if first_layer is None:
        return
    imp = np.abs(first_layer).mean(axis=1)
    if len(imp) != len(feature_names):
        feature_names = feature_names[: len(imp)]
    fig, ax = plt.subplots(figsize=(8, 4))
    idx = np.argsort(imp)[::-1]
    ax.bar(range(len(imp)), imp[idx], color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
    ax.set_ylabel("Mean |weight|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance_dnn_har_svd.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_crisis_windows(
    df_eval: pd.DataFrame,
    crisis_windows: list[tuple[str, str]],
    pred_columns: list[str],
    true_col: str = "TrueVol",
    out_dir: Path = None,
) -> None:
    """For each (start, end) in crisis_windows, plot actual vs predicted for that date range; plus one calm sub-sample."""
    if not _HAS_MPL or df_eval.empty or true_col not in df_eval.columns:
        return
    out_dir = Path(out_dir) if out_dir else Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = df_eval.index
    y_true = df_eval[true_col].values
    for start, end in crisis_windows:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        mask = (idx >= start_ts) & (idx <= end_ts)
        if mask.sum() < 5:
            continue
        sub = df_eval.loc[mask]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(range(len(sub)), sub[true_col].values, label="Actual", color="black", linewidth=0.9)
        for col in pred_columns[:4]:
            if col not in sub.columns:
                continue
            v = sub[col].values
            if np.isfinite(v).any():
                ax.plot(range(len(sub)), v, label=col, alpha=0.7)
        ax.set_xlabel("Observation")
        ax.set_ylabel("Realized variance (%²)")
        ax.set_title(f"Crisis window {start} to {end}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = f"crisis_window_{start}_{end}.pdf".replace(" ", "_")
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    # Calm sub-sample: first chunk of test set where Regime_date == 'Calm' if available
    if "Regime_date" in df_eval.columns:
        calm_mask = (df_eval["Regime_date"] == "Calm").values
        if calm_mask.sum() > 30:
            calm_idx = np.where(calm_mask)[0][: min(100, calm_mask.sum())]
            sub = df_eval.iloc[calm_idx]
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(range(len(sub)), sub[true_col].values, label="Actual", color="black", linewidth=0.9)
            for col in pred_columns[:4]:
                if col not in sub.columns:
                    continue
                v = sub[col].values
                if np.isfinite(v).any():
                    ax.plot(range(len(sub)), v, label=col, alpha=0.7)
            ax.set_xlabel("Observation")
            ax.set_ylabel("Realized variance (%²)")
            ax.set_title("Calm sub-sample")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "calm_subsample.pdf", dpi=150, bbox_inches="tight")
            plt.close(fig)


# ===========================================================================
# F1: Posterior Predictive Forecast
# ===========================================================================

def plot_posterior_forecast(
    df_eval: pd.DataFrame,
    uncertainty: dict,
    crisis_windows: list,
    out_dir: Path,
    true_col: str = "TrueVol",
) -> None:
    """
    F1: 6-panel posterior predictive forecast plot.

    Each panel (M1-M6) shows:
        - Actual realized variance (%²) (black line)
        - Model mean forecast (coloured line)
        - 95% predictive band (shaded)
        - Crisis windows (light red background)

    Bayesian / empirical predictive intervals:
        HAR           -- conjugate Normal-InvGamma Student-t (OLS HAR; same fit sample as M1).
        HAR+SVD       -- block-bootstrap over training (ElasticNet M2; not conjugate).
        DNN+SVD       -- MC Dropout
        LSTM+SVD      -- MC Dropout
        DNN_HAR, LSTM_HAR -- no band

    Y-axis is on log scale for better spike visibility.
    """
    if not _HAS_MPL:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if true_col not in df_eval.columns:
        return

    # Map model labels to (column name in df_eval, uncertainty key)
    panel_models = [
        ("HAR",          "HAR",        "HAR_Bayes",  "#2E86AB"),
        ("HAR+SVD",      "HAR+SVD",    "HAR_SVD_Bootstrap",  "#E63946"),
        ("DNN_HAR",      "DNN_HAR",    None,         "#06D6A0"),
        ("DNN_HAR+SVD",  "DNN_HAR+SVD","DNN+SVD",    "#118AB2"),
        ("LSTM_HAR",     "LSTM_HAR",   None,         "#FFD166"),
        ("LSTM_HAR+SVD", "LSTM_HAR+SVD","LSTM+SVD",  "#073B4C"),
    ]

    y_true = df_eval[true_col].values
    idx = df_eval.index
    n = len(y_true)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes_flat = axes.flatten()

    for ax, (panel_title, col, unc_key, color) in zip(axes_flat, panel_models):
        if col not in df_eval.columns:
            ax.set_visible(False)
            continue

        mu = df_eval[col].values
        valid = np.isfinite(mu)

        # Crisis window shading
        for start, end in crisis_windows:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            ax.axvspan(start_ts, end_ts, alpha=0.10, color="red", zorder=0)

        # Actual RV
        ax.semilogy(idx, y_true, "k-", lw=0.7, alpha=0.55, label="Actual RV")

        # Uncertainty band
        if unc_key and unc_key in uncertainty:
            unc = uncertainty[unc_key]
            lo_log = unc.get("lower_log")
            hi_log = unc.get("upper_log")
            if lo_log is not None and hi_log is not None:
                lo_vol = np.exp(np.clip(lo_log, -30, 30))
                hi_vol = np.exp(np.clip(hi_log, -30, 30))
                # Align length with mu (LSTM/DNN may differ in length due to sequences)
                n_unc = min(n, len(lo_vol))
                ax.fill_between(
                    idx[:n_unc],
                    lo_vol[:n_unc],
                    hi_vol[:n_unc],
                    alpha=0.22,
                    color=color,
                    label="95% predictive",
                )

        # Model mean
        n_mu = min(n, len(mu))
        ax.semilogy(
            idx[:n_mu],
            np.where(valid[:n_mu], mu[:n_mu], np.nan),
            "-",
            color=color,
            lw=1.2,
            label=panel_title,
        )

        ax.set_title(panel_title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

    # Shared x-axis formatting
    for ax in axes_flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle(
        "One-Day-Ahead realized variance (%²): predictive intervals (h=1)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "F1_posterior_predictive_forecast.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] F1 saved.")


# ===========================================================================
# F2: Ablation Heatmap
# ===========================================================================

def plot_ablation_heatmap(
    results_by_horizon: dict,
    out_dir: Path,
    metric: str = "RMSE",
) -> None:
    """
    F2: Ablation heatmap showing % improvement over Tier-0 HAR baseline.

    Rows = model architectures + tiered linear ablations
    Columns = feature sets (HAR-only, SVD-T1, HAR+SVD, SVD-T3)
    Panels = one per forecast horizon h in {1, 5, 22}

    Tiered linear models (HAR, SVD-T1, HAR+SVD, SVD-T3) all use ElasticNet,
    isolating feature contribution from estimator differences. Neural models
    (DNN, LSTM, HARNet, GNN) use their respective architectures with Tier-0
    and Tier-2 feature sets.

    Green cells = improvement over baseline; Red = deterioration.
    Annotated with exact % values. Blank cells where model not available.
    """
    if not _HAS_MPL:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define row (architecture) and column (feature set) structure.
    # Linear rows show the full tiered ablation (T0 -> T1 -> T2 -> T3).
    architectures = ["ElasticNet-T0", "ElasticNet-T1", "ElasticNet-T2", "ElasticNet-T3",
                     "DNN", "LSTM", "HARNet", "GNN"]
    # (label, har_key, svd_key) -- keys in results_by_horizon[h]
    arch_map = {
        "ElasticNet-T0": ("HAR",          None),
        "ElasticNet-T1": ("HAR_SVD_T1",   None),
        "ElasticNet-T2": ("HAR+SVD",      None),
        "ElasticNet-T3": ("HAR_SVD_T3",   None),
        "DNN":    ("DNN_HAR", "DNN_HAR+SVD"),
        "LSTM":   ("LSTM_HAR","LSTM_HAR+SVD"),
        "HARNet": (None,      "HARNet"),
        "GNN":    (None,      "GNN"),
    }
    feat_cols = ["HAR", "HAR+SVD"]
    horizons = sorted(results_by_horizon.keys())

    n_horizons = len(horizons)
    fig, axes = plt.subplots(1, n_horizons, figsize=(5 * n_horizons, 6), sharey=True)
    if n_horizons == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        res = results_by_horizon[h]
        # Get HAR baseline RMSE for this horizon
        har_baseline = res.get("HAR", {}).get(metric)
        if har_baseline is None or not np.isfinite(har_baseline) or har_baseline <= 0:
            ax.set_title(f"h={h} (no baseline)")
            continue

        # Build matrix: rows=architectures, cols=["HAR","HAR+SVD"]
        mat = np.full((len(architectures), len(feat_cols)), np.nan)
        annot = [["" for _ in feat_cols] for _ in architectures]

        for i, arch in enumerate(architectures):
            har_key, svd_key = arch_map[arch]
            for j, feat in enumerate(feat_cols):
                # For linear-only rows (ElasticNet tiers), the single key is stored in har_key
                if arch.startswith("ElasticNet"):
                    key = har_key  # single column for single-tier models
                    if j > 0:      # only one "feature set" column for linear tiers
                        continue
                else:
                    key = har_key if feat == "HAR" else svd_key
                if key is None:
                    continue
                val = res.get(key, {}).get(metric)
                if val is not None and np.isfinite(val):
                    pct_change = (val - har_baseline) / har_baseline * 100
                    mat[i, j] = pct_change
                    annot[i][j] = f"{pct_change:+.1f}%"

        if _HAS_SNS:
            # Mask NaN cells
            mask = np.isnan(mat)
            sns.heatmap(
                mat,
                annot=annot,
                fmt="",
                cmap="RdYlGn_r",
                center=0,
                vmin=-50,
                vmax=50,
                xticklabels=feat_cols,
                yticklabels=architectures,
                mask=mask,
                ax=ax,
                cbar_kws={"label": f"% vs HAR {metric}"},
                linewidths=0.5,
            )
        else:
            # Fallback: plain imshow
            im = ax.imshow(mat, cmap="RdYlGn_r", vmin=-50, vmax=50, aspect="auto")
            ax.set_xticks(range(len(feat_cols)))
            ax.set_xticklabels(feat_cols)
            ax.set_yticks(range(len(architectures)))
            ax.set_yticklabels(architectures)
            for i in range(len(architectures)):
                for j in range(len(feat_cols)):
                    if annot[i][j]:
                        ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, label=f"% vs HAR {metric}")

        ax.set_title(f"h={h} | {metric}", fontweight="bold")

    fig.suptitle(
        "Ablation: Incremental Contribution of SVD Features (% vs HAR Baseline)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "F2_ablation_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] F2 saved.")


# ===========================================================================
# F3: Threshold Sensitivity (4-panel)
# ===========================================================================

def plot_threshold_sensitivity_4panel(
    sens_df: pd.DataFrame,
    out_dir: Path,
    models: list = None,
) -> None:
    """
    F3: 4-panel crisis threshold sensitivity figure.

    Panels (2 rows x 2 cols):
        Top-left:    RMSE in Crisis regime vs threshold tau
        Top-right:   RMSE in Calm regime vs threshold tau
        Bottom-left: R2 in Crisis regime vs threshold tau
        Bottom-right:R2 in Calm regime vs threshold tau

    One line per model. Vertical dashed line at tau=0.80 (paper's default choice).
    If the lines are flat, the tau=0.80 choice is robust; if steep, a robustness
    caveat is warranted.
    """
    if not _HAS_MPL or sens_df.empty:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required_cols = {"Threshold", "Regime", "Model", "RMSE", "R2"}
    if not required_cols.issubset(sens_df.columns):
        return

    if models is None:
        models = sens_df["Model"].unique().tolist()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    # Support both date-based crisis labels and angle-percentile labels.
    regimes_present = set(sens_df["Regime"].unique())
    if "High Rotation" in regimes_present or "Low Rotation" in regimes_present:
        regime_a, regime_b = "High Rotation", "Low Rotation"
        label_a, label_b = "High-Rotation periods", "Low-Rotation periods"
    else:
        regime_a, regime_b = "Crisis", "Calm"
        label_a, label_b = "Crisis periods", "Calm periods"

    panel_specs = [
        (axes[0, 0], "RMSE", regime_a, f"RMSE -- {label_a}"),
        (axes[0, 1], "RMSE", regime_b, f"RMSE -- {label_b}"),
        (axes[1, 0], "R2",   regime_a, f"R\u00b2 -- {label_a}"),
        (axes[1, 1], "R2",   regime_b, f"R\u00b2 -- {label_b}"),
    ]

    # cos-\u03b8 crisis cutoffs are \u2265 0.7; percentile-based rotation uses keys \u2264 0.5.
    pct_mode = sens_df["Threshold"].max() <= 0.5

    for ax, metric, regime, title in panel_specs:
        sub = sens_df[sens_df["Regime"] == regime]
        for i, model in enumerate(models):
            msub = sub[sub["Model"] == model].sort_values("Threshold")
            if msub.empty or metric not in msub.columns:
                continue
            color = colors[i % len(colors)]
            ax.plot(
                msub["Threshold"],
                msub[metric],
                "o-",
                label=model,
                color=color,
                alpha=0.85,
                markersize=5,
                linewidth=1.4,
            )
        if pct_mode:
            ax.axvline(0.20, color="gray", linestyle="--", linewidth=1.2, label="top 20% angle")
            ax.set_xlabel("High-rotation definition (top fraction of angle)")
        else:
            ax.axvline(0.80, color="gray", linestyle="--", linewidth=1.2, label="\u03c4=0.80")
            ax.set_xlabel("Crisis threshold \u03c4 (cos \u03b8)")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel(metric)
        ax.legend(loc="best", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Rotation-regime sensitivity: performance vs threshold"
        if pct_mode
        else "Crisis Threshold Sensitivity: Performance by Regime vs \u03c4",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "F3_threshold_sensitivity.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] F3 saved.")


# ===========================================================================
# F4: Crisis Window Deep-Dives
# ===========================================================================

def plot_crisis_deep_dives(
    df_eval: pd.DataFrame,
    crisis_windows: list,
    uncertainty: dict,
    out_dir: Path,
    true_col: str = "TrueVol",
    pred_models: list = None,
) -> None:
    """
    F4: Crisis window deep-dive panels.

    One panel per crisis episode (COVID-19, Ukraine, SVB). Each panel shows:
        - Actual realized variance (%²)
        - All M1-M6 model forecasts
        - 95% predictive bands for M4 (DNN+SVD) and M6 (LSTM+SVD)

    COVID-19 (Feb-Apr 2020) is the largest event in the test period. The paper
    must discuss whether the crisis indicator flagged it (cos_theta < 0.80) and
    how models responded.
    """
    if not _HAS_MPL or df_eval.empty:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if true_col not in df_eval.columns:
        return

    if pred_models is None:
        pred_models = [
            c for c in ["HAR", "HAR+SVD", "DNN_HAR", "DNN_HAR+SVD",
                        "LSTM_HAR", "LSTM_HAR+SVD"]
            if c in df_eval.columns
        ]

    # Give each window a human-readable title
    window_titles = {}
    for start, end in crisis_windows:
        s = pd.Timestamp(start)
        if s.year == 2020:
            window_titles[(start, end)] = "COVID-19 Crash (Feb\u2013Apr 2020)"
        elif s.year == 2022:
            window_titles[(start, end)] = "Russia\u2013Ukraine Invasion (Feb\u2013Mar 2022)"
        elif s.year == 2023:
            window_titles[(start, end)] = "SVB Collapse (Mar 2023)"
        else:
            window_titles[(start, end)] = f"{start} to {end}"

    colors_map = {
        "HAR":          "#2E86AB",
        "HAR+SVD":      "#E63946",
        "DNN_HAR":      "#06D6A0",
        "DNN_HAR+SVD":  "#118AB2",
        "LSTM_HAR":     "#FFD166",
        "LSTM_HAR+SVD": "#073B4C",
    }
    unc_bands = {
        "DNN_HAR+SVD":  "DNN+SVD",
        "LSTM_HAR+SVD": "LSTM+SVD",
    }

    idx = pd.DatetimeIndex(pd.to_datetime(df_eval.index, errors="coerce"))
    n_windows = len(crisis_windows)
    if n_windows == 0:
        return

    fig, axes = plt.subplots(1, n_windows, figsize=(6 * n_windows, 5), squeeze=False)

    for ax, (start, end) in zip(axes[0], crisis_windows):
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        mask = (idx >= start_ts) & (idx <= end_ts)
        if mask.sum() < 3:
            ax.set_title(f"No data: {start}")
            continue

        sub = df_eval.loc[mask]
        sub_idx = sub.index
        y_act = sub[true_col].values
        n_sub = len(sub)

        ax.plot(sub_idx, y_act, "k-", lw=1.2, alpha=0.85, label="Actual RV", zorder=5)

        for col in pred_models:
            if col not in sub.columns:
                continue
            v = sub[col].values
            valid = np.isfinite(v)
            if not valid.any():
                continue
            color = colors_map.get(col, None)
            ax.plot(
                sub_idx[valid],
                v[valid],
                "-",
                color=color,
                lw=1.0,
                alpha=0.75,
                label=col,
            )
            # Uncertainty band for DNN+SVD and LSTM+SVD
            unc_key = unc_bands.get(col)
            if unc_key and unc_key in uncertainty:
                unc = uncertainty[unc_key]
                lo_vol = unc.get("lower_vol")
                hi_vol = unc.get("upper_vol")
                if lo_vol is not None and hi_vol is not None:
                    # Align the uncertainty arrays with the full test set, then slice
                    full_test_idx = df_eval.index
                    lo_series = pd.Series(
                        lo_vol[: len(full_test_idx)], index=full_test_idx[: len(lo_vol)]
                    ).reindex(sub_idx)
                    hi_series = pd.Series(
                        hi_vol[: len(full_test_idx)], index=full_test_idx[: len(hi_vol)]
                    ).reindex(sub_idx)
                    lo_vals = lo_series.values
                    hi_vals = hi_series.values
                    band_valid = np.isfinite(lo_vals) & np.isfinite(hi_vals)
                    if band_valid.any():
                        ax.fill_between(
                            sub_idx[band_valid],
                            lo_vals[band_valid],
                            hi_vals[band_valid],
                            alpha=0.15,
                            color=color,
                        )

        title = window_titles.get((start, end), f"{start} to {end}")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Realized variance (%²)")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle(
        "F4: Crisis Episode Deep-Dives with 95% Predictive Bands",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "F4_crisis_deep_dives.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] F4 saved.")


# ===========================================================================
# F5: Epistemic Uncertainty + HAR Posterior Coefficients
# ===========================================================================

def plot_uncertainty_figure(
    df_eval: pd.DataFrame,
    uncertainty: dict,
    crisis_windows: list,
    out_dir: Path,
    true_col: str = "TrueVol",
) -> None:
    """
    F5: Uncertainty figure with up to three panels (any subset may appear).

    MC Dropout: predictive std in log space for LSTM+SVD / DNN+SVD (crisis shading).

    M2 block bootstrap: HAR+SVD 95% variance bands and half-width when
    ``lower_vol`` / ``upper_vol`` are present.

    Bayesian HAR: violin plots of posterior coefficient draws (M1 conjugate
    regression only), not ElasticNet (M2) coefficient uncertainty.
    """
    if not _HAS_MPL:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    has_mc = any(k in uncertainty for k in ["DNN+SVD", "LSTM+SVD"])
    has_bayes = "HAR_Bayes" in uncertainty and "coef_samples" in uncertainty.get("HAR_Bayes", {})
    has_boot = (
        "HAR_SVD_Bootstrap" in uncertainty
        and uncertainty["HAR_SVD_Bootstrap"].get("lower_vol") is not None
        and uncertainty["HAR_SVD_Bootstrap"].get("upper_vol") is not None
    )

    if not has_mc and not has_bayes and not has_boot:
        print(
            "[INFO] F5 skipped: no MC Dropout, no M2 bootstrap bands, no HAR Bayes coef_samples."
        )
        return

    n_panels = int(has_mc) + int(has_boot) + int(has_bayes)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.8 * n_panels))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0
    idx = df_eval.index

    if has_mc:
        ax1 = axes[panel_idx]
        panel_idx += 1
        mc_colors = {"DNN+SVD": "#2E86AB", "LSTM+SVD": "#073B4C"}
        for key, color in mc_colors.items():
            if key not in uncertainty:
                continue
            std_arr = uncertainty[key].get("std_log")
            if std_arr is None:
                continue
            n_unc = min(len(idx), len(std_arr))
            ax1.fill_between(
                idx[:n_unc],
                0,
                std_arr[:n_unc],
                alpha=0.50,
                color=color,
                label=f"{key} epistemic \u03c3",
            )
        for start, end in crisis_windows:
            ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.12, color="red")
        ax1.set_title(
            "Epistemic Uncertainty (MC Dropout std dev) -- should peak during crises",
            fontweight="bold",
        )
        ax1.set_ylabel("Predictive Std Dev (log scale)")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.2)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    if has_boot:
        axb = axes[panel_idx]
        panel_idx += 1
        unc_b = uncertainty["HAR_SVD_Bootstrap"]
        lo_v = np.asarray(unc_b["lower_vol"], dtype=float).ravel()
        hi_v = np.asarray(unc_b["upper_vol"], dtype=float).ravel()
        n_b = min(len(idx), len(lo_v), len(hi_v))
        half_w = 0.5 * np.maximum(hi_v[:n_b] - lo_v[:n_b], 1e-20)
        axb.fill_between(idx[:n_b], lo_v[:n_b], hi_v[:n_b], alpha=0.35, color="#E63946", label="95% band")
        axb.semilogy(idx[:n_b], half_w, color="#333", lw=0.9, ls=":", label="Half-width (var space)")
        for start, end in crisis_windows:
            axb.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.12, color="red")
        axb.set_title(
            "M2 HAR+SVD: block-bootstrap predictive band (variance space)",
            fontweight="bold",
        )
        axb.set_ylabel("Variance (%²)")
        axb.legend(fontsize=8, loc="upper right")
        axb.grid(True, alpha=0.2)
        axb.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(axb.xaxis.get_majorticklabels(), rotation=30, ha="right")

    if has_bayes:
        ax2 = axes[panel_idx]
        # Violin plot: conjugate posterior for OLS HAR features only (same design as M1 fit sample).
        coef_samples = uncertainty["HAR_Bayes"]["coef_samples"]  # (2000, p)
        # Columns: [const, ...HAR features...] (statsmodels add_constant prepends const)
        n_coef = coef_samples.shape[1]
        # Build label list dynamically from the stored feature names, falling back to
        # a generic β_k label so the plot never crashes on unexpected feature counts.
        _har_feature_names = uncertainty["HAR_Bayes"].get("feature_names", [])
        if _har_feature_names:
            # feature_names are the columns passed to OLS (excluding the constant added
            # internally by statsmodels); prepend "const" to match coef ordering.
            labels = ["const"] + list(_har_feature_names)
        else:
            # Fallback: generic labels that expand to whatever n_coef is
            _default = [
                "const",
                "\u03b2_d (daily)",
                "\u03b2_w (weekly)",
                "\u03b2_10d (10-day)",
                "\u03b2_m (monthly)",
                "\u03b2_RSV\u207a",
                "\u03b2_RSV\u207b",
                "\u03b2_VIX",
            ]
            labels = (_default + [f"\u03b2_{i}" for i in range(len(_default), n_coef)])[:n_coef]
        labels = labels[:n_coef]  # ensure exact length match
        data = [coef_samples[:, i] for i in range(n_coef)]
        positions = list(range(1, n_coef + 1))
        vp = ax2.violinplot(data, positions=positions, showmedians=True, showextrema=True)
        for body in vp["bodies"]:
            body.set_facecolor("#2E86AB")
            body.set_alpha(0.6)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(labels, fontsize=10)
        ax2.axhline(0, color="gray", lw=0.9, ls="--", alpha=0.7)
        ax2.set_title("HAR Posterior Coefficient Distributions (Bayesian Linear Regression)",
                      fontweight="bold")
        ax2.set_ylabel("Coefficient value")
        ax2.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "F5: Parameter and Epistemic Uncertainty",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "F5_uncertainty.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] F5 saved.")


# ===========================================================================
# F9: Mincer–Zarnowitz calibration summary (all models, h=1)
# ===========================================================================


def plot_mincer_zarnowitz_summary(
    mz_results: dict,
    out_dir: Path,
    z_crit: float = 1.96,
) -> None:
    """
    F9: HAC-based Mincer–Zarnowitz intercept and slope for every model with
    valid MZ output — point estimates ± z_crit * SE, reference lines α=0, β=1.
    """
    if not _HAS_MPL or not mz_results:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(mz_results.keys())
    alphas = np.array([float(mz_results[m].get("mz_alpha", np.nan)) for m in names])
    a_se = np.array([float(mz_results[m].get("mz_alpha_se", np.nan)) for m in names])
    betas = np.array([float(mz_results[m].get("mz_beta", np.nan)) for m in names])
    b_se = np.array([float(mz_results[m].get("mz_beta_se", np.nan)) for m in names])

    x = np.arange(len(names))
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(max(10, 0.35 * len(names)), 8), sharex=True)

    ax0.bar(x, alphas, yerr=z_crit * a_se, capsize=3, color="#2E86AB", alpha=0.85, ecolor="#333")
    ax0.axhline(0.0, color="gray", ls="--", lw=1.0)
    ax0.set_ylabel(r"$\alpha$ (HAC)")
    ax0.set_title("MZ intercept (ideal: 0)", fontsize=11, fontweight="bold")
    ax0.grid(True, axis="y", alpha=0.25)

    ax1.bar(x, betas, yerr=z_crit * b_se, capsize=3, color="#E63946", alpha=0.85, ecolor="#333")
    ax1.axhline(1.0, color="gray", ls="--", lw=1.0)
    ax1.set_ylabel(r"$\beta$ (HAC)")
    ax1.set_title("MZ slope (ideal: 1)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=55, ha="right", fontsize=8)
    ax1.grid(True, axis="y", alpha=0.25)

    # R² as text above bars (small)
    fig.suptitle(
        "F9: Mincer–Zarnowitz calibration (h=1, variance space, HAC SE; R² in mz_results.csv)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "F9_mincer_zarnowitz_summary.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] F9 Mincer–Zarnowitz summary saved.")


# ===========================================================================
# F6: Enhanced Residual Diagnostics
# ===========================================================================

def plot_residual_diagnostics(
    df_eval: pd.DataFrame,
    pred_columns: list,
    out_dir: Path,
    true_col: str = "TrueVol",
    max_models: int = 6,
) -> None:
    """
    F6: 4-panel residual diagnostics for each model.

    For each model in pred_columns (up to max_models), produces a figure with:
        Panel 1 (top-left):   Residual histogram with fitted normal density overlay
        Panel 2 (top-right):  Normal Q-Q plot (residuals vs theoretical quantiles)
        Panel 3 (bottom-left):Residuals vs fitted values (calibration check)
        Panel 4 (bottom-right):ACF of squared residuals (ARCH / volatility clustering check)

    Saved as F6_residual_diagnostics_{model}.pdf for each model.

    Panel 4 uses statsmodels acf; if statsmodels not available, falls back to a
    manual autocorrelation computation.
    """
    if not _HAS_MPL or df_eval.empty or true_col not in df_eval.columns:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from statsmodels.tsa.stattools import acf as sm_acf
        _has_sm_acf = True
    except ImportError:
        _has_sm_acf = False

    y_true = df_eval[true_col].values
    models_done = 0

    for col in pred_columns:
        if col not in df_eval.columns or models_done >= max_models:
            continue
        y_pred = df_eval[col].values
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if valid.sum() < 20:
            continue

        residuals = y_true[valid] - y_pred[valid]
        fitted = y_pred[valid]
        resid_sq = residuals ** 2

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle(
            f"F6: Residual Diagnostics -- {col}",
            fontsize=12,
            fontweight="bold",
        )

        # Panel 1: Histogram + normal fit
        ax = axes[0, 0]
        _, bins, _ = ax.hist(
            residuals, bins=50, density=True,
            color="#2E86AB", alpha=0.7, edgecolor="white",
        )
        x_fit = np.linspace(residuals.min(), residuals.max(), 200)
        mu_fit, std_fit = residuals.mean(), residuals.std(ddof=1)
        ax.plot(
            x_fit,
            stats.norm.pdf(x_fit, mu_fit, std_fit),
            "r-",
            lw=1.8,
            label=f"N({mu_fit:.3f}, {std_fit:.3f})",
        )
        ax.set_xlabel("Residual (actual - predicted)")
        ax.set_ylabel("Density")
        ax.set_title("Residual distribution")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # Panel 2: Q-Q plot
        ax = axes[0, 1]
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        ax.scatter(osm, osr, s=6, alpha=0.5, color="#2E86AB")
        x_qq = np.array([osm[0], osm[-1]])
        ax.plot(x_qq, slope * x_qq + intercept, "r-", lw=1.5, label=f"R²={r**2:.3f}")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.set_title("Normal Q-Q plot")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # Panel 3: Residuals vs fitted
        ax = axes[1, 0]
        ax.scatter(fitted, residuals, s=4, alpha=0.35, color="#2E86AB")
        ax.axhline(0, color="red", lw=1.2, ls="--")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs fitted")
        ax.grid(True, alpha=0.2)

        # Panel 4: ACF of squared residuals
        ax = axes[1, 1]
        n_lags = min(40, len(resid_sq) // 2 - 1)
        if _has_sm_acf and n_lags > 1:
            acf_vals = sm_acf(resid_sq, nlags=n_lags, fft=True)
            lags = np.arange(len(acf_vals))
            ax.bar(lags, acf_vals, color="#2E86AB", alpha=0.75, width=0.6)
            # 95% confidence band (Bartlett's formula: +-1.96/sqrt(n))
            ci = 1.96 / np.sqrt(len(resid_sq))
            ax.axhline(ci, color="red", ls="--", lw=1.0, alpha=0.7)
            ax.axhline(-ci, color="red", ls="--", lw=1.0, alpha=0.7)
        else:
            # Manual ACF fallback
            n_r = len(resid_sq)
            mean_r = resid_sq.mean()
            var_r = ((resid_sq - mean_r) ** 2).sum()
            acf_vals_manual = []
            for lag in range(min(n_lags + 1, n_r)):
                cov_lag = ((resid_sq[lag:] - mean_r) * (resid_sq[: n_r - lag] - mean_r)).sum()
                acf_vals_manual.append(cov_lag / (var_r + 1e-12))
            ax.bar(range(len(acf_vals_manual)), acf_vals_manual,
                   color="#2E86AB", alpha=0.75, width=0.6)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title("ACF of squared residuals (ARCH check)")
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        safe_col = col.replace("+", "_plus_").replace(" ", "_")
        fig.savefig(out_dir / f"F6_residual_diagnostics_{safe_col}.pdf",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        models_done += 1

    if models_done > 0:
        print(f"[INFO] F6 saved ({models_done} models).")


# ===========================================================================
# F7: DM Statistic Heatmap
# ===========================================================================

def plot_dm_matrix(
    dm_df: pd.DataFrame,
    out_dir: Path,
    horizon: int = 1,
) -> None:
    """
    F7: Pairwise DM-statistic heatmap for all model pairs at a given horizon.

    The heatmap is signed: cell (i, j) shows the DM statistic when model i is
    Model1 and model j is Model2. A POSITIVE statistic means model j has LOWER
    squared error (i.e. model j beats model i). Convention: green = row model
    is worse; red = row model is better. This is consistent with interpreting
    positive DM_stat as "Model2 is better than Model1".

    Significance markers:
        *   p < 0.10  (marginal)
        **  p < 0.05  (significant)
        *** p < 0.01  (highly significant)

    A symmetric heatmap is also shown with the raw signed statistics.
    Missing pairs (fewer than 10 valid observations) are left grey/blank.

    Parameters
    ----------
    dm_df : pd.DataFrame
        Output of the DM test loop; columns: Model1, Model2, DM_QLIKE, p_QLIKE,
        DM_MSE, p_MSE (new format) or legacy DM_stat, p_value. The function
        prefers QLIKE-based statistics when both are available.
    out_dir : Path
        Directory for output PDF.
    horizon : int
        Forecast horizon (used in title only).
    """
    if not _HAS_MPL or dm_df is None or dm_df.empty:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect unique model names preserving order
    all_models_ordered = []
    for col in ("Model1", "Model2"):
        if col in dm_df.columns:
            for m in dm_df[col].tolist():
                if m not in all_models_ordered:
                    all_models_ordered.append(m)
    if len(all_models_ordered) < 2:
        return

    # Prefer a canonical ordering if available
    canonical = [
        "HAR", "HAR+SVD", "DNN_HAR", "DNN_HAR+SVD",
        "LSTM_HAR", "LSTM_HAR+SVD", "HARNet", "GNN",
        "GARCH", "IV_baseline",
    ]
    models = [m for m in canonical if m in all_models_ordered]
    # Append any remaining models not in canonical
    for m in all_models_ordered:
        if m not in models:
            models.append(m)

    n = len(models)
    model_idx = {m: i for i, m in enumerate(models)}

    # Build full symmetric stat and annotation matrices
    stat_mat = np.full((n, n), np.nan)
    annot_mat = [["" for _ in range(n)] for _ in range(n)]

    for _, row in dm_df.iterrows():
        m1, m2 = row.get("Model1"), row.get("Model2")
        # Prefer QLIKE-based DM stat (proxy-robust); fall back to MSE-based or legacy DM_stat
        stat = row.get("DM_QLIKE", row.get("DM_MSE", row.get("DM_stat")))
        pval = row.get("p_QLIKE", row.get("p_MSE", row.get("p_value")))
        if m1 not in model_idx or m2 not in model_idx:
            continue
        if not (np.isfinite(stat) if stat is not None else False):
            continue
        i, j = model_idx[m1], model_idx[m2]
        stat_mat[i, j] = stat
        stat_mat[j, i] = -stat  # antisymmetric: swap direction

        # Significance marker
        def _sig(p):
            if p is None or not np.isfinite(p):
                return ""
            if p < 0.01:
                return "***"
            if p < 0.05:
                return "**"
            if p < 0.10:
                return "*"
            return ""

        sig = _sig(pval)
        annot_mat[i][j] = f"{stat:.2f}{sig}"
        annot_mat[j][i] = f"{-stat:.2f}{sig}"

    # Diagonal: self-comparison; mark as 0 with no label
    for k in range(n):
        stat_mat[k, k] = 0.0
        annot_mat[k][k] = "—"

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), max(6, n * 1.0)))

    if _HAS_SNS:
        mask = np.isnan(stat_mat)
        vmax = np.nanquantile(np.abs(stat_mat[~mask]), 0.95) if not mask.all() else 3.0
        vmax = max(vmax, 1.0)
        sns.heatmap(
            stat_mat,
            annot=np.array(annot_mat),
            fmt="",
            cmap="RdYlGn",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            xticklabels=models,
            yticklabels=models,
            mask=mask,
            ax=ax,
            cbar_kws={"label": "DM statistic (+ = col model better)"},
            linewidths=0.4,
            annot_kws={"size": 7},
        )
    else:
        mask = np.isnan(stat_mat)
        safe_mat = np.where(mask, 0.0, stat_mat)
        vmax = float(np.abs(safe_mat).max()) if not mask.all() else 3.0
        vmax = max(vmax, 1.0)
        im = ax.imshow(safe_mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(models, fontsize=8)
        for i in range(n):
            for j in range(n):
                if annot_mat[i][j]:
                    ax.text(j, i, annot_mat[i][j], ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="DM statistic (+ = col model better)")

    ax.set_title(
        f"F7: Pairwise DM Statistic Matrix (h={horizon})\n"
        "Positive = column model has lower squared error\n"
        "* p<0.10  ** p<0.05  *** p<0.01",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel("Model 2 (column)", fontsize=9)
    ax.set_ylabel("Model 1 (row)", fontsize=9)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(
        out_dir / f"F7_dm_matrix_h{horizon}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"[INFO] F7 DM matrix (h={horizon}) saved.")


# ===========================================================================
# Giacomini-White Conditional Predictive Ability Test
# ===========================================================================

def standardize_gw_instruments(Z: np.ndarray) -> np.ndarray:
    """
    Column-wise z-score using **moments of Z itself** (evaluation sample).
    Skips (near-)constant columns so an intercept column of ones is unchanged.
    """
    return apply_gw_zscore_from_reference(Z, Z)


def apply_gw_zscore_from_reference(Z_eval: np.ndarray, Z_ref: np.ndarray) -> np.ndarray:
    """
    Column-wise z-score of ``Z_eval`` using nan-mean / nan-std from **Z_ref**
    (e.g. training-period instruments). Same shape along axis 1 as ``Z_eval``.

    Skips columns that are (near-)constant on ``Z_eval`` or have σ<1e-12 in
    ``Z_ref`` so a column of ones stays unchanged.
    """
    Z_e = np.asarray(Z_eval, dtype=np.float64).copy()
    Z_r = np.asarray(Z_ref, dtype=np.float64)
    if Z_e.shape[1] != Z_r.shape[1]:
        raise ValueError("Z_eval and Z_ref must have the same number of columns")
    for j in range(Z_e.shape[1]):
        col_e = Z_e[:, j]
        if not np.any(np.isfinite(col_e)):
            continue
        spread_e = float(np.nanmax(col_e) - np.nanmin(col_e))
        if spread_e < 1e-15:
            continue
        col_r = Z_r[:, j]
        mu = float(np.nanmean(col_r))
        sig = float(np.nanstd(col_r))
        if sig < 1e-12:
            continue
        Z_e[:, j] = (col_e - mu) / sig
    return Z_e


def giacomini_white_test(
    true_var: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    instruments: np.ndarray,
    nlags: int | None = None,
    eps: float = EPS_DEFAULT,
    standardize_instruments: bool = True,
    instrument_zscore_moments: str | None = None,
    Z_train: np.ndarray | None = None,
    horizon: int = 1,
) -> dict:
    """
    Giacomini-White (2006, Econometrica 74(6), 1545-1578) Conditional Predictive
    Ability (CPA) test.

    **Wald GW** is always computed from **raw** instruments Z (same scaling as
    the moment R_t = Z_t * d_t). Column-wise z-scoring affects **only** the
    auxiliary regression d_t = Z_t'δ + u_t used to report **δ**, HAC **SE**,
    and **t** (so coefficients are interpretable on a common scale). The
    χ² Wald statistic is **invariant** to full-rank column scaling of Z in the
    ideal linear-Gaussian setting; here we keep R_t built from **raw** Z so the
    reported GW statistic does not depend on the z-score choice.

    ``instrument_zscore_moments``:
        - ``"eval"`` (default when ``standardize_instruments`` is True): μ,σ
          from the **evaluation** (test) Z rows used in the GW.
        - ``"train"``: μ,σ from ``Z_train`` (same columns as Z before any
          column drop); requires ``Z_train`` with ``Z_train.shape[1] == Z.shape[1]``.
        - ``"none"``: no z-scoring for the auxiliary regression.

    Parameters
    ----------
    Z_train : optional (T_train, k) matrix aligned with the **same** instrument
              columns as ``instruments`` (e.g. training-period angle, gap).
    """
    import statsmodels.api as sm_gw

    h = np.asarray(true_var, dtype=np.float64).ravel()
    h1 = np.clip(np.asarray(pred1, dtype=np.float64).ravel(), eps, None)
    h2 = np.clip(np.asarray(pred2, dtype=np.float64).ravel(), eps, None)
    h = np.maximum(h, eps)

    Z_raw = np.asarray(instruments, dtype=np.float64)
    if Z_raw.ndim == 1:
        Z_raw = Z_raw.reshape(-1, 1)

    if instrument_zscore_moments is None:
        instrument_zscore_moments = "eval" if standardize_instruments else "none"

    # Align lengths
    n = min(len(h), len(h1), len(h2), len(Z_raw))
    h, h1, h2, Z_raw = h[:n], h1[:n], h2[:n], Z_raw[:n]

    Z_tr = None
    if Z_train is not None:
        Z_tr = np.asarray(Z_train, dtype=np.float64)
        if Z_tr.ndim == 1:
            Z_tr = Z_tr.reshape(-1, 1)
        if Z_tr.shape[1] != Z_raw.shape[1]:
            Z_tr = None  # mismatch — fall back to eval moments

    # QLIKE loss differential
    r1, r2 = h / h1, h / h2
    L1 = r1 - np.log(r1) - 1.0
    L2 = r2 - np.log(r2) - 1.0
    d = L1 - L2

    # Drop rows with non-finite d or Z
    finite_mask = np.isfinite(d) & np.all(np.isfinite(Z_raw), axis=1)
    d = d[finite_mask]
    Z_raw = Z_raw[finite_mask]
    T = len(d)

    if T < 20:
        nan_vec = np.full(Z_raw.shape[1], np.nan)
        return {
            "gw_stat": np.nan, "df": Z_raw.shape[1], "p_value": np.nan,
            "coef": nan_vec, "se": nan_vec, "t_stats": nan_vec,
            "instrument_zscore_moments": instrument_zscore_moments,
            "gw_moment_uses_raw_Z": True,
        }

    # --- Wald GW on raw Z only (R_t = Z_raw * d_t) ---------------------------
    R_t_raw = Z_raw * d[:, np.newaxis]
    col_std = np.std(R_t_raw, axis=0)
    active_cols = col_std > 1e-10
    if int(active_cols.sum()) == 0:
        nan_vec = np.full(1, np.nan)
        return {
            "gw_stat": np.nan, "df": 0, "p_value": np.nan,
            "coef": nan_vec, "se": nan_vec, "t_stats": nan_vec,
            "instrument_zscore_moments": instrument_zscore_moments,
            "gw_moment_uses_raw_Z": True,
        }

    Zr = Z_raw[:, active_cols]
    R_t = Zr * d[:, np.newaxis]
    R_bar = R_t.mean(axis=0)
    k = int(Zr.shape[1])
    if nlags is None:
        nlags = hac_lags_for_horizon(T=int(T), horizon=int(horizon))
    nlags = int(max(nlags, 1))
    gamma0 = (R_t.T @ R_t) / T
    S_hat = gamma0.copy()
    for lag in range(1, nlags + 1):
        w = 1.0 - lag / (nlags + 1)
        gamma_l = (R_t[lag:].T @ R_t[:-lag]) / T
        S_hat += w * (gamma_l + gamma_l.T)

    try:
        S_inv = np.linalg.inv(S_hat)
        gw_stat = float(T * R_bar @ S_inv @ R_bar)
    except np.linalg.LinAlgError:
        gw_stat = np.nan
    p_value = float(1.0 - stats.chi2.cdf(gw_stat, df=k)) if np.isfinite(gw_stat) else np.nan

    # --- Auxiliary δ: OLS of d on Z_aux with HAC SE (z-scored as requested) --
    if instrument_zscore_moments == "train" and Z_tr is not None:
        Z_ref = Z_tr[:, active_cols]
        if Z_ref.shape[0] < 5 or not np.all(np.isfinite(Z_ref)):
            Z_aux = apply_gw_zscore_from_reference(Zr.copy(), Zr)
        else:
            Z_aux = apply_gw_zscore_from_reference(Zr.copy(), Z_ref)
    elif instrument_zscore_moments == "eval" or (
        instrument_zscore_moments == "train" and Z_tr is None
    ):
        Z_aux = standardize_gw_instruments(Zr.copy())
    else:
        Z_aux = Zr.copy()

    try:
        ols = sm_gw.OLS(d, Z_aux).fit(cov_type="HAC", cov_kwds={"maxlags": nlags})
        coef = np.asarray(ols.params, dtype=float).ravel()
        se = np.asarray(ols.bse, dtype=float).ravel()
        t_stats = np.asarray(ols.tvalues, dtype=float).ravel()
    except Exception:
        coef = np.full(k, np.nan)
        se = np.full(k, np.nan)
        t_stats = np.full(k, np.nan)

    return {
        "gw_stat": gw_stat,
        "df": k,
        "p_value": p_value,
        "coef": coef,
        "se": se,
        "t_stats": t_stats,
        "instrument_zscore_moments": instrument_zscore_moments,
        "gw_moment_uses_raw_Z": True,
    }


# ===========================================================================
# Forecast Encompassing Test (Harvey-Leybourne-Newbold 1998)
# ===========================================================================

def forecast_encompassing_test(
    true_var: np.ndarray,
    pred_har: np.ndarray,
    pred_svd: np.ndarray,
    eps: float = EPS_DEFAULT,
    nlags: int | None = None,
    horizon: int = 1,
) -> dict:
    """
    Harvey-Leybourne-Newbold (1998) Forecast Encompassing Test.

    The encompassing regression:
        sigma2_realized = alpha + lambda_1 * sigma2_HAR + lambda_2 * sigma2_SVD + eps

    Tests H0: lambda_2 = 0 (HAR encompasses SVD; SVD adds no incremental information).
    If rejected, SVD contains unique predictive information not in HAR.

    The directional claim of the paper is "SVD forecasts add positive
    incremental information", i.e. ``lambda_2 > 0`` with statistical
    significance.  A statistically significant *negative* ``lambda_2`` means
    SVD forecasts are systematically *over-shooting* relative to HAR in the
    encompassing combination — the OLS / WLS-style minimum-variance fit is
    putting a negative weight on the SVD piece — which is **not** evidence
    that "SVD adds info".  We therefore report:

      * ``svd_adds_info``           — sign-aware: ``lambda_2 > 0`` AND
        one-sided p-value ``< 0.05`` against the alternative ``lambda_2 > 0``.
        This is the directional claim used in the headline grid.
      * ``svd_significantly_negative`` — ``lambda_2 < 0`` AND two-sided
        p-value ``< 0.05``.  Diagnostic: indicates the SVD forecast is
        miscalibrated *given* HAR (the encompassing regression effectively
        subtracts SVD from HAR rather than adding it), and should never be
        sold as "SVD adds incremental information".
      * ``svd_encompasses``         — kept for backward compatibility,
        equals ``svd_adds_info`` (the previous two-sided-only flag was a
        bug; see ``methods/preregistration.md``).

    All variance series are in raw (not log) space so lambda_1, lambda_2 are
    "mixture weights" interpretable as the share of variance explained by
    each model.

    Parameters
    ----------
    true_var  : (T,) realized variance series.
    pred_har  : (T,) HAR variance forecasts.
    pred_svd  : (T,) HAR+SVD variance forecasts.
    nlags     : Newey-West HAC lags for standard errors.

    Returns
    -------
    dict with:
        "alpha", "lambda1", "lambda2" : OLS coefficients
        "alpha_se", "lambda1_se", "lambda2_se" : HAC standard errors
        "alpha_t", "lambda1_t", "lambda2_t"   : HAC t-statistics
        "alpha_p", "lambda1_p", "lambda2_p"   : two-sided p-values
        "lambda2_p_one_sided"         : one-sided p-value vs lambda2 > 0
        "r2"                          : R² of encompassing regression
        "svd_adds_info"               : sign-aware bool (lambda2 > 0 & one-sided p<.05)
        "svd_significantly_negative"  : lambda2 < 0 & two-sided p<.05
        "svd_encompasses"             : alias for ``svd_adds_info`` (legacy key)
        "encompassing_verdict"        : human-readable summary string
    """
    import statsmodels.api as sm_enc

    h = np.asarray(true_var, dtype=np.float64).ravel()
    h1 = np.asarray(pred_har, dtype=np.float64).ravel()
    h2 = np.asarray(pred_svd, dtype=np.float64).ravel()
    h = np.maximum(h, eps)
    h1 = np.maximum(h1, eps)
    h2 = np.maximum(h2, eps)

    n = min(len(h), len(h1), len(h2))
    h, h1, h2 = h[:n], h1[:n], h2[:n]
    finite_mask = np.isfinite(h) & np.isfinite(h1) & np.isfinite(h2)
    h, h1, h2 = h[finite_mask], h1[finite_mask], h2[finite_mask]
    T = len(h)
    nan = float("nan")
    empty_payload = {
        "alpha": nan, "lambda1": nan, "lambda2": nan,
        "alpha_se": nan, "lambda1_se": nan, "lambda2_se": nan,
        "alpha_t": nan, "lambda1_t": nan, "lambda2_t": nan,
        "alpha_p": nan, "lambda1_p": nan, "lambda2_p": nan,
        "lambda2_p_one_sided": nan,
        "r2": nan,
        "svd_adds_info": False,
        "svd_significantly_negative": False,
        "svd_encompasses": False,
        "encompassing_verdict": "insufficient data",
    }
    if T < 10:
        return empty_payload

    X_enc = sm_enc.add_constant(np.column_stack([h1, h2]))
    if nlags is None:
        nlags = hac_lags_for_horizon(T=int(T), horizon=int(horizon))
    nlags = int(max(nlags, 1))
    try:
        res = sm_enc.OLS(h, X_enc).fit(
            cov_type="HAC", cov_kwds={"maxlags": int(nlags)}
        )
        alpha, l1, l2 = float(res.params[0]), float(res.params[1]), float(res.params[2])
        a_se, l1_se, l2_se = float(res.bse[0]), float(res.bse[1]), float(res.bse[2])
        a_t, l1_t, l2_t = float(res.tvalues[0]), float(res.tvalues[1]), float(res.tvalues[2])
        a_p, l1_p, l2_p = float(res.pvalues[0]), float(res.pvalues[1]), float(res.pvalues[2])
        r2 = float(res.rsquared)
    except Exception:
        return empty_payload

    # One-sided p-value against the alternative lambda_2 > 0 (positive
    # incremental information).  Constructed from the two-sided p-value as
    # P(T_{T-3} > t_obs) using the symmetry of the t distribution under H0.
    if np.isfinite(l2_t):
        l2_p_one_sided = float(0.5 * l2_p) if l2_t > 0 else float(1.0 - 0.5 * l2_p)
    else:
        l2_p_one_sided = nan

    svd_adds_info = bool(np.isfinite(l2) and l2 > 0 and l2_p_one_sided < 0.05)
    svd_negative = bool(np.isfinite(l2) and l2 < 0 and l2_p < 0.05)
    if svd_adds_info:
        verdict = "SVD adds info (lambda2>0, p_one<.05)"
    elif svd_negative:
        verdict = "SVD enters with NEGATIVE weight (lambda2<0, p<.05) -- HAR dominates"
    else:
        verdict = "not significant"

    return {
        "alpha": alpha,
        "lambda1": l1,
        "lambda2": l2,
        "alpha_se": a_se,
        "lambda1_se": l1_se,
        "lambda2_se": l2_se,
        "alpha_t": a_t,
        "lambda1_t": l1_t,
        "lambda2_t": l2_t,
        "alpha_p": a_p,
        "lambda1_p": l1_p,
        "lambda2_p": l2_p,
        "lambda2_p_one_sided": l2_p_one_sided,
        "r2": r2,
        "svd_adds_info": svd_adds_info,
        "svd_significantly_negative": svd_negative,
        "svd_encompasses": svd_adds_info,
        "encompassing_verdict": verdict,
    }


# ===========================================================================
# Bates-Granger (1969) Optimal Forecast Combination with Bootstrap CI
# ===========================================================================

def bates_granger_combination(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    eps: float = EPS_DEFAULT,
    nonneg: bool = True,
) -> dict:
    """
    Bates-Granger (1969) minimum-variance forecast combination.

    For two unbiased forecasts ``f_a, f_b`` of ``y`` with squared-error
    covariance matrix
    ::

        Sigma = [[s_aa, s_ab],
                 [s_ab, s_bb]]

    the minimum-variance linear combination ``f^* = w_a f_a + (1-w_a) f_b``
    has weight
    ::

        w_a = (s_bb - s_ab) / (s_aa + s_bb - 2 s_ab) .

    When ``nonneg=True`` we project ``w_a`` onto ``[0, 1]`` (the convex
    combination feasible set) so the combined forecast is interpretable as
    a probability mixture; when ``nonneg=False`` we report the unrestricted
    Bates-Granger weight (which can fall outside ``[0,1]`` if the forecasts
    are highly correlated and one strongly dominates).

    Parameters
    ----------
    y_true   : (T,) realized targets (variance for our application).
    pred_a   : (T,) forecast A (e.g. HAR).
    pred_b   : (T,) forecast B (e.g. HAR+SVD).
    eps      : numerical floor for variance.
    nonneg   : if True, clip ``w_a`` to ``[0,1]``; default True.

    Returns
    -------
    dict with keys ``w_a``, ``w_b``, ``mse_a``, ``mse_b``, ``mse_comb``,
    ``cov_ab`` and ``corr_ab``.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    fa = np.asarray(pred_a, dtype=np.float64).ravel()
    fb = np.asarray(pred_b, dtype=np.float64).ravel()
    n = min(len(y), len(fa), len(fb))
    y, fa, fb = y[:n], fa[:n], fb[:n]
    mask = np.isfinite(y) & np.isfinite(fa) & np.isfinite(fb)
    y, fa, fb = y[mask], fa[mask], fb[mask]
    nan = float("nan")
    if y.size < 10:
        return {
            "w_a": nan, "w_b": nan, "mse_a": nan, "mse_b": nan,
            "mse_comb": nan, "cov_ab": nan, "corr_ab": nan, "T": int(y.size),
        }

    e_a = y - fa
    e_b = y - fb
    s_aa = float(np.mean(e_a ** 2))
    s_bb = float(np.mean(e_b ** 2))
    s_ab = float(np.mean(e_a * e_b))
    denom = s_aa + s_bb - 2.0 * s_ab
    if not np.isfinite(denom) or abs(denom) < eps:
        w_a = 0.5
    else:
        w_a = (s_bb - s_ab) / denom
    if nonneg:
        w_a = float(np.clip(w_a, 0.0, 1.0))
    w_b = 1.0 - w_a
    f_comb = w_a * fa + w_b * fb
    mse_comb = float(np.mean((y - f_comb) ** 2))
    corr = nan
    if s_aa > eps and s_bb > eps:
        corr = float(s_ab / np.sqrt(s_aa * s_bb))
    return {
        "w_a": float(w_a),
        "w_b": float(w_b),
        "mse_a": s_aa,
        "mse_b": s_bb,
        "mse_comb": mse_comb,
        "cov_ab": s_ab,
        "corr_ab": corr,
        "T": int(y.size),
    }


def bates_granger_with_bootstrap_ci(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    block_len: int = 22,
    n_boot: int = 999,
    alpha: float = 0.05,
    seed: int = 42,
    nonneg: bool = True,
    eps: float = EPS_DEFAULT,
) -> dict:
    """
    Bates-Granger weights + stationary-block-bootstrap confidence intervals.

    Uses a Politis-Romano (1994) circular block bootstrap with mean block
    length ``block_len`` to resample the joint ``(y_t, f^a_t, f^b_t)``
    series; recomputes the optimal weight on each bootstrap replicate and
    reports the empirical quantiles.  Bootstrap percentile interval is the
    paper's headline; the basic interval is also returned as a robustness
    check (Davison & Hinkley 1997).

    Returns a dict augmenting :func:`bates_granger_combination` with
    ``w_a_lo``, ``w_a_hi`` (percentile CI for the weight on forecast A) and
    ``mse_comb_lo``, ``mse_comb_hi`` for the combined MSE; both at level
    ``1 - alpha``.
    """
    base = bates_granger_combination(y_true, pred_a, pred_b, eps=eps, nonneg=nonneg)
    if not np.isfinite(base.get("w_a", np.nan)):
        base.update({"w_a_lo": float("nan"), "w_a_hi": float("nan"),
                     "mse_comb_lo": float("nan"), "mse_comb_hi": float("nan"),
                     "n_boot": 0, "block_len": int(block_len)})
        return base

    y = np.asarray(y_true, dtype=np.float64).ravel()
    fa = np.asarray(pred_a, dtype=np.float64).ravel()
    fb = np.asarray(pred_b, dtype=np.float64).ravel()
    n = min(len(y), len(fa), len(fb))
    y, fa, fb = y[:n], fa[:n], fb[:n]
    mask = np.isfinite(y) & np.isfinite(fa) & np.isfinite(fb)
    y, fa, fb = y[mask], fa[mask], fb[mask]
    T = y.size
    if T < max(20, 2 * block_len):
        base.update({"w_a_lo": float("nan"), "w_a_hi": float("nan"),
                     "mse_comb_lo": float("nan"), "mse_comb_hi": float("nan"),
                     "n_boot": 0, "block_len": int(block_len)})
        return base

    rng = np.random.default_rng(int(seed))
    p_geom = 1.0 / max(block_len, 1)
    w_boot = np.empty(n_boot, dtype=np.float64)
    mse_boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = np.empty(T, dtype=np.int64)
        i = 0
        while i < T:
            start = int(rng.integers(0, T))
            length = int(rng.geometric(p_geom)) if p_geom < 1.0 else 1
            length = min(length, T - i)
            for k in range(length):
                idx[i + k] = (start + k) % T
            i += length
        y_b = y[idx]; fa_b = fa[idx]; fb_b = fb[idx]
        e_a = y_b - fa_b
        e_b = y_b - fb_b
        s_aa = float(np.mean(e_a ** 2))
        s_bb = float(np.mean(e_b ** 2))
        s_ab = float(np.mean(e_a * e_b))
        denom = s_aa + s_bb - 2.0 * s_ab
        if not np.isfinite(denom) or abs(denom) < eps:
            w = 0.5
        else:
            w = (s_bb - s_ab) / denom
        if nonneg:
            w = float(np.clip(w, 0.0, 1.0))
        w_boot[b] = w
        f_comb_b = w * fa_b + (1.0 - w) * fb_b
        mse_boot[b] = float(np.mean((y_b - f_comb_b) ** 2))

    lo = float(np.quantile(w_boot, alpha / 2.0))
    hi = float(np.quantile(w_boot, 1.0 - alpha / 2.0))
    mlo = float(np.quantile(mse_boot, alpha / 2.0))
    mhi = float(np.quantile(mse_boot, 1.0 - alpha / 2.0))
    base.update({
        "w_a_lo": lo, "w_a_hi": hi,
        "mse_comb_lo": mlo, "mse_comb_hi": mhi,
        "n_boot": int(n_boot), "block_len": int(block_len), "alpha_ci": float(alpha),
    })
    return base


# ===========================================================================
# F8: CSLD Plot + Rolling DM (Giacomini-Rossi 2010)
# ===========================================================================

def plot_csld_and_rolling_dm(
    true_var: np.ndarray,
    pred_har: np.ndarray,
    pred_svd: np.ndarray,
    index: pd.Index,
    out_dir: Path,
    crisis_windows: list | None = None,
    rolling_window: int = 252,
    eps: float = EPS_DEFAULT,
    horizon: int = 1,
) -> None:
    """
    F8: Two-panel figure combining:

    Top panel: Cumulative Sum of QLIKE Loss Differentials (CSLD).
        CSLD_t = sum_{s=1}^{t} [QLIKE(HAR)_s - QLIKE(HAR+SVD)_s]

        Upward slope = SVD wins (HAR QLIKE higher). Sharp upward moves during
        crises (red shading) prove SVD's advantage is concentrated in stress.
        Reference: Giacomini & Rossi (2010, J. Applied Econometrics).

    Bottom panel: 252-day rolling DM t-statistic.
        Computed as the rolling DM test using MSE-based loss differentials.
        Dashed horizontal lines at ±1.96 mark significance bands.
        Periods above +1.96 mean HAR+SVD is significantly better in that window.

    Parameters
    ----------
    crisis_windows : list of (start, end) string pairs for crisis shading.
    rolling_window : window length for rolling DM (default 252 = 1 year).
    """
    if not _HAS_MPL:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h = np.asarray(true_var, dtype=np.float64).ravel()
    h1 = np.asarray(pred_har, dtype=np.float64).ravel()
    h2 = np.asarray(pred_svd, dtype=np.float64).ravel()
    h = np.maximum(h, eps)
    h1 = np.maximum(h1, eps)
    h2 = np.maximum(h2, eps)
    n = min(len(h), len(h1), len(h2), len(index))
    h, h1, h2, idx = h[:n], h1[:n], h2[:n], index[:n]

    # QLIKE per-period loss differential: positive = HAR worse, SVD better
    r1, r2 = h / h1, h / h2
    L1 = r1 - np.log(r1) - 1.0
    L2 = r2 - np.log(r2) - 1.0
    d_qlike = L1 - L2                 # (+) = HAR worse than SVD
    csld = np.nancumsum(d_qlike)

    # Rolling DM (MSE-based for robustness)
    mse1 = (h - h1) ** 2
    mse2 = (h - h2) ** 2
    d_mse = mse1 - mse2               # (+) = HAR has higher MSE = SVD wins
    rolling_dm = np.full(n, np.nan)
    for t in range(rolling_window, n):
        d_win = d_mse[t - rolling_window : t]
        dm_stat, _ = dmw_test(
            d_win, nlags=None, alternative="two-sided", horizon=int(horizon),
        )
        rolling_dm[t] = dm_stat

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Top: CSLD ---
    ax1 = axes[0]
    ax1.plot(idx, csld, color="#E63946", lw=1.4, label="CSLD (HAR − HAR+SVD QLIKE)")
    ax1.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    if crisis_windows:
        for start, end in crisis_windows:
            ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                        alpha=0.12, color="red", zorder=0)
    ax1.fill_between(idx, 0, csld, where=(csld > 0),
                     alpha=0.25, color="#E63946", label="SVD better")
    ax1.fill_between(idx, 0, csld, where=(csld < 0),
                     alpha=0.25, color="#2E86AB", label="HAR better")
    ax1.set_ylabel("Cumulative QLIKE differential", fontsize=10)
    ax1.set_title(
        "CSLD: SVD vs HAR — upward slope = SVD wins",
        fontweight="bold", fontsize=11,
    )
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.2)

    # --- Bottom: Rolling DM ---
    ax2 = axes[1]
    ax2.plot(idx, rolling_dm, color="#118AB2", lw=1.2,
             label=f"Rolling {rolling_window}-day DM t-stat")
    ax2.axhline(1.96, color="red", ls="--", lw=1.0, alpha=0.7, label="±1.96 (5%)")
    ax2.axhline(-1.96, color="red", ls="--", lw=1.0, alpha=0.7)
    ax2.axhline(0, color="black", lw=0.7, ls="-", alpha=0.4)
    if crisis_windows:
        for start, end in crisis_windows:
            ax2.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                        alpha=0.12, color="red", zorder=0)
    ax2.fill_between(idx, 1.96, rolling_dm, where=(np.nan_to_num(rolling_dm) > 1.96),
                     alpha=0.30, color="#E63946", label="SVD significantly better")
    ax2.set_ylabel("DM t-statistic", fontsize=10)
    ax2.set_title(
        f"Rolling {rolling_window}-day DM statistic — positive = SVD wins in window",
        fontweight="bold", fontsize=11,
    )
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle(
        f"F8: When Does SVD Help? (h={horizon})\n"
        "Top: Cumulative QLIKE differential | Bottom: Rolling DM (252-day)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / f"F8_csld_rolling_dm_h{horizon}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] F8 CSLD+Rolling DM (h={horizon}) saved.")


# ===========================================================================
# VaR Backtesting: Kupiec + Christoffersen
# ===========================================================================

def var_backtest(
    actual_returns: np.ndarray,
    variance_forecasts: np.ndarray,
    alpha_level: float = 0.01,
    eps: float = EPS_DEFAULT,
) -> dict:
    """
    VaR backtesting using Kupiec (1995) Proportion-of-Failures (PoF) test and
    Christoffersen (1998) Conditional Coverage (CC) test.

    VaR_{alpha,t+1} = -z_{alpha} * sqrt(sigma2_hat_{t+1})
    where z_alpha is the standard normal quantile, e.g. 2.326 for alpha=0.01.

    Kupiec PoF test (unconditional coverage):
        H0: E[I_t] = alpha, where I_t = 1 if return < -VaR_t.
        Likelihood ratio statistic: LR_PoF = -2*log[ alpha^m * (1-alpha)^{T-m} /
                                              phat^m * (1-phat)^{T-m} ] ~ chi^2(1).

    Christoffersen CC test (independence of violations):
        H0: I_t is i.i.d. Bernoulli(alpha) — violations are not clustered.
        LR_CC = LR_PoF + LR_ind where LR_ind tests for independence.
        LR_CC ~ chi^2(2).

    If SVD improves crisis QLIKE, it should also reduce VaR violation clustering
    during crises (passes Christoffersen CC test), linking statistical and risk
    management performance.

    Parameters
    ----------
    actual_returns    : (T,) actual log-return series (in %-units, matching variance).
    variance_forecasts: (T,) one-step-ahead conditional variance forecasts (%-squared).
    alpha_level       : VaR confidence level (e.g. 0.01 for 1% VaR).

    Returns
    -------
    dict with:
        "alpha_level"   : the requested confidence level
        "n_obs"         : number of observations
        "n_violations"  : number of VaR violations
        "phat"          : empirical violation rate
        "expected_viol" : expected violations = alpha * T
        "kupiec_stat"   : LR_PoF statistic
        "kupiec_p"      : p-value under chi^2(1)
        "christoffersen_stat" : LR_CC statistic
        "christoffersen_p"    : p-value under chi^2(2)
        "passes_kupiec"       : bool, p > 0.05
        "passes_christoffersen": bool, p > 0.05
        "violations"          : (T,) boolean array of VaR violations
    """
    from scipy.stats import chi2 as chi2_dist

    r = np.asarray(actual_returns, dtype=np.float64).ravel()
    h_hat = np.clip(np.asarray(variance_forecasts, dtype=np.float64).ravel(), eps, None)
    n = min(len(r), len(h_hat))
    r, h_hat = r[:n], h_hat[:n]

    finite_mask = np.isfinite(r) & np.isfinite(h_hat)
    r_f, h_f = r[finite_mask], h_hat[finite_mask]
    T = len(r_f)

    z_alpha = -stats.norm.ppf(alpha_level)          # e.g. 2.326 for 1%
    var_t = z_alpha * np.sqrt(h_f)                  # VaR in return units (positive)
    violations = r_f < -var_t                       # True when return exceeds VaR

    m = int(violations.sum())
    phat = m / T if T > 0 else 0.0
    expected_viol = alpha_level * T

    # Kupiec PoF (LR_uc)
    if m == 0 or m == T or phat <= 0 or phat >= 1:
        lr_uc = np.nan
        kupiec_p = np.nan
    else:
        lr_uc = -2.0 * (
            m * np.log(alpha_level / phat)
            + (T - m) * np.log((1 - alpha_level) / (1 - phat))
        )
        kupiec_p = float(1.0 - chi2_dist.cdf(lr_uc, df=1))

    # Christoffersen CC = PoF + independence (LR_cc = LR_uc + LR_ind)
    # Build transitions: n00, n01, n10, n11 (hits and misses)
    I = violations.astype(float)
    n00 = float(((1 - I[:-1]) * (1 - I[1:])).sum())
    n01 = float(((1 - I[:-1]) * I[1:]).sum())
    n10 = float((I[:-1] * (1 - I[1:])).sum())
    n11 = float((I[:-1] * I[1:]).sum())

    pi01 = n01 / (n00 + n01 + 1e-12)
    pi11 = n11 / (n10 + n11 + 1e-12)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11 + 1e-12)

    if pi <= 0 or pi >= 1 or pi01 <= 0 or pi01 >= 1 or pi11 <= 0 or pi11 >= 1:
        lr_ind = np.nan
        cc_p = np.nan
        lr_cc = np.nan
    else:
        ll_null = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        ll_alt = n00 * np.log(1 - pi01) + n01 * np.log(pi01) + \
                 n10 * np.log(1 - pi11) + n11 * np.log(pi11)
        lr_ind = -2.0 * (ll_null - ll_alt)
        if np.isfinite(lr_uc) and np.isfinite(lr_ind):
            lr_cc = lr_uc + lr_ind
            cc_p = float(1.0 - chi2_dist.cdf(lr_cc, df=2))
        else:
            lr_cc = np.nan
            cc_p = np.nan

    # Expand violations back to full index
    viol_full = np.full(n, np.nan)
    viol_full[finite_mask[:n]] = violations.astype(float)

    return {
        "alpha_level": alpha_level,
        "n_obs": T,
        "n_violations": m,
        "phat": phat,
        "expected_viol": expected_viol,
        "kupiec_stat": float(lr_uc) if np.isfinite(lr_uc) else np.nan,
        "kupiec_p": float(kupiec_p) if np.isfinite(kupiec_p) else np.nan,
        "christoffersen_stat": float(lr_cc) if np.isfinite(lr_cc) else np.nan,
        "christoffersen_p": float(cc_p) if np.isfinite(cc_p) else np.nan,
        "passes_kupiec": bool(np.isfinite(kupiec_p) and kupiec_p > 0.05),
        "passes_christoffersen": bool(np.isfinite(cc_p) and cc_p > 0.05),
        "violations": viol_full,
    }


# ===========================================================================
# F9: Feature Gate Weight Visualization
# ===========================================================================

def plot_gate_weights(
    gate_weights: np.ndarray,
    feature_names: list[str],
    index: pd.Index,
    crisis_col: np.ndarray | None,
    out_dir: Path,
    horizon: int = 1,
) -> None:
    """
    F9: Visualize the learned FeatureGate attention weights over the test period.

    The gate produces softmax weights in [0, 1/n_features] per sample. We plot:
        - Top panel: mean gate weight per feature (bar chart) — shows which
          features the model attends to on average.
        - Bottom panel: time series of SVD vs HAR mean gate weights over the
          test period, with crisis shading. If SVD features receive higher
          average weight during crises, this directly confirms the paper thesis.

    Parameters
    ----------
    gate_weights  : (T, n_features) softmax gate weights (pre-scaling).
    feature_names : list of n_features strings.
    index         : test-period datetime index.
    crisis_col    : (T,) array of 0/1 crisis flags, or None.
    out_dir       : directory for output.
    horizon       : forecast horizon (for filename and title).
    """
    if not _HAS_MPL or gate_weights is None or gate_weights.shape[1] == 0:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n, p = gate_weights.shape
    n_idx = min(n, len(index))
    gw = gate_weights[:n_idx]
    idx = index[:n_idx]

    # Identify SVD and HAR feature groups
    svd_keywords = {"f1", "sigma1", "log_sigma1", "ar", "angle", "delta_f1",
                    "cos_theta", "crisis", "entropy", "gap", "condition", "k_90",
                    "interaction"}
    feature_names_lower = [f.lower() for f in feature_names]
    is_svd = np.array([
        any(kw in fn for kw in svd_keywords) for fn in feature_names_lower
    ])
    svd_weight = gw[:, is_svd].sum(axis=1)
    har_weight = gw[:, ~is_svd].sum(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # --- Top: mean gate weight per feature ---
    ax1 = axes[0]
    mean_w = gw.mean(axis=0)
    order = np.argsort(mean_w)[::-1]
    colors_bar = ["#E63946" if is_svd[i] else "#2E86AB" for i in order]
    ax1.bar(range(p), mean_w[order], color=colors_bar, alpha=0.85, edgecolor="white")
    ax1.set_xticks(range(p))
    ax1.set_xticklabels([feature_names[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Mean gate weight")
    ax1.set_title(
        "Feature Gate: Mean attention weights (red = SVD, blue = HAR)",
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.2, axis="y")

    # --- Bottom: SVD vs HAR attention over time ---
    ax2 = axes[1]
    ax2.plot(idx, svd_weight, color="#E63946", lw=1.2, alpha=0.85,
             label=f"SVD features ({is_svd.sum()} features)")
    ax2.plot(idx, har_weight, color="#2E86AB", lw=1.2, alpha=0.85,
             label=f"HAR features ({(~is_svd).sum()} features)")
    if crisis_col is not None:
        crisis_arr = np.asarray(crisis_col, dtype=float)[:n_idx]
        crisis_mask = crisis_arr == 1
        if crisis_mask.any():
            ax2.fill_between(idx, 0, 1, where=crisis_mask,
                             alpha=0.15, color="red", transform=ax2.get_xaxis_transform(),
                             label="Crisis periods")
    ax2.set_ylabel("Total gate weight (summed over group)")
    ax2.set_title(
        "Gate weight: SVD vs HAR features over time — SVD up-weighted during crises?",
        fontweight="bold",
    )
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle(
        f"F9: DNN Feature Gate Attention Weights (h={horizon})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / f"F9_gate_weights_h{horizon}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] F9 gate weights (h={horizon}) saved.")
