# -*- coding: utf-8 -*-
"""
GARCH(1,1) benchmark for volatility forecasting.
Self-contained implementation: no external arch package.
Fit via MLE (scipy.optimize); rolling 1-step **conditional variance** forecast on test.

**Units:** `returns` must match the pipeline (here: log-returns in **percent** after
`run_experiments` scales by 100). Output is **conditional variance** in **%-squared**,
consistent with the realized variance target — not volatility (sqrt) and not annualized.

Reference: Bollerslev (1986); Student-t likelihood optional (Bollerslev 1987).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


def _garch_variance_recursion(
    returns_sq: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    sigma0_sq: float,
) -> np.ndarray:
    """
    Compute conditional variance series sigma_t^2 for GARCH(1,1).
    returns_sq: array of r_t^2. sigma0_sq: initial variance for t=0.
    Returns array of length len(returns_sq) with sigma_t^2 at each t.
    """
    n = len(returns_sq)
    sigma_sq = np.empty(n)
    sigma_sq[0] = sigma0_sq
    for t in range(1, n):
        sigma_sq[t] = omega + alpha * returns_sq[t - 1] + beta * sigma_sq[t - 1]
    return sigma_sq


def _neg_log_likelihood_gaussian(
    params: np.ndarray,
    returns: np.ndarray,
) -> float:
    """
    Negative log-likelihood for GARCH(1,1) with Gaussian innovations.
    params = [omega, alpha, beta]. Returns scalar to minimize.
    """
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0:
        return 1e20
    if alpha + beta >= 1:
        return 1e20
    # Unconditional variance for initial sigma_0^2
    sigma0_sq = omega / (1.0 - alpha - beta)
    if sigma0_sq <= 0:
        return 1e20
    returns_sq = returns.astype(np.float64) ** 2
    sigma_sq = _garch_variance_recursion(returns_sq, omega, alpha, beta, sigma0_sq)
    # Avoid log(0) or division by zero
    sigma_sq = np.maximum(sigma_sq, 1e-14)
    # Gaussian log-likelihood: -0.5 * sum( log(sigma_t^2) + r_t^2/sigma_t^2 )
    ll = -0.5 * np.sum(np.log(sigma_sq) + returns_sq / sigma_sq)
    return -ll


def _neg_log_likelihood_student_t(
    params: np.ndarray,
    returns: np.ndarray,
    nu: float,
) -> float:
    """GARCH(1,1) with conditional Student-t innovations (fixed df=nu > 2)."""
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or nu <= 2.01:
        return 1e20
    if alpha + beta >= 1:
        return 1e20
    sigma0_sq = omega / (1.0 - alpha - beta)
    if sigma0_sq <= 0:
        return 1e20
    returns_sq = returns.astype(np.float64) ** 2
    r = returns.astype(np.float64)
    sigma_sq = _garch_variance_recursion(returns_sq, omega, alpha, beta, sigma0_sq)
    sigma_sq = np.maximum(sigma_sq, 1e-14)
    n = len(r)
    const = n * (gammaln((nu + 1) / 2) - gammaln(nu / 2))
    ll = const - 0.5 * np.sum(np.log(np.pi * (nu - 2) * sigma_sq))
    ll += -(nu + 1) / 2 * np.sum(np.log(1.0 + r ** 2 / ((nu - 2) * sigma_sq)))
    return -ll


def _fit_garch_mle(
    train_returns: pd.Series,
    innovations: str,
    t_df: float,
) -> tuple[float, float, float, float] | None:
    """
    Fit GARCH(1,1) on train_returns. Returns (omega, alpha, beta, h_next) where
    h_next = omega + alpha * r_T^2 + beta * sigma_T^2 is the proper one-step-ahead
    conditional variance forecast FROM the last training date T, i.e., the first
    out-of-sample forecast h_{T+1}.

    Note: sigma_sq_train[-1] is sigma_T^2 (filtered variance AT date T, not h_{T+1}).
    Using sigma_T^2 directly as the first forecast biases results by one date because
    sigma_T^2 uses r_T as an input (in-sample) while h_{T+1} is truly out-of-sample.
    """
    r = train_returns.values.astype(np.float64)
    if len(r) < 100:
        return None
    var_r = np.var(r)
    x0 = np.array([0.01 * var_r, 0.05, 0.90])
    bounds = [(1e-9, None), (1e-9, None), (1e-9, None)]

    def obj(p):
        if innovations == "t":
            return _neg_log_likelihood_student_t(p, r, t_df)
        return _neg_log_likelihood_gaussian(p, r)

    res = minimize(
        obj,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not res.success:
        return None
    omega, alpha, beta = res.x
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return None
    sigma0_sq = omega / (1.0 - alpha - beta)
    sigma_sq_train = _garch_variance_recursion(r ** 2, omega, alpha, beta, sigma0_sq)
    # sigma_T^2 is the filtered variance at the LAST training observation.
    sigma_T_sq = float(sigma_sq_train[-1])
    # Proper one-step-ahead forecast: h_{T+1} = omega + alpha * r_T^2 + beta * sigma_T^2
    r_T_sq = float(r[-1] ** 2)
    h_next = omega + alpha * r_T_sq + beta * sigma_T_sq
    return omega, alpha, beta, h_next


def build_garch(
    returns: pd.Series,
    test_dates: pd.Index,
    *,
    innovations: str = "gaussian",
    t_df: float = 8.0,
    refit_every: int | None = None,
) -> np.ndarray:
    """
    GARCH(1,1): fit on returns strictly before each forecast origin;
    roll 1-step **conditional variance** through the test period.

    innovations: \"gaussian\" (default) or \"t\" (Student-t with fixed df=t_df).

    refit_every: if None, fit once using returns before the first test date (default).
        If a positive int, refit MLE every `refit_every` test observations using an
        expanding window (all returns strictly before the current test date).
    """
    test_dates = pd.Index(test_dates)
    test_returns = returns.reindex(test_dates).ffill()
    test_vals = test_returns.values.astype(np.float64)
    n_test = len(test_dates)
    forecasts: list[float] = []
    omega = alpha = beta = 0.0
    h_t = 0.0
    params_ready = False

    for j in range(n_test):
        td = test_dates[j]
        do_refit = False
        if refit_every is None:
            do_refit = j == 0
        else:
            if refit_every < 1:
                raise ValueError("refit_every must be None or >= 1")
            do_refit = j == 0 or (j % refit_every == 0)

        if do_refit:
            tr = returns.loc[returns.index < td].dropna()
            fitted = _fit_garch_mle(tr, innovations, t_df)
            if fitted is None:
                forecasts.append(np.nan)
                if refit_every is None and j == 0:
                    return np.full(n_test, np.nan)
                if params_ready:
                    r_test = test_vals[j]
                    h_t = omega + alpha * (r_test ** 2) + beta * h_t
                continue
            omega, alpha, beta, h_t = fitted
            params_ready = True
        elif not params_ready:
            forecasts.append(np.nan)
            continue

        forecasts.append(h_t)
        r_test = test_vals[j]
        h_t = omega + alpha * (r_test ** 2) + beta * h_t

    return np.array(forecasts)


def _gjr_variance_recursion(
    returns_sq: np.ndarray,
    negative_indicator: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    sigma0_sq: float,
) -> np.ndarray:
    """
    Conditional variance recursion for GJR-GARCH(1,1).

    sigma_t^2 = omega + (alpha + gamma * I_{r_{t-1}<0}) * r_{t-1}^2 + beta * sigma_{t-1}^2

    where I_{r<0} = 1 if r < 0 (negative return indicator).
    The gamma term captures asymmetric leverage: negative return shocks inflate
    variance more than positive shocks of equal magnitude (Glosten et al. 1993,
    J. Finance 48(5), 1779-1801).

    For stationarity: alpha + gamma/2 + beta < 1 (expected value of ARCH coefficient
    integrating over the indicator).
    """
    n = len(returns_sq)
    sigma_sq = np.empty(n)
    sigma_sq[0] = sigma0_sq
    for t in range(1, n):
        sigma_sq[t] = (
            omega
            + (alpha + gamma * negative_indicator[t - 1]) * returns_sq[t - 1]
            + beta * sigma_sq[t - 1]
        )
    return sigma_sq


def _neg_log_likelihood_gjr_gaussian(
    params: np.ndarray,
    returns: np.ndarray,
) -> float:
    """Negative Gaussian log-likelihood for GJR-GARCH(1,1)."""
    omega, alpha, gamma, beta = params
    if omega <= 0 or alpha < 0 or gamma < -alpha or beta < 0:
        return 1e20
    # Covariance-stationarity: E[ARCH] = alpha + gamma/2 + beta < 1
    if alpha + gamma / 2.0 + beta >= 1:
        return 1e20
    r = returns.astype(np.float64)
    r_sq = r ** 2
    neg_ind = (r < 0).astype(np.float64)
    # Unconditional variance: omega / (1 - alpha - gamma/2 - beta)
    denom = 1.0 - alpha - gamma / 2.0 - beta
    if denom <= 0:
        return 1e20
    sigma0_sq = omega / denom
    if sigma0_sq <= 0:
        return 1e20
    sigma_sq = _gjr_variance_recursion(r_sq, neg_ind, omega, alpha, gamma, beta, sigma0_sq)
    sigma_sq = np.maximum(sigma_sq, 1e-14)
    ll = -0.5 * np.sum(np.log(sigma_sq) + r_sq / sigma_sq)
    return -ll


def _neg_log_likelihood_gjr_student_t(
    params: np.ndarray,
    returns: np.ndarray,
    nu: float,
) -> float:
    """Negative Student-t log-likelihood for GJR-GARCH(1,1) with fixed df=nu."""
    omega, alpha, gamma, beta = params
    if omega <= 0 or alpha < 0 or gamma < -alpha or beta < 0 or nu <= 2.01:
        return 1e20
    if alpha + gamma / 2.0 + beta >= 1:
        return 1e20
    r = returns.astype(np.float64)
    r_sq = r ** 2
    neg_ind = (r < 0).astype(np.float64)
    denom = 1.0 - alpha - gamma / 2.0 - beta
    if denom <= 0:
        return 1e20
    sigma0_sq = omega / denom
    if sigma0_sq <= 0:
        return 1e20
    sigma_sq = _gjr_variance_recursion(r_sq, neg_ind, omega, alpha, gamma, beta, sigma0_sq)
    sigma_sq = np.maximum(sigma_sq, 1e-14)
    n = len(r)
    const = n * (gammaln((nu + 1) / 2) - gammaln(nu / 2))
    ll = const - 0.5 * np.sum(np.log(np.pi * (nu - 2) * sigma_sq))
    ll += -(nu + 1) / 2 * np.sum(np.log(1.0 + r_sq / ((nu - 2) * sigma_sq)))
    return -ll


def _fit_gjr_garch_mle(
    train_returns: pd.Series,
    innovations: str,
    t_df: float,
) -> tuple[float, float, float, float, float] | None:
    """
    Fit GJR-GARCH(1,1) on train_returns.

    Returns (omega, alpha, gamma, beta, h_next) where h_next is the proper
    one-step-ahead conditional variance forecast from the last training date.

    The GJR model adds a leverage term gamma to capture the asymmetry between
    positive and negative return shocks. If gamma > 0, negative shocks amplify
    variance more — the well-known leverage effect in equity markets (Black 1976;
    Christie 1982).
    """
    r = train_returns.values.astype(np.float64)
    if len(r) < 100:
        return None

    var_r = np.var(r)
    # Initial guess: alpha=0.05, gamma=0.05 (moderate leverage), beta=0.88
    x0 = np.array([0.01 * var_r, 0.05, 0.05, 0.88])
    bounds = [(1e-9, None), (1e-9, None), (0.0, None), (1e-9, None)]

    def obj(p):
        if innovations == "t":
            return _neg_log_likelihood_gjr_student_t(p, r, t_df)
        return _neg_log_likelihood_gjr_gaussian(p, r)

    res = minimize(
        obj,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not res.success:
        return None

    omega, alpha, gamma, beta = res.x
    # Validate stationarity
    if omega <= 0 or alpha < 0 or gamma < -alpha or beta < 0:
        return None
    if alpha + gamma / 2.0 + beta >= 1:
        return None

    denom = 1.0 - alpha - gamma / 2.0 - beta
    sigma0_sq = omega / denom
    r_sq = r ** 2
    neg_ind = (r < 0).astype(np.float64)
    sigma_sq_train = _gjr_variance_recursion(r_sq, neg_ind, omega, alpha, gamma, beta, sigma0_sq)

    # One-step-ahead forecast from T:
    # h_{T+1} = omega + (alpha + gamma * I_{r_T<0}) * r_T^2 + beta * sigma_T^2
    sigma_T_sq = float(sigma_sq_train[-1])
    r_T_sq = float(r[-1] ** 2)
    neg_T = float(r[-1] < 0)
    h_next = omega + (alpha + gamma * neg_T) * r_T_sq + beta * sigma_T_sq
    return omega, alpha, gamma, beta, h_next


def build_gjr_garch(
    returns: pd.Series,
    test_dates: pd.Index,
    *,
    innovations: str = "t",
    t_df: float = 8.0,
    refit_every: int | None = None,
) -> np.ndarray:
    """
    GJR-GARCH(1,1) with optional Student-t innovations.

    Default: Student-t (nu=8) — standard in equity volatility papers since
    Bollerslev (1987) showed normal GARCH systematically under-estimates tail risk.
    The asymmetric term gamma captures the leverage effect (Black 1976), producing
    better crisis-period forecasts than symmetric GARCH.

    Reference: Glosten, Jagannathan & Runkle (1993, J. Finance 48(5), 1779-1801).

    Parameters match `build_garch` for drop-in replacement. Returns conditional
    variance array in the same units as returns² (%-squared).
    """
    test_dates = pd.Index(test_dates)
    test_returns = returns.reindex(test_dates).ffill()
    test_vals = test_returns.values.astype(np.float64)
    n_test = len(test_dates)
    forecasts: list[float] = []
    omega = alpha = gamma = beta = 0.0
    h_t = 0.0
    neg_t = 0.0
    params_ready = False

    for j in range(n_test):
        td = test_dates[j]
        do_refit = (refit_every is None and j == 0) or (
            refit_every is not None and (j == 0 or j % refit_every == 0)
        )

        if do_refit:
            tr = returns.loc[returns.index < td].dropna()
            fitted = _fit_gjr_garch_mle(tr, innovations, t_df)
            if fitted is None:
                forecasts.append(np.nan)
                if refit_every is None and j == 0:
                    return np.full(n_test, np.nan)
                if params_ready:
                    r_test = test_vals[j]
                    neg_t = float(r_test < 0)
                    h_t = omega + (alpha + gamma * neg_t) * (r_test ** 2) + beta * h_t
                continue
            omega, alpha, gamma, beta, h_t = fitted
            params_ready = True

        elif not params_ready:
            forecasts.append(np.nan)
            continue

        forecasts.append(h_t)
        r_test = test_vals[j]
        neg_t = float(r_test < 0)
        h_t = omega + (alpha + gamma * neg_t) * (r_test ** 2) + beta * h_t

    return np.array(forecasts)


def build_gjr_garch_multistep(
    returns: pd.Series,
    test_dates: pd.Index,
    horizon: int,
    *,
    innovations: str = "t",
    t_df: float = 8.0,
    refit_every: int | None = None,
) -> np.ndarray:
    """
    GJR-GARCH(1,1) h-step ahead conditional variance forecast.

    For h > 1, uses the exact multi-step formula exploiting covariance stationarity:
        sigma_bar^2 = omega / (1 - alpha - gamma/2 - beta)
        E[sigma^2_{t+h}] = sigma_bar^2 + (alpha + gamma/2 + beta)^h * (sigma_t^2 - sigma_bar^2)

    This is the standard analytical GJR multi-step formula (see Franses & van Dijk 1996).
    """
    if horizon == 1:
        return build_gjr_garch(
            returns, test_dates,
            innovations=innovations, t_df=t_df, refit_every=refit_every,
        )

    test_dates = pd.Index(test_dates)
    test_returns = returns.reindex(test_dates).ffill()
    test_vals = test_returns.values.astype(np.float64)
    n_test = len(test_dates)
    forecasts: list[float] = []
    omega = alpha = gamma = beta = 0.0
    persistence = 0.0
    sigma_bar_sq = 0.0
    h_t = 0.0
    params_ready = False

    for j in range(n_test):
        td = test_dates[j]
        do_refit = (refit_every is None and j == 0) or (
            refit_every is not None and (j == 0 or j % refit_every == 0)
        )

        if do_refit:
            tr = returns.loc[returns.index < td].dropna()
            fitted = _fit_gjr_garch_mle(tr, innovations, t_df)
            if fitted is None:
                forecasts.append(np.nan)
                if refit_every is None and j == 0:
                    return np.full(n_test, np.nan)
                if params_ready:
                    r_test = test_vals[j]
                    neg_t = float(r_test < 0)
                    h_t = omega + (alpha + gamma * neg_t) * (r_test ** 2) + beta * h_t
                continue
            omega, alpha, gamma, beta, h_t = fitted
            persistence = alpha + gamma / 2.0 + beta
            sigma_bar_sq = omega / (1.0 - persistence)
            params_ready = True

        elif not params_ready:
            forecasts.append(np.nan)
            continue

        h_ahead_sq = sigma_bar_sq + (persistence ** horizon) * (h_t - sigma_bar_sq)
        forecasts.append(max(float(h_ahead_sq), 1e-14))
        r_test = test_vals[j]
        neg_t = float(r_test < 0)
        h_t = omega + (alpha + gamma * neg_t) * (r_test ** 2) + beta * h_t

    return np.array(forecasts)


def build_garch_multistep(
    returns: pd.Series,
    test_dates: pd.Index,
    horizon: int,
    *,
    innovations: str = "gaussian",
    t_df: float = 8.0,
    refit_every: int | None = None,
) -> np.ndarray:
    """
    GARCH(1,1) h-step ahead volatility forecast for horizon > 1.

    For a stationary GARCH(1,1):
        sigma_bar^2 = omega / (1 - alpha - beta)   [unconditional variance]
        E[sigma^2_{t+h}] = sigma_bar^2 + (alpha + beta)^h * (sigma_t^2 - sigma_bar^2)

    This uses the exact multi-step variance formula, exploiting the fact that
    sigma_{t+h}^2 mean-reverts to the unconditional variance at rate (alpha+beta)^h.

    For horizon 1, delegates to build_garch() for consistency.

    Parameters
    ----------
    returns    : daily log-returns (index = dates)
    test_dates : dates at which to produce forecasts
    horizon    : forecast horizon h (days)

    Returns
    -------
    array of length len(test_dates) with **conditional variance** (same units as returns²).
    """
    if horizon == 1:
        return build_garch(
            returns,
            test_dates,
            innovations=innovations,
            t_df=t_df,
            refit_every=refit_every,
        )

    test_dates = pd.Index(test_dates)
    test_returns = returns.reindex(test_dates).ffill()
    test_vals = test_returns.values.astype(np.float64)
    n_test = len(test_dates)
    forecasts: list[float] = []
    omega = alpha = beta = 0.0
    persistence = 0.0
    sigma_bar_sq = 0.0
    h_t = 0.0
    params_ready = False

    for j in range(n_test):
        td = test_dates[j]
        do_refit = False
        if refit_every is None:
            do_refit = j == 0
        else:
            if refit_every < 1:
                raise ValueError("refit_every must be None or >= 1")
            do_refit = j == 0 or (j % refit_every == 0)

        if do_refit:
            tr = returns.loc[returns.index < td].dropna()
            fitted = _fit_garch_mle(tr, innovations, t_df)
            if fitted is None:
                forecasts.append(np.nan)
                if refit_every is None and j == 0:
                    return np.full(n_test, np.nan)
                if params_ready:
                    r_test = test_vals[j]
                    h_t = omega + alpha * (r_test ** 2) + beta * h_t
                continue
            omega, alpha, beta, h_t = fitted
            persistence = alpha + beta
            sigma_bar_sq = omega / (1.0 - persistence)
            params_ready = True
        elif not params_ready:
            forecasts.append(np.nan)
            continue

        h_ahead_sq = sigma_bar_sq + (persistence ** horizon) * (h_t - sigma_bar_sq)
        h_ahead_sq = max(float(h_ahead_sq), 1e-14)
        forecasts.append(h_ahead_sq)
        r_test = test_vals[j]
        h_t = omega + alpha * (r_test ** 2) + beta * h_t

    return np.array(forecasts)
