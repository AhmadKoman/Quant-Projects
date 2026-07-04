"""
Economic evaluation utilities for volatility forecasts.

The module covers three strands of risk-management evaluation discussed in the
manuscript:

1. **VaR / ES backtesting** with strictly consistent joint scoring
   (Fissler & Ziegel, 2016, *Annals of Statistics* 44(4), 1680-1707) and
   the standard suite of unconditional / independence / spectral
   diagnostics: Kupiec (1995, *J. Derivatives*), Christoffersen (1998,
   *Int. Econ. Rev.*) conditional coverage, the Christoffersen-Pelletier
   (2004, *J. Empirical Finance*) duration test, the Engle-Manganelli
   (2004, *JBES*) Dynamic-Quantile test, the Du-Escanciano (2017,
   *Mgmt. Science*) cumulative-violations test, and the
   Acerbi-Szekely (2014, *Risk*) Z2 ES backtest.  A Diebold-Mariano-West
   test on the FZ0 differential implements the joint VaR/ES dominance
   test of Patton-Ziegel-Chen (2019, *J. Financial Economics*).
2. **Strategy-level evaluation**: classical volatility targeting plus the
   *fractional-Kelly* extension that uses posterior-predictive variance
   (\\hat{s}_t^2) to scale the leverage gain (MacLean-Thorp-Ziemba 2010;
   Rujeerapaiboon-Kuhn-Wiesemann 2016, *Mgmt. Science*; Glasserman-Xu
   2014, *Mgmt. Science*).
3. **Standard finance-journal economic-value tests**: Moreira-Muir (2017,
   *J. Finance*) volatility-managed alpha regression and the
   Fleming-Kirby-Ostdiek (2001, *J. Finance*; 2003, *JFE*) certainty-
   equivalent fee for a CRRA investor.

All functions are deterministic and designed for strict, no-leakage
evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

DistName = Literal["gaussian", "student_t"]


def _to_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def var_es_from_variance(
    pred_var: np.ndarray,
    *,
    alpha: float,
    dist: DistName = "gaussian",
    df: float | None = None,
    mu: float = 0.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map a conditional variance forecast to VaR and ES for returns.

    Conventions:
      - returns r_t can be in % units or decimals; VaR/ES will match the same units.
      - VaR_alpha is the alpha-quantile of returns (typically negative for left tail).
      - Violation is r_t < VaR_alpha.
    """
    h = np.maximum(_to_1d(pred_var), eps)
    sigma = np.sqrt(h)
    a = float(alpha)
    if not (0.0 < a < 0.5):
        raise ValueError("alpha must be in (0, 0.5) for left-tail VaR/ES.")

    if dist == "gaussian":
        z = float(stats.norm.ppf(a))
        var = mu + sigma * z
        es = mu - sigma * (stats.norm.pdf(z) / a)
        return var, es

    if dist == "student_t":
        nu = float(df) if df is not None else 8.0
        if nu <= 2.01:
            raise ValueError("Student-t df must be > 2 for finite variance.")
        # Standardize t innovations to unit variance:
        # If u ~ t_nu, then Var(u) = nu/(nu-2). Set e = u / sqrt(nu/(nu-2)).
        scale = np.sqrt(nu / (nu - 2.0))
        q = float(stats.t.ppf(a, df=nu))
        var = mu + sigma * (q / scale)
        # ES for standardized Student-t:
        # ES_alpha = mu + sigma * E[e | e <= q_std], where q_std = q/scale.
        # For u ~ t_nu: E[u | u <= q] = - ( (nu + q^2) / ( (nu-1) * a ) ) * f_t(q)
        # Then divide by scale for standardized e.
        ft = float(stats.t.pdf(q, df=nu))
        es_u = -((nu + q * q) / ((nu - 1.0) * a)) * ft
        es = mu + sigma * (es_u / scale)
        return var, es

    raise ValueError(f"Unknown dist: {dist}")


def fz0_var_es_loss(
    returns: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    *,
    alpha: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Fissler–Ziegel (2016) strictly consistent scoring function for (VaR, ES).

    We implement a common FZ0 form (up to affine transformations) that is
    valid for evaluating joint VaR/ES forecasts.
    """
    r = _to_1d(returns)
    q = _to_1d(var)
    e = _to_1d(es)
    a = float(alpha)
    if r.shape != q.shape or r.shape != e.shape:
        raise ValueError("returns, var, es must have the same shape.")
    # Stabilize ES denominator (ES is negative for left tail; keep magnitude away from 0)
    e_safe = np.where(np.abs(e) < eps, np.sign(e) * eps, e)
    ind = (r <= q).astype(np.float64)
    # FZ0 score (one of standard parametrizations)
    # Lower is better.
    score = (1.0 / a) * (ind - a) * (q - r) + (1.0 / a) * ind * (e - q) + (e - q) / e_safe
    return score


@dataclass
class VaRBacktestResult:
    alpha: float
    dist: str
    df: float | None
    n: int
    phat: float
    kupiec_p: float
    christoffersen_p: float
    passes_kupiec: bool
    passes_christoffersen: bool
    mean_fz0: float


def backtest_var_es(
    returns: np.ndarray,
    pred_var: np.ndarray,
    *,
    alpha: float,
    dist: DistName = "gaussian",
    df: float | None = None,
    mu: float = 0.0,
    eps: float = 1e-12,
) -> tuple[VaRBacktestResult, dict[str, np.ndarray]]:
    """
    Compute VaR/ES series, Kupiec + Christoffersen p-values, and mean FZ0 score.
    Returns the scalar summary and the full series (var, es, violations, fz0).
    """
    r = _to_1d(returns)
    h = _to_1d(pred_var)
    if r.shape != h.shape:
        raise ValueError("returns and pred_var must have same shape.")
    var, es = var_es_from_variance(h, alpha=alpha, dist=dist, df=df, mu=mu, eps=eps)
    violations = (r < var).astype(int)

    # Kupiec + Christoffersen from evaluation.py logic (re-implemented to avoid circular import)
    n = int(len(r))
    x = int(violations.sum())
    a = float(alpha)
    phat = float(x / n) if n > 0 else float("nan")

    # Kupiec LR_uc
    if x == 0 or x == n:
        kupiec_p = 0.0
    else:
        lr_uc = -2.0 * (
            (n - x) * np.log((1 - a) / (1 - phat)) + x * np.log(a / phat)
        )
        kupiec_p = float(1.0 - stats.chi2.cdf(lr_uc, df=1))

    # Christoffersen independence
    v = violations
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        if v[i - 1] == 0 and v[i] == 0:
            n00 += 1
        elif v[i - 1] == 0 and v[i] == 1:
            n01 += 1
        elif v[i - 1] == 1 and v[i] == 0:
            n10 += 1
        else:
            n11 += 1
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi1 = (n01 + n11) / (n00 + n01 + n10 + n11) if (n > 1) else 0.0
    def _ll(p, a_, b_):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return a_ * np.log(1 - p) + b_ * np.log(p)
    ll_ind = _ll(pi01, n00, n01) + _ll(pi11, n10, n11)
    ll_null = _ll(pi1, n00 + n10, n01 + n11)
    lr_ind = -2.0 * (ll_null - ll_ind)
    christoffersen_p = float(1.0 - stats.chi2.cdf(lr_ind, df=1))

    passes_k = bool(kupiec_p >= 0.05)
    passes_c = bool(christoffersen_p >= 0.05)

    fz0 = fz0_var_es_loss(r, var, es, alpha=alpha, eps=eps)
    res = VaRBacktestResult(
        alpha=a,
        dist=str(dist),
        df=float(df) if df is not None else None,
        n=n,
        phat=float(phat),
        kupiec_p=float(kupiec_p),
        christoffersen_p=float(christoffersen_p),
        passes_kupiec=passes_k,
        passes_christoffersen=passes_c,
        mean_fz0=float(np.mean(fz0)),
    )
    series = {"VaR": var, "ES": es, "violations": violations.astype(int), "FZ0": fz0}
    return res, series


@dataclass
class VolTargetResult:
    target_vol_annual: float
    max_leverage: float
    tc_bps: float
    n: int
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    turnover: float


def volatility_targeting(
    returns: np.ndarray,
    pred_var_next: np.ndarray,
    *,
    target_vol_annual: float = 0.10,
    max_leverage: float = 3.0,
    tc_bps: float = 5.0,
    trading_days: int = 252,
    eps: float = 1e-12,
) -> tuple[VolTargetResult, dict[str, np.ndarray]]:
    """
    Volatility targeting using next-period variance forecasts.

    Inputs:
      returns       : realized next-period returns r_{t+1} aligned to the prediction index (same length)
      pred_var_next : variance forecast for that next-period return (same length)

    Weight rule (applied with no look-ahead inside the aligned series):
      w_t = min(max_leverage, target_vol_daily / sqrt(pred_var_next_t))

    Transaction costs:
      cost_t = (tc_bps/1e4) * |w_t - w_{t-1}|   (proportional to turnover)
    """
    r = _to_1d(returns)
    h = np.maximum(_to_1d(pred_var_next), eps)
    if r.shape != h.shape:
        raise ValueError("returns and pred_var_next must have the same shape.")
    n = int(len(r))
    if n < 20:
        raise ValueError("Need at least 20 observations for strategy evaluation.")

    tv_daily = float(target_vol_annual) / np.sqrt(float(trading_days))
    w = np.minimum(float(max_leverage), tv_daily / np.sqrt(h))
    # Apply weights contemporaneously to the aligned next-period returns (no extra shift needed)
    dw = np.empty_like(w)
    dw[0] = w[0]
    dw[1:] = w[1:] - w[:-1]
    tc = float(tc_bps) / 1e4
    costs = tc * np.abs(dw)
    rp = w * r - costs

    # Performance stats (returns are in whatever units r uses; assume decimals if you want Sharpe)
    mu = float(np.mean(rp))
    sig = float(np.std(rp, ddof=1))
    ann_return = (1.0 + mu) ** float(trading_days) - 1.0 if np.all(np.isfinite([mu])) else float("nan")
    ann_vol = sig * np.sqrt(float(trading_days)) if sig > 0 else float("nan")
    sharpe = (mu / sig) * np.sqrt(float(trading_days)) if sig > 0 else float("nan")

    # Drawdown
    eq = np.cumprod(1.0 + rp)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    max_dd = float(np.min(dd))

    turnover = float(np.mean(np.abs(dw)))
    res = VolTargetResult(
        target_vol_annual=float(target_vol_annual),
        max_leverage=float(max_leverage),
        tc_bps=float(tc_bps),
        n=n,
        ann_return=float(ann_return),
        ann_vol=float(ann_vol),
        sharpe=float(sharpe),
        max_drawdown=float(max_dd),
        turnover=float(turnover),
    )
    series = {"w": w, "rp": rp, "costs": costs, "eq": eq, "drawdown": dd}
    return res, series


# ===========================================================================
# Advanced VaR / ES diagnostic tests
# ===========================================================================

def acerbi_szekely_z2(
    returns: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    *,
    alpha: float,
    n_boot: int = 5000,
    block_length: int | None = None,
    seed: int = 7,
) -> dict:
    """
    Acerbi-Szekely (2014, *Risk*) Z2 expected-shortfall backtest.

    Z2_t = r_t · 1{r_t < VaR_t} / (alpha · ES_t),  ZBar = mean(Z2) + 1.

    Under correct ES specification ``E[Z2] = -1`` so the centred statistic
    has mean zero and is asymptotically standard normal in the i.i.d.
    case.  We compute a stationary block-bootstrap p-value (Politis-Romano
    1994) for the alternative ``ZBar > 0`` (ES is *under-estimated*) and a
    two-sided p-value for symmetric departures.  The block length defaults
    to ``floor(T^{1/3})``.

    Returns a dict with the test statistic, the bootstrap p-values, and
    the empirical violation rate (for cross-checks against Kupiec).
    """
    r = _to_1d(returns)
    q = _to_1d(var)
    e = _to_1d(es)
    if r.shape != q.shape or r.shape != e.shape:
        raise ValueError("returns, var, es must be of equal length.")
    a = float(alpha)
    T = int(r.size)
    if T < 30:
        return {
            "alpha": a,
            "T": T,
            "n_violations": int((r < q).sum()),
            "ZBar": float("nan"),
            "p_one_sided": float("nan"),
            "p_two_sided": float("nan"),
        }
    ind = (r < q).astype(np.float64)
    e_safe = np.where(np.abs(e) < 1e-12, np.sign(e) * 1e-12 + 1e-12, e)
    z2 = r * ind / (a * e_safe)
    zbar = float(np.mean(z2) + 1.0)

    rng = np.random.default_rng(int(seed))
    bl = int(block_length) if block_length else max(1, int(np.floor(T ** (1.0 / 3.0))))
    bl = int(np.clip(bl, 1, T))
    n_blocks = int(np.ceil(T / bl))
    boot = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        starts = rng.integers(0, T, size=n_blocks)
        idx = (starts[:, None] + np.arange(bl)[None, :]) % T
        idx = idx.ravel()[:T]
        boot[b] = float(np.mean(z2[idx]) + 1.0)
    centred = boot - boot.mean()
    p_one = float(np.mean(centred >= zbar))
    p_two = float(np.mean(np.abs(centred) >= abs(zbar)))
    return {
        "alpha": a,
        "T": T,
        "n_violations": int(ind.sum()),
        "ZBar": zbar,
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "block_length": bl,
        "n_boot": int(n_boot),
    }


def engle_manganelli_dq_test(
    returns: np.ndarray,
    var: np.ndarray,
    *,
    alpha: float,
    n_lags: int = 4,
) -> dict:
    """
    Engle-Manganelli (2004, *JBES*) Dynamic-Quantile out-of-sample DQ
    test.  Define the de-meaned hit ``H_t = 1{r_t < VaR_t} - alpha``.
    Under correct specification ``H_t`` has mean zero and is uncorrelated
    with any ``F_{t-1}``-measurable instrument; we use the constant, the
    first ``n_lags`` of ``H``, and the contemporaneous VaR level as
    instruments.  The DQ statistic is

    .. math::
        DQ = \\frac{H' Z (Z' Z)^{-1} Z' H}{\\alpha (1 - \\alpha)} \\sim \\chi^2_{k},

    where ``k`` is the column rank of ``Z`` (Engle-Manganelli 2004 Eq. 8).
    """
    r = _to_1d(returns)
    q = _to_1d(var)
    if r.shape != q.shape:
        raise ValueError("returns and var must have the same length.")
    a = float(alpha)
    T = int(r.size)
    if T < 4 * max(n_lags, 1) + 5:
        return {"alpha": a, "DQ_stat": float("nan"), "p_value": float("nan"),
                "df": 0, "T": T}
    H = (r < q).astype(np.float64) - a
    L = int(max(0, n_lags))
    cols = [np.ones(T - L)]
    for k in range(1, L + 1):
        cols.append(H[L - k : T - k])
    cols.append(q[L:])
    Z = np.column_stack(cols)
    Hy = H[L:]
    G = Z.T @ Z
    try:
        Ginv = np.linalg.pinv(G)
    except np.linalg.LinAlgError:
        return {"alpha": a, "DQ_stat": float("nan"), "p_value": float("nan"),
                "df": 0, "T": T}
    score = Z.T @ Hy
    dq = float(score @ Ginv @ score / (a * (1.0 - a) + 1e-30))
    df = int(np.linalg.matrix_rank(G))
    p = float(1.0 - stats.chi2.cdf(dq, df=df)) if df > 0 else float("nan")
    return {"alpha": a, "DQ_stat": dq, "p_value": p, "df": df, "T": int(T - L)}


def christoffersen_pelletier_duration_test(
    returns: np.ndarray,
    var: np.ndarray,
) -> dict:
    """
    Christoffersen-Pelletier (2004, *Journal of Empirical Finance*)
    duration independence test — simplified implementation.

    Gap lengths ``D_i`` between VaR violations are i.i.d. geometric under
    correct unconditional coverage *and* independence.  We embed the
    geometric law in the two-parameter Weibull family (shape ``k``,
    scale ``lambda``) which nests the exponential case ``k = 1``, and
    form the likelihood-ratio statistic ``-2 [\\ell_{exp} -
    \\ell_{Wei}]`` with ``H_0: k = 1`` (CP-2004 §3).

    ``scipy.stats.weibull_min`` supplies stable MLEs for the alternative.
    """
    r = _to_1d(returns)
    q = _to_1d(var)
    if r.shape != q.shape:
        raise ValueError("returns and var must have the same length.")
    H = (r < q).astype(int)
    idx_v = np.where(H == 1)[0]
    if idx_v.size < 3:
        return {"LR_dur": float("nan"), "p_value": float("nan"),
                "n_durations": int(max(idx_v.size - 1, 0)), "shape_hat": float("nan")}
    d = np.maximum(np.diff(idx_v).astype(np.float64), 1.0)
    n_d = int(d.size)
    try:
        from scipy.stats import weibull_min

        k_alt, loc_alt, scale_alt = weibull_min.fit(d, floc=0.0)
        ll_alt = float(np.sum(weibull_min.logpdf(d, k_alt, loc=loc_alt, scale=scale_alt)))
        scale_null = float(np.mean(d))
        ll_null = float(np.sum(stats.expon.logpdf(d, scale=scale_null)))
        lr = float(max(0.0, -2.0 * (ll_null - ll_alt)))
        p = float(1.0 - stats.chi2.cdf(lr, df=1))
        return {
            "LR_dur": lr,
            "p_value": p,
            "n_durations": n_d,
            "shape_hat": float(k_alt),
        }
    except Exception:
        return {"LR_dur": float("nan"), "p_value": float("nan"),
                "n_durations": n_d, "shape_hat": float("nan")}


def du_escanciano_test(
    returns: np.ndarray,
    var: np.ndarray,
    *,
    alpha: float,
    n_lags: int = 5,
) -> dict:
    """
    Du-Escanciano (2017, *Mgmt. Science*) cumulative-violations test.
    The score is the studentised Box-Pierce statistic on the demeaned
    hits ``H_t``,

    .. math::
        BP = T \\sum_{k=1}^{m} \\hat\\rho_k^2 \\to \\chi^2_m,

    with ``\\hat\\rho_k = \\widehat{Cov}(H_t, H_{t-k}) / [\\alpha(1-\\alpha)]``.
    A small p-value indicates serial dependence in the violations.
    """
    r = _to_1d(returns)
    q = _to_1d(var)
    if r.shape != q.shape:
        raise ValueError("returns and var must have the same length.")
    a = float(alpha)
    T = int(r.size)
    if T < n_lags + 5:
        return {"alpha": a, "BP_stat": float("nan"), "p_value": float("nan"),
                "n_lags": int(n_lags), "T": T}
    H = (r < q).astype(np.float64) - a
    var_h = max(a * (1.0 - a), 1e-12)
    bp = 0.0
    for k in range(1, n_lags + 1):
        gamma_k = float(np.mean(H[k:] * H[:-k]))
        rho_k = gamma_k / var_h
        bp += T * (rho_k ** 2)
    p = float(1.0 - stats.chi2.cdf(bp, df=int(n_lags)))
    return {"alpha": a, "BP_stat": float(bp), "p_value": p,
            "n_lags": int(n_lags), "T": T}


def fz0_dm_test(
    fz0_a: np.ndarray,
    fz0_b: np.ndarray,
    *,
    horizon: int = 1,
    base_factor: float = 1.5,
    min_lag: int = 1,
) -> dict:
    """
    Diebold-Mariano-West test on the FZ0 differential between two
    models, implementing the joint VaR-ES dominance test of
    Patton-Ziegel-Chen (2019, *J. Financial Economics*) Eq. 3.

    The HAC bandwidth follows the horizon-aware rule
    ``max(h-1, floor(base_factor * T^{1/3}))`` matching the rest of the
    codebase.
    """
    a = _to_1d(fz0_a)
    b = _to_1d(fz0_b)
    if a.shape != b.shape:
        raise ValueError("fz0_a and fz0_b must have the same length.")
    d = a - b
    T = int(d.size)
    if T < 5:
        return {"DM_stat": float("nan"), "p_value": float("nan"),
                "mean_diff": float(np.mean(d)) if T else float("nan"),
                "T": T, "nlags": 0}

    # Horizon-aware NW bandwidth (delegated to evaluation.py to share the
    # exact same formula as the QLIKE-DM tests).
    overlap = max(int(horizon) - 1, 0)
    rule = int(np.floor(base_factor * (T ** (1.0 / 3.0))))
    L = int(max(overlap, rule, int(min_lag)))
    L = int(min(L, T - 1))

    dm = d - d.mean()
    var = float(np.dot(dm, dm)) / T
    for ell in range(1, L + 1):
        w = 1.0 - ell / (L + 1.0)  # Bartlett kernel
        var += 2.0 * w * float(np.dot(dm[ell:], dm[:-ell])) / T
    var = max(var, 1e-12)
    se = float(np.sqrt(var / T))
    stat = float(np.mean(d) / se) if se > 0 else float("nan")
    p_two = float(2.0 * (1.0 - stats.norm.cdf(abs(stat)))) if np.isfinite(stat) else float("nan")
    return {
        "DM_stat": stat,
        "p_value": p_two,
        "mean_diff": float(np.mean(d)),
        "T": T,
        "nlags": L,
    }


# ===========================================================================
# VaR / ES tournament across multiple alphas, models, regimes, distributions
# ===========================================================================

def var_es_tournament(
    *,
    returns_dec: np.ndarray,
    pred_var_dec: dict,
    alphas: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05),
    distributions: tuple[tuple[str, float | None], ...] = (
        ("gaussian", None),
        ("student_t", 8.0),
    ),
    regime_labels: np.ndarray | None = None,
    horizon: int = 1,
    fz0_baseline: str | None = None,
    z2_n_boot: int = 5000,
    z2_seed: int = 7,
    du_escanciano_lags: int = 5,
    dq_lags: int = 4,
) -> dict:
    """
    Run the full VaR/ES tournament across an arbitrary collection of
    models, tail levels ``alphas``, return distributions, and (optionally)
    regime strata.

    Parameters
    ----------
    returns_dec   : (T,) realised next-period returns, in *decimal* units,
                    aligned to the prediction index.
    pred_var_dec  : dict ``{model_name: (T,) variance forecast in decimal
                    units}``; all arrays must have length ``T``.
    alphas        : tuple of left-tail probabilities.
    distributions : tuple ``((dist_name, df_or_None), ...)``.
    regime_labels : optional (T,) string labels (e.g. ``"Calm"`` /
                    ``"Crisis"``) used to stratify every test.
    horizon       : forecast horizon used by the FZ0-DM HAC bandwidth.
    fz0_baseline  : if not ``None``, the FZ0-DM test is run for every
                    other model against this baseline (typically ``"HAR"``).
    z2_n_boot, z2_seed, du_escanciano_lags, dq_lags : hyper-parameters
                    forwarded to the corresponding individual tests.

    Returns
    -------
    dict with keys:
        ``"per_model"`` : nested dict of per-(model, alpha, dist, regime)
                          backtests.  Each leaf carries the Kupiec /
                          Christoffersen / DQ / Du-Escanciano /
                          duration / Z2 results plus mean FZ0.
        ``"fz0_dm"``    : pairwise FZ0-DM results when ``fz0_baseline``
                          is given.
        ``"violation_overlap"`` : a sparse cross-tabulation of the
                          1{r_t<VaR_t} indicator series across models —
                          the diagnostic for the "identical-violations"
                          collision flagged in the manuscript plan.
    """
    r = _to_1d(returns_dec)
    T = int(r.size)
    models = list(pred_var_dec.keys())
    for m in models:
        h = _to_1d(pred_var_dec[m])
        if h.size != T:
            raise ValueError(
                f"variance forecast for {m!r} has length {h.size}, "
                f"expected {T}."
            )

    if regime_labels is None:
        regime_labels = np.full(T, "All", dtype=object)
    else:
        regime_labels = np.asarray(regime_labels)
        if regime_labels.shape[0] != T:
            raise ValueError("regime_labels must have length T.")

    unique_regimes = ["All"] + [r_ for r_ in pd_unique(regime_labels) if r_ != "All"]

    per_model: dict[str, dict] = {}
    fz0_panel: dict[tuple, np.ndarray] = {}

    for m in models:
        h = np.maximum(_to_1d(pred_var_dec[m]), 1e-30)
        per_model[m] = {}
        for dist_name, dist_df in distributions:
            for a in alphas:
                var_arr, es_arr = var_es_from_variance(
                    h, alpha=float(a), dist=dist_name, df=dist_df, mu=0.0
                )
                fz0 = fz0_var_es_loss(r, var_arr, es_arr, alpha=float(a))
                fz0_panel[(m, dist_name, float(a))] = fz0
                key = f"{dist_name}_a{float(a):.4f}"
                per_model[m][key] = {}
                for reg in unique_regimes:
                    if reg == "All":
                        mask = np.ones(T, dtype=bool)
                    else:
                        mask = regime_labels == reg
                    if mask.sum() < 30:
                        continue
                    rr = r[mask]
                    vv = var_arr[mask]
                    ee = es_arr[mask]
                    ff = fz0[mask]
                    res, _ = backtest_var_es(rr, h[mask],
                                             alpha=float(a), dist=dist_name,
                                             df=dist_df, mu=0.0)
                    z2 = acerbi_szekely_z2(rr, vv, ee, alpha=float(a),
                                           n_boot=z2_n_boot, seed=z2_seed)
                    dq = engle_manganelli_dq_test(rr, vv, alpha=float(a),
                                                  n_lags=dq_lags)
                    de = du_escanciano_test(rr, vv, alpha=float(a),
                                            n_lags=du_escanciano_lags)
                    cp = christoffersen_pelletier_duration_test(rr, vv)
                    per_model[m][key][reg] = {
                        "n": int(mask.sum()),
                        "phat": res.phat,
                        "kupiec_p": res.kupiec_p,
                        "christoffersen_p": res.christoffersen_p,
                        "duration_p": cp["p_value"],
                        "duration_LR": cp["LR_dur"],
                        "duration_shape": cp["shape_hat"],
                        "dq_p": dq["p_value"],
                        "dq_stat": dq["DQ_stat"],
                        "dq_df": dq["df"],
                        "du_escanciano_p": de["p_value"],
                        "du_escanciano_BP": de["BP_stat"],
                        "z2_ZBar": z2["ZBar"],
                        "z2_p_one_sided": z2["p_one_sided"],
                        "z2_p_two_sided": z2["p_two_sided"],
                        "mean_fz0": float(np.mean(ff)),
                        "mean_VaR": float(np.mean(vv)),
                        "mean_ES": float(np.mean(ee)),
                    }

    # FZ0-DM grid against optional baseline -----------------------------------
    fz0_dm_results: dict[str, dict] = {}
    if fz0_baseline is not None and fz0_baseline in models:
        for dist_name, dist_df in distributions:
            for a in alphas:
                base_loss = fz0_panel[(fz0_baseline, dist_name, float(a))]
                for m in models:
                    if m == fz0_baseline:
                        continue
                    cand_loss = fz0_panel[(m, dist_name, float(a))]
                    res = fz0_dm_test(base_loss, cand_loss, horizon=int(horizon))
                    key = f"{m}_vs_{fz0_baseline}_{dist_name}_a{float(a):.4f}"
                    fz0_dm_results[key] = {
                        "candidate": m,
                        "baseline": fz0_baseline,
                        "alpha": float(a),
                        "distribution": dist_name,
                        "DM_stat": res["DM_stat"],
                        "p_value": res["p_value"],
                        "mean_diff": res["mean_diff"],
                        "T": res["T"],
                        "nlags": res["nlags"],
                    }

    # Identical-violations cross-tabulation -----------------------------------
    overlap = identical_violation_diagnostic(
        returns_dec=r,
        pred_var_dec=pred_var_dec,
        alphas=alphas,
        distributions=distributions,
    )

    return {
        "per_model": per_model,
        "fz0_dm": fz0_dm_results,
        "violation_overlap": overlap,
        "regimes": list(unique_regimes),
        "alphas": list(map(float, alphas)),
        "distributions": [list(d) for d in distributions],
        "horizon": int(horizon),
    }


def pd_unique(arr: np.ndarray) -> list:
    """
    Order-preserving unique that works on object/string arrays without
    requiring pandas at this layer.
    """
    seen: set = set()
    out: list = []
    for v in arr:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def identical_violation_diagnostic(
    *,
    returns_dec: np.ndarray,
    pred_var_dec: dict,
    alphas: tuple[float, ...],
    distributions: tuple[tuple[str, float | None], ...],
) -> dict:
    """
    Cross-tabulate the violation indicator ``1{r_t < VaR_t}`` across
    every (model, alpha, dist) cell.  The plan flagged the bit-for-bit
    identical 1%-VaR violations of HAR and HAR+SVD as a likely
    quantile-clipping collision; this diagnostic prints the pairwise
    violation overlap and the corresponding Hamming distances so the
    manuscript can either (i) confirm the clipping artefact or
    (ii) document it as a genuine consequence of the location-scale
    quantile mapping.

    Returns a dict ``{(dist, alpha): {"models": [...], "violations":
    DataFrame-like 2-d list of int, "pairwise_overlap": dict, "hamming":
    dict}}``.
    """
    r = _to_1d(returns_dec)
    T = int(r.size)
    models = list(pred_var_dec.keys())
    out: dict = {}
    for dist_name, dist_df in distributions:
        for a in alphas:
            viols = {}
            for m in models:
                h = np.maximum(_to_1d(pred_var_dec[m]), 1e-30)
                v_arr, _ = var_es_from_variance(
                    h, alpha=float(a), dist=dist_name, df=dist_df, mu=0.0
                )
                viols[m] = (r < v_arr).astype(int)
            pairs = {}
            hamming = {}
            for i, m1 in enumerate(models):
                for m2 in models[i + 1:]:
                    overlap = int(np.sum(viols[m1] & viols[m2]))
                    only_a = int(np.sum(viols[m1] & (1 - viols[m2])))
                    only_b = int(np.sum((1 - viols[m1]) & viols[m2]))
                    pairs[f"{m1}|{m2}"] = {
                        "both": overlap,
                        f"only_{m1}": only_a,
                        f"only_{m2}": only_b,
                    }
                    hamming[f"{m1}|{m2}"] = only_a + only_b
            key = f"{dist_name}_a{float(a):.4f}"
            out[key] = {
                "models": models,
                "T": T,
                "n_violations": {m: int(v.sum()) for m, v in viols.items()},
                "pairwise_overlap": pairs,
                "hamming_distance": hamming,
            }
    return out


# ===========================================================================
# Fractional-Kelly volatility targeting
# ===========================================================================

@dataclass
class FractionalKellyResult:
    rule: str
    target_vol_annual: float
    max_leverage: float
    tc_bps: float
    n: int
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    turnover: float
    delta_g_log: float
    mean_fraction: float


def kelly_fractional_targeting(
    returns: np.ndarray,
    pred_var_next: np.ndarray,
    posterior_var_log: np.ndarray | None,
    *,
    rule: Literal["bayes", "rkw", "kl_tilt"] = "bayes",
    kappa: float = 1.0,
    target_vol_annual: float = 0.10,
    max_leverage: float = 3.0,
    tc_bps: float = 5.0,
    trading_days: int = 252,
    eps: float = 1e-12,
) -> tuple[FractionalKellyResult, dict[str, np.ndarray]]:
    """
    Fractional-Kelly volatility-targeting strategy.  Strictly nests
    :func:`volatility_targeting` (recovered when ``posterior_var_log``
    is ``None`` or zero).

    Three risk-aversion rules are supported, each with a published
    citation:

    * ``"bayes"`` -- Bayesian shrinkage ``f^*_t = 1 / (1 + s_t^2)`` from
      MacLean-Thorp-Ziemba (2010, *Quant. Finance*) Eq. 5; matches the
      maximum-expected-log-growth rule under conjugate-normal predictive
      uncertainty.
    * ``"rkw"`` -- Rujeerapaiboon-Kuhn-Wiesemann (2016, *Mgmt. Science*)
      distributionally-robust Kelly fraction ``f^*_t = max(0, 1 -
      \\kappa s_t)``.
    * ``"kl_tilt"`` -- Glasserman-Xu (2014, *Mgmt. Science*) KL-tilted
      Kelly ``f^*_t = max(0, 1 - \\kappa s_t^2)``.

    The leverage rule then becomes
    ``w_t = min(L_max, f^*_t * sigma_target_daily / sqrt(h_t))``.

    Posterior-predictive log-variance ``s_t^2 = Var(log h_{t+1} | F_t)``
    is harvested from :mod:`bayesian_uncertainty` outputs (e.g. HAR
    conjugate posterior, HAR+SVD block-bootstrap quantile width,
    MC-Dropout dispersion).  When ``posterior_var_log is None`` the
    fraction collapses to one and the strategy is identical to the
    deterministic vol target.

    The recovered Kelly growth gain (MacLean-Thorp-Ziemba 2010 Eq. 7)

    .. math::
        \\Delta G_{\\log} = \\tfrac{1}{2}\\, \\mathrm{SR}^2 \\, \\bar{s}^2

    is reported on the same axis as Sharpe so the panel can be
    interpreted as Pareto-improvement evidence over the deterministic
    vol target.
    """
    r = _to_1d(returns)
    h = np.maximum(_to_1d(pred_var_next), eps)
    if r.shape != h.shape:
        raise ValueError("returns and pred_var_next must have the same shape.")
    n = int(len(r))
    if n < 20:
        raise ValueError("Need at least 20 observations for strategy evaluation.")

    if posterior_var_log is None:
        s2 = np.zeros_like(h)
    else:
        s2 = np.maximum(_to_1d(posterior_var_log), 0.0)
        if s2.shape != h.shape:
            raise ValueError("posterior_var_log must align with pred_var_next.")
    s_t = np.sqrt(s2)

    rule_l = str(rule).lower()
    if rule_l == "bayes":
        fraction = 1.0 / (1.0 + s2)
    elif rule_l == "rkw":
        fraction = np.clip(1.0 - float(kappa) * s_t, 0.0, 1.0)
    elif rule_l == "kl_tilt":
        fraction = np.clip(1.0 - float(kappa) * s2, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown Kelly rule: {rule!r}")

    tv_daily = float(target_vol_annual) / np.sqrt(float(trading_days))
    w = np.minimum(float(max_leverage), fraction * tv_daily / np.sqrt(h))
    dw = np.empty_like(w)
    dw[0] = w[0]
    dw[1:] = w[1:] - w[:-1]
    tc = float(tc_bps) / 1e4
    costs = tc * np.abs(dw)
    rp = w * r - costs

    mu = float(np.mean(rp))
    sig = float(np.std(rp, ddof=1))
    ann_return = (1.0 + mu) ** float(trading_days) - 1.0 if np.isfinite(mu) else float("nan")
    ann_vol = sig * np.sqrt(float(trading_days)) if sig > 0 else float("nan")
    sharpe_d = (mu / sig) if sig > 0 else float("nan")
    sharpe_ann = sharpe_d * np.sqrt(float(trading_days)) if np.isfinite(sharpe_d) else float("nan")

    eq = np.cumprod(1.0 + rp)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    max_dd = float(np.min(dd))
    turnover = float(np.mean(np.abs(dw)))

    s2_bar = float(np.mean(s2))
    delta_g_log = 0.5 * (sharpe_d ** 2) * s2_bar if np.isfinite(sharpe_d) else float("nan")

    res = FractionalKellyResult(
        rule=rule_l,
        target_vol_annual=float(target_vol_annual),
        max_leverage=float(max_leverage),
        tc_bps=float(tc_bps),
        n=n,
        ann_return=float(ann_return),
        ann_vol=float(ann_vol),
        sharpe=float(sharpe_ann),
        max_drawdown=float(max_dd),
        turnover=float(turnover),
        delta_g_log=float(delta_g_log),
        mean_fraction=float(np.mean(fraction)),
    )
    series = {
        "w": w,
        "fraction": fraction,
        "rp": rp,
        "costs": costs,
        "eq": eq,
        "drawdown": dd,
        "posterior_var_log": s2,
    }
    return res, series


# ===========================================================================
# Moreira-Muir (2017) alpha regression and FKO certainty-equivalent fee
# ===========================================================================

def moreira_muir_alpha(
    returns: np.ndarray,
    pred_var_next: np.ndarray,
    *,
    target_vol_annual: float = 0.10,
    max_leverage: float = 3.0,
    tc_bps: float = 0.0,
    trading_days: int = 252,
    horizon: int = 1,
    base_factor: float = 1.5,
    eps: float = 1e-12,
) -> dict:
    """
    Moreira-Muir (2017, *J. Finance* Eq. 1) volatility-managed alpha.

    Form the volatility-managed return
    ``r^{sigma}_{t+1} = (sigma^* / sqrt(h_t)) * r_{t+1}`` (capped at the
    same ``max_leverage`` as the volatility-targeting strategy) and run

    .. math::
        r^{sigma}_{t+1} = \\alpha + \\beta\\, r_{t+1} + \\varepsilon_{t+1}.

    The intercept ``alpha`` (annualised, in %/year) tests whether
    volatility timing earns a positive alpha after controlling for the
    static long exposure.  HAC standard errors use the same horizon-
    aware Newey-West bandwidth as the rest of the codebase.

    Returns a dict with ``alpha_daily``, ``alpha_annualised_bp``,
    ``alpha_t``, ``alpha_p``, ``beta``, ``beta_t``, ``r_squared``,
    ``hac_lags``.
    """
    r = _to_1d(returns)
    h = np.maximum(_to_1d(pred_var_next), eps)
    if r.shape != h.shape:
        raise ValueError("returns and pred_var_next must have the same shape.")
    T = int(r.size)
    if T < 30:
        return {"alpha_daily": float("nan"), "alpha_t": float("nan"),
                "alpha_p": float("nan"), "alpha_annualised_bp": float("nan"),
                "beta": float("nan"), "beta_t": float("nan"),
                "r_squared": float("nan"), "T": T, "hac_lags": 0}

    sigma_target = float(target_vol_annual) / np.sqrt(float(trading_days))
    scale = np.minimum(sigma_target / np.sqrt(h),
                       float(max_leverage) * np.ones_like(h))
    rsigma = scale * r
    if float(tc_bps) > 0.0:
        dw_scale = np.empty_like(scale)
        dw_scale[0] = scale[0]
        dw_scale[1:] = scale[1:] - scale[:-1]
        rsigma = rsigma - (float(tc_bps) / 1e4) * np.abs(dw_scale)

    X = np.column_stack([np.ones(T), r])
    XtX = X.T @ X
    try:
        coef = np.linalg.solve(XtX, X.T @ rsigma)
    except np.linalg.LinAlgError:
        return {"alpha_daily": float("nan"), "alpha_t": float("nan"),
                "alpha_p": float("nan"), "alpha_annualised_bp": float("nan"),
                "beta": float("nan"), "beta_t": float("nan"),
                "r_squared": float("nan"), "T": T, "hac_lags": 0}
    alpha_d, beta = coef
    resid = rsigma - X @ coef
    rss = float(np.dot(resid, resid))
    tss = float(np.dot(rsigma - rsigma.mean(), rsigma - rsigma.mean()))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")

    overlap = max(int(horizon) - 1, 0)
    rule = int(np.floor(base_factor * (T ** (1.0 / 3.0))))
    L = int(max(overlap, rule, 1))
    L = int(min(L, T - 1))

    u = X * resid[:, None]
    S = u.T @ u / T
    for ell in range(1, L + 1):
        wbart = 1.0 - ell / (L + 1.0)
        Gamma = u[ell:].T @ u[:-ell] / T
        S += wbart * (Gamma + Gamma.T)
    XtX_inv = np.linalg.inv(XtX / T)
    cov = XtX_inv @ S @ XtX_inv / T
    se = np.sqrt(np.diag(cov))
    alpha_se = float(se[0])
    beta_se = float(se[1])
    alpha_t = float(alpha_d / alpha_se) if alpha_se > 0 else float("nan")
    beta_t = float(beta / beta_se) if beta_se > 0 else float("nan")
    alpha_p = float(2.0 * (1.0 - stats.norm.cdf(abs(alpha_t)))) \
        if np.isfinite(alpha_t) else float("nan")

    return {
        "alpha_daily": float(alpha_d),
        "alpha_annualised_bp": float(alpha_d * float(trading_days) * 1e4),
        "alpha_t": alpha_t,
        "alpha_p": alpha_p,
        "alpha_se": alpha_se,
        "beta": float(beta),
        "beta_t": beta_t,
        "beta_se": beta_se,
        "r_squared": float(r2),
        "T": T,
        "hac_lags": L,
    }


def fko_ce_fee(
    rp_baseline: np.ndarray,
    rp_candidate: np.ndarray,
    *,
    gammas: tuple[float, ...] = (1.0, 5.0, 10.0),
    trading_days: int = 252,
) -> dict:
    """
    Fleming-Kirby-Ostdiek (2001, *J. Finance*; 2003, *JFE*)
    certainty-equivalent (CE) fee.

    For a power-utility CRRA investor with relative risk aversion
    ``gamma``, the CE rate of return at the daily frequency satisfies

    .. math::
        \\sum_t U(1 + r^{ce} - \\Delta) = \\sum_t U(1 + r_{cand,t}) - \\sum_t
        U(1 + r_{base,t}) + \\sum_t U(1 + r^{ce}),

    where the certainty-equivalent fee ``\\Delta`` is the maximum
    constant fee per period that an investor would pay to switch from
    ``r_baseline`` to ``r_candidate``.  We solve for ``\\Delta`` by
    setting the average utilities equal:

    .. math::
        \\bar{U}(1 + r_{cand} - \\Delta) = \\bar{U}(1 + r_{base}).

    Returns a dict ``{gamma: fee_annualised_bp}``.  Negative fees
    indicate a CRRA investor would *pay* to avoid switching.
    """
    a = _to_1d(rp_baseline)
    b = _to_1d(rp_candidate)
    if a.shape != b.shape:
        raise ValueError("rp_baseline and rp_candidate must have the same length.")
    out: dict = {"T": int(a.size), "gammas": list(map(float, gammas))}
    for g in gammas:
        gamma = float(g)
        def _U(x: np.ndarray) -> np.ndarray:
            x = np.clip(x, 1e-12, None)
            if abs(gamma - 1.0) < 1e-9:
                return np.log(x)
            return (x ** (1.0 - gamma) - 1.0) / (1.0 - gamma)
        Ub = float(np.mean(_U(1.0 + a)))
        # Find Delta s.t. mean(U(1 + b - Delta)) == Ub via bisection.
        lo, hi = -0.10, 0.10  # ±10% per period brackets
        f_lo = float(np.mean(_U(1.0 + b - lo))) - Ub
        f_hi = float(np.mean(_U(1.0 + b - hi))) - Ub
        if f_lo * f_hi > 0:
            # No root in the bracket; expand.
            lo, hi = -1.0, 1.0
            f_lo = float(np.mean(_U(1.0 + b - lo))) - Ub
            f_hi = float(np.mean(_U(1.0 + b - hi))) - Ub
            if f_lo * f_hi > 0:
                out[f"gamma_{gamma:.1f}"] = float("nan")
                continue
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fm = float(np.mean(_U(1.0 + b - mid))) - Ub
            if abs(fm) < 1e-12:
                lo = hi = mid
                break
            if fm * f_lo < 0:
                hi = mid
                f_hi = fm
            else:
                lo = mid
                f_lo = fm
        delta = 0.5 * (lo + hi)
        out[f"gamma_{gamma:.1f}"] = {
            "fee_daily": float(delta),
            "fee_annualised_bp": float(delta * float(trading_days) * 1e4),
        }
    return out


def stationary_block_bootstrap_ci(
    series_a: np.ndarray,
    series_b: np.ndarray,
    *,
    statistic: Literal["sharpe_diff", "fz0_diff"] = "sharpe_diff",
    n_boot: int = 5000,
    block_length: int | None = None,
    alpha: float = 0.05,
    seed: int = 13,
    trading_days: int = 252,
) -> dict:
    """
    Stationary block-bootstrap (Politis-Romano 1994) confidence interval
    for the difference of either annualised Sharpe ratios or mean FZ0
    losses between candidate and baseline.

    For ``statistic="sharpe_diff"`` ``series_a`` and ``series_b`` are
    realised P&L series (e.g. ``rp_candidate``, ``rp_baseline``); the
    statistic is ``SR(a) - SR(b)`` annualised by ``sqrt(trading_days)``.
    For ``statistic="fz0_diff"`` they are FZ0 loss series; the statistic
    is ``mean(b) - mean(a)`` so that a *positive* value means the
    candidate has lower (better) FZ0 loss.
    """
    a = _to_1d(series_a)
    b = _to_1d(series_b)
    if a.shape != b.shape:
        raise ValueError("series_a and series_b must align.")
    T = int(a.size)
    if T < 50:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "T": T, "n_boot": int(n_boot)}
    rng = np.random.default_rng(int(seed))
    bl = int(block_length) if block_length else max(2, int(np.floor(T ** (1.0 / 3.0))))
    bl = int(np.clip(bl, 1, T))
    p_geom = 1.0 / bl

    def _stat(aa: np.ndarray, bb: np.ndarray) -> float:
        if statistic == "sharpe_diff":
            sa = float(np.std(aa, ddof=1))
            sb = float(np.std(bb, ddof=1))
            sr_a = float(np.mean(aa)) / sa * np.sqrt(trading_days) if sa > 0 else float("nan")
            sr_b = float(np.mean(bb)) / sb * np.sqrt(trading_days) if sb > 0 else float("nan")
            return float(sr_a - sr_b)
        if statistic == "fz0_diff":
            return float(np.mean(bb) - np.mean(aa))
        raise ValueError(f"Unknown statistic {statistic!r}")

    point = _stat(a, b)
    boot = np.empty(int(n_boot), dtype=np.float64)
    for k in range(int(n_boot)):
        idx = np.empty(T, dtype=np.int64)
        i = 0
        while i < T:
            start = int(rng.integers(0, T))
            ell = max(1, int(rng.geometric(p_geom)))
            for j in range(ell):
                if i >= T:
                    break
                idx[i] = (start + j) % T
                i += 1
        boot[k] = _stat(a[idx], b[idx])

    lo = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "T": T,
        "n_boot": int(n_boot),
        "block_length": bl,
        "alpha": float(alpha),
        "statistic": statistic,
    }
