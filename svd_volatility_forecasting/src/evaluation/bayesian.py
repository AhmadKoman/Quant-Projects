# -*- coding: utf-8 -*-
"""
Bayesian uncertainty quantification for volatility forecasting.

Two complementary approaches:

A) MC Dropout for DNN and LSTM (Gal & Ghahramani 2016)
   -------------------------------------------------------
   Dropout at test time with `training=True` makes the neural network approximate
   a Bayesian posterior via Monte Carlo integration. T=200 stochastic forward
   passes yield an empirical distribution over predictions; the standard deviation
   captures epistemic (model) uncertainty.

   Reference:
       Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation:
       Representing Model Uncertainty in Deep Learning. ICML 2016.

B) Conjugate Bayesian Linear Regression for HAR
   -----------------------------------------------
   The Normal-Inverse-Gamma prior conjugate to the Gaussian linear model yields
   an exact closed-form posterior predictive distribution -- a Student-t at each
   test point. No MCMC sampling is required.

   Model:
       y | x, beta, sigma^2 ~ N(x^T beta, sigma^2)
       beta | sigma^2 ~ N(mu_0, sigma^2 Lambda_0^{-1})
       sigma^2 ~ InvGamma(alpha_0, beta_0)

   Posterior (after observing n training points):
       Lambda_n = X_tr^T X_tr + Lambda_0
       mu_n = Lambda_n^{-1} X_tr^T y_tr
       alpha_n = alpha_0 + n/2
       beta_n = beta_0 + 0.5 (y_tr - X_tr mu_n)^T (y_tr - X_tr mu_n)

   Posterior predictive for new point x*:
       p(y* | x*, X_tr, y_tr) = t_{2 alpha_n}(
           x*^T mu_n,
           (beta_n / alpha_n) * (1 + x*^T Lambda_n^{-1} x*)
       )
   where t_nu denotes the Student-t with nu degrees of freedom.

   The 95% predictive interval at each test point is:
       mu_pred ± t_{0.975, df=2*alpha_n} * scale

   We also draw posterior samples of the regression coefficients for F5 violin
   plots:
       sigma^2 | data ~ InvGamma(alpha_n, beta_n)
       beta | sigma^2, data ~ N(mu_n, sigma^2 Lambda_n^{-1})
   which we approximate by sampling sigma^2 from InvGamma and then sampling
   beta | sigma^2 from the Gaussian.
"""

import numpy as np
from scipy import stats
from sklearn.base import clone


# ---------------------------------------------------------------------------
# A) MC Dropout
# ---------------------------------------------------------------------------

def mc_dropout_predict(
    model,
    X_test: np.ndarray,
    T: int = 200,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MC Dropout predictive distribution for a Keras DNN or LSTM.

    Runs T stochastic forward passes with `training=True`, which keeps Dropout
    layers active at inference time (Gal & Ghahramani 2016). The resulting
    empirical distribution approximates the Bayesian posterior predictive.

    Parameters
    ----------
    model : Keras Model
        Trained DNN or LSTM with Dropout layers.
    X_test : np.ndarray
        Test inputs.
        - DNN: shape (n_test, n_features)
        - LSTM: shape (n_test, seq_len, n_features)
    T : int
        Number of Monte Carlo forward passes. 200 is sufficient for stable
        95% intervals per Gal & Ghahramani (2016).
    batch_size : int
        Batch size for each forward pass (tunable for memory efficiency).

    Returns
    -------
    mean_pred : (n_test,) -- posterior predictive mean (log-scale)
    std_pred  : (n_test,) -- posterior predictive standard deviation (epistemic)
    lower_95  : (n_test,) -- 2.5th percentile (empirical; approx. 95% CI lower)
    upper_95  : (n_test,) -- 97.5th percentile (empirical; approx. 95% CI upper)
    """
    import tensorflow as tf

    X = np.asarray(X_test)
    n_test = X.shape[0]
    samples = np.empty((T, n_test), dtype=np.float32)

    for t in range(T):
        # training=True activates dropout stochastically at inference
        preds = model(X, training=True)
        samples[t] = np.asarray(preds).ravel()

    mean_pred = samples.mean(axis=0)
    std_pred = samples.std(axis=0)
    lower_95 = np.percentile(samples, 2.5, axis=0)
    upper_95 = np.percentile(samples, 97.5, axis=0)

    return (
        mean_pred.astype(np.float64),
        std_pred.astype(np.float64),
        lower_95.astype(np.float64),
        upper_95.astype(np.float64),
    )


# ---------------------------------------------------------------------------
# B) Conjugate Bayesian Linear Regression for HAR
# ---------------------------------------------------------------------------

def bayesian_har_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    lambda0_scale: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact posterior predictive intervals for Bayesian linear regression (HAR).

    Uses the Normal-Inverse-Gamma conjugate model. The posterior predictive
    distribution at each test point x* is a Student-t:

        p(y*|x*, data) = t_{2 alpha_n}( x*^T mu_n,
                                         (beta_n/alpha_n)(1 + x*^T Lambda_n^{-1} x*) )

    Parameters
    ----------
    X_train : (n, p) -- training features (with constant column if desired)
    y_train : (n,)   -- training targets (log-scale)
    X_test  : (m, p) -- test features
    alpha0  : prior shape for InvGamma (uninformative: 1.0)
    beta0   : prior rate for InvGamma (uninformative: 1.0)
    lambda0_scale : ridge-like regularisation on the prior precision matrix.
                    Lambda_0 = lambda0_scale * I. Default 1e-4 is weakly informative.

    Returns
    -------
    pred_mean  : (m,) -- posterior predictive mean
    lower_95   : (m,) -- 2.5th percentile of Student-t predictive
    upper_95   : (m,) -- 97.5th percentile of Student-t predictive
    """
    X_tr = np.asarray(X_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64).ravel()
    X_te = np.asarray(X_test, dtype=np.float64)
    n, p = X_tr.shape

    # Prior precision: Lambda_0 = lambda0_scale * I  (ridge-like, very diffuse)
    Lambda_0 = lambda0_scale * np.eye(p)

    # Posterior precision and mean
    Lambda_n = X_tr.T @ X_tr + Lambda_0
    mu_n = np.linalg.solve(Lambda_n, X_tr.T @ y_tr)

    # Posterior shape and rate
    alpha_n = alpha0 + n / 2.0
    resid = y_tr - X_tr @ mu_n
    beta_n = beta0 + 0.5 * (resid @ resid)

    # Posterior predictive: Student-t at each test point
    pred_mean = X_te @ mu_n
    Lambda_n_inv = np.linalg.inv(Lambda_n)
    # Predictive variance factor: (beta_n / alpha_n) * (1 + x*^T Lambda_n^{-1} x*)
    quad_form = np.sum((X_te @ Lambda_n_inv) * X_te, axis=1)  # (m,)
    pred_scale = np.sqrt((beta_n / alpha_n) * (1.0 + quad_form))
    dof = 2.0 * alpha_n

    # t-distribution quantiles for 95% interval
    t_crit = stats.t.ppf(0.975, df=dof)
    lower_95 = pred_mean - t_crit * pred_scale
    upper_95 = pred_mean + t_crit * pred_scale

    return (
        pred_mean.astype(np.float64),
        lower_95.astype(np.float64),
        upper_95.astype(np.float64),
    )


def bayesian_har_predictive_variance_log(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    lambda0_scale: float = 1e-4,
) -> np.ndarray:
    """
    Posterior predictive variance of ``Y^*`` (log-variance) at each test row,
    under the same conjugate Normal-Inverse-Gamma model as
    :func:`bayesian_har_predict`.

    For Student ``t_{2\\alpha_n}`` with scale ``s``, ``Var = s^2 \\cdot \\nu /
    (\\nu - 2)`` for degrees of freedom ``\\nu > 2``.
    """
    X_tr = np.asarray(X_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64).ravel()
    X_te = np.asarray(X_test, dtype=np.float64)
    n, p = X_tr.shape
    Lambda_0 = lambda0_scale * np.eye(p)
    Lambda_n = X_tr.T @ X_tr + Lambda_0
    mu_n = np.linalg.solve(Lambda_n, X_tr.T @ y_tr)
    alpha_n = alpha0 + n / 2.0
    resid = y_tr - X_tr @ mu_n
    beta_n = beta0 + 0.5 * (resid @ resid)
    Lambda_n_inv = np.linalg.inv(Lambda_n)
    quad_form = np.sum((X_te @ Lambda_n_inv) * X_te, axis=1)
    pred_scale_sq = (beta_n / alpha_n) * (1.0 + quad_form)
    dof = 2.0 * alpha_n
    if dof > 2.0 + 1e-9:
        factor = dof / (dof - 2.0)
    else:
        factor = 1.0
    return (pred_scale_sq * factor).astype(np.float64)


def posterior_var_log_from_mc_dropout(std_log: np.ndarray) -> np.ndarray:
    """``ŝ_t^2`` from MC-Dropout ``std_log`` on the log-variance forecast."""
    s = np.asarray(std_log, dtype=np.float64).ravel()
    return np.maximum(s ** 2, 0.0)


def posterior_var_log_from_log_quantiles(
    lower_log: np.ndarray,
    upper_log: np.ndarray,
    *,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Approximate ``Var(\\log \\hat{h})`` from symmetric predictive intervals,
    assuming approximate normality on the log scale (bootstrap bands).
    """
    lo = np.asarray(lower_log, dtype=np.float64).ravel()
    hi = np.asarray(upper_log, dtype=np.float64).ravel()
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    half = (hi - lo) / 2.0
    sig = half / z
    return np.maximum(sig ** 2, 0.0)


def bayesian_har_posterior_samples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_samples: int = 2000,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    lambda0_scale: float = 1e-4,
) -> np.ndarray:
    """
    Draw posterior samples of HAR regression coefficients for F5 violin plots.

    Sampling procedure:
        1. Draw sigma^2 ~ InvGamma(alpha_n, beta_n) -- marginal posterior.
        2. Draw beta | sigma^2 ~ N(mu_n, sigma^2 * Lambda_n^{-1}) -- conditional posterior.

    Parameters
    ----------
    X_train : (n, p)
    y_train : (n,)
    n_samples : int -- number of posterior draws
    alpha0, beta0, lambda0_scale : prior hyperparameters (same as bayesian_har_predict)

    Returns
    -------
    samples : (n_samples, p) array of posterior coefficient draws.
              Columns correspond to columns of X_train (e.g. intercept, RV_d, RV_w, RV_m).
    """
    X_tr = np.asarray(X_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64).ravel()
    n, p = X_tr.shape

    Lambda_0 = lambda0_scale * np.eye(p)
    Lambda_n = X_tr.T @ X_tr + Lambda_0
    mu_n = np.linalg.solve(Lambda_n, X_tr.T @ y_tr)
    alpha_n = alpha0 + n / 2.0
    resid = y_tr - X_tr @ mu_n
    beta_n = beta0 + 0.5 * (resid @ resid)

    Lambda_n_inv = np.linalg.inv(Lambda_n)

    # Step 1: sample sigma^2 from InvGamma(alpha_n, beta_n)
    # scipy uses the shape/scale parameterisation: InvGamma(a, scale=b)
    sigma2_samples = stats.invgamma.rvs(a=alpha_n, scale=beta_n, size=n_samples)

    # Step 2: sample beta | sigma^2 ~ N(mu_n, sigma^2 * Lambda_n^{-1})
    # Cholesky of Lambda_n^{-1} for efficient sampling
    L = np.linalg.cholesky(Lambda_n_inv)
    z = np.random.randn(n_samples, p)  # (n_samples, p)
    # beta[i] = mu_n + sqrt(sigma2[i]) * L @ z[i]
    beta_samples = mu_n[None, :] + (np.sqrt(sigma2_samples)[:, None]) * (z @ L.T)

    return beta_samples.astype(np.float64)


# ---------------------------------------------------------------------------
# C) Block bootstrap predictive intervals (ElasticNet / non-conjugate linear)
# ---------------------------------------------------------------------------


def _circular_block_bootstrap_idx(
    n: int, block_len: int, rng: np.random.Generator
) -> np.ndarray:
    """Contiguous circular blocks until length n (Politis–Romano style)."""
    block_len = max(1, min(block_len, n))
    out: list[int] = []
    while len(out) < n:
        s = int(rng.integers(0, n))
        for j in range(block_len):
            out.append((s + j) % n)
    return np.asarray(out[:n], dtype=int)


def block_bootstrap_predictive_quantiles(
    fitted_pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_boot: int,
    block_len: int,
    random_state: int | None = None,
    min_success_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Empirical 95% predictive intervals for log-variance: refit a sklearn Pipeline
    on circular block-bootstrap resamples of the training set, predict X_test each time.

    Use for M2 (ElasticNet) where conjugate NIG intervals do not apply. Intervals
    are *not* exact posterior predictive; they reflect sampling variability of the
    fitted penalized regression.

    Parameters
    ----------
    fitted_pipeline : sklearn Pipeline
        A *fitted* pipeline (used only as a template for clone()).
    X_train, y_train : training arrays (same as used to fit M2).
    X_test : test design matrix (same rows as test point forecasts).
    n_boot : number of bootstrap replicates.
    block_len : length of each resampled block (e.g. 22 trading days).
    random_state : RNG seed.
    min_success_frac : fraction of successful fits required; else raises RuntimeError.

    Returns
    -------
    lower_log, upper_log : (m,) arrays, 2.5% and 97.5% quantiles per test row.
    """
    X_tr = np.asarray(X_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64).ravel()
    X_te = np.asarray(X_test, dtype=np.float64)
    n = X_tr.shape[0]
    m = X_te.shape[0]
    if n < 5 or m < 1:
        raise ValueError("block_bootstrap_predictive_quantiles: insufficient data.")

    rng = np.random.default_rng(random_state)
    preds_list: list[np.ndarray] = []
    n_fail = 0
    for b in range(n_boot):
        idx = _circular_block_bootstrap_idx(n, block_len, rng)
        pipe = clone(fitted_pipeline)
        try:
            pipe.fit(X_tr[idx], y_tr[idx])
            pr = pipe.predict(X_te)
            if pr.shape[0] != m or not np.all(np.isfinite(pr)):
                n_fail += 1
                continue
            preds_list.append(np.asarray(pr, dtype=np.float64).ravel())
        except Exception:
            n_fail += 1
            continue

    min_ok = int(np.ceil(n_boot * min_success_frac))
    if len(preds_list) < min_ok:
        raise RuntimeError(
            f"Block bootstrap: only {len(preds_list)}/{n_boot} successful fits "
            f"(need >= {min_ok})."
        )
    mat = np.stack(preds_list, axis=0)
    lower = np.percentile(mat, 2.5, axis=0)
    upper = np.percentile(mat, 97.5, axis=0)
    return lower.astype(np.float64), upper.astype(np.float64)
