# -*- coding: utf-8 -*-
"""
Linear models for SVD volatility forecasting.

M1: HAR (OLS)       -- plain Corsi (2009) HAR with 4 RMS-based lags
M2: HAR+SVD (ElasticNet) -- HAR augmented with 5 SVD features, using
    ElasticNetCV with TimeSeriesSplit cross-validation.

ElasticNet justification for M2:
    The 5 SVD features (f1, log_sigma1, AR, angle, crisis_*) are derived from
    the same eigendecomposition of C_t and are therefore highly collinear.
    OLS inflates out-of-sample prediction variance under collinearity.
    Elastic Net (L1+L2 penalty) performs automatic feature selection (L1) and
    handles collinearity (L2). TimeSeriesSplit ensures CV folds respect
    chronological order -- no future information leakage during fold construction.

Reference: Audrino & Knaus (2016), Buncic & Gisler (2016).
"""

import warnings

import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def default_elastic_alphas(n: int = 40) -> np.ndarray:
    """Log-spaced alpha grid for ElasticNetCV (faster than 100 defaults)."""
    return np.logspace(-4, 2, int(n))


def default_l1_ratios() -> list[float]:
    return [0.5, 0.7, 0.9, 0.95, 1.0]


def _make_elasticnet_cv(
    cv_splits: int,
    *,
    enet_n_jobs: int | None = None,
    n_alphas: int = 40,
) -> ElasticNetCV:
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    enet_kw: dict = dict(
        l1_ratio=default_l1_ratios(),
        cv=tscv,
        max_iter=15_000,
        tol=1e-4,
        fit_intercept=True,
        alphas=default_elastic_alphas(n_alphas),
    )
    if enet_n_jobs is not None:
        enet_kw["n_jobs"] = int(enet_n_jobs)
    return ElasticNetCV(**enet_kw)


def _make_elasticnet_frozen(alpha: float, l1_ratio: float) -> ElasticNet:
    return ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        max_iter=15_000,
        tol=1e-4,
        fit_intercept=True,
    )


def _elastic_hyperparams_from_pipe(pipe: Pipeline) -> tuple[float, float]:
    enet = pipe.named_steps["enet"]
    return float(enet.alpha_), float(enet.l1_ratio_)


def elastic_hyperparams_from_model(model: Pipeline | object) -> dict[str, float]:
    """Extract frozen ElasticNet (alpha, l1_ratio) for scheduled walk-forward refits."""
    if hasattr(model, "pipe_har") and hasattr(model, "pipe_spec"):
        ah, lh = _elastic_hyperparams_from_pipe(model.pipe_har)
        asp, lsp = _elastic_hyperparams_from_pipe(model.pipe_spec)
        return {
            "alpha_har": ah,
            "l1_ratio_har": lh,
            "alpha_spec": asp,
            "l1_ratio_spec": lsp,
        }
    a, l = _elastic_hyperparams_from_pipe(model)
    return {"alpha": a, "l1_ratio": l}


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Clip each feature column to the [lo_pct, hi_pct] percentile range computed
    from the training data, then pass through unchanged.

    This is the standard robust preprocessing step for linear models with
    skewed or heavy-tailed features (e.g. `angle = 1-cos_theta` which is near 0
    on calm days but spikes to ~1-2 during regime changes). Without Winsorization,
    StandardScaler divides by a tiny training std, amplifying test-period spikes
    to extreme standardized values that corrupt ElasticNet coefficients.

    Importantly, this preserves the linear relationships between features and target
    (unlike QuantileTransformer which applies a non-linear mapping), making it
    appropriate for linear models such as ElasticNet.

    Parameters
    ----------
    lo_pct, hi_pct : float
        Lower and upper percentile bounds (inclusive). Default: 1st-99th.
    """

    def __init__(self, lo_pct: float = 1.0, hi_pct: float = 99.0) -> None:
        self.lo_pct = lo_pct
        self.hi_pct = hi_pct
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None):
        self.lo_ = np.percentile(X, self.lo_pct, axis=0)
        self.hi_ = np.percentile(X, self.hi_pct, axis=0)
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return np.clip(X, self.lo_, self.hi_)


class RandomFourierFeatures(BaseEstimator, TransformerMixin):
    """
    Random Fourier Features for an RBF kernel (Rahimi & Recht, 2007).

    Produces a rich smooth basis so that a *linear* estimator with strong
    shrinkage (Ridge) can approximate nonlinear relationships.
    """

    def __init__(self, n_components: int = 256, gamma: float = 1.0, seed: int = 42) -> None:
        self.n_components = int(n_components)
        self.gamma = float(gamma)
        self.seed = int(seed)
        self.W_: np.ndarray | None = None
        self.b_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None):
        X = np.asarray(X, dtype=np.float64)
        d = int(X.shape[1])
        if self.n_components < 8:
            raise ValueError("RandomFourierFeatures: n_components must be >= 8.")
        rng = np.random.default_rng(self.seed)
        self.W_ = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(d, self.n_components))
        self.b_ = rng.uniform(0.0, 2.0 * np.pi, size=(self.n_components,))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.W_ is None or self.b_ is None:
            raise ValueError("RandomFourierFeatures must be fit before transform.")
        X = np.asarray(X, dtype=np.float64)
        proj = X @ self.W_ + self.b_
        Z = np.concatenate([np.cos(proj), np.sin(proj)], axis=1)
        return np.sqrt(1.0 / self.n_components) * Z


def ridge_alpha_from_model(model: Pipeline) -> float:
    est = model.named_steps["ridge"]
    if hasattr(est, "alpha_"):
        return float(est.alpha_)
    return float(getattr(est, "alpha", 1.0))


def train_ridge_rff(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: int = 5,
    *,
    tune: bool = True,
    alpha: float | None = None,
    n_components: int = 256,
    gamma: float = 1.0,
    seed: int = 42,
) -> Pipeline:
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    if tune or alpha is None:
        ridge = RidgeCV(alphas=np.logspace(-4, 4, 25), cv=tscv, fit_intercept=True)
    else:
        ridge = Ridge(alpha=float(alpha), fit_intercept=True)
    pipe = Pipeline(
        [
            ("winsor", Winsorizer(lo_pct=1.0, hi_pct=99.0)),
            ("scaler", StandardScaler()),
            ("rff", RandomFourierFeatures(n_components=int(n_components), gamma=float(gamma), seed=int(seed))),
            ("ridge", ridge),
        ]
    )
    pipe.fit(np.asarray(X_train, dtype=np.float64), np.asarray(y_train, dtype=np.float64))
    return pipe


def train_har(X_train: np.ndarray, y_train: np.ndarray, add_constant: bool = True):
    """
    OLS fit for HAR (M1).
    X_train: (n, 4) array of [log(RV_d), log(RV_w), log(RV_10d), log(RV_m)].
    y_train: log(realized_vol), shape (n,).

    Note: ``sm.add_constant`` is called with ``has_constant='add'`` so the
    intercept column is *always* prepended, even on degenerate inputs (e.g.
    walk-forward mode where ``X_test`` has a single row and every column would
    otherwise be flagged as constant).
    """
    if add_constant:
        X = sm.add_constant(X_train, has_constant="add")
    else:
        X = X_train
    model = sm.OLS(y_train, X).fit()
    return model


def predict_har(model, X_test: np.ndarray, add_constant: bool = True) -> np.ndarray:
    """
    Predict log(vol); caller should apply exp() for vol-level predictions.

    ``has_constant='add'`` is required because ``sm.add_constant`` silently
    refuses to prepend an intercept when *every* column of ``X_test`` looks
    constant — exactly what happens when the walk-forward engine calls this
    function with a one-row test slice. The training-time ``train_har`` uses
    the same flag, keeping the design matrices consistent.
    """
    if add_constant:
        X = sm.add_constant(X_test, has_constant="add")
    else:
        X = X_test
    return model.predict(X)


def train_har_svd(X_train: np.ndarray, y_train: np.ndarray, add_constant: bool = True):
    """
    OLS fit for HAR+SVD (kept for comparison / transparency reporting).
    X_train: (n, 9) array of [4 HAR + 5 SVD features].
    """
    return train_har(X_train, y_train, add_constant)


def predict_har_svd(model, X_test: np.ndarray, add_constant: bool = True) -> np.ndarray:
    return predict_har(model, X_test, add_constant)


def train_har_svd_elastic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: int = 5,
    *,
    enet_n_jobs: int | None = None,
    tune: bool = True,
    alpha: float | None = None,
    l1_ratio: float | None = None,
) -> Pipeline:
    """
    ElasticNetCV with TimeSeriesSplit for HAR+SVD (M2, primary estimator).

    Pipeline: Winsorizer (1st–99th pct) -> StandardScaler -> ElasticNetCV.

    Winsorizer clips extreme values using training percentiles only, preserving
    linear structure while preventing `angle` and similar features from exploding
    under StandardScaler on heavy-tailed test periods.

    TimeSeriesSplit(n_splits=5) creates 5 sequential folds, each with expanding
    training set, which respects temporal ordering and avoids look-ahead.

    l1_ratio grid: [0.1, 0.5, 0.7, 0.9, 0.95, 1.0] covers the full range from
    near-Ridge to pure-LASSO; the optimal l1_ratio is selected by CV.

    Parameters
    ----------
    X_train : ndarray, shape (n, p)
        HAR + stable SVD features (p = 12 in the current configuration)
    y_train : ndarray, shape (n,)
        log(realized_variance)
    cv_splits : int
        Number of TimeSeriesSplit folds for alpha/l1_ratio selection.
    enet_n_jobs : int | None
        If not ``None``, passed to ``ElasticNetCV(n_jobs=...)`` so cross-validated
        alpha search can use multiple CPU cores.  Does not change the estimator
        definition; with ``n_jobs > 1`` bit-wise reproducibility vs single-threaded
        runs is not guaranteed.  Default ``None`` leaves sklearn's default.

    Returns
    -------
    pipe : fitted sklearn Pipeline (qt + scaler + elasticnet)
    """
    if tune or alpha is None or l1_ratio is None:
        enet_est = _make_elasticnet_cv(cv_splits, enet_n_jobs=enet_n_jobs)
    else:
        enet_est = _make_elasticnet_frozen(alpha, l1_ratio)
    pipe = Pipeline([
        ("winsor", Winsorizer(lo_pct=1.0, hi_pct=99.0)),
        ("scaler", StandardScaler()),
        ("enet", enet_est),
    ])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Objective did not converge",
            category=UserWarning,
            module="sklearn.linear_model._coordinate_descent",
        )
        pipe.fit(X_train, y_train)
    return pipe


def predict_har_svd_elastic(pipe: Pipeline, X_test: np.ndarray) -> np.ndarray:
    """
    Predict log(vol) using fitted ElasticNet pipeline.
    StandardScaler transform is applied automatically inside the pipeline.
    """
    return pipe.predict(X_test)


def orthogonalize_spectral_on_har(
    Z_fit: np.ndarray,
    S_fit: np.ndarray,
    Z_pred: np.ndarray,
    S_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each column of S, take residuals from OLS of S_j on [1, Z] fitted on
    the training slice only; apply the same coefficients to the prediction
    slice.  This is the multivariate Frisch–Waugh step for spectral blocks.

    Returns
    -------
    out_fit : ndarray, shape (n_fit, n_spec)
        Training-slice residuals  S_fit_j - [1 | Z_fit] @ beta_j.
    out_pred : ndarray, shape (n_pred, n_spec)
        Prediction-slice residuals using the *same* training coefficients.
    beta_orth : ndarray, shape (1 + p_har, n_spec)
        Stacked OLS coefficients (column j contains [intercept, slope_1,
        ..., slope_{p_har}] for orthogonalising the j-th spectral feature).
        Returning these explicitly is essential for any downstream predictor
        that must orthogonalise *new* observations (e.g. permutation-based
        XAI in the walk-forward engine) without re-fitting on the test
        window.

    Inputs must be finite.  The caller (e.g. the walk-forward driver in
    ``run_experiments.py``) is responsible for filtering rows with feature
    warm-up NaN *before* slicing into train/test windows, so that ``Z_fit``
    and ``S_fit`` remain row-aligned and no implicit imputation is performed
    inside the regression.  We assert finiteness here to surface upstream
    misalignment early with an actionable message rather than letting
    scikit-learn raise a generic ``Input y contains NaN`` from deep inside
    ``check_X_y``.
    """
    Z_fit = np.asarray(Z_fit, dtype=np.float64)
    S_fit = np.asarray(S_fit, dtype=np.float64)
    Z_pred = np.asarray(Z_pred, dtype=np.float64)
    S_pred = np.asarray(S_pred, dtype=np.float64)
    if not (np.isfinite(Z_fit).all() and np.isfinite(S_fit).all()):
        bad_S = int(np.flatnonzero(~np.isfinite(S_fit).all(axis=1)).size)
        bad_Z = int(np.flatnonzero(~np.isfinite(Z_fit).all(axis=1)).size)
        raise ValueError(
            "orthogonalize_spectral_on_har: training inputs contain NaN/Inf. "
            f"Non-finite rows in S_fit: {bad_S}, Z_fit: {bad_Z}. "
            "Tighten the upstream validity mask to require feature finiteness "
            "across every active tier (HAR + T1 + T2 + T3) before slicing."
        )
    Zf = np.column_stack([np.ones(len(Z_fit), dtype=np.float64), Z_fit])
    Zp = np.column_stack([np.ones(len(Z_pred), dtype=np.float64), Z_pred])
    # Single least-squares solve for all spectral columns — algebraically identical
    # to column-wise sklearn LinearRegression(fit_intercept=False), but avoids
    # Python overhead per feature (walk-forward calls this thousands of times).
    beta_orth, _, _, _ = np.linalg.lstsq(Zf, S_fit, rcond=None)
    beta_orth = np.asarray(beta_orth, dtype=np.float64)
    out_fit = S_fit - Zf @ beta_orth
    out_pred = S_pred - Zp @ beta_orth
    return out_fit, out_pred, beta_orth


class OsiPredictor:
    """
    Two-stage Orthogonalised Spectral Increment predictor.

    Wraps the artefacts produced by :func:`train_predict_osi_elastic` so that
    a single ``predict(X)`` call evaluates the *full* OSI pipeline on a
    generic feature matrix ``X = [Z | S]`` whose first ``har_col_count``
    columns are HAR features and whose remaining columns are spectral
    features (in the same order used at training time).

    The orthogonalisation coefficients ``beta_orth`` are *frozen at training
    time* — predicting on a new ``X`` reproduces the test-time logic
    inside :func:`train_predict_osi_elastic`:

        m       = pipe_har.predict(Z)                       (stage 1)
        S_orth  = S - [1 | Z] @ beta_orth                   (FW projection)
        u_hat   = pipe_spec.predict(S_orth)                 (stage 2)
        pred    = m + u_hat

    Exposing this object as ``FitPredictResult.model`` lets the walk-forward
    XAI hook treat OSI like any other ``predict``-able model, with no
    refitting on the test window and no leakage.
    """

    def __init__(
        self,
        pipe_har: Pipeline,
        pipe_spec: Pipeline,
        beta_orth: np.ndarray,
        har_col_count: int,
    ) -> None:
        self.pipe_har = pipe_har
        self.pipe_spec = pipe_spec
        self.beta_orth = np.asarray(beta_orth, dtype=np.float64)
        self.har_col_count = int(har_col_count)
        if self.beta_orth.ndim != 2 or self.beta_orth.shape[0] != self.har_col_count + 1:
            raise ValueError(
                "OsiPredictor: beta_orth must have shape (1 + har_col_count, n_spec); "
                f"got shape {self.beta_orth.shape} with har_col_count={self.har_col_count}."
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] < self.har_col_count + 1:
            raise ValueError(
                f"OsiPredictor.predict: X must be 2-D with at least "
                f"{self.har_col_count + 1} columns; got shape {X.shape}."
            )
        Z = X[:, : self.har_col_count]
        S = X[:, self.har_col_count :]
        if S.shape[1] != self.beta_orth.shape[1]:
            raise ValueError(
                "OsiPredictor.predict: spectral block width "
                f"({S.shape[1]}) does not match training-time width "
                f"({self.beta_orth.shape[1]})."
            )
        m = np.asarray(self.pipe_har.predict(Z), dtype=np.float64).ravel()
        Zf = np.column_stack([np.ones(len(Z), dtype=np.float64), Z])
        S_orth = S - Zf @ self.beta_orth
        u_hat = np.asarray(self.pipe_spec.predict(S_orth), dtype=np.float64).ravel()
        return m + u_hat


def train_predict_osi_elastic(
    X_full_fit: np.ndarray,
    X_full_test: np.ndarray,
    y_fit: np.ndarray,
    har_col_count: int,
    cv_splits: int = 5,
    *,
    enet_n_jobs: int | None = None,
    tune_har: bool = True,
    tune_spec: bool = True,
    alpha_har: float | None = None,
    l1_ratio_har: float | None = None,
    alpha_spec: float | None = None,
    l1_ratio_spec: float | None = None,
) -> tuple[np.ndarray, np.ndarray, "OsiPredictor"]:
    """
    Orthogonalized Spectral Increment (OSI): ElasticNet on HAR (stage 1), then
    ElasticNet on HAR-orthogonalized spectral columns predicting
    (y - HAR_pred).

    Returns
    -------
    pred_log_test, pred_log_fit, predictor

    where ``predictor`` is an :class:`OsiPredictor` exposing ``predict(X)`` for
    a generic ``X = [Z | S]`` matrix in the same column order as
    ``X_full_fit``.  The previous 4-tuple form ``(..., pipe_har, pipe_spec)``
    is intentionally retired so callers cannot accidentally treat the
    two-stage pipeline as a single sklearn ``Pipeline`` (the bug surfaced by
    the walk-forward XAI hook).  The component pipes are still reachable via
    ``predictor.pipe_har`` and ``predictor.pipe_spec``.
    """
    if har_col_count <= 0 or har_col_count >= X_full_fit.shape[1]:
        raise ValueError("har_col_count must split HAR and spectral columns strictly.")
    Z_fit = np.asarray(X_full_fit[:, :har_col_count], dtype=np.float64)
    Z_test = np.asarray(X_full_test[:, :har_col_count], dtype=np.float64)
    S_fit = np.asarray(X_full_fit[:, har_col_count:], dtype=np.float64)
    S_test = np.asarray(X_full_test[:, har_col_count:], dtype=np.float64)
    pipe_har = train_har_svd_elastic(
        Z_fit, y_fit, cv_splits=cv_splits, enet_n_jobs=enet_n_jobs,
        tune=tune_har, alpha=alpha_har, l1_ratio=l1_ratio_har,
    )
    m_fit = predict_har_svd_elastic(pipe_har, Z_fit)
    m_test = predict_har_svd_elastic(pipe_har, Z_test)
    u_fit = np.asarray(y_fit, dtype=np.float64) - m_fit
    St_fit, St_test, beta_orth = orthogonalize_spectral_on_har(Z_fit, S_fit, Z_test, S_test)
    pipe_spec = train_har_svd_elastic(
        St_fit, u_fit, cv_splits=cv_splits, enet_n_jobs=enet_n_jobs,
        tune=tune_spec, alpha=alpha_spec, l1_ratio=l1_ratio_spec,
    )
    u_hat_fit = predict_har_svd_elastic(pipe_spec, St_fit)
    u_hat_test = predict_har_svd_elastic(pipe_spec, St_test)
    pred_log_fit = m_fit + u_hat_fit
    pred_log_test = m_test + u_hat_test
    predictor = OsiPredictor(
        pipe_har=pipe_har,
        pipe_spec=pipe_spec,
        beta_orth=beta_orth,
        har_col_count=har_col_count,
    )
    return pred_log_test, pred_log_fit, predictor


def get_elastic_coefs(pipe: Pipeline, feature_names: list[str]) -> dict:
    """
    Extract ElasticNet coefficients in the Winsorized + standardized feature space.

    Pipeline: Winsorizer -> StandardScaler -> ElasticNet.

    After Winsorization and StandardScaler all features are approximately N(0,1)
    (with outliers capped), so coefficients are directly comparable across features.
    A coefficient of 1.0 means a 1-sigma increase in the Winsorized+Scaled feature
    increases the log-variance prediction by 1.0.

    Note: The Winsorizer preserves linear relationships (only clips extremes), so
    coefficients retain their economic interpretation unlike QuantileTransformer.
    Zero coefficients (L1 zeroed out) indicate the feature was excluded.
    """
    enet = pipe.named_steps["enet"]
    # Coefficients are in Winsorized+Standardized space
    coefs = enet.coef_
    return {name: float(c) for name, c in zip(feature_names, coefs)}
