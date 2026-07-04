import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.bayesian import block_bootstrap_predictive_quantiles


def test_block_bootstrap_runs_on_linear_pipeline():
    rng = np.random.default_rng(42)
    n, p = 80, 3
    X = rng.normal(size=(n, p))
    y = X @ np.array([1.0, -0.5, 0.2]) + rng.normal(0, 0.3, size=n)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    pipe.fit(X, y)
    X_test = rng.normal(size=(10, p))
    lo, hi = block_bootstrap_predictive_quantiles(
        pipe, X, y, X_test, n_boot=30, block_len=10, random_state=0, min_success_frac=0.5
    )
    assert lo.shape == (10,) and hi.shape == (10,)
    assert np.all(hi >= lo)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
