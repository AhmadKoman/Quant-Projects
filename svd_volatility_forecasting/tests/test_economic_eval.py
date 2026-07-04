import numpy as np

from evaluation import economic as econ


def test_var_es_backtest_shapes_and_pvalues():
    rng = np.random.default_rng(0)
    n = 500
    # Simulate daily returns in decimals
    r = rng.normal(0.0, 0.01, size=n)
    # Constant variance forecast
    h = np.full(n, 0.01**2)
    res, series = econ.backtest_var_es(r, h, alpha=0.05, dist="gaussian")
    assert res.n == n
    assert 0.0 <= res.kupiec_p <= 1.0
    assert 0.0 <= res.christoffersen_p <= 1.0
    assert series["VaR"].shape == (n,)
    assert series["ES"].shape == (n,)
    assert series["violations"].shape == (n,)
    assert series["FZ0"].shape == (n,)


def test_volatility_targeting_runs_and_returns_series():
    rng = np.random.default_rng(1)
    n = 300
    r = rng.normal(0.0, 0.01, size=n)
    h = np.full(n, 0.01**2)
    res, ser = econ.volatility_targeting(r, h, target_vol_annual=0.10, max_leverage=2.0, tc_bps=5.0)
    assert res.n == n
    assert ser["w"].shape == (n,)
    assert ser["rp"].shape == (n,)
    assert ser["eq"].shape == (n,)
    assert np.isfinite(res.turnover)

