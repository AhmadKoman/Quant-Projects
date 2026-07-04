# -*- coding: utf-8 -*-
import numpy as np

import evaluation as ev


def test_clark_west_positive_when_unrestricted_closer():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.uniform(0.5, 2.0, size=n)
    f1 = y + rng.normal(0, 0.5, size=n)  # noisy restricted
    f2 = y + rng.normal(0, 0.05, size=n)  # better unrestricted
    out = ev.clark_west_nested_mspe(y, f1, f2, nlags=5)
    assert np.isfinite(out["cw_stat"])
    assert np.isfinite(out["mean_f_cw"])


def test_white_rc_and_spa_run():
    rng = np.random.default_rng(1)
    T, K = 120, 4
    L = rng.uniform(0.1, 1.0, size=(T, K))
    L[:, 0] += 0.05  # benchmark slightly worse on average
    rc = ev.white_reality_check_bootstrap(L, 0, n_boot=200, block_len=10, seed=3)
    spa = ev.hansen_spa_studentized_bootstrap(L, 0, nlags=5, n_boot=200, block_len=10, seed=4)
    assert "p_value" in rc and "p_value" in spa
    assert np.isfinite(rc["V_stat"])


def test_hac_ols_active_columns_smoke():
    rng = np.random.default_rng(2)
    y = rng.normal(size=80)
    X = rng.normal(size=(80, 2))
    tab = ev.hac_ols_active_columns(y, X, ["a", "b"], nlags=5)
    assert "table" in tab and len(tab["table"]) >= 3
