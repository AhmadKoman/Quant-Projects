# -*- coding: utf-8 -*-
import numpy as np

import evaluation as ev


def test_apply_gw_zscore_from_reference():
    rng = np.random.default_rng(0)
    Z_ref = rng.normal(size=(500, 2))
    Z_eval = rng.normal(loc=1.0, scale=2.0, size=(80, 2))
    Zs = ev.apply_gw_zscore_from_reference(Z_eval, Z_ref)
    assert Zs.shape == Z_eval.shape
    assert np.isfinite(Zs).all()
    # First column scaled with ref moments → unit-ish scale on eval slice
    assert 0.2 < float(np.std(Zs[:, 0])) < 5.0


def test_giacomini_white_train_vs_eval_auxiliary_differs():
    rng = np.random.default_rng(1)
    n = 100
    h = rng.uniform(0.5, 2.0, size=n)
    p1 = h + rng.normal(0, 0.1, size=n)
    p2 = h + rng.normal(0, 0.1, size=n)
    Z_eval = np.column_stack([np.ones(n), rng.normal(size=(n,))])
    Z_train = np.column_stack([np.ones(300), rng.normal(scale=5.0, size=(300,))])
    out_train = ev.giacomini_white_test(
        h, p1, p2, Z_eval, nlags=3,
        standardize_instruments=True,
        instrument_zscore_moments="train",
        Z_train=Z_train,
    )
    out_eval = ev.giacomini_white_test(
        h, p1, p2, Z_eval, nlags=3,
        standardize_instruments=True,
        instrument_zscore_moments="eval",
    )
    assert out_train["gw_moment_uses_raw_Z"] is True
    np.testing.assert_allclose(out_train["gw_stat"], out_eval["gw_stat"], rtol=1e-9)
    assert np.all(np.isfinite(out_train["coef"]))
    assert np.all(np.isfinite(out_eval["coef"]))
