# -*- coding: utf-8 -*-

import numpy as np

import features as fe


def test_extract_svd_features_includes_adaptive_k():
    rng = np.random.default_rng(0)
    # Simple SPD covariance
    A = rng.normal(size=(10, 10))
    C = A @ A.T + np.eye(10) * 1e-3
    out = fe.extract_svd_features(C, prev_u1=None, K=6, crisis_thresholds=[0.8], eps=1e-12)
    assert "K_eff" in out and "AR_eff" in out
    assert np.isfinite(out["K_eff"])
    assert 1 <= int(out["K_eff"]) <= 10
    assert np.isfinite(out["AR_eff"])
    assert 0.0 < float(out["AR_eff"]) <= 1.0

