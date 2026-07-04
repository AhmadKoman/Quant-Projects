"""Smoke tests for VoC ridge + RFF helpers."""
import numpy as np

from models import voc_ridge as voc


def test_ridge_primal_and_dual_match_small_design():
    rng = np.random.default_rng(2)
    T, F = 30, 50
    Phi = rng.standard_normal(size=(T, F))
    y = rng.standard_normal(size=T)
    lam = 0.5
    coef_p, yhat_p = voc.ridge_fit_primal_dual(y, Phi, lam)
    assert coef_p.shape == (F,)
    assert yhat_p.shape == (T,)
    # Dual path explicit
    coef_d, yhat_d = voc.ridge_fit_primal_dual(y, Phi[:, :15], lam)
    assert coef_d.shape == (15,)
    assert np.allclose(yhat_p, Phi @ coef_p)


def test_voc_rff_oos_curve_runs():
    rng = np.random.default_rng(0)
    n_tr, n_va, n_te, d = 120, 40, 40, 6
    G_tr = rng.standard_normal(size=(n_tr, d))
    G_va = rng.standard_normal(size=(n_va, d))
    G_te = rng.standard_normal(size=(n_te, d))
    y_tr = np.exp(rng.standard_normal(size=n_tr) * 0.3)
    y_va = np.exp(rng.standard_normal(size=n_va) * 0.3)
    y_te = np.exp(rng.standard_normal(size=n_te) * 0.3)
    out = voc.voc_rff_oos_curve(
        G_tr, y_tr, G_va, y_va, G_te, y_te,
        P=32,
        gamma=1.0,
        log_lam_grid=np.linspace(-3, 2, 12),
        rng=rng,
    )
    assert np.isfinite(out["qlike_test"])
    assert out["P"] == 32
