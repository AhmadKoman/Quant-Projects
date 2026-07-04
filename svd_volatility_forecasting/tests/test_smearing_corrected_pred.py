import numpy as np

from transforms import smearing_corrected_pred


def test_smearing_monotone_in_pred_log_interior():
    y_tr = np.full(50, 0.5)
    train_pred = np.full(50, 0.5)
    pred_lo = np.full(10, 0.3)
    pred_hi = np.full(10, 0.4)
    out_lo = smearing_corrected_pred(pred_lo, train_pred, y_tr)
    out_hi = smearing_corrected_pred(pred_hi, train_pred, y_tr)
    assert np.all(out_hi >= out_lo)


def test_smearing_identity_when_train_residuals_zero():
    y_tr = np.array([1.0, 1.0, 1.0])
    train_pred = np.array([1.0, 1.0, 1.0])
    pred = np.array([1.0, 1.0])
    out = smearing_corrected_pred(pred, train_pred, y_tr)
    np.testing.assert_allclose(out, np.exp(1.0), rtol=1e-5)
