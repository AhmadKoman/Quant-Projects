# -*- coding: utf-8 -*-
"""Target transforms for log variance models (smearing, back transformation)."""

from __future__ import annotations

import numpy as np

import config


def smearing_corrected_pred(
    pred_log: np.ndarray,
    train_pred_log: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """
    Duan (1983) non-parametric smearing estimator for Jensen's inequality bias.

    When a model is trained on log(variance) and predictions are recovered via
    exp(pred_log), the result is the geometric mean E[var|X] ≈ exp(E[log_var|X]),
    which is systematically lower than the arithmetic mean E[var|X] by a factor
    of exp(0.5 * Var(log_var)). The smearing estimator corrects this:

        E[var|X] ≈ exp(pred_log) * mean(exp(train_residuals))

    Clip bounds come from config training.smearing_factor_bounds and
    training.pred_log_clip_width (not tuned on test).
    """
    tcfg = config.config.get("training", {})
    sf_lo, sf_hi = tcfg.get("smearing_factor_bounds", [0.1, 50.0])
    clip_w = float(tcfg.get("pred_log_clip_width", 4.0))
    residuals = y_train - train_pred_log
    smearing_factor = float(np.exp(residuals[np.isfinite(residuals)]).mean())
    smearing_factor = float(np.clip(smearing_factor, sf_lo, sf_hi))
    y_tr_mean = float(np.nanmean(y_train))
    lo_l, hi_l = y_tr_mean - clip_w, y_tr_mean + clip_w
    pred_log_safe = np.clip(pred_log, lo_l, hi_l)
    result = np.exp(pred_log_safe) * smearing_factor
    n_clipped = int(np.sum((pred_log < lo_l) | (pred_log > hi_l)))
    if n_clipped > 0:
        frac = n_clipped / len(pred_log)
        print(
            f"  [WARN] smearing: {n_clipped} ({frac:.1%}) test predictions outside"
            f" [{lo_l:.2f}, {hi_l:.2f}] — model may be diverged."
        )
    return result


def pred_var_from_log_raw(
    pred_log: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """exp(pred_log) with log clip only (no Duan smearing); for sensitivity tables."""
    tcfg = config.config.get("training", {})
    clip_w = float(tcfg.get("pred_log_clip_width", 4.0))
    y_tr_mean = float(np.nanmean(y_train))
    lo_l, hi_l = y_tr_mean - clip_w, y_tr_mean + clip_w
    pl = np.clip(np.asarray(pred_log, dtype=np.float64), lo_l, hi_l)
    return np.exp(np.clip(pl, -80.0, 80.0))
