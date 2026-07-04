# -*- coding: utf-8 -*-
"""
Pre-registered leave-one-group-out ablations for ElasticNet HAR+SVD (Tier 2).

Group definitions match methods/inference_spec.md. Uses the same numpy
column order as features.build_feature_sets / models_linear.train_har_svd_elastic.
"""
from __future__ import annotations

import numpy as np

# Columns removed when dropping each group (exact names from Tier 2 build).
_G1_EIGEN = frozenset({"f1", "log_sigma1"})
_G2_SVD_BASE = frozenset({
    "AR", "angle", "delta_f1", "entropy", "spectral_gap", "log_condition", "k_90",
})
_G3_IXN = frozenset({"angle_x_logRSV_minus"})
_G4_XS = frozenset({"log_CSD_d", "log_turb", "corr_surprise"})


def _g2_drop_names(feature_names: list[str]) -> frozenset[str]:
    s = set(_G2_SVD_BASE)
    for n in feature_names:
        if n.startswith("crisis_"):
            s.add(n)
    return frozenset(s)


def dropped_column_indices(feature_names: list[str], group_to_drop: str) -> list[int]:
    """0-based indices of columns removed when ablating `group_to_drop`."""
    if group_to_drop == "G1_eigen":
        drop = _G1_EIGEN
    elif group_to_drop == "G2_svd":
        drop = _g2_drop_names(feature_names)
    elif group_to_drop == "G3_ixn":
        drop = _G3_IXN
    elif group_to_drop == "G4_xs":
        drop = _G4_XS
    elif group_to_drop == "G0_har":
        drop = frozenset({
            "log_RSV_d_minus", "log_RSV_d_plus", "RV_d", "RV_w", "RV_10d", "RV_m",
        })
    else:
        raise KeyError(f"Unknown ablation group: {group_to_drop}")
    return [i for i, n in enumerate(feature_names) if n in drop]


def group_keep_mask(feature_names: list[str], group_to_drop: str) -> np.ndarray:
    """Boolean length-F mask: True = column retained for M2_minus_G."""
    drop_idx = set(dropped_column_indices(feature_names, group_to_drop))
    return np.array([i not in drop_idx for i in range(len(feature_names))], dtype=bool)


def fit_predict_m2_ablation(
    X_fit: np.ndarray,
    X_test: np.ndarray,
    y_fit: np.ndarray,
    y_train_for_smear: np.ndarray,
    feature_names: list[str],
    group_to_drop: str,
    train_har_svd_elastic,
    predict_har_svd_elastic,
    smearing_corrected_pred,
    eps: float,
):
    """
    Fit ElasticNet on a column subset (all features except `group_to_drop`) and
    return (pred_var_test, pred_var_train_slice) using the same smearing pattern
    as the main M2 block in run_experiments.

    Returns None if too few columns remain.
    """
    mask = group_keep_mask(feature_names, group_to_drop)
    if int(mask.sum()) < 3:
        return None
    X_sub_fit = X_fit[:, mask]
    X_sub_test = X_test[:, mask]
    pipe = train_har_svd_elastic(X_sub_fit, y_fit)
    pred_log_test = predict_har_svd_elastic(pipe, X_sub_test)
    pred_log_train = predict_har_svd_elastic(pipe, X_sub_fit)
    pred_var_test = smearing_corrected_pred(pred_log_test, pred_log_train, y_fit)
    return pred_var_test, pipe, mask


def ablation_model_key(group_to_drop: str, prefix: str = "HAR+SVD_minus_") -> str:
    return f"{prefix}{group_to_drop}"
