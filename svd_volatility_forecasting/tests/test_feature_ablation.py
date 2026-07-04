# -*- coding: utf-8 -*-
import numpy as np

from evaluation import feature_ablation as fa


def test_ablation_model_key():
    assert fa.ablation_model_key("G2_svd") == "HAR+SVD_minus_G2_svd"
    assert fa.ablation_model_key("G2_svd", prefix="P_") == "P_G2_svd"


def test_group_keep_mask_length_matches_features():
    names = [
        "log_RSV_d_minus", "log_RSV_d_plus", "RV_w", "RV_10d", "RV_m",
        "f1", "log_sigma1",
        "AR", "angle", "delta_f1", "crisis_0.8", "entropy",
        "spectral_gap", "log_condition", "k_90",
        "angle_x_logRSV_minus",
    ]
    m = fa.group_keep_mask(names, "G1_eigen")
    assert m.shape == (len(names),)
    assert m.sum() == len(names) - 2
    drop = fa.dropped_column_indices(names, "G1_eigen")
    assert set(names[i] for i in drop) == {"f1", "log_sigma1"}


def test_g2_includes_dynamic_crisis_columns():
    names = ["RV_w", "crisis_0.75", "crisis_0.8", "AR", "angle", "entropy"]
    drop = set(names[i] for i in fa.dropped_column_indices(names, "G2_svd"))
    assert "crisis_0.75" in drop and "crisis_0.8" in drop
    assert "RV_w" not in drop


def test_g4_xs_only_tier3_columns():
    names = ["AR", "log_CSD_d", "log_turb", "corr_surprise"]
    drop = [names[i] for i in fa.dropped_column_indices(names, "G4_xs")]
    assert set(drop) == {"log_CSD_d", "log_turb", "corr_surprise"}


def test_g0_har_drop_set():
    names = ["RV_d", "RV_w", "f1"]
    drop = set(names[i] for i in fa.dropped_column_indices(names, "G0_har"))
    assert drop == {"RV_d", "RV_w"}
