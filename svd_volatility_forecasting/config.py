# -*- coding: utf-8 -*-
"""
Central configuration for SVD volatility forecasting experiments.
All hyperparameters and paths in one place for reproducibility and paper table.
Notation: S_t = sample covariance, C_t = Ledoit-Wolf shrinkage estimator; SVD features from C_t.
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

config = {
    "data": {
        # tickers_100.csv contains 135 ticker candidates. After applying the
        # max_missing_pct = 5% filter, N = 115 stocks remain in the final panel.
        # Previous runs erroneously documented N = 100; the actual panel is 115.
        "n_stocks_universe": 135,   # tickers in candidate CSV (for documentation)
        "n_stocks_filtered": 115,   # post-filter panel size reported in paper
        "tickers_path": DATA_DIR / "tickers_100.csv",
        "returns_cache_path": DATA_DIR / "returns_100.csv",
        "vix_path": DATA_DIR / "vix_daily_vol.csv",  # IV baseline: VIX level -> daily vol; fetch via data/fetch_vix.py
        "start_date": "2000-01-15",
        "end_date": "2024-12-31",
        "max_missing_pct": 0.05,
    },
    "features": {
        # Main window: 252 trading days (1 year). N/M = 115/252 ≈ 0.46 (high-dimensional).
        # Shorter window (126 = 6 months) tested in robustness Appendix: faster-reacting
        # features may help at short horizons but reduce estimation precision.
        "svd_window": 252,
        "svd_window_sensitivity": [126, 252],   # Appendix robustness; set to [252] for quick run
        # K eigenvalues in absorption ratio. Sensitivity over K = [3,4,5,6,8,10] in Appendix.
        "K": 6,
        "K_sensitivity": [3, 4, 5, 6, 8, 10],
        "rv_windows": {"daily": 1, "weekly": 5, "biweekly": 10, "monthly": 22},
        "crisis_thresholds": [0.7, 0.75, 0.8, 0.85, 0.9],
        "default_crisis_threshold": 0.8,
        # Primary paper spec: BOTH Tier-2 (HAR+SVD) and Tier-3 (HAR_SVD_T3) are
        # pre-registered as primary alternatives to HAR (see
        # `methods/preregistration.md`).  Romano-Wolf step-down on QLIKE-DM
        # treats both on equal footing; the cross-sectional features (G4_xs)
        # therefore enter the primary feature pipeline by default.
        "include_cross_section_primary": True,
        # Orthogonalized Spectral Increment (Frisch–Waugh): second-stage spectral on HAR residual.
        "osi_elastic_net_enabled": True,
        # Covariance estimator used to build the SVD feature panel:
        #   "qis"               — Ledoit-Wolf (2020) Quadratic-Inverse Shrinkage
        #                         (PRIMARY; KMZ-2024 Prop. 2 rate-optimal).
        #   "linear_shrinkage"  — Ledoit-Wolf (2003) constant-correlation linear
        #                         shrinkage (legacy).
        #   "bbp_rie"           — Bouchaud-Bun-Potters (2017) RIE eigenvalue
        #                         cleaning (Marchenko-Pastur bulk clip + BBP
        #                         observable shrinkage on outliers).
        "cov_estimator": "qis",
        "cov_estimator_sensitivity": ["linear_shrinkage", "qis", "bbp_rie"],
    },
    "models": {
        "dnn": {
            # Same architecture for ALL DNN tiers (HAR, Tier1, Tier2, Tier3).
            # Using a wider net for SVD variants was a capacity confound that
            # could produce improvements from extra parameters, not SVD features.
            # Both "hidden_layers" and "hidden_layers_svd" are kept for backward
            # compatibility with call sites; they must be identical here.
            "hidden_layers": [64, 32, 16],
            "hidden_layers_svd": [64, 32, 16],
            "dropout": 0.2,
            "l2_reg": 1e-4,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 500,
            "patience": 20,
            # Horizon-specific overrides: stronger regularisation at longer horizons.
            # At h=22 the target is a 22-day average (much smoother) so a smaller
            # capacity model with higher dropout reduces the overfitting (R2=-0.50)
            # observed with the same [64,32,16] net used for h=1.
            "horizon_overrides": {
                5:  {"dropout": 0.3, "l2_reg": 5e-4},
                22: {"dropout": 0.4, "l2_reg": 1e-3, "hidden_layers": [32, 16],
                     "hidden_layers_svd": [32, 16]},
            },
        },
        "lstm": {
            "seq_len": 22,
            "hidden_size": 64,
            "dropout": 0.2,
            "recurrent_dropout": 0.2,
            "l2_reg": 1e-4,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 500,
            "patience": 20,
            # Horizon-specific overrides for LSTM capacity.
            # LSTM+SVD at h=5 achieved R2=0.19 vs LSTM_HAR 0.31 — same overfitting
            # pattern as DNN.  Reduce hidden_size and increase dropout at h≥5.
            "horizon_overrides": {
                5:  {"hidden_size": 32, "dropout": 0.3, "l2_reg": 5e-4},
                22: {"hidden_size": 32, "dropout": 0.4, "l2_reg": 1e-3},
            },
        },
        "harnet": {
            "seq_len": 22,
            "filters": 32,          # Conv1D filters per dilated branch
            "dilations": [1, 5, 11], # cover HAR time-scales 1, 5, 22 days
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 500,
            "patience": 20,
        },
        "gnn": {
            "hidden_dim": 32,           # GCN hidden dimension
            "adj_threshold": 0.3,       # correlation threshold for adjacency
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 300,
            "patience": 20,
        },
        "garch_benchmark": {
            "innovations": "gaussian",
            "t_df": 8.0,
            # None = fit once on pre-test sample; int = refit MLE every N test days (expanding window).
            "refit_every": None,
        },
        "mc_dropout_samples": 200,  # T forward passes for MC Dropout uncertainty
        "har_svd_bootstrap": {
            "enabled": True,
            "n_boot": 150,
            "block_len": 22,
            "min_success_frac": 0.5,
        },
    },
    "training": {
        "train_split": 0.7,
        # Three-split robustness check (Appendix). Multiplies runtime by 3; set to [0.7]
        # for a quick run and expand to [0.65, 0.70, 0.75] for the full paper run.
        "train_split_sensitivity": [0.65, 0.70, 0.75],
        "horizons": [1, 5, 22],
        "eps": 1e-8,
        "min_valid_rows": 500,
        "min_aligned_per_horizon": 300,
        "min_assets": 50,
        "smearing_factor_bounds": [0.1, 50.0],
        "pred_log_clip_width": 4.0,
        "feature_clip_std": 5.0,
    },
    # ---------------------------------------------------------------------
    # Walk-forward out-of-sample evaluation (rolling + expanding)
    # ---------------------------------------------------------------------
    # Journal-grade rules:
    # - Splits are deterministic and date-index based.
    # - No leakage: scalers, tuning, smearing, and any moments are fit on train/val only.
    # - All horizons evaluated under both expanding and rolling protocols.
    "walk_forward": {
        # Active profile for ``python run_experiments.py --walk-forward`` (override via
        # ``--walk-forward-profile``).  ``headline`` = prereg confirmatory linear models.
        "profile": "headline",
        "profiles": {
            # Plan evaluation: all SVD-fix linear variants + baselines (expanding only).
            "svd_fix": {
                "protocols": ["expanding"],
                "models": [
                    "HAR",
                    "HAR_SVD_T1",
                    "HAR+SVD",
                    "HAR+SVD_GATED_AR",
                    "HAR+SVD_DYN",
                    "HAR+SVD_RFF_RIDGE",
                    "HAR_SVD_T3",
                ],
                "refit_cadence": {
                    "HAR": 1,
                    "HAR_SVD_T1": 21,
                    "HAR+SVD": 21,
                    "HAR+SVD_GATED_AR": 21,
                    "HAR+SVD_DYN": 21,
                    "HAR+SVD_RFF_RIDGE": 21,
                    "HAR_SVD_T3": 21,
                },
                "tuning_policy": "scheduled",
                "retune_every": 63,
                "xai": {"enabled": False},
                "ensemble_sub_seeds": [42, 1337, 2024],
                "rff_n_components": 256,
                "rff_gamma": 1.0,
            },
            "headline": {
                "protocols": ["expanding"],
                "models": ["HAR", "HAR+SVD", "HAR_SVD_T3"],
                "refit_cadence": {
                    "HAR": 1,
                    "HAR+SVD": 21,
                    "HAR_SVD_T3": 21,
                },
                "tuning_policy": "scheduled",
                "retune_every": 63,
                "xai": {"enabled": False},
                "ensemble_sub_seeds": [42, 1337, 2024],
            },
            "full": {
                "protocols": ["expanding", "rolling"],
                "models": [
                    "HAR",
                    "HAR_SVD_T1",
                    "HAR+SVD",
                    "HAR+SVD_GATED_AR",
                    "HAR+SVD_DYN",
                    "HAR+SVD_RFF_RIDGE",
                    "HAR_SVD_T3",
                    "DNN_HAR",
                    "DNN_HAR+SVD",
                    "LSTM_HAR",
                    "LSTM_HAR+SVD",
                    "HARNet",
                    "GNN",
                ],
                "refit_cadence": {
                    "HAR": 1,
                    "HAR_SVD_T1": 21,
                    "HAR+SVD": 21,
                    "HAR+SVD_GATED_AR": 21,
                    "HAR+SVD_DYN": 21,
                    "HAR+SVD_RFF_RIDGE": 21,
                    "HAR_SVD_T3": 21,
                    "DNN_HAR": 21,
                    "DNN_HAR+SVD": 21,
                    "LSTM_HAR": 21,
                    "LSTM_HAR+SVD": 21,
                    "HARNet": 21,
                    "GNN": 21,
                },
                "tuning_policy": "scheduled",
                "retune_every": 63,
                "xai": {
                    "enabled": True,
                    "run_every_n_steps": 21,
                },
                "ensemble_sub_seeds": [42, 1337, 2024, 314, 999],
                "rff_n_components": 256,
                "rff_gamma": 1.0,
            },
        },
        # Run BOTH protocols when no profile override (legacy base; profiles merge on top).
        "protocols": ["expanding", "rolling"],
        # Initial in-sample length before the first OOS forecast is produced.
        # 1260 ≈ 5 trading years; chosen to stabilize covariance/SVD and NN training.
        "initial_train_len": 1260,
        # Rolling window length (only used for protocol == "rolling").
        "rolling_train_len": 1260,
        # Validation block length carved out immediately before test within each step.
        # Used for early stopping and for smearing-factor estimation (never touches test).
        "val_len": 252,  # ≈ 1 trading year
        # Forecast step size (how often to advance the walk-forward origin).
        "step": 1,  # daily
        # ElasticNetCV parallelism (walk-forward only passes this through model_runners).
        # -1 uses all cores; None = sklearn default (usually single-threaded).
        # Bit-identical reproduction vs single-thread may require None or 1.
        "elastic_net_cv_n_jobs": -1,
        # Terminal progress: print every N walk-forward splits (ETA from recent mean rate).
        "progress_log_every": 25,
        # Deep models are expensive to refit daily. We make refit cadence explicit and fixed ex ante.
        # cadence = number of walk-forward steps between full refits. 1 means refit every step.
        "refit_cadence": {
            "HAR": 1,
            "HAR+SVD": 1,
            "HAR_SVD_T1": 1,
            "HAR_SVD_T3": 1,
            "GARCH": 1,
            "GJR-GARCH-t": 1,
            # Neural + graph models: refit every ~month by default (21 trading days).
            "DNN_HAR": 21,
            "DNN_HAR+SVD": 21,
            "LSTM_HAR": 21,
            "LSTM_HAR+SVD": 21,
            "HARNet": 21,
            "GNN": 21,
            # Combination/ensemble models (if used): refit cadence follows their constituents.
            "Combo_EqWt": 21,
            "Combo_InvMSE": 21,
        },
        # Hyperparameter tuning policy for walk-forward:
        # - "initial_only": tune once on the first train+val block and freeze thereafter.
        # - "scheduled": re-tune on a fixed schedule using only past data (train+val).
        # - "every_refit": re-tune at each refit (strict, most expensive).
        "tuning_policy": "scheduled",
        # If tuning_policy == "scheduled": how often to re-tune (in steps).
        "retune_every": 63,  # ≈ quarterly
        # Deterministic seed offset for walk-forward refits/tunes.
        "seed_offset": 10_000,
        # Ensemble sub-seeds for neural models (deterministic, reported).
        # Used when `walk_forward_use_ensembles=True` in orchestration.
        "ensemble_sub_seeds": [42, 1337, 2024, 314, 999],
        # XAI settings (used for blocked permutation importance on validation blocks)
        "xai": {
            "enabled": True,
            "loss": "qlike",
            "reps": 10,
            "block_len": 5,
            "seed": 42,
            # Run blocked permutation importance only every k step_ids (0, k, 2k, ...).
            # Does not change OOS pred_log / pred_var; only reduces XAI record count.
            # Use 1 for a full time-aligned XAI panel (slow); 21 ~= monthly on daily data.
            "run_every_n_steps": 21,
        },
        # HAC bandwidth policy for forecast comparison tests under overlapping targets.
        # Pre-registered rule (see methods/preregistration.md):
        #   nlags(h, T) = max(h - 1, floor(base_factor * T^{1/3}))
        # i.e. Andrews-(1991) cube-root width with West-(1996) overlap floor.
        # The legacy "base_plus_h_minus_1" policy is retained for runs that
        # need the old behaviour (set policy to that string explicitly).
        "dm_hac_nlags_base": 5,
        "dm_hac_base_factor": 1.5,
        "dm_hac_nlags_policy": "max_h_minus_1_andrews",
    },
    "robustness": {
        # Cross-universe / subperiod grid (scripts/run_robustness_grid.py)
        "universes": ["panel115", "mega_cap50", "sector_balanced"],
        "subperiods": "see robustness_config.SUBPERIODS",
        "gap_screened_K": {"K_max": 10, "rho_ar": 0.70, "gap_threshold_factor": 2.0},
        "grid_stride": 42,
        "grid_max_windows": 60,
        "quick_wf_step": 7,
    },
    "regime": {
        # Date-based crisis windows for performance-by-regime (Crisis vs Calm)
        # COVID-19 is the largest volatility event in the test period; must be addressed
        "crisis_windows": [
            ("2020-02-20", "2020-03-31"),  # COVID-19 crash (largest event in test period)
            ("2022-02-20", "2022-03-31"),  # Russia-Ukraine invasion
            ("2023-03-01", "2023-03-20"),  # SVB collapse
        ],
    },
    "seed": 42,
    # Journal-grade inference exports (DM/MZ/encompassing/GW/MCS); see methods/inference_spec.md
    "inference": {
        # MCS CSV: restrict to core forecasting models (excludes ablations, metadata).
        "mcs_core_models_only": True,
        "mcs_core_model_names": [
            "HAR",
            "HAR_SVD_T1",
            "HAR+SVD",
            "HAR+SVD_OSI",
            "HAR_SVD_T3",
            "DNN_HAR",
            "DNN_HAR+SVD",
            "LSTM_HAR",
            "LSTM_HAR+SVD",
            "HARNet",
            "GNN",
            "GARCH",
            "GJR-GARCH-t",
            "Combo_EqWt",
            "Combo_InvMSE",
            "Combo_BG",
        ],
        "dm_export_detailed_columns": True,
        "gw_standardize_instruments": True,
        # Instrument z-scores for GW *auxiliary* δ regression only (Wald uses raw Z).
        # "train" = μ,σ from training-period Z; "eval" = from evaluation sample; "none" = raw.
        "gw_zscore_moments": "train",
        # Clark–West nested MSPE, White (2000) RC, Hansen (2005)-style SPA bootstrap,
        # and split-sample HAC on M2 active set (appendix / referee responses).
        "optional_inference": {
            "enabled": True,
            "clark_west_pairs": [
                ["HAR", "HAR+SVD"],
                ["HAR", "HAR_SVD_T3"],
                ["DNN_HAR", "DNN_HAR+SVD"],
                ["LSTM_HAR", "LSTM_HAR+SVD"],
            ],
            # Reality Check / SPA on QLIKE is the pre-registered primary
            # (kept on MSE as well for backward compatibility / sensitivity).
            "reality_check_spa_mse": True,
            "reality_check_qlike": True,
            "reality_check_benchmark": "HAR",
            "bootstrap_n_boot": 1999,
            "bootstrap_block_len": 22,
            # Romano-Wolf step-down on QLIKE (FWER-controlled headline family).
            "rw_n_boot": 9999,
            "rw_alpha": 0.05,
            # Acerbi–Szekely Z2 stationary bootstrap for VaR/ES tournament (`run_experiments` walk-forward).
            "var_es_z2_n_boot": 1999,
            "split_sample_hac_m2_h1": True,
            "split_sample_mid_frac": 0.5,
        },
    },
    # Leave-one-group-out ElasticNet M2 refits (Tier 2 by default; G4 uses Tier 3 if enabled)
    "feature_ablation": {
        "enabled": True,
        "groups_to_drop": ["G2_svd", "G3_ixn", "G1_eigen"],
        "include_tier3_ablation": True,
        "tier3_additional_groups": ["G4_xs"],
        "qlike_noninferiority_delta": 0.0,
        "include_ablations_in_mcs": False,
        "model_key_prefix": "HAR+SVD_minus_",
    },
}


def get_hyperparameter_table():
    """Return a flat dict suitable for paper hyperparameter table (Section 2 / Appendix)."""
    c = config
    return {
        "n_stocks_universe": c["data"]["n_stocks_universe"],
        "n_stocks_filtered": c["data"]["n_stocks_filtered"],
        "referee_response_cross_section_N": (
            "Initial manuscript used a 7×7 covariance illustration; the revised "
            "pipeline sets data.n_stocks_filtered=115 (see config.py / fetch_stocks)."
        ),
        "svd_window_M": c["features"]["svd_window"],
        "K": c["features"]["K"],
        "crisis_threshold_default": c["features"]["default_crisis_threshold"],
        "crisis_thresholds_sensitivity": c["features"]["crisis_thresholds"],
        "rv_windows": list(c["features"]["rv_windows"].values()),
        "train_val_test_split": [c["training"]["train_split"], 0.15, 0.15],
        "horizons": c["training"]["horizons"],
        "min_assets": c["training"].get("min_assets", 50),
        "dnn_hidden_har": c["models"]["dnn"]["hidden_layers"],
        "dnn_hidden_svd": c["models"]["dnn"]["hidden_layers_svd"],
        "dnn_dropout": c["models"]["dnn"]["dropout"],
        "dnn_l2": c["models"]["dnn"]["l2_reg"],
        "dnn_lr": c["models"]["dnn"]["learning_rate"],
        "dnn_batch_size": c["models"]["dnn"]["batch_size"],
        "dnn_epochs_max": c["models"]["dnn"]["epochs"],
        "dnn_patience": c["models"]["dnn"]["patience"],
        "lstm_seq_len": c["models"]["lstm"]["seq_len"],
        "lstm_hidden": c["models"]["lstm"]["hidden_size"],
        "lstm_dropout": c["models"]["lstm"]["dropout"],
        "lstm_recurrent_dropout": c["models"]["lstm"]["recurrent_dropout"],
        "lstm_l2": c["models"]["lstm"]["l2_reg"],
        "lstm_lr": c["models"]["lstm"]["learning_rate"],
        "lstm_epochs_max": c["models"]["lstm"]["epochs"],
        "lstm_patience": c["models"]["lstm"]["patience"],
        "harnet_seq_len": c["models"]["harnet"]["seq_len"],
        "harnet_filters": c["models"]["harnet"]["filters"],
        "harnet_dilations": c["models"]["harnet"]["dilations"],
        "harnet_dropout": c["models"]["harnet"]["dropout"],
        "gnn_hidden_dim": c["models"]["gnn"]["hidden_dim"],
        "gnn_adj_threshold": c["models"]["gnn"]["adj_threshold"],
        "mc_dropout_samples": c["models"]["mc_dropout_samples"],
        "seed": c["seed"],
        "crisis_windows_date": c.get("regime", {}).get("crisis_windows", []),
        "train_split_sensitivity": c["training"].get("train_split_sensitivity", [c["training"]["train_split"]]),
        "smearing_factor_bounds": c["training"].get("smearing_factor_bounds", [0.1, 50.0]),
        "pred_log_clip_width": c["training"].get("pred_log_clip_width", 4.0),
        "feature_clip_std": c["training"].get("feature_clip_std", 5.0),
        "garch_innovations": c.get("models", {}).get("garch_benchmark", {}).get("innovations", "gaussian"),
        "garch_refit_every": c.get("models", {}).get("garch_benchmark", {}).get("refit_every"),
        "har_svd_bootstrap_n_boot": c.get("models", {}).get("har_svd_bootstrap", {}).get("n_boot", 150),
        "inference_dm_export_detailed": c.get("inference", {}).get("dm_export_detailed_columns", True),
        "inference_gw_standardize": c.get("inference", {}).get("gw_standardize_instruments", True),
        "inference_gw_zscore_moments": c.get("inference", {}).get("gw_zscore_moments", "train"),
        "feature_ablation_enabled": c.get("feature_ablation", {}).get("enabled", False),
        "feature_ablation_groups": c.get("feature_ablation", {}).get("groups_to_drop", []),
        "feature_ablation_include_tier3": c.get("feature_ablation", {}).get("include_tier3_ablation", False),
        "feature_ablation_in_mcs": c.get("feature_ablation", {}).get("include_ablations_in_mcs", False),
        "feature_ablation_qlike_delta": c.get("feature_ablation", {}).get("qlike_noninferiority_delta", 0.0),
        "optional_inference_enabled": c.get("inference", {}).get("optional_inference", {}).get("enabled", False),
        "optional_inference_n_boot": c.get("inference", {}).get("optional_inference", {}).get("bootstrap_n_boot", 1999),
    }
