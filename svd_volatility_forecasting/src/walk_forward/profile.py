# -*- coding: utf-8 -*-
"""
Resolve walk-forward configuration profiles and tuning schedules.

Profiles (``headline`` vs ``full``) merge into the base ``config["walk_forward"]``
dict so ``walk_forward_engine`` reads a single effective config without
scattering profile logic across the orchestrator.

Checkpoint I/O is implemented in :mod:`experiment_checkpoint`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from experiment_checkpoint import (
    load_walk_forward_checkpoint,
    save_walk_forward_checkpoint,
    walk_forward_checkpoint_file as _walk_forward_checkpoint_file,
)

ALL_WF_MODELS = [
    "HAR",
    "HAR_SVD_T1",
    "HAR+SVD",
    "HAR_SVD_T3",
    "DNN_HAR",
    "DNN_HAR+SVD",
    "LSTM_HAR",
    "LSTM_HAR+SVD",
    "HARNet",
    "GNN",
]

DEEP_WF_MODELS = frozenset({
    "DNN_HAR",
    "DNN_HAR+SVD",
    "LSTM_HAR",
    "LSTM_HAR+SVD",
    "HARNet",
    "GNN",
})


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_walk_forward_config(
    walk_forward_base: dict,
    profile_name: str | None = None,
) -> dict:
    """
    Build effective walk-forward config for one run.

    Parameters
    ----------
    walk_forward_base : dict
        ``config.config["walk_forward"]``.
    profile_name : str | None
        If None, uses ``walk_forward_base.get("profile", "headline")``.
    """
    base = copy.deepcopy(walk_forward_base)
    name = profile_name or str(base.get("profile", "headline"))
    profiles = base.get("profiles") or {}
    if name not in profiles:
        raise ValueError(
            f"Unknown walk_forward profile {name!r}; "
            f"available: {sorted(profiles.keys())}"
        )
    merged = _deep_merge(base, profiles[name])
    merged["profile_active"] = name
    merged["models"] = list(merged.get("models", ALL_WF_MODELS))
    cadence = dict(merged.get("refit_cadence", {}))
    for m in merged["models"]:
        if m not in cadence:
            raise ValueError(f"Profile {name}: refit_cadence missing model {m!r}")
    merged["refit_cadence"] = cadence
    return merged


def should_tune_elasticnet(step_id: int, walk_cfg: dict) -> bool:
    """Whether this walk-forward step should run ElasticNetCV (vs frozen alpha/l1)."""
    policy = str(walk_cfg.get("tuning_policy", "scheduled"))
    if policy == "every_refit":
        return True
    if policy == "initial_only":
        return int(step_id) == 0
    if policy == "scheduled":
        every = int(walk_cfg.get("retune_every", 63))
        if every < 1:
            every = 1
        return (int(step_id) % every) == 0
    raise ValueError(f"Unknown tuning_policy: {policy}")


def needs_refit(
    model: str,
    step_id: int,
    cadence: dict[str, int],
    cache: dict,
    cache_step_last_fit: dict[str, int],
) -> bool:
    """True if model must be (re)fitted on this step."""
    c = int(cadence[model])
    if model not in cache:
        return True
    return (int(step_id) - int(cache_step_last_fit[model])) >= c


def model_enabled(model: str, active_models: list[str]) -> bool:
    return model in active_models


def walk_forward_checkpoint_file(checkpoint_dir: Path) -> Path:
    return _walk_forward_checkpoint_file(checkpoint_dir)
