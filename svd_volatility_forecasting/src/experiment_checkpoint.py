# -*- coding: utf-8 -*-
"""
Skip / resume / on-disk cache for fixed-split and walk-forward experiment runs.

- **Skip**: finished jobs are not re-run when outputs + fingerprint match.
- **Resume**: interrupted jobs continue from the last saved checkpoint.
- **Cache**: partial progress is written during long runs (WF: every N splits;
  fixed-split: after each training phase per horizon).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

JOB_COMPLETE = "job_complete.json"
WF_STATE = "walk_forward_state.json"
EXPORTS_COMPLETE = "pipeline_exports_complete.json"
HORIZON_COMPLETE = "horizon_complete.json"
PROGRESS_STATE = "progress_state.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, pd.Index):
        return [str(x) for x in obj]
    raise TypeError(f"Not JSON-serializable: {type(obj)}")


def config_fingerprint(
    *,
    seed: int,
    horizons: list[int],
    train_split: float | None = None,
    returns_path: Path | None = None,
    extra: dict | None = None,
) -> str:
    """Short hash identifying a reproducible experiment configuration."""
    payload: dict[str, Any] = {
        "seed": int(seed),
        "horizons": [int(x) for x in horizons],
    }
    if train_split is not None:
        payload["train_split"] = float(train_split)
    if extra:
        payload.update(extra)
    if returns_path is not None:
        p = Path(returns_path)
        if p.is_file():
            st = p.stat()
            payload["returns_mtime"] = float(st.st_mtime)
            payload["returns_size"] = int(st.st_size)
    raw = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def walk_forward_state_path(job_dir: Path) -> Path:
    return Path(job_dir) / WF_STATE


def walk_forward_checkpoint_file(checkpoint_dir: Path) -> Path:
    """Alias for resume state path (same file as :func:`walk_forward_state_path`)."""
    return walk_forward_state_path(checkpoint_dir)


def walk_forward_job_complete_path(job_dir: Path) -> Path:
    return Path(job_dir) / JOB_COMPLETE


def expected_wf_oos_rows(
    idx_len: int,
    *,
    initial_train_len: int,
    val_len: int,
    step: int,
) -> int:
    """Number of OOS test rows for daily walk-forward (matches walk_forward.py)."""
    t0 = int(initial_train_len) + int(val_len)
    if idx_len <= t0:
        return 0
    return int((idx_len - t0 + int(step) - 1) // int(step))


def is_walk_forward_job_complete(
    job_dir: Path,
    *,
    protocol: str,
    horizon: int,
    profile: str,
    fingerprint: str,
    expected_rows: int | None = None,
) -> bool:
    """True if this (protocol, horizon, profile) finished successfully."""
    job_dir = Path(job_dir)
    marker = _read_json(walk_forward_job_complete_path(job_dir))
    if marker is not None:
        if (
            str(marker.get("protocol")) == str(protocol)
            and int(marker.get("horizon", -1)) == int(horizon)
            and str(marker.get("profile")) == str(profile)
            and str(marker.get("fingerprint")) == str(fingerprint)
        ):
            return True
        # Another profile/fingerprint finished in this directory — do not treat as complete.
        return False
    pred_path = job_dir / "predictions_log.csv"
    if not pred_path.is_file() or expected_rows is None or expected_rows < 1:
        return False
    try:
        n = sum(1 for _ in open(pred_path, encoding="utf-8")) - 1
    except OSError:
        return False
    return n >= int(expected_rows)


def mark_walk_forward_job_complete(
    job_dir: Path,
    *,
    protocol: str,
    horizon: int,
    profile: str,
    fingerprint: str,
    n_rows: int,
) -> None:
    job_dir = Path(job_dir)
    _write_json(
        walk_forward_job_complete_path(job_dir),
        {
            "protocol": str(protocol),
            "horizon": int(horizon),
            "profile": str(profile),
            "fingerprint": str(fingerprint),
            "n_rows": int(n_rows),
        },
    )
    clear_walk_forward_resume_state(job_dir)


def clear_walk_forward_resume_state(job_dir: Path) -> None:
    p = walk_forward_state_path(Path(job_dir))
    if p.is_file():
        p.unlink()


def save_walk_forward_checkpoint(
    job_dir: Path,
    *,
    protocol: str,
    horizon: int,
    profile: str,
    fingerprint: str,
    last_step_id: int,
    last_si: int,
    preds_log: dict[str, list],
    preds_var: dict[str, list],
    test_dates: list,
) -> None:
    """Persist partial WF state (imported by walk_forward_profile for compatibility)."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": str(protocol),
        "horizon": int(horizon),
        "profile": str(profile),
        "fingerprint": str(fingerprint),
        "last_step_id": int(last_step_id),
        "last_si": int(last_si),
        "preds_log": {k: list(v) for k, v in preds_log.items()},
        "preds_var": {k: list(v) for k, v in preds_var.items()},
        "test_dates": [str(d) for d in test_dates],
    }
    walk_forward_state_path(job_dir).write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def load_walk_forward_checkpoint(
    job_dir: Path,
    *,
    protocol: str,
    horizon: int,
    profile: str,
    fingerprint: str,
) -> dict | None:
    path = walk_forward_state_path(Path(job_dir))
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        str(data.get("protocol")) != str(protocol)
        or int(data.get("horizon", -1)) != int(horizon)
        or str(data.get("profile")) != str(profile)
        or str(data.get("fingerprint", "")) != str(fingerprint)
    ):
        return None
    return data


# ---------------------------------------------------------------------------
# Fixed-split (per train_split × horizon)
# ---------------------------------------------------------------------------

class FixedSplitHorizonCache:
    """
    Phase-level cache for one (train_split, horizon) training block.

    Phases: ``linear``, ``deep``, ``gnn``, ``garch_combo``, ``ablation``.
    """

    PHASES = ("linear", "deep", "gnn", "garch_combo", "ablation")

    def __init__(
        self,
        cache_root: Path,
        *,
        train_split: float,
        horizon: int,
        fingerprint: str,
        force: bool = False,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.train_split = float(train_split)
        self.horizon = int(horizon)
        self.fingerprint = str(fingerprint)
        self.force = bool(force)
        split_tag = f"split_{self.train_split:.4f}".replace(".", "_")
        self.job_dir = self.cache_root / "fixed_split" / split_tag / f"h{self.horizon}"
        self.progress_dir = self.job_dir / "progress"

    def _progress_state(self) -> dict:
        p = self.progress_dir / PROGRESS_STATE
        data = _read_json(p)
        return data if data is not None else {"completed_phases": [], "fingerprint": self.fingerprint}

    def _save_progress_state(self, completed: list[str]) -> None:
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            self.progress_dir / PROGRESS_STATE,
            {"completed_phases": list(completed), "fingerprint": self.fingerprint},
        )

    def is_horizon_complete(self) -> bool:
        if self.force:
            return False
        marker = _read_json(self.job_dir / HORIZON_COMPLETE)
        if marker is None:
            return False
        return (
            str(marker.get("fingerprint")) == self.fingerprint
            and float(marker.get("train_split", -1)) == self.train_split
            and int(marker.get("horizon", -1)) == self.horizon
        )

    def load_horizon_pack(self) -> dict | None:
        pack_path = self.job_dir / "horizon_pack.json"
        if not pack_path.is_file():
            return None
        data = json.loads(pack_path.read_text(encoding="utf-8"))
        if str(data.get("fingerprint")) != self.fingerprint:
            return None
        return data

    def mark_horizon_complete(self, rb_h: dict, ap_h: dict, extras: dict | None = None) -> None:
        self.job_dir.mkdir(parents=True, exist_ok=True)
        pack = {
            "fingerprint": self.fingerprint,
            "train_split": self.train_split,
            "horizon": self.horizon,
            "rb": rb_h,
            "ap": {k: _json_default(v) if isinstance(v, np.ndarray) else v for k, v in ap_h.items()},
            "extras": extras or {},
        }
        for k, v in pack["ap"].items():
            if isinstance(v, np.ndarray):
                pack["ap"][k] = v.tolist()
            elif isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], (np.floating, float)):
                pack["ap"][k] = [float(x) for x in v]
            elif hasattr(v, "__iter__") and not isinstance(v, (str, dict)):
                try:
                    pack["ap"][k] = [str(x) for x in v]
                except Exception:
                    pass
        _write_json(self.job_dir / "horizon_pack.json", pack)
        _write_json(
            self.job_dir / HORIZON_COMPLETE,
            {
                "fingerprint": self.fingerprint,
                "train_split": self.train_split,
                "horizon": self.horizon,
            },
        )
        if self.progress_dir.is_dir():
            shutil.rmtree(self.progress_dir, ignore_errors=True)

    def phase_done(self, phase: str) -> bool:
        if self.force:
            return False
        st = self._progress_state()
        if str(st.get("fingerprint")) != self.fingerprint:
            return False
        return phase in st.get("completed_phases", [])

    def load_phase(self, phase: str) -> dict[str, Any]:
        path = self.progress_dir / f"{phase}.json"
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        out: dict[str, Any] = {}
        for k, v in data.items():
            if k.startswith("_arr_"):
                out[k[5:]] = np.asarray(v, dtype=np.float64)
            else:
                out[k] = v
        return out

    def save_phase(self, phase: str, artifacts: dict[str, Any]) -> None:
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        serial: dict[str, Any] = {}
        for k, v in artifacts.items():
            if isinstance(v, np.ndarray):
                serial[f"_arr_{k}"] = np.asarray(v, dtype=np.float64).tolist()
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (np.floating, np.integer)):
                serial[f"_arr_{k}"] = [float(x) for x in v]
            else:
                serial[k] = v
        _write_json(self.progress_dir / f"{phase}.json", serial)
        st = self._progress_state()
        done = list(st.get("completed_phases", []))
        if phase not in done:
            done.append(phase)
        self._save_progress_state(done)
        print(f"[CACHE] fixed_split h={self.horizon} phase={phase!r} saved", flush=True)

    def has_partial_progress(self) -> bool:
        st_path = self.progress_dir / PROGRESS_STATE
        return st_path.is_file() and not self.is_horizon_complete()

    def restore_arrays(self, data: dict[str, Any]) -> dict[str, Any]:
        """Re-hydrate arrays after ``load_horizon_pack``."""
        out: dict[str, Any] = {}
        ap = data.get("ap", data)
        non_numeric = {"test_dates", "_h1_train_dates_for_gw"}
        for k, v in ap.items():
            if k in non_numeric:
                out[k] = v
            elif isinstance(v, list):
                try:
                    out[k] = np.asarray(v, dtype=np.float64)
                except (ValueError, TypeError):
                    out[k] = v
            else:
                out[k] = v
        return out


# ---------------------------------------------------------------------------
# Fixed-split export / post-processing
# ---------------------------------------------------------------------------

def exports_complete_path(results_dir: Path) -> Path:
    return Path(results_dir) / EXPORTS_COMPLETE


def is_exports_complete(results_dir: Path, fingerprint: str) -> bool:
    marker = _read_json(exports_complete_path(results_dir))
    if marker is None:
        return False
    return str(marker.get("fingerprint")) == str(fingerprint)


def all_fixed_split_horizons_complete(
    cache_root: Path,
    *,
    train_split: float,
    horizons: list[int],
    fingerprint: str,
    force: bool = False,
) -> bool:
    if force:
        return False
    for h in horizons:
        fsc = FixedSplitHorizonCache(
            cache_root,
            train_split=float(train_split),
            horizon=int(h),
            fingerprint=str(fingerprint),
            force=False,
        )
        if not fsc.is_horizon_complete():
            return False
    return True


def load_fixed_split_horizon_results(
    cache_root: Path,
    *,
    train_split: float,
    horizons: list[int],
    fingerprint: str,
) -> tuple[dict, dict]:
    rb: dict = {}
    ap: dict = {}
    for h in horizons:
        fsc = FixedSplitHorizonCache(
            cache_root,
            train_split=float(train_split),
            horizon=int(h),
            fingerprint=str(fingerprint),
        )
        pack = fsc.load_horizon_pack()
        if pack is None:
            raise FileNotFoundError(f"Missing horizon pack for h={h} split={train_split}")
        rb[int(h)] = pack["rb"]
        ap[int(h)] = fsc.restore_arrays(pack)
    return rb, ap


def mark_exports_complete(results_dir: Path, fingerprint: str) -> None:
    _write_json(
        exports_complete_path(results_dir),
        {"fingerprint": str(fingerprint), "complete": True},
    )
