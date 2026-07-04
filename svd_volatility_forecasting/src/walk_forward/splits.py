"""
Walk-forward (rolling + expanding) out-of-sample evaluation utilities.

This module is intentionally strict:
  - date-index based splits only (no fractional splits)
  - no overlap between train/val/test within a step
  - deterministic, fully specified by config

It does NOT fit models; it only defines the splitting protocol and provides
metadata needed by orchestration code in `run_experiments.py` (or another runner).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

Protocol = Literal["expanding", "rolling"]


@dataclass(frozen=True)
class WalkForwardSplit:
    """A single walk-forward step split (all indices are date labels, not integer positions)."""

    protocol: Protocol
    step_id: int
    horizon: int
    train_idx: pd.Index
    val_idx: pd.Index
    test_idx: pd.Index

    @property
    def origin(self) -> pd.Timestamp:
        """Forecast origin date: first (and typically only) date in test_idx."""
        return pd.Timestamp(self.test_idx[0])


def _validate_index(idx: pd.Index) -> pd.Index:
    if not isinstance(idx, pd.Index):
        raise TypeError("idx must be a pandas Index of dates.")
    if len(idx) < 10:
        raise ValueError("idx too short for walk-forward evaluation.")
    if not idx.is_monotonic_increasing:
        raise ValueError("idx must be strictly increasing (chronological).")
    if idx.has_duplicates:
        raise ValueError("idx must not contain duplicate dates.")
    # Force Timestamp index for safety
    try:
        idx = pd.Index(pd.to_datetime(idx))
    except Exception as e:
        raise ValueError("idx must be coercible to datetime.") from e
    return idx


def _validate_step_slices(train_pos: slice, val_pos: slice, test_pos: slice, n: int) -> None:
    # Convert slices to (start, stop)
    tr0, tr1 = int(train_pos.start), int(train_pos.stop)
    va0, va1 = int(val_pos.start), int(val_pos.stop)
    te0, te1 = int(test_pos.start), int(test_pos.stop)
    if not (0 <= tr0 < tr1 <= n):
        raise ValueError(f"Invalid train slice: {train_pos} for n={n}")
    if not (0 <= va0 < va1 <= n):
        raise ValueError(f"Invalid val slice: {val_pos} for n={n}")
    if not (0 <= te0 < te1 <= n):
        raise ValueError(f"Invalid test slice: {test_pos} for n={n}")
    # Enforce strict chronological separation
    if not (tr1 <= va0):
        raise ValueError("train must end on/before val starts (no overlap).")
    if not (va1 <= te0):
        raise ValueError("val must end on/before test starts (no overlap).")
    # No empty blocks
    if (tr1 - tr0) < 1 or (va1 - va0) < 1 or (te1 - te0) < 1:
        raise ValueError("train/val/test blocks must all be non-empty.")


def iter_walk_forward_splits(
    idx: pd.Index,
    *,
    protocol: Protocol,
    horizons: list[int],
    initial_train_len: int,
    rolling_train_len: int,
    val_len: int,
    step: int = 1,
) -> Iterable[WalkForwardSplit]:
    """
    Yield walk-forward splits for each horizon under a given protocol.

    Split layout at each step (three-block):
      - train_core: used for fitting model parameters
      - val_block : used for early stopping / tuning diagnostics and smearing estimation
      - test_point: the forecast evaluation point(s), typically length=1

    Important: This splitter is agnostic to horizon alignment. The calling code must
    ensure that y_h and X_h are pre-aligned so that test_idx corresponds to the
    correct realized target for that horizon (your pipeline already builds forward RV).
    """
    idx = _validate_index(idx)
    n = len(idx)
    if step < 1:
        raise ValueError("step must be >= 1.")
    if initial_train_len < 50:
        raise ValueError("initial_train_len must be sufficiently large (>= 50).")
    if val_len < 10:
        raise ValueError("val_len must be >= 10 for meaningful validation.")
    if protocol == "rolling" and rolling_train_len < 50:
        raise ValueError("rolling_train_len must be sufficiently large (>= 50).")
    if not horizons:
        raise ValueError("horizons must be non-empty.")
    if any(int(h) < 1 for h in horizons):
        raise ValueError(f"Invalid horizons: {horizons}")

    # Define the first test position (integer index) at which we have:
    #   train_core length >= initial_train_len - val_len
    #   val_block length == val_len
    # With a three-block scheme, we anchor the first test at position:
    #   t0 = initial_train_len + val_len
    # where train = [0, initial_train_len), val = [initial_train_len, initial_train_len+val_len)
    t0 = initial_train_len + val_len
    if t0 >= n:
        raise ValueError(
            f"Not enough data for walk-forward: need n > initial_train_len+val_len = {t0}, got n={n}."
        )

    step_id = 0
    for test_pos0 in range(t0, n, step):
        test_pos = slice(test_pos0, min(test_pos0 + 1, n))
        # Validation is the contiguous block immediately before test
        val_pos = slice(test_pos0 - val_len, test_pos0)

        if protocol == "expanding":
            train_pos = slice(0, test_pos0 - val_len)
        elif protocol == "rolling":
            tr_end = test_pos0 - val_len
            tr_start = max(0, tr_end - rolling_train_len)
            train_pos = slice(tr_start, tr_end)
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        _validate_step_slices(train_pos, val_pos, test_pos, n)

        train_idx = idx[train_pos]
        val_idx = idx[val_pos]
        test_idx = idx[test_pos]

        for h in horizons:
            yield WalkForwardSplit(
                protocol=protocol,
                step_id=step_id,
                horizon=int(h),
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )

        step_id += 1


def splits_to_frame(splits: Iterable[WalkForwardSplit]) -> pd.DataFrame:
    """Convert an iterable of splits to a compact DataFrame for logging/debugging."""
    rows = []
    for s in splits:
        rows.append(
            {
                "protocol": s.protocol,
                "step_id": int(s.step_id),
                "horizon": int(s.horizon),
                "train_start": s.train_idx[0],
                "train_end": s.train_idx[-1],
                "val_start": s.val_idx[0],
                "val_end": s.val_idx[-1],
                "test_date": s.test_idx[0],
                "n_train": int(len(s.train_idx)),
                "n_val": int(len(s.val_idx)),
                "n_test": int(len(s.test_idx)),
            }
        )
    return pd.DataFrame(rows)

