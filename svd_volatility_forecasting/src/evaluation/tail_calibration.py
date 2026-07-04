# -*- coding: utf-8 -*-
"""
Tail calibration layers for variance forecasts using FZ0 scoring.

Goal
----
Variance forecasts that are competitive on QLIKE can still be miscalibrated for
VaR/ES (hit rates far above alpha). This module fits small, leakage-safe scale
maps that *only* adjust the conditional scale (sigma) using state variables.

We focus on a simple, robust structure:
  - two-regime scale multiplier on sigma: low vs high gate state
  - optimized to minimize average FZ0 score on a training sample

This provides a concrete “tail objective” path without changing the mean model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import economic as econ


@dataclass(frozen=True)
class TwoRegimeScale:
    gate_name: str
    q_high: float
    threshold: float
    sigma_mult_low: float
    sigma_mult_high: float


def _to_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def apply_two_regime_scale(
    pred_var: np.ndarray,
    *,
    gate: np.ndarray,
    params: TwoRegimeScale,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return scaled variance forecast (variance scale)."""
    h = np.maximum(_to_1d(pred_var), eps)
    g = _to_1d(gate)
    if g.shape != h.shape:
        raise ValueError("gate and pred_var must have same shape.")
    high = g >= float(params.threshold)
    # sigma_adj = sigma * m  => var_adj = var * m^2
    m = np.where(high, float(params.sigma_mult_high), float(params.sigma_mult_low))
    return h * (m ** 2)


def fit_two_regime_scale_fz0(
    returns: np.ndarray,
    pred_var: np.ndarray,
    *,
    gate: np.ndarray,
    gate_name: str,
    alpha: float,
    q_high: float = 0.8,
    dist: econ.DistName = "gaussian",
    df: float | None = None,
    grid_log10: tuple[float, float, int] = (-0.5, 0.8, 41),
    eps: float = 1e-12,
) -> TwoRegimeScale:
    """
    Fit (sigma_mult_low, sigma_mult_high) on a training sample.

    We search over a multiplicative grid for sigma multipliers. This is cheap,
    deterministic, and stable; it also aligns with the main failure mode we
    see in the WF VaR backtests (systematic underprediction of tail risk).
    """
    r = _to_1d(returns)
    h = np.maximum(_to_1d(pred_var), eps)
    g = _to_1d(gate)
    if r.shape != h.shape or r.shape != g.shape:
        raise ValueError("returns, pred_var, gate must have same shape.")

    gf = g[np.isfinite(g)]
    if gf.size < 200:
        raise ValueError("Gate has insufficient finite observations.")
    thr = float(np.quantile(gf, float(q_high)))
    high = g >= thr
    if int(high.sum()) < 50 or int((~high).sum()) < 50:
        raise ValueError("Two-regime split too imbalanced for fitting.")

    lo, hi, n = grid_log10
    grid = np.logspace(float(lo), float(hi), int(n))
    best = None
    best_score = float("inf")

    # Pre-split to avoid repeated masking work
    r0, h0 = r[~high], h[~high]
    r1, h1 = r[high], h[high]

    for m0 in grid:
        h0s = h0 * (float(m0) ** 2)
        bt0, _ = econ.backtest_var_es(r0, h0s, alpha=alpha, dist=dist, df=df, eps=eps)
        for m1 in grid:
            h1s = h1 * (float(m1) ** 2)
            bt1, _ = econ.backtest_var_es(r1, h1s, alpha=alpha, dist=dist, df=df, eps=eps)
            score = float((bt0.mean_fz0 * len(r0) + bt1.mean_fz0 * len(r1)) / len(r))
            if score < best_score:
                best_score = score
                best = (float(m0), float(m1))

    assert best is not None
    return TwoRegimeScale(
        gate_name=str(gate_name),
        q_high=float(q_high),
        threshold=float(thr),
        sigma_mult_low=float(best[0]),
        sigma_mult_high=float(best[1]),
    )

