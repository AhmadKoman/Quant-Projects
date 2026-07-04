#!/usr/bin/env python3
"""Build README summary charts from saved metrics."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics" / "results_by_horizon.json"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

HEADLINE = [
    "HAR",
    "HAR+SVD",
    "HAR+SVD_GATED_AR",
    "HAR+SVD_TAILCAL_AR",
    "HAR+SVD_OSI",
    "HAR_SVD_T3",
    "GJR-GARCH-t",
    "GARCH",
    "DNN_HAR",
    "DNN_HAR+SVD",
    "LSTM_HAR",
    "LSTM_HAR+SVD",
    "HARNet",
    "GNN",
]

PDFS = [
    "F2_ablation_heatmap",
    "F7_dm_matrix_h1",
    "scatter_actual_vs_predicted",
    "full_sample_actual_vs_predicted",
    "F1_posterior_predictive_forecast",
    "F3_threshold_sensitivity",
    "crisis_window_actual_vs_predicted",
    "F6_residual_diagnostics_HAR_plus_SVD",
]


def convert_pdfs() -> None:
    src = ROOT / "results" / "figures"
    for stem in PDFS:
        pdf = src / f"{stem}.pdf"
        if not pdf.is_file():
            print(f"[skip] missing {pdf.name}")
            continue
        out = OUT / f"{stem}.png"
        subprocess.run(
            ["pdftoppm", "-png", "-r", "144", "-singlefile", str(pdf), str(OUT / stem)],
            check=True,
        )
        generated = OUT / f"{stem}.png"
        if generated.is_file():
            print(f"[ok] {generated.name}")


def rmse_heatmap() -> None:
    data = json.loads(METRICS.read_text())
    horizons = ["1", "5", "22"]
    labels = []
    rows = []
    for model in HEADLINE:
        vals = []
        ok = False
        for h in horizons:
            m = data.get(h, {}).get(model, {})
            rmse = m.get("RMSE")
            if rmse is None or (isinstance(rmse, float) and np.isnan(rmse)):
                vals.append(np.nan)
            else:
                vals.append(float(rmse))
                ok = True
        if ok:
            labels.append(model.replace("_", " "))
            rows.append(vals)

    arr = np.array(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, max(4.0, 0.35 * len(labels))))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
    ax.set_title("Out of sample RMSE by model and horizon")
    fig.colorbar(im, ax=ax, shrink=0.8, label="RMSE")
    fig.tight_layout()
    fig.savefig(OUT / "rmse_by_horizon.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[ok] rmse_by_horizon.png")


def qlike_h1_bar() -> None:
    data = json.loads(METRICS.read_text())
    h1 = data["1"]
    items = [(m, h1[m]["QLIKE"]) for m in HEADLINE if m in h1 and np.isfinite(h1[m]["QLIKE"])]
    items.sort(key=lambda x: x[1])
    models = [x[0].replace("_", " ") for x in items]
    qlikes = [x[1] for x in items]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2ecc71" if "SVD" in m or "svd" in m else "#3498db" for m in models]
    ax.barh(models, qlikes, color=colors)
    ax.set_xlabel("QLIKE (lower is better)")
    ax.set_title("Fixed split h=1 forecast loss (QLIKE)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "qlike_h1_bar.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[ok] qlike_h1_bar.png")


def main() -> None:
    convert_pdfs()
    rmse_heatmap()
    qlike_h1_bar()


if __name__ == "__main__":
    main()
