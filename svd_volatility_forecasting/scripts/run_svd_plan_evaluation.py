#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the full evaluation pipeline from fix_svd_underperformance plan.

Steps:
  1. pytest
  2. Fixed-split experiments (all new SVD variants)
  3. Walk-forward expanding, profile ``svd_fix`` (linear SVD hypotheses + benchmarks)
  4. Regime diagnostics (h=1,5,22)
  5. Feature redundancy / stability diagnostics
  6. Tail calibration (h=1)
  7. Consolidated markdown report

Usage:
  python scripts/run_svd_plan_evaluation.py
  python scripts/run_svd_plan_evaluation.py --skip-wf   # diagnostics only on existing preds
  python scripts/run_svd_plan_evaluation.py --wf-profile full  # include deep models (slow)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT
RESULTS_DIR = ROOT / "results"
REPORT_DIR = RESULTS_DIR / "svd_plan_evaluation"


def _run(cmd: list[str], *, log: Path, step: str) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    header = f"\n{'=' * 72}\n[{datetime.now(timezone.utc).isoformat()}] {step}\n{'=' * 72}\n"
    with log.open("a", encoding="utf-8") as f:
        f.write(header)
        f.write(" ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    print(f"[{step}] exit={p.returncode} (log: {log})")
    return int(p.returncode)


def _write_report(report_path: Path, *, wf_profile: str) -> None:
    import pandas as pd

    lines = [
        "# SVD plan evaluation report",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}",
        f"\nWalk-forward profile: `{wf_profile}`",
        "\n## Walk-forward metrics (expanding)\n",
    ]
    wf_root = RESULTS_DIR / "walk_forward" / "expanding"
    focus = [
        "IV_baseline",
        "HAR",
        "HAR_SVD_T1",
        "HAR+SVD",
        "HAR+SVD_GATED_AR",
        "HAR+SVD_DYN",
        "HAR+SVD_RFF_RIDGE",
        "HAR_SVD_T3",
        "GARCH",
        "GJR-GARCH-t",
    ]
    for h in (1, 5, 22):
        mp = wf_root / f"h{h}" / "metrics.csv"
        if not mp.is_file():
            lines.append(f"\n### h={h}\n\n(missing metrics.csv)\n")
            continue
        m = pd.read_csv(mp)
        if "model" in m.columns:
            m = m.set_index("model")
        sub = m.reindex([x for x in focus if x in m.index]).dropna(how="all")
        lines.append(f"\n### h={h}\n\n")
        if sub.empty:
            lines.append("(no matching models in metrics.csv)\n")
        else:
            cols = [c for c in ["QLIKE", "RMSE", "R2", "MALE"] if c in sub.columns]
            lines.append(sub[cols].sort_values("QLIKE").to_markdown() + "\n")

        reg = wf_root / f"h{h}" / "diagnostics" / f"wf_regime_diagnostics_expanding_h{h}.csv"
        if reg.is_file():
            rd = pd.read_csv(reg)
            svd_models = [x for x in rd["model"].unique() if "SVD" in x or x == "HAR"]
            for rv in rd["regime_var"].unique()[:2]:
                hi = rd[(rd["regime_var"] == rv) & (rd["bin"] == rd["bin"].max())]
                hi = hi[hi["model"].isin(svd_models)].sort_values("QLIKE").head(3)
                if not hi.empty:
                    lines.append(
                        f"\n**High-{rv} bin (top QLIKE):** "
                        + ", ".join(
                            f"{r['model']}={r['QLIKE']:.4f}" for _, r in hi.iterrows()
                        )
                        + "\n"
                    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"[OK] report: {report_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--wf-profile",
        default="svd_fix",
        choices=["svd_fix", "full", "headline"],
        help="Walk-forward profile (svd_fix = plan linear variants).",
    )
    ap.add_argument("--skip-wf", action="store_true", help="Skip training; run post-hoc only.")
    ap.add_argument("--skip-fixed", action="store_true", help="Skip fixed-split run.")
    ap.add_argument(
        "--no-force-rerun",
        action="store_true",
        help="Allow skip/resume from cache (default: force rerun so new SVD variants are trained).",
    )
    args = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    master_log = REPORT_DIR / f"run_{ts}.log"
    force = [] if args.no_force_rerun else ["--force-rerun"]
    rc = 0

    if _run([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"], log=master_log, step="pytest"):
        return 1

    if not args.skip_fixed:
        if _run(
            [sys.executable, str(ROOT / "scripts" / "run_experiments.py"), *force],
            log=master_log,
            step="fixed_split",
        ):
            rc = 1

    if not args.skip_wf:
        if _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_experiments.py"),
                "--walk-forward",
                "--walk-forward-profile",
                args.wf_profile,
                *force,
            ],
            log=master_log,
            step="walk_forward",
        ):
            rc = 1

    for h in (1, 5, 22):
        _run(
            [
                sys.executable,
                "scripts/wf_regime_diagnostics.py",
                "--protocol",
                "expanding",
                "--horizon",
                str(h),
            ],
            log=master_log,
            step=f"regime_diagnostics_h{h}",
        )
        _run(
            [
                sys.executable,
                "scripts/wf_feature_redundancy.py",
                "--protocol",
                "expanding",
                "--horizon",
                str(h),
            ],
            log=master_log,
            step=f"redundancy_h{h}",
        )

    for model in ("HAR+SVD", "HAR+SVD_GATED_AR", "HAR"):
        _run(
            [
                sys.executable,
                "scripts/wf_tail_calibration.py",
                "--protocol",
                "expanding",
                "--horizon",
                "1",
                "--model",
                model,
                "--gate",
                "AR",
            ],
            log=master_log,
            step=f"tailcal_{model.replace('+', '_')}",
        )

    _write_report(REPORT_DIR / "svd_plan_report.md", wf_profile=args.wf_profile)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
