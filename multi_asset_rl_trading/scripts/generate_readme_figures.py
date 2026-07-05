"""Generate summary figures for the project README from saved backtest metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "normal"
FIGURES_DIR = ROOT / "docs" / "figures"


def load_metrics() -> pd.DataFrame:
    rows = []
    for test_dir in sorted(RESULTS_DIR.glob("test_*")):
        metrics_path = test_dir / "model_comparison_metrics.csv"
        if not metrics_path.exists():
            continue
        year = test_dir.name.replace("test_", "")
        df = pd.read_csv(metrics_path)
        df["Test Year"] = year
        rows.append(df)
    if not rows:
        raise FileNotFoundError("No model_comparison_metrics.csv files found under results/normal/")
    return pd.concat(rows, ignore_index=True)


def plot_sharpe_heatmap(metrics: pd.DataFrame, output_path: Path) -> None:
    pivot = metrics.pivot(index="Model", columns="Test Year", values="Sharpe_Ratio_mean")
    model_order = ["Random", "A2C", "LSTM", "CNN", "ANN", "ARIMA"]
    pivot = pivot.reindex([m for m in model_order if m in pivot.index])

    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=0.5)
    plt.title("Out-of-Sample Sharpe Ratio by Model and Test Year")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_return_bars(metrics: pd.DataFrame, output_path: Path) -> None:
    model_order = ["Random", "A2C", "LSTM", "CNN", "ANN", "ARIMA"]
    metrics = metrics.copy()
    metrics["Model"] = pd.Categorical(metrics["Model"], categories=model_order, ordered=True)
    metrics = metrics.sort_values(["Test Year", "Model"])

    g = sns.catplot(
        data=metrics,
        kind="bar",
        x="Model",
        y="Return_Rate_mean",
        hue="Test Year",
        palette="muted",
        height=5,
        aspect=1.6,
    )
    g.set_axis_labels("Model", "Return Rate (%)")
    g.set(title="Out-of-Sample Return by Model")
    g.legend.set_title("Test Year")
    plt.xticks(rotation=30)
    g.savefig(output_path, dpi=160)
    plt.close()


def plot_drawdown(metrics: pd.DataFrame, output_path: Path) -> None:
    model_order = ["Random", "A2C", "LSTM", "CNN", "ANN", "ARIMA"]
    metrics = metrics.copy()
    metrics["Model"] = pd.Categorical(metrics["Model"], categories=model_order, ordered=True)
    metrics = metrics.sort_values(["Test Year", "Model"])

    g = sns.catplot(
        data=metrics,
        kind="bar",
        x="Model",
        y="Max_Drawdown_mean",
        hue="Test Year",
        palette="Reds_r",
        height=5,
        aspect=1.6,
    )
    g.set_axis_labels("Model", "Max Drawdown (%)")
    g.set(title="Out-of-Sample Max Drawdown by Model")
    g.legend.set_title("Test Year")
    plt.xticks(rotation=30)
    g.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()
    plot_sharpe_heatmap(metrics, FIGURES_DIR / "sharpe_heatmap.png")
    plot_return_bars(metrics, FIGURES_DIR / "return_by_model.png")
    plot_drawdown(metrics, FIGURES_DIR / "max_drawdown_by_model.png")
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
