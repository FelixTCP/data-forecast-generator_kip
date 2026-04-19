from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot actual target series with train/test split")
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory")
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: <run-dir>/inference_eval/actual_timeseries_split.png)",
    )
    parser.add_argument(
        "--zoom-output",
        default=None,
        help=(
            "Output PNG path for zoomed test prediction vs ground truth "
            "(default: <run-dir>/inference_eval/test_zoom_pred_vs_truth.png)"
        ),
    )
    parser.add_argument(
        "--zoom-points",
        type=int,
        default=500,
        help="How many last test points to show in zoomed chart",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)

    step12_path = run_dir / "step-12-features.json"
    features_path = run_dir / "features.parquet"

    if not step12_path.exists():
        raise FileNotFoundError(f"Missing file: {step12_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing file: {features_path}")

    step12 = json.loads(step12_path.read_text(encoding="utf-8"))
    context = step12["context"]
    target = context["target_column"]
    split_mode = context["split_strategy"]["resolved_mode"]

    df = pl.read_parquet(features_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {features_path}")

    y = df.select(pl.col(target).cast(pl.Float64, strict=False)).to_numpy().reshape(-1)
    n = len(y)

    if split_mode == "time_series":
        split_idx = int(n * 0.8)
    else:
        split_idx = int(n * 0.8)

    out_path = (
        Path(args.output)
        if args.output
        else run_dir / "inference_eval" / "actual_timeseries_split.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(n)
    ax.plot(x, y, linewidth=1.0, alpha=0.9, color="#1f77b4", label=f"Actual {target}")
    ax.axvline(
        split_idx,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label=f"Test starts at index {split_idx}",
    )
    ax.axvspan(0, split_idx, color="#2ca02c", alpha=0.08, label="Train region")
    ax.axvspan(split_idx, n, color="#ff7f0e", alpha=0.08, label="Test region")
    ax.set_title(f"Actual {target} with Train/Test Split ({split_mode})")
    ax.set_xlabel("Row index (time-ordered after feature extraction)")
    ax.set_ylabel(target)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    holdout_path = run_dir / "holdout.npz"
    preds_path = run_dir / "inference_eval" / "predictions.csv"
    if holdout_path.exists() and preds_path.exists():
        holdout = np.load(holdout_path)
        y_test = np.asarray(holdout["y_test"]).reshape(-1)

        preds_df = pl.read_csv(preds_path)
        if "prediction" not in preds_df.columns:
            raise ValueError(f"Expected 'prediction' column in {preds_path}")
        y_pred = preds_df.select(pl.col("prediction").cast(pl.Float64, strict=False)).to_numpy().reshape(-1)

        if len(y_test) != len(y_pred):
            raise ValueError(
                f"Prediction length {len(y_pred)} does not match y_test length {len(y_test)}"
            )

        zoom_n = max(50, min(int(args.zoom_points), len(y_test)))
        start = len(y_test) - zoom_n
        xx = np.arange(start, len(y_test))

        zoom_out = (
            Path(args.zoom_output)
            if args.zoom_output
            else run_dir / "inference_eval" / "test_zoom_pred_vs_truth.png"
        )
        zoom_out.parent.mkdir(parents=True, exist_ok=True)

        fig2, ax2 = plt.subplots(figsize=(16, 5))
        ax2.plot(xx, y_test[start:], linewidth=1.8, color="#1f77b4", label="Ground truth (test)")
        ax2.plot(xx, y_pred[start:], linewidth=1.5, color="#d62728", alpha=0.9, label="Prediction (test)")
        ax2.set_title(f"Zoomed Test Window: Prediction vs Ground Truth ({zoom_n} points)")
        ax2.set_xlabel("Test index")
        ax2.set_ylabel(target)
        ax2.grid(alpha=0.25)
        ax2.legend(loc="upper right")
        fig2.tight_layout()
        fig2.savefig(zoom_out, dpi=170)
        plt.close(fig2)

        print(zoom_out)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
