from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference + quality evaluation utility")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Pipeline run directory containing model.joblib and holdout.npz",
    )
    parser.add_argument("--model", default=None, help="Path to model.joblib")
    parser.add_argument("--csv", default=None, help="Path to input CSV for inference")
    parser.add_argument(
        "--holdout",
        default=None,
        help="Optional holdout.npz path with X_test and y_test for verification",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Optional target column for quick evaluation (R2/RMSE/MAE)",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Optional path to step-12-features.json to enforce feature order on CSV inference",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save predictions CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write metrics and charts (default: <run-dir>/inference_eval or ./inference_eval)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for prediction loops with tqdm progress",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="How many predictions to print",
    )
    return parser


def _extract_predictor(artifact: Any) -> Any:
    if hasattr(artifact, "predict"):
        return artifact

    if isinstance(artifact, dict):
        for key in ("model", "estimator", "best_model", "pipeline"):
            candidate = artifact.get(key)
            if hasattr(candidate, "predict"):
                return candidate
        raise ValueError(
            "model.joblib contains a dict without a fitted estimator under keys "
            "['model', 'estimator', 'best_model', 'pipeline']."
        )

    raise ValueError(f"Unsupported model artifact type: {type(artifact).__name__}")


def _predict_in_batches(model: Any, x: np.ndarray, batch_size: int) -> np.ndarray:
    if x.size == 0:
        return np.array([], dtype=float)

    outputs: list[np.ndarray] = []
    total = x.shape[0]
    for start in tqdm(range(0, total, batch_size), desc="predict", unit="batch"):
        stop = min(start + batch_size, total)
        chunk = x[start:stop]
        outputs.append(np.asarray(model.predict(chunk)).reshape(-1))
    return np.concatenate(outputs, axis=0)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    if np.all(np.abs(y_true) > 1e-12):
        metrics["mape"] = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    return metrics


def _plot_quality(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> None:
    residuals = y_true - y_pred
    abs_err = np.abs(residuals)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, s=8, alpha=0.45, edgecolors="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.25)
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    ax = axes[0, 1]
    ax.hist(residuals, bins=50, alpha=0.85)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Count")

    ax = axes[1, 0]
    ax.scatter(y_pred, residuals, s=8, alpha=0.45, edgecolors="none")
    ax.axhline(0.0, linestyle="--", linewidth=1.25)
    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")

    ax = axes[1, 1]
    sorted_err = np.sort(abs_err)
    q = np.linspace(0.0, 100.0, len(sorted_err), endpoint=True)
    ax.plot(q, sorted_err, linewidth=1.5)
    ax.set_title("Absolute Error by Percentile")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Absolute Error")

    fig.tight_layout()
    fig_path = out_dir / "quality_overview.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None, Path]:
    run_dir = Path(args.run_dir) if args.run_dir else None

    if run_dir is not None:
        model_path = Path(args.model) if args.model else run_dir / "model.joblib"
        holdout_path = Path(args.holdout) if args.holdout else run_dir / "holdout.npz"
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = run_dir / "inference_eval"
    else:
        if not args.model:
            raise ValueError("Provide --model or --run-dir")
        model_path = Path(args.model)
        holdout_path = Path(args.holdout) if args.holdout else None
        output_dir = Path(args.output_dir) if args.output_dir else Path("inference_eval")

    csv_path = Path(args.csv) if args.csv else None
    return model_path, holdout_path, csv_path, output_dir


def _load_feature_order(features_json: Path | None) -> list[str] | None:
    if features_json is None:
        return None
    payload = json.loads(features_json.read_text(encoding="utf-8"))
    values = payload.get("features")
    if isinstance(values, list) and all(isinstance(v, str) for v in values):
        return values
    return None


def main() -> int:
    args = _build_parser().parse_args()

    model_path, holdout_path, csv_path, output_dir = _resolve_paths(args)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if csv_path is not None and not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(model_path)
    model = _extract_predictor(artifact)

    y_true: np.ndarray | None = None
    preds: np.ndarray

    if holdout_path is not None and holdout_path.exists():
        holdout = np.load(holdout_path)
        x_test = np.asarray(holdout["X_test"])
        y_true = np.asarray(holdout["y_test"]).reshape(-1)
        preds = _predict_in_batches(model, x_test, args.batch_size)

        metrics = _compute_metrics(y_true, preds)
        metrics_path = output_dir / "test_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _plot_quality(y_true, preds, output_dir)

        print("Inference + verification successful on holdout test data")
        print(f"Rows: {len(preds)}")
        print(
            "Metrics -> "
            + ", ".join([f"{k.upper()}={v:.6f}" for k, v in metrics.items()])
        )
        print(f"Saved: {metrics_path}")
        print(f"Saved: {output_dir / 'quality_overview.png'}")
    else:
        if csv_path is None:
            raise ValueError(
                "No holdout file found and no --csv provided. Provide --run-dir with holdout.npz or pass --csv."
            )

        df = pl.read_csv(csv_path, try_parse_dates=True)
        feature_order = _load_feature_order(Path(args.features) if args.features else None)

        if args.target_column and args.target_column in df.columns:
            y_true = df.select(args.target_column).to_numpy().reshape(-1)
            df = df.drop(args.target_column)

        if feature_order:
            missing = [f for f in feature_order if f not in df.columns]
            if missing:
                raise ValueError(
                    f"CSV is missing required features from features JSON: {missing[:10]}"
                )
            x = df.select(
                [pl.col(c).cast(pl.Float64, strict=False).fill_null(float("nan")).alias(c) for c in feature_order]
            ).to_numpy()
            used_features = feature_order
        else:
            numeric_cols = [
                c for c, dt in zip(df.columns, df.dtypes, strict=False) if dt.is_numeric()
            ]
            if not numeric_cols:
                raise ValueError("No numeric columns found for inference input")
            x = df.select(numeric_cols).to_numpy()
            used_features = numeric_cols

        preds = _predict_in_batches(model, x, args.batch_size)

        print("Inference successful on CSV")
        print(f"Rows: {len(preds)}")
        print(f"Features used count: {len(used_features)}")
        print(
            "Prediction stats: "
            f"min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}, std={preds.std():.4f}"
        )

        if y_true is not None and len(y_true) == len(preds):
            metrics = _compute_metrics(y_true, preds)
            metrics_path = output_dir / "csv_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            _plot_quality(y_true, preds, output_dir)
            print(
                "CSV metrics -> "
                + ", ".join([f"{k.upper()}={v:.6f}" for k, v in metrics.items()])
            )
            print(f"Saved: {metrics_path}")
            print(f"Saved: {output_dir / 'quality_overview.png'}")

    print(f"First {min(args.head, len(preds))} predictions: {preds[: args.head].tolist()}")

    out_csv = Path(args.output_csv) if args.output_csv else output_dir / "predictions.csv"
    pl.DataFrame({"prediction": preds}).write_csv(out_csv)
    print(f"Saved predictions: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
