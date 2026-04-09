from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference using model.joblib")
    parser.add_argument("--model", required=True, help="Path to model.joblib")
    parser.add_argument("--csv", required=True, help="Path to input CSV for inference")
    parser.add_argument(
        "--target-column",
        default=None,
        help="Optional target column for quick evaluation (R2/RMSE/MAE)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save predictions CSV",
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


def main() -> int:
    args = _build_parser().parse_args()

    model_path = Path(args.model)
    csv_path = Path(args.csv)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    artifact = joblib.load(model_path)
    model = _extract_predictor(artifact)
    df = pl.read_csv(csv_path, try_parse_dates=True)

    y_true = None
    if args.target_column:
        if args.target_column not in df.columns:
            raise ValueError(
                f"target-column '{args.target_column}' not found. Available: {df.columns}"
            )
        y_true = df.select(args.target_column).to_numpy().reshape(-1)
        df = df.drop(args.target_column)

    numeric_cols = [
        c for c, dt in zip(df.columns, df.dtypes, strict=False) if dt.is_numeric()
    ]
    if not numeric_cols:
        raise ValueError("No numeric columns found for inference input")

    x = df.select(numeric_cols).to_numpy()
    preds = np.asarray(model.predict(x)).reshape(-1)

    print("✅ Inference successful")
    print(f"Rows: {len(preds)}")
    print(f"Features used: {numeric_cols}")
    print(
        "Prediction stats: "
        f"min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}, std={preds.std():.4f}"
    )
    print(f"First {min(args.head, len(preds))} predictions: {preds[: args.head].tolist()}")

    if y_true is not None and len(y_true) == len(preds):
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        mae = float(mean_absolute_error(y_true, preds))
        r2 = float(r2_score(y_true, preds))
        print(f"Quick eval -> R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_df = pl.DataFrame({"prediction": preds})
        out_df.write_csv(out_path)
        print(f"Saved predictions: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
