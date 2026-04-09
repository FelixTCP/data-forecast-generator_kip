"""Minimal agentic pipeline runtime entrypoint for next-step execution.

This script implements the canonical contracts from docs/agentic-pipeline/contracts.md
It performs steps 10-13 minimally so the next agentic step can run deterministically.

Usage (env vars or CLI):
  CSV_PATH, TARGET_COLUMN, OUTPUT_DIR, RUN_ID, SPLIT_MODE

Outputs (written under OUTPUT_DIR):
  - progress.json
  - step-10-cleanse.json
  - step-11-exploration.json
  - step-12-features.json
  - step-13-training.json

This file is intentionally minimal and deterministic (random_state used).
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from time import perf_counter

import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Config from environment (fallback to defaults)
CSV_PATH = os.environ.get("CSV_PATH", "./data/appliances_energy_prediction.csv")
TARGET_COLUMN = os.environ.get("TARGET_COLUMN", "appliances")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "artifacts/run-local"))
RUN_ID = os.environ.get("RUN_ID", "run-local")
SPLIT_MODE = os.environ.get("SPLIT_MODE", "auto")
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict):
    with path.open("w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# Step 10: load and clean csv
def load_and_clean_csv(csv_path: str) -> tuple[pl.DataFrame, dict]:
    df = pl.read_csv(csv_path, try_parse_dates=True)
    quality = {
        "row_count_before": df.height,
        "column_count": df.width,
        "null_rate": {c: float(df.select(pl.col(c).is_null().mean()).item()) for c in df.columns},
        "fixes": [],
    }
    # normalize column names to snake_case
    normalized = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if normalized != list(df.columns):
        df = df.rename(dict(zip(df.columns, normalized)))
        quality["fixes"].append("normalized_column_names")
    return df, quality


# Step 11: basic exploration
def explore_data(df: pl.DataFrame, target_hint: str | None = None) -> dict:
    profile = {
        "shape": [df.height, df.width],
        "columns": list(df.columns),
        "target_hint": target_hint,
        "numeric_columns": [c for c, dt in zip(df.columns, df.dtypes) if dt in (pl.Int64, pl.Float64, pl.Int32, pl.Float32, pl.Float64)],
    }
    return profile


# Step 12: build minimal features (X, y) and feature_meta
def build_features(df: pl.DataFrame, target_column: str) -> tuple[list[list[float]], list[float], dict]:
    if target_column not in df.columns:
        raise ValueError(f"target_column={target_column} not in dataframe")
    # select numeric columns except target
    numeric_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt in (pl.Int64, pl.Float64, pl.Int32, pl.Float32, pl.Float64)]
    features = [c for c in numeric_cols if c != target_column]
    if not features:
        raise RuntimeError("no numeric features available to build X")
    X = df.select(features).to_pandas()
    y = df.select(target_column).to_pandas().iloc[:, 0]
    feature_meta = {"features": features, "n_features": len(features)}
    return X, y, feature_meta


# Step 13: train minimal candidate(s)
def train_models(X, y, config: dict) -> dict:
    random_state = int(config.get("random_state", RANDOM_STATE))
    split_mode = config.get("split_mode", "random")
    n_splits = int(config.get("n_splits", 5))

    results = {"run_id": RUN_ID, "candidates": [], "best_model": None}

    # single candidate: Ridge with default alpha grid
    candidate_name = "ridge-default"
    model = Ridge()
    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", model),
    ])

    start = perf_counter()
    if split_mode == "time_series":
        # use simple train_test_split without shuffle to emulate time-series holdout
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        pipe.fit(X_train, y_train)
        score = float(pipe.score(X_test, y_test))
        cv_mean = score
        cv_std = 0.0
    else:
        # random split and simple KFold CV
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        kf = KFold(n_splits=min(n_splits, 5), shuffle=True, random_state=random_state)
        scores = []
        for tr_idx, va_idx in kf.split(X_train):
            Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            ytr, yva = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            pipe.fit(Xtr, ytr)
            scores.append(float(pipe.score(Xva, yva)))
        cv_mean = float(np.mean(scores))
        cv_std = float(np.std(scores))
        pipe.fit(X_train, y_train)
        score = float(pipe.score(X_test, y_test))
    fit_time = perf_counter() - start

    candidate_report = {
        "model_name": candidate_name,
        "params": {"model": "Ridge(default)"},
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "test_score": score,
        "fit_time_sec": fit_time,
    }
    results["candidates"].append(candidate_report)
    results["best_model"] = candidate_report
    return results


def main():
    progress = {
        "run_id": RUN_ID,
        "csv_path": CSV_PATH,
        "target_column": TARGET_COLUMN,
        "status": "running",
        "current_step": "10-csv-read-cleansing",
        "completed_steps": [],
        "errors": [],
    }
    write_json(OUTPUT_DIR / "progress.json", progress)

    # Step 10
    try:
        df, quality = load_and_clean_csv(CSV_PATH)
        step10 = {"quality_report": quality, "columns": list(df.columns)}
        write_json(OUTPUT_DIR / "step-10-cleanse.json", step10)
        progress["completed_steps"].append("10-csv-read-cleansing")
        progress["current_step"] = "11-data-exploration"
        write_json(OUTPUT_DIR / "progress.json", progress)
    except Exception as e:
        progress["status"] = "error"
        progress["errors"].append(str(e))
        write_json(OUTPUT_DIR / "progress.json", progress)
        raise

    # Step 11
    try:
        exploration = explore_data(df, TARGET_COLUMN)
        write_json(OUTPUT_DIR / "step-11-exploration.json", exploration)
        progress["completed_steps"].append("11-data-exploration")
        progress["current_step"] = "12-feature-extraction"
        write_json(OUTPUT_DIR / "progress.json", progress)
    except Exception as e:
        progress["status"] = "error"
        progress["errors"].append(str(e))
        write_json(OUTPUT_DIR / "progress.json", progress)
        raise

    # Step 12
    try:
        X, y, feature_meta = build_features(df, TARGET_COLUMN)
        step12 = {"feature_meta": feature_meta}
        write_json(OUTPUT_DIR / "step-12-features.json", step12)
        progress["completed_steps"].append("12-feature-extraction")
        progress["current_step"] = "13-model-training"
        write_json(OUTPUT_DIR / "progress.json", progress)
    except Exception as e:
        progress["status"] = "error"
        progress["errors"].append(str(e))
        write_json(OUTPUT_DIR / "progress.json", progress)
        raise

    # Step 13
    try:
        training = train_models(X, y, {"random_state": RANDOM_STATE, "split_mode": ("time_series" if SPLIT_MODE == "time_series" else "random"), "n_splits": 5})
        write_json(OUTPUT_DIR / "step-13-training.json", training)
        progress["completed_steps"].append("13-model-training")
        progress["current_step"] = "done"
        progress["status"] = "succeeded"
        write_json(OUTPUT_DIR / "progress.json", progress)
    except Exception as e:
        progress["status"] = "error"
        progress["errors"].append(str(e))
        write_json(OUTPUT_DIR / "progress.json", progress)
        raise


if __name__ == "__main__":
    main()
