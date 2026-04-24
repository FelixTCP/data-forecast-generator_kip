#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from runtime_utils import mark_step_error, mark_step_start, mark_step_success, read_json, update_code_audit, write_json


STEP_NAME = "13-model-training"


def build_candidates() -> list[tuple[str, object]]:
    candidates: list[tuple[str, object]] = [
        ("ridge", Pipeline([("scaler", StandardScaler()), ("model", Ridge())])),
        ("random_forest", RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42, n_jobs=-1)),
        ("gradient_boosting", GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=42)),
    ]
    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBRegressor

        candidates.append(
            (
                "xgboost",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                ),
            )
        )
    return candidates


def make_json_safe_params(params: dict) -> dict:
    safe = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = repr(value)
    return safe


def training(output_dir: Path, run_id: str, split_mode: str) -> None:
    mark_step_start(output_dir, STEP_NAME)
    try:
        step12 = read_json(output_dir / "step-12-features.json")
        df = pl.read_parquet(output_dir / "features.parquet")
        target = step12["target_column"]
        feature_columns = step12["features"]
        leakage_names = {target, f"{target}_lag_0", f"{target}_rolling_mean_0", f"{target}_rolling_std_0"}
        if any(name in leakage_names for name in feature_columns):
            raise RuntimeError("Forbidden leakage feature name detected in training input.")

        X = df.select(feature_columns).to_pandas()
        y = df[target].to_pandas()
        resolved_mode = step12["split_strategy"]["resolved_mode"]
        if split_mode in {"random", "time_series"}:
            resolved_mode = split_mode

        if resolved_mode == "time_series":
            split_at = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
            y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
            cv = TimeSeriesSplit(n_splits=3)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )
            cv = KFold(n_splits=3, shuffle=True, random_state=42)

        lag_1_name = f"{target}_lag_1"
        naive_baseline = X_test[lag_1_name].to_numpy() if lag_1_name in X_test.columns else np.full(len(y_test), float(np.mean(y_train)))
        np.savez(
            output_dir / "holdout.npz",
            X_train=X_train.to_numpy(),
            y_train=y_train.to_numpy(),
            X_test=X_test.to_numpy(),
            y_test=y_test.to_numpy(),
            naive_baseline_pred=naive_baseline,
            feature_names=np.array(feature_columns, dtype=object),
        )

        candidate_results = []
        best_model_name = None
        best_score = -np.inf
        for name, estimator in build_candidates():
            start = time.perf_counter()
            fold_scores = []
            for train_index, val_index in cv.split(X_train):
                X_fold_train = X_train.iloc[train_index]
                X_fold_val = X_train.iloc[val_index]
                y_fold_train = y_train.iloc[train_index]
                y_fold_val = y_train.iloc[val_index]
                model = clone(estimator)
                model.fit(X_fold_train, y_fold_train)
                fold_scores.append(float(r2_score(y_fold_val, model.predict(X_fold_val))))

            fitted = clone(estimator)
            fitted.fit(X_train, y_train)
            preds = fitted.predict(X_test)
            result = {
                "model_name": name,
                "params": make_json_safe_params(fitted.get_params()),
                "cv_mean_r2": float(np.mean(fold_scores)),
                "cv_std_r2": float(np.std(fold_scores)),
                "r2": float(r2_score(y_test, preds)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                "mae": float(mean_absolute_error(y_test, preds)),
                "fit_time_sec": float(time.perf_counter() - start),
                "artifact": str(output_dir / f"candidate-{name}.joblib"),
            }
            joblib.dump(fitted, output_dir / f"candidate-{name}.joblib")
            candidate_results.append(result)
            if result["cv_mean_r2"] > best_score:
                best_score = result["cv_mean_r2"]
                best_model_name = name

        best_artifact = output_dir / f"candidate-{best_model_name}.joblib"
        best_estimator = joblib.load(best_artifact)
        joblib.dump(best_estimator, output_dir / "model.joblib")
        payload = {
            "step": STEP_NAME,
            "run_id": run_id,
            "split_mode": resolved_mode,
            "target_column": target,
            "feature_names": feature_columns,
            "candidates": candidate_results,
            "best_model_name": best_model_name,
            "artifacts": {
                "model_joblib": str(output_dir / "model.joblib"),
                "holdout_npz": str(output_dir / "holdout.npz"),
            },
        }
        write_json(output_dir / "step-13-training.json", payload)
        mark_step_success(output_dir, STEP_NAME)
        update_code_audit(output_dir, Path(__file__).resolve().parent)
    except Exception as exc:
        mark_step_error(output_dir, STEP_NAME, str(exc))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-mode", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-column", required=True)
    args = parser.parse_args()
    training(Path(args.output_dir), args.run_id, args.split_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
