#!/usr/bin/env python3
"""Step 13: model training and artifact persistence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

STEP_NAME = "13-model-training"
STEP12_JSON = "step-12-features.json"
STEP13_JSON = "step-13-training.json"
FEATURES_PARQUET = "features.parquet"
HOLDOUT_NPZ = "holdout.npz"
MODEL_JOBLIB = "model.joblib"
PROGRESS_FILE = "progress.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _update_progress(
    output_dir: Path,
    *,
    run_id: str,
    csv_path: str,
    target_column: str,
    status: str,
    current_step: str,
    error: str | None = None,
    mark_completed: bool = False,
) -> None:
    path = output_dir / PROGRESS_FILE
    progress = _load_json(path)
    if not progress:
        progress = {
            "run_id": run_id,
            "csv_path": csv_path,
            "target_column": target_column,
            "status": "running",
            "current_step": current_step,
            "completed_steps": [],
            "errors": [],
        }

    progress["run_id"] = run_id
    progress["csv_path"] = csv_path
    progress["target_column"] = target_column
    progress["status"] = status
    progress["current_step"] = current_step

    completed = progress.setdefault("completed_steps", [])
    if mark_completed and current_step not in completed:
        completed.append(current_step)

    if error:
        progress.setdefault("errors", []).append({"step": current_step, "error": error})

    _write_json(path, progress)


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _build_candidates(random_state: int) -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = [
        (
            "ridge",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
        ),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingRegressor(random_state=random_state),
        ),
    ]

    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBRegressor  # type: ignore

        candidates.append(
            (
                "xgboost",
                XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        )

    return candidates


def _split_data(
    X: np.ndarray,
    y: np.ndarray,
    split_mode: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if split_mode == "time_series":
        split_idx = max(int(len(X) * 0.8), 1)
        split_idx = min(split_idx, len(X) - 1)
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
    )
    return X_train, X_test, y_train, y_test


def _make_cv(split_mode: str, random_state: int) -> Any:
    if split_mode == "time_series":
        return TimeSeriesSplit(n_splits=5)
    return KFold(n_splits=5, shuffle=True, random_state=random_state)


def _check_for_forbidden_leakage_features(features: list[str]) -> None:
    forbidden = [
        f
        for f in features
        if f.startswith("target_rolling_") and "shift" not in f
    ]
    if forbidden:
        raise ValueError(
            f"Forbidden leaked features present in training input: {forbidden}"
        )


def run_step(output_dir: Path, run_id: str, split_mode_arg: str) -> dict[str, Any]:
    step12 = _load_json(output_dir / STEP12_JSON)
    if not step12:
        raise FileNotFoundError("Missing step-12-features.json.")

    features_path = Path(step12["artifacts"]["features_parquet"])
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features parquet at {features_path}")

    df = pl.read_parquet(features_path)

    context = dict(step12.get("context", {}))
    target = context.get("target_column")
    if not target or target not in df.columns:
        raise ValueError("Target column missing from step 12 context or feature frame.")

    features = list(step12.get("features", []))
    if not features:
        raise ValueError("Step 12 produced an empty feature list.")

    _check_for_forbidden_leakage_features(features)

    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' is missing from features.parquet")

    requested_mode = split_mode_arg
    if requested_mode == "auto":
        requested_mode = step12.get("split_strategy", {}).get("resolved_mode", "random")
    if requested_mode not in {"random", "time_series"}:
        raise ValueError(f"Invalid split mode '{requested_mode}'")

    X = df.select(features).to_numpy()
    y = df[target].cast(pl.Float64).to_numpy()

    if X.shape[0] < 30:
        raise ValueError("Insufficient rows for robust training (<30).")

    random_state = 42
    X_train, X_test, y_train, y_test = _split_data(
        X=X,
        y=y,
        split_mode=requested_mode,
        random_state=random_state,
    )

    cv = _make_cv(split_mode=requested_mode, random_state=random_state)
    candidate_defs = _build_candidates(random_state=random_state)

    candidates: list[dict[str, Any]] = []
    successful: list[tuple[str, Any, float]] = []
    candidate_paths: dict[str, str] = {}

    for name, estimator in candidate_defs:
        start = perf_counter()
        try:
            cv_scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_test)

            r2 = float(r2_score(y_test, pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            mae = float(mean_absolute_error(y_test, pred))

            candidate_path = (output_dir / f"candidate-{name}.joblib").resolve()
            joblib.dump(estimator, candidate_path)
            candidate_paths[name] = str(candidate_path)

            fit_time_sec = perf_counter() - start
            rec = {
                "model_name": name,
                "status": "ok",
                "cv_mean_r2": float(np.mean(cv_scores)),
                "cv_std_r2": float(np.std(cv_scores)),
                "fit_time_sec": fit_time_sec,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "artifact": str(candidate_path),
            }
            candidates.append(rec)
            successful.append((name, estimator, r2))
        except Exception as exc:  # noqa: BLE001
            fit_time_sec = perf_counter() - start
            candidates.append(
                {
                    "model_name": name,
                    "status": "failed",
                    "fit_time_sec": fit_time_sec,
                    "error": str(exc),
                }
            )

    if not successful:
        raise RuntimeError("All model candidates failed during training.")

    successful.sort(key=lambda x: x[2], reverse=True)
    best_name, best_estimator, best_r2 = successful[0]

    model_path = (output_dir / MODEL_JOBLIB).resolve()
    joblib.dump(best_estimator, model_path)

    holdout_path = (output_dir / HOLDOUT_NPZ).resolve()
    np.savez(
        holdout_path,
        X_test=X_test,
        y_test=y_test,
        feature_names=np.array(features, dtype=object),
        split_mode=np.array([requested_mode], dtype=object),
    )

    if requested_mode == "time_series":
        baseline = np.empty_like(y_test)
        baseline[0] = y_train[-1] if len(y_train) > 0 else y_test[0]
        if len(y_test) > 1:
            baseline[1:] = y_test[:-1]
    else:
        baseline = np.full_like(y_test, fill_value=float(np.mean(y_train)))

    baseline_metrics = {
        "r2": float(r2_score(y_test, baseline)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, baseline))),
        "mae": float(mean_absolute_error(y_test, baseline)),
    }

    if not any(_is_finite(float(c.get("r2", float("nan")))) for c in candidates if c.get("status") == "ok"):
        raise RuntimeError("No trained candidate has finite holdout r2.")

    context.setdefault("split_strategy", {})
    context["split_strategy"] = {
        "requested_mode": split_mode_arg,
        "resolved_mode": requested_mode,
        "random_state": random_state,
    }
    context["model_candidates"] = [c for c in candidates if c.get("status") == "ok"]
    context.setdefault("metrics", {})
    context["metrics"]["best_holdout_r2"] = best_r2
    context["metrics"]["naive_baseline_r2"] = baseline_metrics["r2"]
    context.setdefault("artifacts", {})
    context["artifacts"]["model_joblib"] = str(model_path)
    context["artifacts"]["holdout_npz"] = str(holdout_path)

    payload = {
        "step": STEP_NAME,
        "split_strategy": {
            "requested_mode": split_mode_arg,
            "resolved_mode": requested_mode,
            "random_state": random_state,
        },
        "candidates": candidates,
        "best_model": {
            "model_name": best_name,
            "r2": best_r2,
            "artifact": str(model_path),
        },
        "naive_baseline": baseline_metrics,
        "artifacts": {
            "model_joblib": str(model_path),
            "holdout_npz": str(holdout_path),
            "candidate_joblibs": candidate_paths,
            "step_13_json": str((output_dir / STEP13_JSON).resolve()),
        },
        "context": context,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 13 model training")
    parser.add_argument("--split-mode", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-column", required=False)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    progress = _load_json(output_dir / PROGRESS_FILE)
    csv_path = progress.get("csv_path", "")
    target = progress.get("target_column", "")

    _update_progress(
        output_dir,
        run_id=args.run_id,
        csv_path=csv_path,
        target_column=target,
        status="running",
        current_step=STEP_NAME,
    )

    try:
        payload = run_step(output_dir=output_dir, run_id=args.run_id, split_mode_arg=args.split_mode)
        _write_json(output_dir / STEP13_JSON, payload)

        _update_progress(
            output_dir,
            run_id=args.run_id,
            csv_path=csv_path,
            target_column=target,
            status="running",
            current_step=STEP_NAME,
            mark_completed=True,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        err = f"{exc}\n{traceback.format_exc()}"
        _update_progress(
            output_dir,
            run_id=args.run_id,
            csv_path=csv_path,
            target_column=target,
            status="failed",
            current_step=STEP_NAME,
            error=err,
        )
        print(err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
