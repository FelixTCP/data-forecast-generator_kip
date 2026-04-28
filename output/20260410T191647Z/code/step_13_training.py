from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

RANDOM_STATE = 42


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def update_progress(
    progress_path: Path,
    run_id: str,
    csv_path: str,
    target_column: str,
    current_step: str,
    completed_steps: list[str],
    errors: list[str],
    status: str,
) -> None:
    write_json(
        progress_path,
        {
            "run_id": run_id,
            "csv_path": csv_path,
            "target_column": target_column,
            "status": status,
            "current_step": current_step,
            "completed_steps": completed_steps,
            "errors": errors,
        },
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 13 model training")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def make_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {
            "name": "ridge",
            "complexity": 1,
            "estimator": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        },
        {
            "name": "random_forest",
            "complexity": 4,
            "estimator": RandomForestRegressor(
                n_estimators=120,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "name": "gradient_boosting",
            "complexity": 3,
            "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
        },
    ]

    try:
        from xgboost import XGBRegressor

        candidates.append(
            {
                "name": "xgboost",
                "complexity": 5,
                "estimator": XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    objective="reg:squarederror",
                    n_jobs=1,
                ),
            }
        )
    except Exception:
        pass

    return candidates


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    step12 = read_json(output_dir / "step-12-features.json")
    context = step12["context"]
    target = context["target_column"]
    feature_columns = step12["features"]
    split_mode = context["split_strategy"]["resolved_mode"]

    progress_path = output_dir / "progress.json"
    progress = read_json(progress_path)
    completed_steps = [str(s) for s in progress.get("completed_steps", [])]
    errors = [str(e) for e in progress.get("errors", [])]

    update_progress(
        progress_path,
        args.run_id,
        progress["csv_path"],
        target,
        "13-model-training",
        completed_steps,
        errors,
        "running",
    )

    try:
        df_features = pl.read_parquet(step12["artifacts"]["features_parquet"])
        if not feature_columns:
            raise ValueError("No feature columns were produced in step 12")

        for col in feature_columns + [target]:
            if col not in df_features.columns:
                raise ValueError(f"Missing expected column in features parquet: {col}")

        df_features = (
            df_features.with_columns(pl.col(target).cast(pl.Float64, strict=False).alias(target))
            .filter(pl.col(target).is_not_null())
        )

        if df_features.height < 30:
            raise ValueError("Not enough non-null target rows for robust training")

        rows = np.arange(df_features.height)
        if split_mode == "time_series":
            split_idx = int(df_features.height * 0.8)
            train_idx = rows[:split_idx]
            test_idx = rows[split_idx:]
        else:
            train_idx, test_idx = train_test_split(
                rows,
                test_size=0.2,
                random_state=RANDOM_STATE,
                shuffle=True,
            )

        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError("Train/test split produced empty partition")

        x_all = df_features.select(
            [pl.col(c).cast(pl.Float64, strict=False).fill_null(float("nan")).alias(c) for c in feature_columns]
        ).to_numpy()
        y_all = df_features.select(pl.col(target).cast(pl.Float64, strict=False)).to_numpy().reshape(-1)

        x_train = x_all[train_idx]
        y_train = y_all[train_idx]
        x_test = x_all[test_idx]
        y_test = y_all[test_idx]

        if x_test.shape[0] == 0:
            raise ValueError("Holdout set is empty")

        holdout_path = output_dir / "holdout.npz"
        np.savez(holdout_path, X_test=x_test, y_test=y_test)

        if split_mode == "time_series":
            n_splits = max(2, min(5, len(y_train) // 100))
            cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        candidate_records: list[dict[str, Any]] = []
        model_candidates: list[dict[str, Any]] = []

        for candidate in tqdm(make_candidates(), desc="step13: train candidates", unit="model"):
            pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", candidate["estimator"]),
                ]
            )

            pipe.fit(x_train, y_train)
            preds = pipe.predict(x_test)
            cv_scores = cross_val_score(pipe, x_train, y_train, cv=cv, scoring="r2", n_jobs=1)

            model_path = output_dir / f"candidate-{candidate['name']}.joblib"
            joblib.dump(pipe, model_path)

            model_candidates.append({"name": candidate["name"], "complexity": candidate["complexity"]})
            candidate_records.append(
                {
                    "model_name": candidate["name"],
                    "complexity": candidate["complexity"],
                    "r2": float(r2_score(y_test, preds)),
                    "rmse": rmse(y_test, preds),
                    "mae": float(mean_absolute_error(y_test, preds)),
                    "cv_scores_r2": [float(v) for v in cv_scores],
                    "cv_mean_r2": float(np.mean(cv_scores)),
                    "cv_std_r2": float(np.std(cv_scores)),
                    "artifact_path": str(model_path),
                }
            )

        best_model = max(candidate_records, key=lambda item: item["cv_mean_r2"])
        best_model_artifact = Path(best_model["artifact_path"])
        final_model = joblib.load(best_model_artifact)
        model_path = output_dir / "model.joblib"
        joblib.dump(final_model, model_path)

        context.setdefault("artifacts", {})["model_joblib"] = str(model_path)
        context["artifacts"]["holdout_npz"] = str(holdout_path)
        context["model_candidates"] = model_candidates
        context["metrics"] = {
            "best_cv_mean_r2": float(best_model["cv_mean_r2"]),
            "best_holdout_r2": float(best_model["r2"]),
        }

        step_output = {
            "step": "13-model-training",
            "split_mode": split_mode,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "feature_count": len(feature_columns),
            "candidates": candidate_records,
            "best_model_for_artifact": best_model["model_name"],
            "context": context,
            "artifacts": {
                "model_joblib": str(model_path),
                "holdout_npz": str(holdout_path),
            },
        }
        write_json(output_dir / "step-13-training.json", step_output)

        if "13-model-training" not in completed_steps:
            completed_steps.append("13-model-training")

        update_progress(
            progress_path,
            args.run_id,
            progress["csv_path"],
            target,
            "13-model-training",
            completed_steps,
            errors,
            "running",
        )
        return 0
    except Exception as exc:
        errors.append(str(exc))
        update_progress(
            progress_path,
            args.run_id,
            progress["csv_path"],
            target,
            "13-model-training",
            completed_steps,
            errors,
            "error",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
