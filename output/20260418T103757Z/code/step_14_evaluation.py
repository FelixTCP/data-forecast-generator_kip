from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 14 model evaluation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def evaluate_one(model: Any, x_test: np.ndarray, y_test: np.ndarray, base: dict[str, Any]) -> dict[str, Any]:
    preds = model.predict(x_test)
    residuals = y_test - preds

    result = {
        "model_name": base["model_name"],
        "complexity": int(base.get("complexity", 99)),
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "cv_mean_r2": float(base.get("cv_mean_r2", 0.0)),
        "cv_std_r2": float(base.get("cv_std_r2", 0.0)),
        "residual_mean": float(np.mean(residuals)),
        "residual_max_abs": float(np.max(np.abs(residuals))),
        "artifact_path": base.get("artifact_path"),
    }
    result["model_worse_than_mean_baseline"] = bool(result["r2"] < 0)

    if np.all(np.abs(y_test) > 1e-12):
        result["mape"] = float(np.mean(np.abs((y_test - preds) / y_test)) * 100.0)

    return result


def quality_from_r2(best_r2: float) -> str:
    if best_r2 >= 0.50:
        return "acceptable"
    if best_r2 >= 0.25:
        return "marginal"
    return "subpar"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    step12 = read_json(output_dir / "step-12-features.json")
    step13 = read_json(output_dir / "step-13-training.json")
    context = step13["context"]
    target = context["target_column"]

    progress_path = output_dir / "progress.json"
    progress = read_json(progress_path)
    completed_steps = [str(s) for s in progress.get("completed_steps", [])]
    errors = [str(e) for e in progress.get("errors", [])]

    update_progress(
        progress_path,
        args.run_id,
        progress["csv_path"],
        target,
        "14-model-evaluation",
        completed_steps,
        errors,
        "running",
    )

    try:
        holdout = np.load(output_dir / "holdout.npz")
        x_test = holdout["X_test"]
        y_test = holdout["y_test"]

        df_features = pl.read_parquet(step12["artifacts"]["features_parquet"])
        target_stats = {
            "mean": float(df_features.select(pl.col(target).mean()).item() or 0.0),
            "std": float(df_features.select(pl.col(target).std()).item() or 0.0),
            "min": float(df_features.select(pl.col(target).min()).item() or 0.0),
            "max": float(df_features.select(pl.col(target).max()).item() or 0.0),
        }

        candidate_reports: list[dict[str, Any]] = []
        for candidate in tqdm(step13["candidates"], desc="step14: evaluate", unit="model"):
            model = joblib.load(candidate["artifact_path"])
            row = evaluate_one(model, x_test, y_test, candidate)
            for key in ("r2", "rmse", "mae"):
                if not math.isfinite(float(row[key])):
                    raise ValueError(f"Non-finite metric {key} for {candidate['model_name']}")
            candidate_reports.append(row)

        best_r2 = max(float(r["r2"]) for r in candidate_reports)
        quality_assessment = quality_from_r2(best_r2)

        expansion_diagnosis = ""
        expansion_candidates: list[dict[str, Any]] = []

        if quality_assessment == "subpar":
            cv_low = all(float(c.get("cv_mean_r2", 0.0)) < 0.10 for c in candidate_reports)
            holdout_gap = any((float(c.get("cv_mean_r2", 0.0)) - float(c["r2"])) > 0.20 for c in candidate_reports)
            rmse_big = any(float(c["rmse"]) > target_stats["std"] for c in candidate_reports)
            skewed = (abs(target_stats["mean"]) > 1e-9) and (target_stats["max"] > 3.0 * max(1e-9, abs(target_stats["mean"])))

            diagnosis_parts = []
            if rmse_big:
                diagnosis_parts.append("RMSE exceeds target std for at least one candidate; models may not be learning target scale.")
            if cv_low:
                diagnosis_parts.append("Training CV R2 is near zero, suggesting weak feature signal.")
            if holdout_gap:
                diagnosis_parts.append("CV-to-holdout gap indicates potential overfitting or split mismatch.")
            if skewed:
                diagnosis_parts.append("Target appears heavy-tailed; consider log-transform in next iteration.")
            if not diagnosis_parts:
                diagnosis_parts.append("General underfit detected; expansion models attempted.")
            expansion_diagnosis = " ".join(diagnosis_parts)

            split_mode = context["split_strategy"]["resolved_mode"]
            features = list(step12["features"])
            x_all = df_features.select([pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0).alias(c) for c in features]).to_numpy()
            y_all = df_features.select(pl.col(target).cast(pl.Float64, strict=False).fill_null(0.0)).to_numpy().reshape(-1)

            rows = np.arange(len(y_all))
            if split_mode == "time_series":
                split_idx = int(len(rows) * 0.8)
                train_idx = rows[:split_idx]
                test_idx = rows[split_idx:]
            else:
                train_idx, test_idx = train_test_split(rows, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)

            x_train = x_all[train_idx]
            y_train = y_all[train_idx]
            x_eval = x_all[test_idx]
            y_eval = y_all[test_idx]

            expansion_models = [
                (
                    "elastic_net",
                    2,
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)),
                        ]
                    ),
                ),
                (
                    "hist_gradient_boosting",
                    3,
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
                        ]
                    ),
                ),
                (
                    "svr_rbf",
                    6,
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("model", SVR(kernel="rbf", C=5.0, epsilon=0.1)),
                        ]
                    ),
                ),
            ]

            for name, complexity, model in tqdm(expansion_models, desc="step14: expansion", unit="model"):
                model.fit(x_train, y_train)
                row = evaluate_one(
                    model,
                    x_eval,
                    y_eval,
                    {
                        "model_name": name,
                        "complexity": complexity,
                        "cv_mean_r2": 0.0,
                        "cv_std_r2": 0.0,
                        "artifact_path": "",
                    },
                )
                expansion_candidates.append(row)

            if expansion_candidates:
                best_expanded_r2 = max(float(r["r2"]) for r in expansion_candidates + candidate_reports)
                if best_expanded_r2 >= 0.50:
                    quality_assessment = "acceptable"
                elif best_expanded_r2 >= 0.25:
                    quality_assessment = "marginal"
                else:
                    quality_assessment = "subpar_after_expansion"

        step_output = {
            "step": "14-model-evaluation",
            "target_stats": target_stats,
            "candidate_reports": candidate_reports,
            "candidates": candidate_reports,
            "quality_assessment": quality_assessment,
            "expansion_diagnosis": expansion_diagnosis,
            "expansion_candidates": expansion_candidates,
            "context": context,
        }

        write_json(output_dir / "step-14-evaluation.json", step_output)

        if "14-model-evaluation" not in completed_steps:
            completed_steps.append("14-model-evaluation")

        update_progress(
            progress_path,
            args.run_id,
            progress["csv_path"],
            target,
            "14-model-evaluation",
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
            "14-model-evaluation",
            completed_steps,
            errors,
            "error",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())