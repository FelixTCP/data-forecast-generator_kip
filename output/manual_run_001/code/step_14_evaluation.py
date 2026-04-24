#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from runtime_utils import finite_float, mark_status, mark_step_error, mark_step_start, mark_step_success, read_json, update_code_audit, write_json


STEP_NAME = "14-model-evaluation"


def evaluate(output_dir: Path, run_id: str) -> None:
    mark_step_start(output_dir, STEP_NAME)
    try:
        step12 = read_json(output_dir / "step-12-features.json")
        step13 = read_json(output_dir / "step-13-training.json")
        holdout = np.load(output_dir / "holdout.npz", allow_pickle=True)
        df = __import__("polars").read_parquet(output_dir / "features.parquet")
        target = step12["target_column"]
        X_train = holdout["X_train"]
        y_train = holdout["y_train"]
        X_test = holdout["X_test"]
        y_test = holdout["y_test"]
        naive_pred = holdout["naive_baseline_pred"]
        feature_names = [str(item) for item in holdout["feature_names"]]

        def metrics_for_predictions(y_true: np.ndarray, preds: np.ndarray) -> dict:
            return {
                "r2": float(r2_score(y_true, preds)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
                "mae": float(mean_absolute_error(y_true, preds)),
                "residual_mean": float(np.mean(y_true - preds)),
                "residual_max_abs": float(np.max(np.abs(y_true - preds))),
            }

        naive_metrics = metrics_for_predictions(y_test, naive_pred)
        candidates = []
        best_r2 = -np.inf
        best_name = None
        suspicious = False
        for item in step13["candidates"]:
            model = joblib.load(item["artifact"])
            preds = model.predict(X_test)
            row = {
                "model_name": item["model_name"],
                **metrics_for_predictions(y_test, preds),
                "cv_mean_r2": item["cv_mean_r2"],
                "cv_std_r2": item["cv_std_r2"],
                "naive_baseline_r2": naive_metrics["r2"],
                "naive_baseline_rmse": naive_metrics["rmse"],
                "naive_baseline_mae": naive_metrics["mae"],
                "model_worse_than_mean_baseline": float(r2_score(y_test, preds)) < 0,
            }
            if np.all(y_test != 0):
                row["mape"] = float(np.mean(np.abs((y_test - preds) / y_test)))
            candidates.append(row)
            if row["r2"] > best_r2:
                best_r2 = row["r2"]
                best_name = item["model_name"]
            if row["r2"] > 0.98:
                suspicious = True

        quality = "acceptable" if best_r2 >= 0.50 else "marginal" if best_r2 >= 0.25 else "subpar"
        leakage_probe = {"triggered": suspicious, "status": "pass", "details": []}
        if suspicious:
            target_derived = [idx for idx, name in enumerate(feature_names) if name.startswith(f"{target}_")]
            rolling_derived = [idx for idx, name in enumerate(feature_names) if f"{target}_rolling_" in name]
            if target_derived:
                keep = [idx for idx in range(len(feature_names)) if idx not in target_derived]
                probe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
                probe.fit(X_train[:, keep], y_train)
                score = float(r2_score(y_test, probe.predict(X_test[:, keep])))
                leakage_probe["details"].append({"probe": "drop_target_derived", "r2": score})
            if rolling_derived:
                keep = [idx for idx in range(len(feature_names)) if idx not in rolling_derived]
                probe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
                probe.fit(X_train[:, keep], y_train)
                score = float(r2_score(y_test, probe.predict(X_test[:, keep])))
                leakage_probe["details"].append({"probe": "drop_target_rolling", "r2": score})
            if target_derived:
                probe = LinearRegression()
                probe.fit(X_train[:, target_derived], y_train)
                score = float(r2_score(y_test, probe.predict(X_test[:, target_derived])))
                leakage_probe["details"].append({"probe": "target_only_reconstruction", "r2": score})
                if score > 0.98:
                    leakage_probe["status"] = "fail"
                    quality = "leakage_suspected"

        expansion_diagnosis = None
        expansion_candidates = []
        if quality == "subpar":
            mark_status(output_dir, "expansion_required")
            target_stats = {
                "mean": float(np.mean(y_test)),
                "std": float(np.std(y_test)),
                "min": float(np.min(y_test)),
                "max": float(np.max(y_test)),
            }
            if max(item["cv_mean_r2"] for item in step13["candidates"]) < 0.10:
                expansion_diagnosis = "Training CV R2 is near zero; the current feature set is likely weak."
            elif best_r2 < max(item["cv_mean_r2"] for item in step13["candidates"]) - 0.20:
                expansion_diagnosis = "Holdout performance is materially below CV, suggesting overfitting or split mismatch."
            elif abs(np.mean(y_train) - np.median(y_train)) > np.std(y_train):
                expansion_diagnosis = "The target distribution appears skewed; a target transform may help."
            else:
                expansion_diagnosis = "Baseline models underfit the signal; expansion candidates were trained."

            for name, estimator in [
                ("elastic_net", Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(random_state=42))])),
                ("hist_gradient_boosting", HistGradientBoostingRegressor(random_state=42)),
                ("svr", Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))])),
            ]:
                estimator.fit(X_train, y_train)
                preds = estimator.predict(X_test)
                expansion_candidates.append(
                    {
                        "model_name": name,
                        **metrics_for_predictions(y_test, preds),
                        "cv_mean_r2": None,
                        "cv_std_r2": None,
                        "model_worse_than_mean_baseline": float(r2_score(y_test, preds)) < 0,
                    }
                )
            best_expansion = max(item["r2"] for item in expansion_candidates)
            if best_expansion >= 0.50:
                quality = "acceptable"
            elif best_expansion >= 0.25:
                quality = "marginal"
            else:
                quality = "subpar_after_expansion"
        target_stats = {
            "mean": float(np.mean(df[target].to_numpy())),
            "std": float(np.std(df[target].to_numpy())),
            "min": float(np.min(df[target].to_numpy())),
            "max": float(np.max(df[target].to_numpy())),
        }
        payload = {
            "step": STEP_NAME,
            "run_id": run_id,
            "target_stats": target_stats,
            "candidates": candidates,
            "quality_assessment": quality,
            "expansion_diagnosis": expansion_diagnosis,
            "expansion_candidates": expansion_candidates,
            "leakage_probe": leakage_probe,
            "best_candidate": best_name,
        }
        if not all(finite_float(item["r2"]) and finite_float(item["rmse"]) and finite_float(item["mae"]) for item in candidates):
            raise RuntimeError("Non-finite evaluation metric detected.")
        write_json(output_dir / "step-14-evaluation.json", payload)
        mark_step_success(output_dir, STEP_NAME)
        update_code_audit(output_dir, Path(__file__).resolve().parent)
    except Exception as exc:
        mark_step_error(output_dir, STEP_NAME, str(exc))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    evaluate(Path(args.output_dir), args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
