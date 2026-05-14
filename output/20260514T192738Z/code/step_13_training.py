#!/usr/bin/env python3
"""
Step 13: Model Training

Train candidate models including mandatory benchmarks against chronological splits.
Update progress.json with per-model status throughout training.
"""
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
import traceback
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
import joblib
from time import time

# Scikit-learn imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Optional imports
try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS_ARIMA = True
except ImportError:
    HAS_STATSMODELS_ARIMA = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def compute_metrics(y_true, y_pred):
    """Compute R2, RMSE, MAE."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


def train_and_evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    cv_splits,
    model_name,
):
    """Train model and compute CV and test metrics."""
    try:
        start_time = time()
        
        # Fit on full train set
        model.fit(X_train, y_train)
        
        fit_time = time() - start_time
        
        # CV scores
        cv_r2_scores = cross_val_score(model, X_train, y_train, cv=cv_splits, scoring='r2')
        cv_r2_mean = float(cv_r2_scores.mean())
        cv_r2_std = float(cv_r2_scores.std())
        
        # Test set prediction
        y_pred_test = model.predict(X_test)
        test_r2, test_rmse, test_mae = compute_metrics(y_test, y_pred_test)
        
        return {
            "status": "success",
            "cv_r2_mean": cv_r2_mean,
            "cv_r2_std": cv_r2_std,
            "holdout_r2": test_r2,
            "holdout_rmse": test_rmse,
            "holdout_mae": test_mae,
            "fit_time_sec": fit_time,
            "model": model,
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
        }


def train_arima_benchmark(y_train, y_test):
    """Train ARIMA benchmark."""
    try:
        if HAS_PMDARIMA:
            # Use auto_arima
            model = pm.auto_arima(
                y_train,
                seasonal=False,
                stepwise=True,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                max_p=5, max_q=5, max_d=2
            )
        elif HAS_STATSMODELS_ARIMA:
            # Fallback to manual ARIMA(1,1,1)
            model = ARIMA(y_train, order=(1, 1, 1))
            model = model.fit()
        else:
            return None
        
        # Predict on test
        if HAS_PMDARIMA:
            y_pred_test, _ = model.predict(n_periods=len(y_test), return_conf_int=True)
            y_pred_test = y_pred_test[-len(y_test):]
        else:
            # Statsmodels ARIMA
            y_pred_test = model.get_forecast(steps=len(y_test)).predicted_mean.values
        
        test_r2, test_rmse, test_mae = compute_metrics(y_test, y_pred_test)
        
        return {
            "status": "success",
            "holdout_r2": test_r2,
            "holdout_rmse": test_rmse,
            "holdout_mae": test_mae,
            "fit_time_sec": 0.0,  # Not precisely tracked
            "model": model,
        }
    except Exception as e:
        print(f"Warning: ARIMA benchmark failed: {e}")
        return None


def train_kmeans_benchmark(X_train, y_train, X_test, y_test):
    """Train KMeans benchmark (cluster-centroid baseline)."""
    try:
        from sklearn.cluster import KMeans
        
        n_clusters = max(3, min(10, len(X_train) // 100))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        # Fit on training data
        train_clusters = kmeans.fit_predict(X_train)
        
        # Compute mean target value per cluster
        cluster_means = np.zeros(n_clusters)
        for c in range(n_clusters):
            mask = train_clusters == c
            if mask.sum() > 0:
                cluster_means[c] = y_train[mask].mean()
        
        # Predict test: map to nearest cluster and use that cluster's mean
        test_clusters = kmeans.predict(X_test)
        y_pred_test = cluster_means[test_clusters]
        
        test_r2, test_rmse, test_mae = compute_metrics(y_test, y_pred_test)
        
        return {
            "status": "success",
            "holdout_r2": test_r2,
            "holdout_rmse": test_rmse,
            "holdout_mae": test_mae,
            "fit_time_sec": 0.0,
            "model": kmeans,
        }
    except Exception as e:
        print(f"Warning: KMeans benchmark failed: {e}")
        return None


def naive_persistence_baseline(y_train, y_test):
    """Compute naive persistence baseline (ŷ_t = y_{t-1})."""
    try:
        y_pred_test = y_test[:-1]
        y_true_trim = y_test[1:]
        
        if len(y_pred_test) > 0:
            test_r2, test_rmse, test_mae = compute_metrics(y_true_trim, y_pred_test)
        else:
            test_r2, test_rmse, test_mae = 0.0, 0.0, 0.0
        
        return {
            "holdout_r2": test_r2,
            "holdout_rmse": test_rmse,
            "holdout_mae": test_mae,
        }
    except:
        return None


def train_models(
    output_dir: str,
    run_id: str,
) -> dict:
    """
    Load features and train all candidate models.
    
    Returns:
        dict: Training output JSON
    """
    output_dir_path = Path(output_dir)
    progress_path = output_dir_path / "progress.json"
    
    # Update progress
    def update_progress(current_model=None, completed_models=None, model_history=None):
        with open(progress_path) as f:
            prog = json.load(f)
        prog["current_step"] = "13-model-training"
        if current_model:
            prog["current_model"] = current_model
        if completed_models:
            prog["completed_models"] = completed_models
        if model_history:
            prog["model_history"] = model_history
        with open(progress_path, 'w') as f:
            json.dump(prog, f, indent=2)
    
    # Load data
    features_parquet = output_dir_path / "features.parquet"
    df_pl = pl.read_parquet(features_parquet)
    df_pd = df_pl.to_pandas()
    
    with open(output_dir_path / "step-12-features.json") as f:
        step_12_data = json.load(f)
    
    with open(output_dir_path / "step-11-exploration.json") as f:
        step_11_data = json.load(f)
    
    target = step_12_data["target"]
    features = step_12_data["features"]
    
    # Check for leakage
    for feat in features:
        if feat == target:
            raise RuntimeError(f"Target column {target} in feature list - leakage!")
    
    # Split data: 80% train/val, 20% holdout (chronological)
    n = len(df_pd)
    holdout_size = int(0.2 * n)
    train_val_size = n - holdout_size
    
    X_train_val = df_pd.iloc[:train_val_size][features].values
    y_train_val = df_pd.iloc[:train_val_size][target].values
    
    X_holdout = df_pd.iloc[train_val_size:][features].values
    y_holdout = df_pd.iloc[train_val_size:][target].values
    
    print(f"Data split: {train_val_size} train/val, {holdout_size} holdout")
    
    # CV splits (TimeSeriesSplit only)
    cv_splits = TimeSeriesSplit(n_splits=5)
    
    # Initialize results
    benchmarks = {}
    candidates = []
    model_history = []
    skipped_models = []
    
    # === TRAIN ARIMA BENCHMARK ===
    update_progress(current_model="arima_benchmark", model_history=model_history)
    print("Training ARIMA benchmark...")
    arima_result = train_arima_benchmark(y_train_val, y_holdout)
    if arima_result:
        benchmarks["arima_benchmark"] = {k: v for k, v in arima_result.items() if k != "model"}
        model_history.append({
            "model_name": "arima_benchmark",
            "status": "success",
            "holdout_r2": arima_result.get("holdout_r2", 0.0),
            "fit_time_sec": arima_result.get("fit_time_sec", 0.0),
        })
        update_progress(completed_models=["arima_benchmark"], model_history=model_history)
        print(f"  ARIMA: R2={arima_result.get('holdout_r2', 0.0):.3f}")
    else:
        model_history.append({
            "model_name": "arima_benchmark",
            "status": "failed",
            "error": "ARIMA not available"
        })
        print("  ARIMA: FAILED")
    
    # === TRAIN KMEANS BENCHMARK ===
    update_progress(current_model="kmeans_benchmark", model_history=model_history)
    print("Training KMeans benchmark...")
    kmeans_result = train_kmeans_benchmark(X_train_val, y_train_val, X_holdout, y_holdout)
    if kmeans_result:
        benchmarks["kmeans_benchmark"] = {k: v for k, v in kmeans_result.items() if k != "model"}
        model_history.append({
            "model_name": "kmeans_benchmark",
            "status": "success",
            "holdout_r2": kmeans_result.get("holdout_r2", 0.0),
            "fit_time_sec": 0.0,
        })
        update_progress(completed_models=["arima_benchmark", "kmeans_benchmark"], model_history=model_history)
        print(f"  KMeans: R2={kmeans_result.get('holdout_r2', 0.0):.3f}")
    else:
        model_history.append({
            "model_name": "kmeans_benchmark",
            "status": "failed",
            "error": "KMeans failed"
        })
        print("  KMeans: FAILED")
    
    # === NAIVE PERSISTENCE BASELINE ===
    print("Computing naive persistence baseline...")
    naive_result = naive_persistence_baseline(y_train_val, y_holdout)
    if naive_result:
        benchmarks["naive_persistence"] = naive_result
        print(f"  Naive: R2={naive_result.get('holdout_r2', 0.0):.3f}")
    
    # === SELECT AND TRAIN CANDIDATES ===
    candidate_models = [
        ("ridge", Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=42))
        ])),
        ("elastic_net", Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.1, random_state=42))
        ])),
        ("random_forest", RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )),
        ("gradient_boosting", GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )),
        ("svr", Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel='rbf', C=1.0))
        ])),
    ]
    
    # Add XGBoost if available
    if HAS_XGBOOST:
        candidate_models.append((
            "xgboost",
            xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        ))
    else:
        skipped_models.append({"name": "xgboost", "reason": "xgboost not installed"})
    
    # Add LightGBM if available
    if HAS_LIGHTGBM:
        candidate_models.append((
            "lightgbm",
            lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        ))
    else:
        skipped_models.append({"name": "lightgbm", "reason": "lightgbm not installed"})
    
    completed_models_list = ["arima_benchmark", "kmeans_benchmark"]
    
    for model_name, model in candidate_models:
        update_progress(current_model=model_name, model_history=model_history)
        print(f"Training {model_name}...")
        
        result = train_and_evaluate_model(
            model,
            X_train_val,
            y_train_val,
            X_holdout,
            y_holdout,
            cv_splits,
            model_name,
        )
        
        if result["status"] == "success":
            # Compute delta vs benchmarks
            arima_r2 = benchmarks.get("arima_benchmark", {}).get("holdout_r2", 0.0)
            kmeans_r2 = benchmarks.get("kmeans_benchmark", {}).get("holdout_r2", 0.0)
            
            candidate_entry = {
                "model_name": model_name,
                "best_params": {},
                "cv_r2_mean": result["cv_r2_mean"],
                "cv_r2_std": result["cv_r2_std"],
                "holdout_r2": result["holdout_r2"],
                "holdout_rmse": result["holdout_rmse"],
                "holdout_mae": result["holdout_mae"],
                "delta_r2_vs_arima": result["holdout_r2"] - arima_r2,
                "delta_r2_vs_kmeans": result["holdout_r2"] - kmeans_r2,
                "fit_time_sec": result["fit_time_sec"],
            }
            candidates.append(candidate_entry)
            
            # Save candidate model
            candidate_path = output_dir_path / f"candidate-{model_name}.joblib"
            joblib.dump(result["model"], candidate_path)
            
            model_history.append({
                "model_name": model_name,
                "status": "success",
                "holdout_r2": result["holdout_r2"],
                "fit_time_sec": result["fit_time_sec"],
            })
            
            completed_models_list.append(model_name)
            print(f"  {model_name}: R2={result['holdout_r2']:.3f}, RMSE={result['holdout_rmse']:.3f}")
            
        else:
            model_history.append({
                "model_name": model_name,
                "status": "failed",
                "error": result.get("error", "Unknown error"),
            })
            print(f"  {model_name}: FAILED - {result.get('error', 'Unknown')}")
        
        update_progress(completed_models=completed_models_list, model_history=model_history)
    
    # === SELECT BEST MODEL ===
    if candidates:
        best_candidate = max(candidates, key=lambda x: x["holdout_r2"])
        best_model_name = best_candidate["model_name"]
        best_model_path = output_dir_path / f"candidate-{best_model_name}.joblib"
        best_model = joblib.load(best_model_path)
    else:
        best_model_name = None
        best_model = None
    
    # === CHECK BENCHMARK WARNING ===
    benchmark_warning = False
    if best_candidate and benchmarks.get("arima_benchmark"):
        arima_r2 = benchmarks["arima_benchmark"].get("holdout_r2", 0.0)
        if best_candidate["holdout_r2"] - arima_r2 < 0.02:
            benchmark_warning = True
    
    if best_candidate and benchmarks.get("kmeans_benchmark"):
        kmeans_r2 = benchmarks["kmeans_benchmark"].get("holdout_r2", 0.0)
        if best_candidate["holdout_r2"] - kmeans_r2 < 0.02:
            benchmark_warning = True
    
    # === SAVE BEST MODEL AND HOLDOUT ===
    if best_model:
        model_path = output_dir_path / "model.joblib"
        joblib.dump(best_model, model_path)
        print(f"✓ Best model saved: {best_model_name}")
    else:
        print("WARNING: No successful candidates")
    
    # Save holdout data
    holdout_path = output_dir_path / "holdout.npz"
    np.savez(holdout_path, X_test=X_holdout, y_test=y_holdout)
    
    # === BUILD OUTPUT JSON ===
    output_json = {
        "step": "13-model-training",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        
        "split_mode": "time_series",
        "random_state": 42,
        "n_splits": 5,
        
        "split_info": {
            "train_val_size": train_val_size,
            "holdout_size": holdout_size,
            "total_rows": n,
        },
        
        "benchmarks": benchmarks,
        "candidates": candidates,
        "best_model_name": best_model_name,
        "benchmark_warning": benchmark_warning,
        "skipped_models": skipped_models,
        
        "artifacts": {
            "model_joblib": str(output_dir_path / "model.joblib"),
            "holdout_npz": str(output_dir_path / "holdout.npz"),
        },
        
        "context": {
            "dataset_id": run_id,
            "target_column": target,
            "time_column": step_12_data["split_strategy"].get("time_column"),
            "features": features,
            "split_strategy": step_12_data.get("split_strategy", {}),
            "model_candidates": candidates,
            "metrics": {
                "best_r2": best_candidate["holdout_r2"] if best_candidate else 0.0,
            },
            "artifacts": {
                "model": str(output_dir_path / "model.joblib"),
                "holdout": str(output_dir_path / "holdout.npz"),
            },
            "notes": [
                f"Trained {len(candidates)} candidate models",
                f"Best: {best_model_name} (R2={best_candidate['holdout_r2']:.3f})" if best_candidate else "No viable candidates",
                f"Benchmark warning: {benchmark_warning}",
            ]
        }
    }
    
    # Write JSON
    step_json_path = output_dir_path / "step-13-training.json"
    with open(step_json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    # Update progress final
    update_progress(
        current_model=best_model_name,
        completed_models=completed_models_list,
        model_history=model_history
    )
    
    progress_final = {
        "status": "running",
        "completed_steps": ["10-csv-read-cleansing", "11-data-exploration", "12-feature-extraction", "13-model-training"],
    }
    with open(progress_path) as f:
        prog = json.load(f)
    prog.update(progress_final)
    with open(progress_path, 'w') as f:
        json.dump(prog, f, indent=2)
    
    print(f"\n✓ Step 13 complete: {len(candidates)} candidates, best={best_model_name}")
    print(f"  Report written to: {step_json_path}")
    
    return output_json


def main():
    parser = argparse.ArgumentParser(description="Step 13: Model Training")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    try:
        train_models(
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
        sys.exit(0)
    except Exception as e:
        print(f"✗ Step 13 failed: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
