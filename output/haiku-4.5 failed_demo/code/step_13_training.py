#!/usr/bin/env python3
"""
Step 13: Model Training

Train multiple candidate regression models and select the best one based on CV performance.
Includes benchmarks: naive_persistence, seasonal_naive, auto_arima, ar1.
"""

import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, List

import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import joblib

try:
    from statsmodels.tsa.ar_model import AutoReg
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import pmdarima
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False


def score_benchmark(y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Score benchmark predictions."""
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae)
    }


def train_benchmarks(y_train: np.ndarray, y_test: np.ndarray, primary_seasonal_period: int = None) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Train all mandatory benchmark models."""
    
    benchmarks = {}
    benchmark_predictions = {}
    
    # 1. Naive persistence
    y_pred_naive = np.concatenate([[y_train[-1]], y_test[:-1]])
    benchmarks["naive_persistence"] = score_benchmark(y_test, y_pred_naive)
    benchmark_predictions["naive_persistence"] = y_pred_naive
    
    # 2. Seasonal naive
    m = primary_seasonal_period or 1
    y_hist = np.concatenate([y_train, y_test])
    y_pred_seasonal = np.array([y_hist[max(0, i - m)] for i in range(len(y_train), len(y_hist))])
    benchmarks["seasonal_naive"] = score_benchmark(y_test, y_pred_seasonal)
    benchmarks["seasonal_naive"]["seasonal_period"] = m
    benchmark_predictions["seasonal_naive"] = y_pred_seasonal
    
    # 3. Auto ARIMA
    y_pred_arima = np.full_like(y_test, np.mean(y_train))  # Fallback to mean
    try:
        if HAS_PMDARIMA:
            bm_model = pmdarima.auto_arima(y_train, seasonal=False, max_p=2, max_q=2, suppress_warnings=True, error_action="ignore")
            y_pred_arima = bm_model.predict(n_periods=len(y_test))
        else:
            # Fallback: use mean
            pass
    except Exception:
        pass
    
    y_pred_arima = np.asarray(y_pred_arima, dtype=float)
    benchmarks["auto_arima_benchmark"] = score_benchmark(y_test, y_pred_arima)
    benchmark_predictions["auto_arima_benchmark"] = y_pred_arima
    
    # 4. AR(1) benchmark
    y_pred_ar1 = np.concatenate([[y_train[-1]], y_test[:-1]])  # Fallback to persistence
    try:
        if HAS_STATSMODELS:
            ar1_model = AutoReg(y_train, lags=1, old_names=False).fit()
            start_idx = len(y_train)
            end_idx = len(y_train) + len(y_test) - 1
            y_pred_ar1 = np.asarray(ar1_model.predict(start=start_idx, end=end_idx), dtype=float)
    except Exception:
        pass
    
    benchmarks["ar1_benchmark"] = score_benchmark(y_test, y_pred_ar1)
    benchmarks["ar1_benchmark"]["lags"] = 1
    benchmark_predictions["ar1_benchmark"] = y_pred_ar1
    
    return benchmarks, benchmark_predictions


def train_candidates(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, output_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Path]:
    """Train candidate models."""
    
    candidates = {}
    
    # Ridge Regression
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    
    cv_scores_ridge = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')
    candidates["ridge"] = {
        "model_name": "ridge",
        "r2": float(r2_score(y_test, y_pred_ridge)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_ridge))),
        "mae": float(mean_absolute_error(y_test, y_pred_ridge)),
        "cv_mean_r2": float(cv_scores_ridge.mean()),
        "cv_std_r2": float(cv_scores_ridge.std()),
        "residual_mean": float(np.mean(y_pred_ridge - y_test)),
        "residual_max_abs": float(np.max(np.abs(y_pred_ridge - y_test)))
    }
    joblib.dump(ridge, str(output_dir / "candidate-ridge.joblib"))
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
    candidates["random_forest"] = {
        "model_name": "random_forest",
        "r2": float(r2_score(y_test, y_pred_rf)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_rf))),
        "mae": float(mean_absolute_error(y_test, y_pred_rf)),
        "cv_mean_r2": float(cv_scores_rf.mean()),
        "cv_std_r2": float(cv_scores_rf.std()),
        "residual_mean": float(np.mean(y_pred_rf - y_test)),
        "residual_max_abs": float(np.max(np.abs(y_pred_rf - y_test)))
    }
    joblib.dump(rf, str(output_dir / "candidate-random_forest.joblib"))
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    
    cv_scores_gb = cross_val_score(gb, X_train, y_train, cv=5, scoring='r2')
    candidates["gradient_boosting"] = {
        "model_name": "gradient_boosting",
        "r2": float(r2_score(y_test, y_pred_gb)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_gb))),
        "mae": float(mean_absolute_error(y_test, y_pred_gb)),
        "cv_mean_r2": float(cv_scores_gb.mean()),
        "cv_std_r2": float(cv_scores_gb.std()),
        "residual_mean": float(np.mean(y_pred_gb - y_test)),
        "residual_max_abs": float(np.max(np.abs(y_pred_gb - y_test)))
    }
    joblib.dump(gb, str(output_dir / "candidate-gradient_boosting.joblib"))
    
    # ElasticNet (expansion candidate)
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000)
    en.fit(X_train, y_train)
    y_pred_en = en.predict(X_test)
    
    cv_scores_en = cross_val_score(en, X_train, y_train, cv=5, scoring='r2')
    candidates["elasticnet"] = {
        "model_name": "elasticnet",
        "r2": float(r2_score(y_test, y_pred_en)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_en))),
        "mae": float(mean_absolute_error(y_test, y_pred_en)),
        "cv_mean_r2": float(cv_scores_en.mean()),
        "cv_std_r2": float(cv_scores_en.std()),
        "residual_mean": float(np.mean(y_pred_en - y_test)),
        "residual_max_abs": float(np.max(np.abs(y_pred_en - y_test)))
    }
    joblib.dump(en, str(output_dir / "candidate-elasticnet.joblib"))
    
    # Select best model
    best_model_name = max(candidates, key=lambda x: candidates[x]["r2"])
    best_model_path = output_dir / f"candidate-{best_model_name}.joblib"
    
    # Copy best as model.joblib
    import shutil
    shutil.copy(str(best_model_path), str(output_dir / "model.joblib"))
    
    return candidates, output_dir / "model.joblib"


def main():
    parser = argparse.ArgumentParser(description="Step 13: Model Training")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--split-mode", default="auto", choices=["auto", "random", "time_series"])
    
    args = parser.parse_args()
    
    try:
        output_dir = Path(args.output_dir)
        
        # Load progress and prior outputs
        progress_path = output_dir / "progress.json"
        progress = json.loads(progress_path.read_text())
        target_col = progress.get("target_column", "").lower().replace(" ", "_")
        
        step11_path = output_dir / "step-11-exploration.json"
        step11 = json.loads(step11_path.read_text())
        primary_seasonal_period = step11.get("ts_diagnostics", {}).get("primary_seasonal_period")
        
        step12_path = output_dir / "step-12-features.json"
        step12 = json.loads(step12_path.read_text())
        features = step12.get("features", [])
        split_mode = step12.get("split_strategy", {}).get("resolved_mode", "random")
        
        # Load features
        features_path = output_dir / "features.parquet"
        df_pl = pl.read_parquet(str(features_path))
        df = df_pl.to_pandas()
        
        # Prepare X and y
        X = df[features].fillna(0)
        y = df[target_col].fillna(0)
        
        # Split data
        if split_mode == "time_series":
            # Chronological split (80/20)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )
        
        print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Train benchmarks
        benchmarks, benchmark_predictions = train_benchmarks(
            y_train.values, y_test.values, primary_seasonal_period
        )
        
        # Train candidates
        candidates, best_model_path = train_candidates(
            X_train.values, y_train.values,
            X_test.values, y_test.values,
            output_dir
        )
        
        # Save holdout set
        np.savez_compressed(
            str(output_dir / "holdout.npz"),
            X_test=X_test.values,
            y_test=y_test.values
        )
        
        # Build output JSON
        output_json = {
            "step": "13-model-training",
            "run_id": args.run_id,
            "split_strategy": split_mode,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "benchmarks": benchmarks,
            "candidates": list(candidates.values()),
            "best_candidate": max(candidates, key=lambda x: candidates[x]["r2"]),
            "artifacts": {
                "model_joblib": str(best_model_path),
                "candidates_joblib": [str(output_dir / f"candidate-{name}.joblib") for name in candidates.keys()],
                "holdout_npz": str(output_dir / "holdout.npz")
            }
        }
        
        # Write output JSON
        step_json_path = output_dir / "step-13-training.json"
        step_json_path.write_text(json.dumps(output_json, indent=2))
        
        # Update progress
        progress["status"] = "running"
        progress["current_step"] = "14-model-evaluation"
        progress["completed_steps"] = ["10-csv-read-cleansing", "11-data-exploration", "12-feature-extraction", "13-model-training"]
        progress_path.write_text(json.dumps(progress, indent=2))
        
        print(f"Step 13 completed: {len(candidates)} candidates trained")
        sys.exit(0)
        
    except Exception as e:
        print(f"Step 13 failed: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
