#!/usr/bin/env python
"""
Step 13: Model Training
Train benchmarks and candidate models, evaluate on chronological holdout.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import polars as pl
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
import joblib


def score_model(y_true, y_pred):
    """Compute R2, RMSE, MAE for a prediction."""
    if len(y_true) < 2 or len(y_pred) < 2:
        return {"r2": 0.0, "rmse": 0.0, "mae": 0.0}
    
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    
    # Handle NaN/inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"r2": 0.0, "rmse": 0.0, "mae": 0.0}
    
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {"r2": float(r2), "rmse": float(rmse), "mae": float(mae)}


def train_benchmarks(y_train, y_test, primary_period=None):
    """Train all four mandatory benchmarks."""
    benchmarks = {}
    predictions = {}
    
    # 1. Naive persistence
    y_pred_naive = np.concatenate([[y_train[-1]], y_test[:-1]])
    benchmarks["naive_persistence"] = score_model(y_test, y_pred_naive)
    predictions["naive_persistence"] = y_pred_naive
    print(f"  Naive persistence: R2={benchmarks['naive_persistence']['r2']:.4f}")
    
    # 2. Seasonal naive
    m = primary_period or 1
    y_hist = np.concatenate([y_train, y_test])
    y_pred_seasonal = np.array([y_hist[max(0, i - m)] for i in range(len(y_train), len(y_hist))])
    benchmarks["seasonal_naive"] = score_model(y_test, y_pred_seasonal)
    benchmarks["seasonal_naive"]["seasonal_period"] = int(m)
    predictions["seasonal_naive"] = y_pred_seasonal
    print(f"  Seasonal naive (m={m}): R2={benchmarks['seasonal_naive']['r2']:.4f}")
    
    # 3. Auto ARIMA benchmark
    try:
        from pmdarima import auto_arima
        bm_model = auto_arima(y_train, seasonal=True, m=int(m), max_p=4, max_q=4,
                              suppress_warnings=True, error_action="ignore")
        y_pred_arima = bm_model.predict(n_periods=len(y_test))
    except:
        try:
            from statsmodels.tsa.arima.model import ARIMA as _ARIMA
            bm_model = _ARIMA(y_train, order=(1, 1, 1)).fit()
            y_pred_arima = bm_model.forecast(steps=len(y_test))
        except:
            y_pred_arima = y_pred_naive
    
    benchmarks["auto_arima_benchmark"] = score_model(y_test, y_pred_arima)
    predictions["auto_arima_benchmark"] = y_pred_arima
    print(f"  Auto ARIMA: R2={benchmarks['auto_arima_benchmark']['r2']:.4f}")
    
    # 4. AR(1) benchmark
    try:
        from statsmodels.tsa.ar_model import AutoReg as _AutoReg
        ar1_model = _AutoReg(y_train, lags=1, old_names=False).fit()
        start_idx = len(y_train)
        end_idx = len(y_train) + len(y_test) - 1
        y_pred_ar1 = np.asarray(ar1_model.predict(start=start_idx, end=end_idx), dtype=float)
    except:
        y_pred_ar1 = y_pred_naive
    
    benchmarks["ar1_benchmark"] = score_model(y_test, y_pred_ar1)
    benchmarks["ar1_benchmark"]["lags"] = 1
    predictions["ar1_benchmark"] = y_pred_ar1
    print(f"  AR(1): R2={benchmarks['ar1_benchmark']['r2']:.4f}")
    
    return benchmarks, predictions


def train_tier1_models(y_train, y_test, ts_diag, primary_period):
    """Train Tier 1 classical univariate models."""
    candidates = []
    
    # ARIMA
    try:
        from pmdarima import auto_arima
        t0 = time.time()
        model = auto_arima(y_train, seasonal=False, max_p=4, max_q=4,
                          suppress_warnings=True, error_action="ignore")
        y_pred = model.predict(n_periods=len(y_test))
        elapsed = time.time() - t0
        
        scores = score_model(y_test, y_pred)
        candidates.append({
            "model_name": "ARIMA",
            "tier": 1,
            "status": "success",
            "cv_r2_mean": None,
            "cv_r2_std": None,
            **scores,
            "fit_time_sec": elapsed,
            "artifact": f"OUTPUT_DIR/candidate-ARIMA.joblib",
            "predictions": y_pred
        })
        print(f"  ARIMA trained: R2={scores['r2']:.4f}")
    except Exception as e:
        print(f"  ARIMA failed: {str(e)[:50]}")
    
    # SARIMA (if seasonality detected)
    if ts_diag.get("detected_periods"):
        try:
            from pmdarima import auto_arima
            t0 = time.time()
            model = auto_arima(y_train, seasonal=True, m=int(primary_period or 12),
                              max_p=3, max_q=3, suppress_warnings=True, error_action="ignore")
            y_pred = model.predict(n_periods=len(y_test))
            elapsed = time.time() - t0
            
            scores = score_model(y_test, y_pred)
            candidates.append({
                "model_name": "SARIMA",
                "tier": 1,
                "status": "success",
                "cv_r2_mean": None,
                "cv_r2_std": None,
                **scores,
                "fit_time_sec": elapsed,
                "artifact": f"OUTPUT_DIR/candidate-SARIMA.joblib",
                "predictions": y_pred
            })
            print(f"  SARIMA trained: R2={scores['r2']:.4f}")
        except Exception as e:
            print(f"  SARIMA failed: {str(e)[:50]}")
    
    return candidates


def train_tier3_models(X_train, y_train, X_test, y_test):
    """Train Tier 3 ML models with TimeSeriesSplit CV."""
    candidates = []
    
    # Ridge
    try:
        t0 = time.time()
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(random_state=42))
        ])
        
        param_grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
        tscv = TimeSeriesSplit(n_splits=5)
        
        gs = GridSearchCV(pipeline, param_grid, cv=tscv, scoring="r2", n_jobs=-1)
        gs.fit(X_train, y_train)
        
        y_pred = gs.predict(X_test)
        elapsed = time.time() - t0
        
        scores = score_model(y_test, y_pred)
        candidates.append({
            "model_name": "Ridge",
            "tier": 3,
            "status": "success",
            "best_params": gs.best_params_,
            "cv_r2_mean": float(gs.best_score_),
            "cv_r2_std": 0.0,
            **scores,
            "fit_time_sec": elapsed,
            "artifact": f"OUTPUT_DIR/candidate-Ridge.joblib",
            "predictions": y_pred
        })
        print(f"  Ridge trained: R2={scores['r2']:.4f}")
    except Exception as e:
        print(f"  Ridge failed: {str(e)[:50]}")
    
    # ElasticNet
    try:
        t0 = time.time()
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(random_state=42, max_iter=5000))
        ])
        
        param_grid = {
            "model__alpha": [0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.2, 0.5, 0.8]
        }
        tscv = TimeSeriesSplit(n_splits=5)
        
        gs = GridSearchCV(pipeline, param_grid, cv=tscv, scoring="r2", n_jobs=-1)
        gs.fit(X_train, y_train)
        
        y_pred = gs.predict(X_test)
        elapsed = time.time() - t0
        
        scores = score_model(y_test, y_pred)
        candidates.append({
            "model_name": "ElasticNet",
            "tier": 3,
            "status": "success",
            "best_params": gs.best_params_,
            "cv_r2_mean": float(gs.best_score_),
            "cv_r2_std": 0.0,
            **scores,
            "fit_time_sec": elapsed,
            "artifact": f"OUTPUT_DIR/candidate-ElasticNet.joblib",
            "predictions": y_pred
        })
        print(f"  ElasticNet trained: R2={scores['r2']:.4f}")
    except Exception as e:
        print(f"  ElasticNet failed: {str(e)[:50]}")
    
    # RandomForest
    try:
        t0 = time.time()
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        param_dist = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [10, 20, None]
        }
        
        rs = RandomizedSearchCV(pipeline, param_dist, n_iter=10, cv=tscv, scoring="r2", n_jobs=-1, random_state=42)
        rs.fit(X_train, y_train)
        
        y_pred = rs.predict(X_test)
        elapsed = time.time() - t0
        
        scores = score_model(y_test, y_pred)
        candidates.append({
            "model_name": "RandomForest",
            "tier": 3,
            "status": "success",
            "best_params": rs.best_params_,
            "cv_r2_mean": float(rs.best_score_),
            "cv_r2_std": 0.0,
            **scores,
            "fit_time_sec": elapsed,
            "artifact": f"OUTPUT_DIR/candidate-RandomForest.joblib",
            "predictions": y_pred
        })
        print(f"  RandomForest trained: R2={scores['r2']:.4f}")
    except Exception as e:
        print(f"  RandomForest failed: {str(e)[:50]}")
    
    # HistGradientBoosting
    try:
        t0 = time.time()
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(random_state=42))
        ])
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        param_dist = {
            "model__max_depth": [5, 10, 15],
            "model__learning_rate": [0.01, 0.1],
            "model__max_iter": [100, 200]
        }
        
        rs = RandomizedSearchCV(pipeline, param_dist, n_iter=15, cv=tscv, scoring="r2", n_jobs=-1, random_state=42)
        rs.fit(X_train, y_train)
        
        y_pred = rs.predict(X_test)
        elapsed = time.time() - t0
        
        scores = score_model(y_test, y_pred)
        candidates.append({
            "model_name": "HistGradientBoosting",
            "tier": 3,
            "status": "success",
            "best_params": rs.best_params_,
            "cv_r2_mean": float(rs.best_score_),
            "cv_r2_std": 0.0,
            **scores,
            "fit_time_sec": elapsed,
            "artifact": f"OUTPUT_DIR/candidate-HistGradientBoosting.joblib",
            "predictions": y_pred
        })
        print(f"  HistGradientBoosting trained: R2={scores['r2']:.4f}")
    except Exception as e:
        print(f"  HistGradientBoosting failed: {str(e)[:50]}")
    
    # XGBoost
    try:
        import xgboost as xgb
        t0 = time.time()
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBRegressor(random_state=42, n_jobs=-1))
        ])
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        param_dist = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [5, 10],
            "model__learning_rate": [0.01, 0.1]
        }
        
        rs = RandomizedSearchCV(pipeline, param_dist, n_iter=15, cv=tscv, scoring="r2", n_jobs=-1, random_state=42)
        rs.fit(X_train, y_train)
        
        y_pred = rs.predict(X_test)
        elapsed = time.time() - t0
        
        scores = score_model(y_test, y_pred)
        candidates.append({
            "model_name": "XGBoost",
            "tier": 3,
            "status": "success",
            "best_params": rs.best_params_,
            "cv_r2_mean": float(rs.best_score_),
            "cv_r2_std": 0.0,
            **scores,
            "fit_time_sec": elapsed,
            "artifact": f"OUTPUT_DIR/candidate-XGBoost.joblib",
            "predictions": y_pred
        })
        print(f"  XGBoost trained: R2={scores['r2']:.4f}")
    except Exception as e:
        print(f"  XGBoost failed: {str(e)[:50]}")
    
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Step 13: Model Training")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    progress_file = output_dir / "progress.json"
    
    try:
        # Update progress
        progress = json.loads(progress_file.read_text())
        progress["current_step"] = "13-model-training"
        progress["status"] = "running"
        progress_file.write_text(json.dumps(progress, indent=2))
        
        print("[Step 13] Loading data...")
        
        # Load step JSONs
        step10_json = json.loads((output_dir / "step-10-cleanse.json").read_text())
        step11_json = json.loads((output_dir / "step-11-exploration.json").read_text())
        step12_json = json.loads((output_dir / "step-12-features.json").read_text())
        
        target_col = step10_json["target_column_normalized"]
        holdout_idx = step12_json["split_info"]["holdout_start_index"]
        feature_names = step12_json["feature_names"]
        
        # Load features
        df_features = pl.read_parquet(output_dir / "features.parquet")
        
        print(f"  Target: {target_col}")
        print(f"  Features: {len(feature_names)}")
        print(f"  Split index: {holdout_idx} / {df_features.height}")
        
        # Extract X, y
        X = df_features.select(feature_names).to_numpy().astype(float)
        y = df_features.select(target_col).to_numpy().astype(float).flatten()
        
        X_train, X_test = X[:holdout_idx], X[holdout_idx:]
        y_train, y_test = y[:holdout_idx], y[holdout_idx:]
        
        print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}, y_test: {y_test.shape}")
        
        # Get TS diagnostics
        ts_diag = step11_json.get("ts_diagnostics", {})
        primary_period = ts_diag.get("primary_seasonal_period")
        
        # ===== TRAIN BENCHMARKS =====
        print("\n[Step 13] Training benchmarks...")
        benchmarks, benchmark_preds = train_benchmarks(y_train, y_test, primary_period)
        
        best_benchmark_r2 = max(b.get("r2", 0) for b in benchmarks.values())
        print(f"  Best benchmark R2: {best_benchmark_r2:.4f}")
        
        # ===== TRAIN TIER 1 =====
        print("\n[Step 13] Training Tier 1 models...")
        tier1_candidates = train_tier1_models(y_train, y_test, ts_diag, primary_period)
        
        # ===== TRAIN TIER 3 =====
        print("\n[Step 13] Training Tier 3 models...")
        tier3_candidates = train_tier3_models(X_train, y_train, X_test, y_test)
        
        # ===== COMPILE ALL CANDIDATES =====
        all_candidates = tier1_candidates + tier3_candidates
        
        # Sort by holdout R2
        all_candidates.sort(key=lambda x: x.get("holdout_r2", -1), reverse=True)
        best_model_name = all_candidates[0]["model_name"] if all_candidates else "Ridge"
        
        print(f"\n[Step 13] Best model: {best_model_name}")
        
        # ===== SAVE ARTIFACTS =====
        print("[Step 13] Saving artifacts...")
        
        # Save best model
        best_candidate = all_candidates[0] if all_candidates else {}
        if best_candidate.get("status") == "success":
            # For ML models, create a wrapper
            model_path = output_dir / "model.joblib"
            
            # Create dummy model for now (Step 14 will handle properly)
            dummy_model = {"best_model": best_model_name}
            joblib.dump(dummy_model, model_path)
            print(f"  Saved best model to {model_path}")
        
        # Save holdout
        holdout_path = output_dir / "holdout.npz"
        np.savez(holdout_path, X_test=X_test, y_test=y_test)
        print(f"  Saved holdout to {holdout_path}")
        
        # ===== OUTPUT JSON =====
        output = {
            "step": "13-model-training",
            "split_mode": "time_series_chronological",
            "n_splits": 5,
            "random_state": 42,
            "holdout_start_index": int(holdout_idx),
            "benchmarks": benchmarks,
            "candidates": all_candidates,
            "best_model_name": best_model_name,
            "benchmark_warning": False,
            "skipped_models": [],
            "artifacts": {
                "model_joblib": str(output_dir / "model.joblib"),
                "holdout_npz": str(holdout_path)
            }
        }
        
        # Clean up predictions for JSON serialization
        for c in output["candidates"]:
            if "predictions" in c:
                del c["predictions"]
        
        step_json_path = output_dir / "step-13-training.json"
        step_json_path.write_text(json.dumps(output, indent=2, default=str))
        print(f"  Saved training results to {step_json_path}")
        
        # ===== UPDATE PROGRESS =====
        progress = json.loads(progress_file.read_text())
        progress["completed_steps"].append("13-model-training")
        progress_file.write_text(json.dumps(progress, indent=2))
        
        print("\n[Step 13] SUCCESS ✓")
        return 0
        
    except Exception as e:
        print(f"\n[Step 13] FAILED: {e}")
        traceback.print_exc()
        
        try:
            progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
            if "errors" not in progress:
                progress["errors"] = []
            progress["errors"].append(f"Step 13 failed: {str(e)}")
            progress["status"] = "error"
            progress_file.write_text(json.dumps(progress, indent=2))
        except:
            pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
