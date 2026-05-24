#!/usr/bin/env python3
"""
Step 13: Model Training
Train candidate regression models with cross-validation and benchmarks.
"""
import argparse
import json
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

warnings.filterwarnings("ignore")


def step_13_main(output_dir: str, run_id: str, split_mode: str = "auto") -> int:
    """Main step 13 logic."""
    try:
        output_path = Path(output_dir)
        
        # Load prior outputs
        with open(output_path / "step-12-features.json") as f:
            step12 = json.load(f)
        with open(output_path / "step-10-cleanse.json") as f:
            step10 = json.load(f)
        with open(output_path / "step-11-exploration.json") as f:
            step11 = json.load(f)
        
        target_col = step10["target_column_normalized"]
        features = step12["features"]
        resolved_split = step12["split_strategy"]["resolved_mode"]
        
        print(f"[Step 13] Loading features.parquet...")
        df = pl.read_parquet(output_path / "features.parquet").to_pandas()
        
        print(f"[Step 13] Shape: {df.shape}")
        print(f"[Step 13] Features: {len(features)}")
        
        # Prepare data
        X = df[features].values
        y = df[target_col].values
        
        print(f"[Step 13] X shape: {X.shape}, y shape: {y.shape}")
        
        # ============ TRAIN/TEST SPLIT ============
        print(f"[Step 13] Performing {resolved_split} split...")
        
        if resolved_split == "time_series":
            # Time series split: 80/20
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            print(f"[Step 13] Time series split: train {X_train.shape[0]}, test {X_test.shape[0]}")
        else:
            # Random split: 80/20
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print(f"[Step 13] Random split: train {X_train.shape[0]}, test {X_test.shape[0]}")
        
        # ============ BENCHMARK MODELS ============
        print(f"[Step 13] Computing benchmarks...")
        benchmarks = {}
        
        # Naive persistence: y_pred(t) = y(t-1)
        y_pred_naive = np.concatenate([[y_train[-1]], y_test[:-1]])
        rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive))
        mae_naive = mean_absolute_error(y_test, y_pred_naive)
        r2_naive = r2_score(y_test, y_pred_naive)
        benchmarks["naive_persistence"] = {
            "r2": r2_naive, "rmse": rmse_naive, "mae": mae_naive
        }
        
        # Seasonal naive (use lag 7 for weekly)
        y_hist = np.concatenate([y_train, y_test])
        seasonal_lag = 7
        y_pred_seasonal = np.array([y_hist[max(0, i - seasonal_lag)] for i in range(len(y_train), len(y_hist))])
        rmse_seasonal = np.sqrt(mean_squared_error(y_test, y_pred_seasonal))
        mae_seasonal = mean_absolute_error(y_test, y_pred_seasonal)
        r2_seasonal = r2_score(y_test, y_pred_seasonal)
        benchmarks["seasonal_naive"] = {
            "r2": r2_seasonal, "rmse": rmse_seasonal, "mae": mae_seasonal
        }
        
        print(f"[Step 13] Benchmarks: Naive R²={r2_naive:.4f}, Seasonal R²={r2_seasonal:.4f}")
        
        # ============ TRAIN CANDIDATES ============
        print(f"[Step 13] Training candidate models...")
        
        candidates_data = []
        
        # Ridge Regression
        print(f"[Step 13] Training Ridge...")
        try:
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(X_train, y_train)
            
            # CV scores
            cv_scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring="r2")
            
            # Test predictions
            y_pred = ridge.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            candidates_data.append({
                "model_name": "ridge",
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "cv_mean_r2": np.mean(cv_scores),
                "cv_std_r2": np.std(cv_scores),
                "residual_mean": np.mean(y_test - y_pred),
                "residual_max_abs": np.max(np.abs(y_test - y_pred)),
                "model_worse_than_mean_baseline": r2 < 0
            })
            
            # Save model
            joblib.dump(ridge, output_path / "candidate-ridge.joblib")
            print(f"[Step 13] Ridge R²={r2:.4f}")
        except Exception as e:
            print(f"[Step 13] Ridge failed: {e}")
        
        # ElasticNet
        print(f"[Step 13] Training ElasticNet...")
        try:
            elastic = ElasticNet(alpha=0.1, random_state=42)
            elastic.fit(X_train, y_train)
            
            cv_scores = cross_val_score(elastic, X_train, y_train, cv=5, scoring="r2")
            
            y_pred = elastic.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            candidates_data.append({
                "model_name": "elasticnet",
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "cv_mean_r2": np.mean(cv_scores),
                "cv_std_r2": np.std(cv_scores),
                "residual_mean": np.mean(y_test - y_pred),
                "residual_max_abs": np.max(np.abs(y_test - y_pred)),
                "model_worse_than_mean_baseline": r2 < 0
            })
            
            joblib.dump(elastic, output_path / "candidate-elasticnet.joblib")
            print(f"[Step 13] ElasticNet R²={r2:.4f}")
        except Exception as e:
            print(f"[Step 13] ElasticNet failed: {e}")
        
        # Random Forest
        print(f"[Step 13] Training Random Forest...")
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            
            cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")
            
            y_pred = rf.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            candidates_data.append({
                "model_name": "random_forest",
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "cv_mean_r2": np.mean(cv_scores),
                "cv_std_r2": np.std(cv_scores),
                "residual_mean": np.mean(y_test - y_pred),
                "residual_max_abs": np.max(np.abs(y_test - y_pred)),
                "model_worse_than_mean_baseline": r2 < 0
            })
            
            joblib.dump(rf, output_path / "candidate-random_forest.joblib")
            print(f"[Step 13] Random Forest R²={r2:.4f}")
        except Exception as e:
            print(f"[Step 13] Random Forest failed: {e}")
        
        # Gradient Boosting
        print(f"[Step 13] Training Gradient Boosting...")
        try:
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb.fit(X_train, y_train)
            
            cv_scores = cross_val_score(gb, X_train, y_train, cv=5, scoring="r2")
            
            y_pred = gb.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            candidates_data.append({
                "model_name": "gradient_boosting",
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "cv_mean_r2": np.mean(cv_scores),
                "cv_std_r2": np.std(cv_scores),
                "residual_mean": np.mean(y_test - y_pred),
                "residual_max_abs": np.max(np.abs(y_test - y_pred)),
                "model_worse_than_mean_baseline": r2 < 0
            })
            
            joblib.dump(gb, output_path / "candidate-gradient_boosting.joblib")
            print(f"[Step 13] Gradient Boosting R²={r2:.4f}")
        except Exception as e:
            print(f"[Step 13] Gradient Boosting failed: {e}")
        
        # ============ SELECT BEST MODEL ============
        if not candidates_data:
            print(f"[Step 13] ✗ No candidates trained!")
            return 1
        
        # Sort by R² descending
        candidates_data.sort(key=lambda x: x["r2"], reverse=True)
        best = candidates_data[0]
        
        print(f"[Step 13] Best model: {best['model_name']} with R²={best['r2']:.4f}")
        
        # Copy best to model.joblib
        best_model_file = output_path / f"candidate-{best['model_name']}.joblib"
        best_model = joblib.load(best_model_file)
        joblib.dump(best_model, output_path / "model.joblib")
        
        # ============ SAVE HOLDOUT ============
        print(f"[Step 13] Saving holdout set...")
        np.savez(output_path / "holdout.npz", X_test=X_test, y_test=y_test)
        
        # ============ OUTPUT JSON ============
        output_json = {
            "step": "13-model-training",
            "run_id": run_id,
            "split_mode": resolved_split,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "benchmarks": benchmarks,
            "candidates": candidates_data,
            "best_model": best["model_name"],
            "artifacts": {
                "model_joblib": str(output_path / "model.joblib"),
                "holdout_npz": str(output_path / "holdout.npz")
            }
        }
        
        # Write output JSON
        step13_json = output_path / "step-13-training.json"
        with open(step13_json, "w") as f:
            json.dump(output_json, f, indent=2)
        
        print(f"[Step 13] ✓ Completed successfully")
        return 0
        
    except Exception as e:
        print(f"[Step 13] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 13: Model Training")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--split-mode", default="auto", help="Split mode: auto|random|time_series")
    
    args = parser.parse_args()
    sys.exit(step_13_main(args.output_dir, args.run_id, args.split_mode))
