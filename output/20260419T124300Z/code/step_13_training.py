#!/usr/bin/env python3
"""
Step 13: Model Training
Train candidate regression models and persist artifacts.
"""

import json
import os
import sys
import argparse
import pickle
import numpy as np
import polars as pl
from pathlib import Path
from time import perf_counter
from typing import Any

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score
import joblib

def load_step12_output(output_dir: str) -> dict:
    """Load step 12 feature extraction output."""
    step12_path = os.path.join(output_dir, "step-12-features.json")
    with open(step12_path, 'r') as f:
        return json.load(f)

def train_models(output_dir: str, run_id: str, target_column: str, split_mode: str = "auto") -> dict:
    """Train candidate models."""
    
    results = {
        "step": "13-model-training",
        "run_id": run_id,
        "candidates": [],
        "best_model": None,
        "artifacts": {}
    }
    
    try:
        # Load features
        features_path = os.path.join(output_dir, "features.parquet")
        print(f"[Step 13] Loading features from {features_path}...")
        df = pl.read_parquet(features_path)
        print(f"  ✓ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Load step 12 output
        step12_output = load_step12_output(output_dir)
        features = step12_output.get("features", [])
        split_strategy = step12_output.get("split_strategy", {})
        time_column = split_strategy.get("time_column")
        resolved_split = split_strategy.get("resolved_mode", "random")
        
        print(f"  ✓ Features: {len(features)}")
        print(f"  ✓ Split strategy: {resolved_split}")
        
        # Prepare X and y
        X = df.select(features).to_numpy().astype(np.float64)
        y = df[target_column].to_numpy().astype(np.float64)
        
        print(f"  ✓ X shape: {X.shape}, y shape: {y.shape}")
        
        # Determine split indices
        if resolved_split == "time_series":
            # Use TimeSeriesSplit
            print("[Step 13] Using time-series split...")
            tscv = TimeSeriesSplit(n_splits=5)
            split_indices = list(tscv.split(X))
            # Get the last split for train/test
            train_idx, test_idx = split_indices[-1]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        else:
            # Use random split
            print("[Step 13] Using random split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )
        
        print(f"  ✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        # Define candidate models
        candidates_config = [
            {
                "name": "ridge",
                "estimator": Ridge(alpha=1.0, random_state=42),
                "cv": 5,
            },
            {
                "name": "random_forest",
                "estimator": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
                "cv": 5,
            },
            {
                "name": "gradient_boosting",
                "estimator": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
                "cv": 5,
            },
        ]
        
        # Try to add XGBoost if available
        try:
            import xgboost as xgb
            candidates_config.append({
                "name": "xgboost",
                "estimator": xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
                "cv": 5,
            })
            print("  ✓ XGBoost available - will train")
        except ImportError:
            print("  - XGBoost not available - skipping")
        
        # Train candidates
        best_r2 = -np.inf
        best_model_name = None
        
        for config in candidates_config:
            name = config["name"]
            estimator = config["estimator"]
            n_cv = config["cv"]
            
            print(f"\n[Step 13] Training {name}...")
            
            # Create preprocessing pipeline
            scaler = StandardScaler()
            pipe = Pipeline([
                ("scaler", scaler),
                ("model", estimator),
            ])
            
            start = perf_counter()
            
            # Fit on training data
            pipe.fit(X_train, y_train)
            
            # Cross-validation scores (on training data)
            if resolved_split == "time_series":
                tscv = TimeSeriesSplit(n_splits=n_cv)
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=tscv, scoring="r2")
            else:
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=n_cv, scoring="r2")
            
            # Test scores
            y_pred = pipe.predict(X_test)
            test_r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
            test_rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            test_mae = np.mean(np.abs(y_test - y_pred))
            
            fit_time = perf_counter() - start
            
            candidate_result = {
                "model_name": name,
                "params": str(estimator),
                "cv_mean_r2": float(np.mean(cv_scores)),
                "cv_std_r2": float(np.std(cv_scores)),
                "test_r2": float(test_r2),
                "test_rmse": float(test_rmse),
                "test_mae": float(test_mae),
                "fit_time_sec": float(fit_time),
            }
            
            results["candidates"].append(candidate_result)
            
            print(f"  ✓ CV R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            print(f"  ✓ Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
            print(f"  ✓ Fit time: {fit_time:.2f}s")
            
            # Track best model
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_model_name = name
                
                # Save best model
                model_path = os.path.join(output_dir, "model.joblib")
                joblib.dump(pipe, model_path)
                print(f"  ✓ Saved best model to {model_path}")
            
            # Save candidate model
            candidate_path = os.path.join(output_dir, f"candidate-{name}.joblib")
            joblib.dump(pipe, candidate_path)
        
        results["best_model"] = best_model_name
        results["best_r2"] = float(best_r2)
        results["artifacts"]["model"] = os.path.join(output_dir, "model.joblib")
        
        # Save holdout set
        holdout_path = os.path.join(output_dir, "holdout.npz")
        np.savez(holdout_path, X_test=X_test, y_test=y_test)
        results["artifacts"]["holdout"] = holdout_path
        print(f"\n✓ Saved holdout set to {holdout_path}")
        
        return results, X_test, y_test
        
    except Exception as e:
        print(f"✗ Step 13 failed: {e}", file=sys.stderr)
        raise

def main():
    parser = argparse.ArgumentParser(description="Step 13: Model Training")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--target-column", required=True, help="Target column")
    parser.add_argument("--split-mode", default="auto", help="Split mode: auto|random|time_series")
    
    args = parser.parse_args()
    
    try:
        results, X_test, y_test = train_models(args.output_dir, args.run_id, args.target_column, args.split_mode)
        
        # Write results
        report_path = os.path.join(args.output_dir, "step-13-training.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Wrote training report to {report_path}")
        
        # Update progress
        progress_path = os.path.join(args.output_dir, "progress.json")
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        
        progress["current_step"] = "13-model-training"
        progress["completed_steps"].append("13-model-training")
        
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("✓ Step 13 completed successfully")
        sys.exit(0)
        
    except Exception as e:
        print(f"✗ Step 13 failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
