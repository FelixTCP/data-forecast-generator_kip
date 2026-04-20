#!/usr/bin/env python3
"""
Step 13: Model Training
Trains multiple regression models and selects the best one.
"""

import json
import argparse
from pathlib import Path
import polars as pl
import numpy as np
import sys
import joblib
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def run_step_13(output_dir: str) -> dict:
    """
    Train models: Ridge, RandomForest, GradientBoosting.
    
    Returns step output dict.
    """
    print(f"run_step_13 called with output_dir={output_dir}", flush=True)
    sys.stdout.flush()
    
    output_path = Path(output_dir)
    
    # Load prior steps
    step_12_path = output_path / "step-12-features.json"
    features_parquet = output_path / "features.parquet"
    
    print(f"Checking for {step_12_path} and {features_parquet}...", flush=True)
    sys.stdout.flush()
    
    if not step_12_path.exists() or not features_parquet.exists():
        sys.exit("Step 12 outputs not found")
    
    print("Loading step 12 JSON...", flush=True)
    sys.stdout.flush()
    
    with open(step_12_path) as f:
        step_12 = json.load(f)
    
    print("Loading features parquet...", flush=True)
    sys.stdout.flush()
    
    df = pl.read_parquet(features_parquet)
    
    print(f"Data shape: {df.shape}", flush=True)
    sys.stdout.flush()
    
    feature_names = step_12["features"]
    target_column = "appliances"  # From step 10
    
    X = df.select(feature_names).to_numpy().astype(np.float64)
    y = df[target_column].to_numpy().astype(np.float64)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}", flush=True)
    sys.stdout.flush()
    
    # Determine split strategy
    split_mode = step_12["split_strategy"]["resolved_mode"]
    random_state = 42
    
    if split_mode == "time_series":
        # Use TimeSeriesSplit
        tss = TimeSeriesSplit(n_splits=5)
        split = list(tss.split(X))
        train_idx, test_idx = split[-1]  # Use last split as holdout
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
    
    # Save holdout
    holdout_path = output_path / "holdout.npz"
    np.savez(holdout_path, X_test=X_test, y_test=y_test)
    
    # Train models
    models = {
        "ridge": Ridge(random_state=random_state),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    }
    
    candidates = []
    best_model = None
    best_r2 = -float('inf')
    
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            
            # Compute CV scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Compute test scores
            r2_test = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            rmse_test = np.sqrt(np.mean((y_test - y_pred) ** 2))
            mae_test = np.mean(np.abs(y_test - y_pred))
            
            candidate = {
                "model": model_name,
                "cv_r2_mean": float(cv_scores.mean()),
                "cv_r2_std": float(cv_scores.std()),
                "test_r2": float(r2_test),
                "test_rmse": float(rmse_test),
                "test_mae": float(mae_test)
            }
            candidates.append(candidate)
            
            # Save model
            model_path = output_path / f"candidate-{model_name}.joblib"
            joblib.dump(model, model_path)
            
            if r2_test > best_r2:
                best_r2 = r2_test
                best_model = model
                best_model_name = model_name
        except Exception as e:
            print(f"Warning: {model_name} training failed: {e}", file=sys.stderr)
    
    if best_model is None:
        sys.exit("All models failed to train")
    
    # Save best model
    model_path = output_path / "model.joblib"
    joblib.dump(best_model, model_path)
    
    step_output = {
        "step": "13-model-training",
        "best_model": best_model_name,
        "candidates": candidates,
        "holdout_shape": {"rows": X_test.shape[0], "features": X_test.shape[1]},
        "artifacts": {
            "model_joblib": str(model_path),
            "holdout_npz": str(holdout_path)
        }
    }
    
    # Save step JSON
    step_json_path = output_path / "step-13-training.json"
    with open(step_json_path, "w") as f:
        json.dump(step_output, f, indent=2)
    
    # Update progress
    progress_path = output_path / "progress.json"
    with open(progress_path, "r") as f:
        progress = json.load(f)
    progress["status"] = "running"
    progress["current_step"] = "14-model-evaluation"
    progress["completed_steps"].append("13-model-training")
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)
    
    return step_output


if __name__ == "__main__":
    print("Step 13 starting...", flush=True)
    sys.stdout.flush()
    parser = argparse.ArgumentParser(description="Step 13: Model Training")
    parser.add_argument("--output-dir", required=True, help="Output directory for artifacts")
    
    args = parser.parse_args()
    print(f"Output dir: {args.output_dir}", flush=True)
    sys.stdout.flush()
    
    try:
        print("Calling run_step_13...", flush=True)
        sys.stdout.flush()
        result = run_step_13(args.output_dir)
        print(json.dumps(result, indent=2), flush=True)
        sys.stdout.flush()
        print("Step 13 completed", flush=True)
        sys.stdout.flush()
        sys.exit(0)
    except Exception as e:
        print(f"Error in Step 13: {e}", file=sys.stderr, flush=True)
        sys.stderr.flush()
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
