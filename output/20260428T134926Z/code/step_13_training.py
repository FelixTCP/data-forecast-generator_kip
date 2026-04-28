#!/usr/bin/env python3
"""
Step 13: Model Training
Trains multiple candidate models using chronological time-series split.
Performs model selection and evaluation across candidates.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
import joblib
import numpy as np
import pandas as pd
import polars as pl
from time import time

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

def load_step_output(output_dir: str, step: str) -> dict:
    """Load output JSON from a previous step."""
    path = Path(output_dir) / f"step-{step}.json"
    with open(path, 'r') as f:
        return json.load(f)

def create_time_series_split(X: np.ndarray, n_splits: int = 3) -> list:
    """Create chronological train/val/test splits."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_idx, test_idx in tscv.split(X):
        splits.append((train_idx, test_idx))
    return splits

def train_sklearn_model(
    model_class,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    use_pipeline: bool = True,
    **kwargs
) -> tuple[object, dict]:
    """Train a scikit-learn compatible model with optional pipeline."""
    try:
        start_time = time()
        
        # Create pipeline with scaler
        if use_pipeline:
            scaler = StandardScaler()
            model = Pipeline([
                ('scaler', scaler),
                ('model', model_class(**kwargs, random_state=42))
            ])
        else:
            model = model_class(**kwargs, random_state=42)
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = model.predict(X_val)
        
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        elapsed = time() - start_time
        
        result = {
            "model": model,
            "metrics": {
                "r2": float(r2),
                "rmse": float(rmse),
                "mae": float(mae)
            },
            "fit_time": elapsed
        }
        
        print(f"[Step 13]   ✓ {model_name}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        return model, result
        
    except Exception as e:
        print(f"[Step 13]   ✗ {model_name} failed: {e}", file=sys.stderr)
        return None, {
            "error": str(e),
            "status": "failed"
        }

def train_candidate_models(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    model_recommendations: list,
    time_series_detected: bool = False,
    progress_path: str = None
) -> tuple[dict, object]:
    """
    Train multiple candidate models and track progress.
    
    Returns:
        (training_result_dict, best_model)
    """
    
    # Create chronological splits
    n_splits = min(5, len(X) // 100)  # Adaptive number of folds
    splits = create_time_series_split(X, n_splits=n_splits)
    
    print(f"[Step 13] Created {len(splits)} chronological train/val splits")
    
    # Use last split for final holdout
    train_idx, test_idx = splits[-1]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"[Step 13] Final split: {len(train_idx)} train, {len(test_idx)} test")
    
    candidates = {}
    best_r2 = -np.inf
    best_model = None
    best_name = None
    
    model_history = []
    
    # Define candidate models (with pipelines)
    candidate_configs = [
        ("ridge", Ridge, {"alpha": 1.0}),
        ("ridge_strong", Ridge, {"alpha": 10.0}),
        ("elasticnet", ElasticNet, {"alpha": 0.1, "l1_ratio": 0.5}),
        ("elasticnet_l1", ElasticNet, {"alpha": 0.1, "l1_ratio": 0.9}),
        ("random_forest", RandomForestRegressor, {"n_estimators": 100, "max_depth": 15}),
        ("hist_gradient_boosting", HistGradientBoostingRegressor, {"max_depth": 7}),
        ("gradient_boosting", GradientBoostingRegressor, {"n_estimators": 100, "max_depth": 5}),
    ]
    
    # Add XGBoost if available
    if HAS_XGBOOST:
        candidate_configs.append(("xgboost", xgb.XGBRegressor, {"n_estimators": 100, "max_depth": 5}))
    
    print(f"[Step 13] Training {len(candidate_configs)} candidate models...")
    
    for idx, (model_name, model_class, kwargs) in enumerate(candidate_configs):
        print(f"[Step 13] [{idx+1}/{len(candidate_configs)}] Training {model_name}...")
        
        # Update progress
        if progress_path:
            try:
                with open(progress_path, 'r') as f:
                    progress = json.load(f)
                progress["current_model"] = model_name
                progress["model_progress"] = {"current": idx + 1, "total": len(candidate_configs)}
                with open(progress_path, 'w') as f:
                    json.dump(progress, f, indent=2)
            except:
                pass
        
        # Train model with pipeline
        use_pipeline = model_name not in ["random_forest", "gradient_boosting", "xgboost"]
        model, result = train_sklearn_model(
            model_class,
            model_name,
            X_train,
            y_train,
            X_test,
            y_test,
            use_pipeline=use_pipeline,
            **kwargs
        )
        
        if model is not None:
            # Save candidate
            candidate_path = str(Path(output_dir) / f"candidate-{model_name}.joblib")
            try:
                joblib.dump(model, candidate_path)
                
                candidates[model_name] = {
                    "path": candidate_path,
                    "r2": result["metrics"]["r2"],
                    "rmse": result["metrics"]["rmse"],
                    "mae": result["metrics"]["mae"],
                    "fit_time": result["fit_time"],
                    "status": "success"
                }
                
                # Track best
                if result["metrics"]["r2"] > best_r2:
                    best_r2 = result["metrics"]["r2"]
                    best_model = model
                    best_name = model_name
                
                # Save to model history
                model_history.append({
                    "model": model_name,
                    "r2": result["metrics"]["r2"],
                    "rmse": result["metrics"]["rmse"],
                    "mae": result["metrics"]["mae"],
                    "fit_time": result["fit_time"],
                    "status": "completed"
                })
                
            except Exception as e:
                print(f"[Step 13]   ✗ Failed to save {model_name}: {e}", file=sys.stderr)
                model_history.append({
                    "model": model_name,
                    "status": "failed_to_save",
                    "error": str(e)
                })
        else:
            model_history.append({
                "model": model_name,
                "status": "training_failed",
                "error": result.get("error", "Unknown error")
            })
    
    if best_model is None:
        raise RuntimeError("All candidate models failed to train")
    
    print(f"[Step 13] ✓ Best model: {best_name} with R²={best_r2:.4f}")
    
    # Save best model
    best_model_path = str(Path(output_dir) / "model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"[Step 13] Saved best model to: {best_model_path}")
    
    # Save holdout data
    holdout_path = str(Path(output_dir) / "holdout.npz")
    np.savez(holdout_path, X_test=X_test, y_test=y_test)
    print(f"[Step 13] Saved holdout data to: {holdout_path}")
    
    return {
        "candidates": candidates,
        "best_model": best_name,
        "best_r2": best_r2,
        "model_history": model_history,
        "holdout_size": len(test_idx)
    }, best_model

def main():
    parser = argparse.ArgumentParser(
        description="Step 13: Model Training"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for artifacts"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier"
    )
    
    args = parser.parse_args()
    output_dir = args.output_dir
    run_id = args.run_id
    progress_path = str(Path(output_dir) / "progress.json")
    
    try:
        print("[Step 13] Starting model training...")
        
        # Load step outputs
        step10 = load_step_output(output_dir, "10-cleanse")
        step11 = load_step_output(output_dir, "11-exploration")
        step12 = load_step_output(output_dir, "12-features")
        
        target_column = step10["target_column_normalized"]
        features_list = step12["features"]
        features_parquet_path = step12["artifacts"]["features_parquet"]
        
        model_recommendations = step11.get("model_recommendations", [])
        time_series_detected = step11.get("time_series_detected", False)
        
        # Load features
        print("[Step 13] Loading feature matrix...")
        df = pl.read_parquet(features_parquet_path)
        df_pd = df.to_pandas()
        
        print(f"[Step 13] Loaded {len(df_pd)} rows × {len(features_list)} features")
        
        X = df_pd[features_list].values.astype(np.float64)
        y = df_pd[target_column].values.astype(np.float64)
        
        # Train candidate models
        training_result, best_model = train_candidate_models(
            X=X,
            y=y,
            output_dir=output_dir,
            model_recommendations=model_recommendations,
            time_series_detected=time_series_detected,
            progress_path=progress_path
        )
        
        # Build step output
        step_output = {
            "step": "13-model-training",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "model": {
                "type": "sklearn-compatible",
                "best_candidate": training_result["best_model"],
                "path": str(Path(output_dir) / "model.joblib")
            },
            "model_candidates": training_result["candidates"],
            "best_r2": training_result["best_r2"],
            "model_history": training_result["model_history"],
            "split_strategy": {
                "type": "time_series",
                "n_splits": 5,
                "holdout_size": training_result["holdout_size"]
            },
            "artifacts": {
                "model": str(Path(output_dir) / "model.joblib"),
                "holdout": str(Path(output_dir) / "holdout.npz")
            },
            "notes": [
                f"Trained {len(training_result['candidates'])} candidate models",
                f"Best model: {training_result['best_model']} (R²={training_result['best_r2']:.4f})",
                f"Used chronological time-series split for evaluation"
            ]
        }
        
        # Write step output
        output_json_path = Path(output_dir) / "step-13-training.json"
        with open(output_json_path, 'w') as f:
            json.dump(step_output, f, indent=2)
        print(f"[Step 13] Output written to: {output_json_path}")
        
        # Update progress
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        progress["current_step"] = "13-model-training"
        progress["current_model"] = training_result["best_model"]
        progress["completed_models"] = [c for c in training_result["candidates"].keys()]
        progress["completed_steps"].append("12-feature-extraction")
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("[Step 13] ✓ Completed successfully")
        sys.exit(0)
        
    except Exception as e:
        print(f"[Step 13] ✗ Failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
