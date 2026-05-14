#!/usr/bin/env python3
"""
Step 14: Model Evaluation

Load all candidate models and holdout data, compute detailed evaluation metrics.
Diagnose quality and decide whether expansion training is needed.
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
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate_models(
    output_dir: str,
    run_id: str,
) -> dict:
    """
    Evaluate all trained models on holdout data.
    Diagnose quality and decide on expansion.
    
    Returns:
        dict: Evaluation output JSON
    """
    output_dir_path = Path(output_dir)
    
    # Load training results
    with open(output_dir_path / "step-13-training.json") as f:
        step_13_data = json.load(f)
    
    with open(output_dir_path / "step-12-features.json") as f:
        step_12_data = json.load(f)
    
    # Load holdout data
    holdout_path = output_dir_path / "holdout.npz"
    holdout = np.load(holdout_path)
    X_test = holdout["X_test"]
    y_test = holdout["y_test"]
    
    # Load best model for prediction
    model_path = output_dir_path / "model.joblib"
    best_model = joblib.load(model_path)
    
    # Compute target statistics
    target_mean = float(np.mean(y_test))
    target_std = float(np.std(y_test))
    target_min = float(np.min(y_test))
    target_max = float(np.max(y_test))
    
    # Evaluate best model
    y_pred = best_model.predict(X_test)
    best_r2 = r2_score(y_test, y_pred)
    best_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    best_mae = mean_absolute_error(y_test, y_pred)
    residuals = y_test - y_pred
    
    # Quality assessment
    if best_r2 >= 0.50:
        quality_assessment = "acceptable"
    elif best_r2 >= 0.25:
        quality_assessment = "marginal"
    else:
        quality_assessment = "subpar"
    
    # === EXPANSION TRAINING IF SUBPAR ===
    expansion_results = []
    expansion_diagnosis = ""
    
    if quality_assessment == "subpar":
        expansion_diagnosis = (
            f"Best model R²={best_r2:.3f} is below acceptable threshold (0.50). "
            f"Attempting expansion with alternative models."
        )
        
        print(f"Quality is subpar (R²={best_r2:.3f}). Training expansion candidates...")
        
        # Try additional models
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.linear_model import ElasticNet
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.svm import SVR
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit
            
            # Load training data for CV
            features_parquet = output_dir_path / "features.parquet"
            import polars as pl
            df_pl = pl.read_parquet(features_parquet)
            
            # Get split point
            train_val_size = step_13_data["split_info"]["train_val_size"]
            X_train_val = df_pl.to_pandas().iloc[:train_val_size][step_12_data["features"]].values
            y_train_val = df_pl.to_pandas().iloc[:train_val_size][step_12_data["target"]].values
            
            cv_splits = TimeSeriesSplit(n_splits=5)
            
            expansion_models = [
                ("elastic_net_expanded", Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=0.01, l1_ratio=0.1, random_state=42, max_iter=10000))
                ])),
                ("hist_gradient_boosting", HistGradientBoostingRegressor(
                    max_iter=100,
                    learning_rate=0.01,
                    random_state=42
                )),
                ("svr_expanded", Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel='rbf', C=10.0, gamma='scale'))
                ])),
            ]
            
            for exp_name, exp_model in expansion_models:
                try:
                    # CV scores
                    cv_scores = cross_val_score(exp_model, X_train_val, y_train_val, cv=cv_splits, scoring='r2')
                    cv_r2_mean = float(cv_scores.mean())
                    
                    # Train and predict on holdout
                    exp_model.fit(X_train_val, y_train_val)
                    y_pred_exp = exp_model.predict(X_test)
                    
                    exp_r2 = r2_score(y_test, y_pred_exp)
                    exp_rmse = np.sqrt(mean_squared_error(y_test, y_pred_exp))
                    exp_mae = mean_absolute_error(y_test, y_pred_exp)
                    
                    expansion_results.append({
                        "model_name": exp_name,
                        "cv_r2_mean": cv_r2_mean,
                        "holdout_r2": exp_r2,
                        "holdout_rmse": exp_rmse,
                        "holdout_mae": exp_mae,
                    })
                    
                    print(f"  {exp_name}: R2={exp_r2:.3f}")
                    
                    # Save expansion candidate
                    exp_path = output_dir_path / f"candidate-{exp_name}.joblib"
                    joblib.dump(exp_model, exp_path)
                    
                except Exception as e:
                    print(f"  {exp_name}: FAILED - {e}")
            
            # Update quality if expansion helped
            if expansion_results:
                best_expansion = max(expansion_results, key=lambda x: x["holdout_r2"])
                if best_expansion["holdout_r2"] >= 0.25:
                    quality_assessment = "subpar_after_expansion"
                    print(f"Expansion improved model to R²={best_expansion['holdout_r2']:.3f}")
        
        except Exception as e:
            print(f"Expansion training failed: {e}")
    
    # Build output JSON
    output_json = {
        "step": "14-model-evaluation",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        
        "best_model_name": step_13_data.get("best_model_name"),
        "best_model_metrics": {
            "r2": best_r2,
            "rmse": best_rmse,
            "mae": best_mae,
        },
        
        "residuals": {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "max_abs_error": float(np.max(np.abs(residuals))),
        },
        
        "target_stats": {
            "mean": target_mean,
            "std": target_std,
            "min": target_min,
            "max": target_max,
        },
        
        "quality_assessment": quality_assessment,
        "expansion_diagnosis": expansion_diagnosis,
        "expansion_results": expansion_results,
        
        "model_worse_than_mean": best_r2 < 0.0,
        
        "candidate_summary": [
            {
                "model_name": cand["model_name"],
                "holdout_r2": cand["holdout_r2"],
                "holdout_rmse": cand["holdout_rmse"],
                "holdout_mae": cand["holdout_mae"],
                "model_worse_than_mean_baseline": cand["holdout_r2"] < 0.0,
            }
            for cand in step_13_data.get("candidates", [])
        ],
        
        "artifacts": {
            "model_joblib": str(output_dir_path / "model.joblib"),
            "holdout_npz": str(output_dir_path / "holdout.npz"),
        },
        
        "context": {
            "dataset_id": run_id,
            "target_column": step_12_data["target"],
            "features": step_12_data["features"],
            "split_strategy": step_12_data.get("split_strategy", {}),
            "model_candidates": step_13_data.get("candidates", []),
            "metrics": {
                "best_r2": best_r2,
                "best_rmse": best_rmse,
                "best_mae": best_mae,
            },
            "artifacts": {},
            "notes": [
                f"Evaluated {len(step_13_data.get('candidates', []))} candidate models",
                f"Best: {step_13_data.get('best_model_name')} (R²={best_r2:.3f})",
                f"Quality: {quality_assessment}",
                f"Expansion: {len(expansion_results)} candidates trained" if expansion_results else "No expansion needed",
            ]
        }
    }
    
    # Write JSON
    step_json_path = output_dir_path / "step-14-evaluation.json"
    with open(step_json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    # Update progress
    progress_path = output_dir_path / "progress.json"
    with open(progress_path) as f:
        progress = json.load(f)
    
    progress["completed_steps"].append("14-model-evaluation")
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"\n✓ Step 14 complete")
    print(f"  Quality assessment: {quality_assessment}")
    print(f"  Best R²: {best_r2:.3f}")
    print(f"  Report written to: {step_json_path}")
    
    return output_json


def main():
    parser = argparse.ArgumentParser(description="Step 14: Model Evaluation")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    try:
        evaluate_models(
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
        sys.exit(0)
    except Exception as e:
        print(f"✗ Step 14 failed: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
