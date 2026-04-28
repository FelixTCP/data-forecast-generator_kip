#!/usr/bin/env python3
"""
Step 14: Model Evaluation
Evaluates all candidate models on holdout test set and produces quality assessment.
Performs expansion training if quality is subpar.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def load_step_output(output_dir: str, step: str) -> dict:
    """Load output JSON from a previous step."""
    path = Path(output_dir) / f"step-{step}.json"
    with open(path, 'r') as f:
        return json.load(f)

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a single model on test set."""
    try:
        y_pred = model.predict(X_test)
        
        # Compute metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Residual statistics
        residuals = y_test - y_pred
        residual_mean = float(np.mean(residuals))
        residual_std = float(np.std(residuals))
        residual_max_abs = float(np.max(np.abs(residuals)))
        
        return {
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "mse": float(mse),
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "residual_max_abs": residual_max_abs,
            "status": "evaluated",
            "model_worse_than_mean_baseline": r2 < 0
        }
    except Exception as e:
        print(f"[Step 14] Error evaluating model: {e}", file=sys.stderr)
        return {
            "status": "evaluation_failed",
            "error": str(e)
        }

def assess_quality(best_r2: float, all_r2_scores: list) -> str:
    """Assess model quality based on R² score."""
    if best_r2 >= 0.50:
        return "acceptable"
    elif best_r2 >= 0.25:
        return "marginal"
    else:
        return "subpar"

def train_expansion_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """Train expansion models for subpar performance."""
    
    print("[Step 14] Training expansion models...")
    
    expansion_candidates = {}
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Expansion candidate configs
    expansion_configs = [
        ("elasticnet_tuned", ElasticNet, {"alpha": 0.05, "l1_ratio": 0.5, "max_iter": 5000}),
        ("ridge_light", Ridge, {"alpha": 0.1}),
        ("hist_gradient_boost_expanded", HistGradientBoostingRegressor, {"max_depth": 10, "learning_rate": 0.01}),
    ]
    
    for model_name, model_class, kwargs in expansion_configs:
        try:
            model = model_class(**kwargs, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            expansion_candidates[model_name] = {
                "r2": float(r2),
                "rmse": float(rmse),
                "mae": float(mae),
                "status": "evaluated"
            }
            
            print(f"[Step 14]   {model_name}: R²={r2:.4f}")
            
        except Exception as e:
            print(f"[Step 14]   {model_name} failed: {e}", file=sys.stderr)
            expansion_candidates[model_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    return expansion_candidates

def main():
    parser = argparse.ArgumentParser(
        description="Step 14: Model Evaluation"
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
    
    try:
        print("[Step 14] Starting model evaluation...")
        
        # Load step outputs
        step13 = load_step_output(output_dir, "13-training")
        
        # Load holdout data
        holdout_path = str(Path(output_dir) / "holdout.npz")
        holdout_data = np.load(holdout_path)
        X_test = holdout_data["X_test"]
        y_test = holdout_data["y_test"]
        
        print(f"[Step 14] Loaded holdout data: {X_test.shape}, {y_test.shape}")
        
        # Compute target statistics
        target_stats = {
            "mean": float(np.mean(y_test)),
            "std": float(np.std(y_test)),
            "min": float(np.min(y_test)),
            "max": float(np.max(y_test)),
            "median": float(np.median(y_test))
        }
        
        print(f"[Step 14] Target stats: mean={target_stats['mean']:.2f}, std={target_stats['std']:.2f}")
        
        # Evaluate all candidates
        print("[Step 14] Evaluating candidate models...")
        candidate_evaluations = {}
        
        candidates = step13.get("model_candidates", {})
        all_r2_scores = []
        
        for model_name, candidate_info in candidates.items():
            if candidate_info.get("status") != "success":
                print(f"[Step 14] Skipping {model_name} (not successfully trained)")
                continue
            
            candidate_path = candidate_info.get("path")
            
            try:
                # Load model
                model = joblib.load(candidate_path)
                
                # Evaluate
                metrics = evaluate_model(model, X_test, y_test)
                
                candidate_evaluations[model_name] = {
                    **metrics,
                    "training_r2": candidate_info.get("r2"),  # From training
                    "training_rmse": candidate_info.get("rmse")
                }
                
                if metrics.get("status") == "evaluated":
                    all_r2_scores.append(metrics.get("r2", -np.inf))
                    print(f"[Step 14]   {model_name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
                
            except Exception as e:
                print(f"[Step 14]   Error loading {model_name}: {e}", file=sys.stderr)
                candidate_evaluations[model_name] = {
                    "status": "failed_to_load",
                    "error": str(e)
                }
        
        if not all_r2_scores:
            raise RuntimeError("No candidates could be evaluated")
        
        best_r2 = max(all_r2_scores)
        print(f"[Step 14] Best R² on holdout: {best_r2:.4f}")
        
        # Assess quality
        quality_assessment = assess_quality(best_r2, all_r2_scores)
        print(f"[Step 14] Quality assessment: {quality_assessment}")
        
        expansion_results = {}
        expansion_diagnosis = ""
        
        # Expansion training if subpar
        if quality_assessment == "subpar":
            print("[Step 14] Performance is subpar, training expansion models...")
            
            # Need training data for expansion
            # Load features and create train/test split
            step12 = load_step_output(output_dir, "12-features")
            features_parquet = step12["artifacts"]["features_parquet"]
            target_column = step12["target_column"]
            features_list = step12["features"]
            
            import polars as pl
            df = pl.read_parquet(features_parquet)
            df_pd = df.to_pandas()
            
            # Use same split indices as training
            # For simplicity, use 80/20 split
            split_point = int(len(df_pd) * 0.8)
            X_train = df_pd[features_list].iloc[:split_point].values.astype(np.float64)
            y_train = df_pd[target_column].iloc[:split_point].values.astype(np.float64)
            
            # Re-scale holdout for expansion training
            expansion_results = train_expansion_models(X_train, y_train, X_test, y_test)
            
            # Check if expansion helped
            expansion_r2_scores = [
                m.get("r2", -np.inf) 
                for m in expansion_results.values() 
                if m.get("status") == "evaluated"
            ]
            
            if expansion_r2_scores and max(expansion_r2_scores) > best_r2:
                print(f"[Step 14] Expansion improved R² to {max(expansion_r2_scores):.4f}")
                quality_assessment = "subpar_after_expansion"
                expansion_diagnosis = (
                    f"Expansion training improved R² from {best_r2:.4f} to {max(expansion_r2_scores):.4f}. "
                    f"Best expansion model: {max(expansion_results.items(), key=lambda x: x[1].get('r2', -np.inf))[0]}"
                )
            else:
                expansion_diagnosis = (
                    f"Expansion training did not improve performance. "
                    f"Best expansion R²: {max(expansion_r2_scores) if expansion_r2_scores else 'N/A'}"
                )
        
        # Build step output
        step_output = {
            "step": "14-model-evaluation",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "candidate_evaluations": candidate_evaluations,
            "target_stats": target_stats,
            "best_r2_on_holdout": best_r2,
            "quality_assessment": quality_assessment,
            "expansion_diagnosis": expansion_diagnosis,
            "expansion_candidates": expansion_results if expansion_results else None,
            "notes": [
                f"Evaluated {len(candidate_evaluations)} candidate models on holdout set",
                f"Best holdout R²: {best_r2:.4f}",
                f"Quality assessment: {quality_assessment}",
                f"Target variable: mean={target_stats['mean']:.2f}, std={target_stats['std']:.2f}"
            ]
        }
        
        # Write step output
        output_json_path = Path(output_dir) / "step-14-evaluation.json"
        with open(output_json_path, 'w') as f:
            json.dump(step_output, f, indent=2)
        print(f"[Step 14] Output written to: {output_json_path}")
        
        # Update progress
        progress_path = Path(output_dir) / "progress.json"
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        progress["current_step"] = "14-model-evaluation"
        progress["completed_steps"].append("13-model-training")
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("[Step 14] ✓ Completed successfully")
        sys.exit(0)
        
    except Exception as e:
        print(f"[Step 14] ✗ Failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
