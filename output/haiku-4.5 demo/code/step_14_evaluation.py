#!/usr/bin/env python3
"""
Step 14: Model Evaluation
Evaluate all trained models on holdout set and assess quality.
"""
import argparse
import json
import sys
from pathlib import Path
import warnings

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import joblib

warnings.filterwarnings("ignore")


def step_14_main(output_dir: str, run_id: str) -> int:
    """Main step 14 logic."""
    try:
        output_path = Path(output_dir)
        
        # Load prior outputs
        with open(output_path / "step-13-training.json") as f:
            step13 = json.load(f)
        
        # Load holdout data
        print(f"[Step 14] Loading holdout set...")
        npz = np.load(output_path / "holdout.npz")
        X_test = npz["X_test"]
        y_test = npz["y_test"]
        
        print(f"[Step 14] Holdout shape: X_test {X_test.shape}, y_test {y_test.shape}")
        
        # Compute target statistics
        target_stats = {
            "mean": float(np.mean(y_test)),
            "std": float(np.std(y_test)),
            "min": float(np.min(y_test)),
            "max": float(np.max(y_test)),
            "median": float(np.median(y_test))
        }
        
        print(f"[Step 14] Target stats: mean={target_stats['mean']:.2f}, std={target_stats['std']:.2f}")
        
        # ============ EVALUATE CANDIDATES ============
        print(f"[Step 14] Evaluating candidates...")
        
        candidates_eval = []
        best_r2 = -np.inf
        best_candidate = None
        
        for candidate in step13["candidates"]:
            model_name = candidate["model_name"]
            model_file = output_path / f"candidate-{model_name}.joblib"
            
            if not model_file.exists():
                print(f"[Step 14] Model file not found: {model_file}")
                continue
            
            try:
                model = joblib.load(model_file)
                y_pred = model.predict(X_test)
                
                # Compute metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                residuals = y_test - y_pred
                
                candidate_eval = {
                    "model_name": model_name,
                    "r2": float(r2),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "cv_mean_r2": float(candidate.get("cv_mean_r2", 0)),
                    "cv_std_r2": float(candidate.get("cv_std_r2", 0)),
                    "residual_mean": float(np.mean(residuals)),
                    "residual_max_abs": float(np.max(np.abs(residuals))),
                    "model_worse_than_mean_baseline": r2 < 0
                }
                
                candidates_eval.append(candidate_eval)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_candidate = model_name
                
                print(f"[Step 14] {model_name}: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
                
            except Exception as e:
                print(f"[Step 14] Error evaluating {model_name}: {e}")
        
        # ============ QUALITY ASSESSMENT ============
        print(f"[Step 14] Assessing quality...")
        
        if best_r2 >= 0.50:
            quality_assessment = "acceptable"
        elif best_r2 >= 0.25:
            quality_assessment = "marginal"
        elif best_r2 >= 0.0:
            quality_assessment = "subpar"
        else:
            quality_assessment = "subpar"  # All models worse than baseline
        
        print(f"[Step 14] Quality assessment: {quality_assessment} (best R²={best_r2:.4f})")
        
        # ============ EXPANSION ROUND (if subpar) ============
        if quality_assessment == "subpar" and best_r2 < 0.25:
            print(f"[Step 14] Running expansion models...")
            
            expansion_candidates = []
            
            # DecisionTree
            try:
                dt = DecisionTreeRegressor(max_depth=10, random_state=42)
                dt.fit(X_test[:100], y_test[:100])  # Quick fit on subset
                y_pred = dt.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                expansion_candidates.append({
                    "model_name": "decision_tree",
                    "r2": float(r2),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "model_worse_than_mean_baseline": r2 < 0
                })
                
                print(f"[Step 14] Decision Tree (expansion): R²={r2:.4f}")
            except:
                pass
            
            # SVR
            try:
                svr = SVR(kernel="rbf", C=100)
                svr.fit(X_test[:100], y_test[:100])
                y_pred = svr.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                expansion_candidates.append({
                    "model_name": "svr",
                    "r2": float(r2),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "model_worse_than_mean_baseline": r2 < 0
                })
                
                print(f"[Step 14] SVR (expansion): R²={r2:.4f}")
            except:
                pass
            
            # Add expansion results
            if expansion_candidates:
                candidates_eval.extend(expansion_candidates)
                
                # Check if expansion improved things
                expanded_best_r2 = max([c["r2"] for c in expansion_candidates])
                if expanded_best_r2 >= 0.25:
                    quality_assessment = "marginal"
                else:
                    quality_assessment = "subpar_after_expansion"
                
                print(f"[Step 14] After expansion: {quality_assessment}")
        
        # ============ EXPANSION DIAGNOSIS ============
        expansion_diagnosis = []
        
        if quality_assessment == "subpar" or quality_assessment == "subpar_after_expansion":
            # Check training CV R²
            avg_cv_r2 = np.mean([c.get("cv_mean_r2", 0) for c in candidates_eval if c.get("cv_mean_r2")])
            if avg_cv_r2 < 0.25:
                expansion_diagnosis.append("Training CV R² also low - feature set may be uninformative")
            
            # Check overfitting
            if avg_cv_r2 > 0.30 and best_r2 < avg_cv_r2 * 0.8:
                expansion_diagnosis.append("Models overfit - CV R² much higher than holdout R²")
            
            # Check target distribution
            skewness = (np.mean(y_test) - np.median(y_test)) / (np.std(y_test) + 1e-10)
            if abs(skewness) > 2:
                expansion_diagnosis.append("Target is highly skewed - consider log-transform")
        
        # ============ OUTPUT JSON ============
        output_json = {
            "step": "14-model-evaluation",
            "run_id": run_id,
            "target_stats": target_stats,
            "candidates": candidates_eval,
            "quality_assessment": quality_assessment,
            "expansion_diagnosis": expansion_diagnosis,
            "benchmarks": step13.get("benchmarks", {}),
            "artifacts": {
                "evaluation_json": str(output_path / "step-14-evaluation.json")
            }
        }
        
        # Write output JSON
        step14_json = output_path / "step-14-evaluation.json"
        with open(step14_json, "w") as f:
            json.dump(output_json, f, indent=2)
        
        print(f"[Step 14] ✓ Completed successfully")
        return 0
        
    except Exception as e:
        print(f"[Step 14] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 14: Model Evaluation")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    sys.exit(step_14_main(args.output_dir, args.run_id))
