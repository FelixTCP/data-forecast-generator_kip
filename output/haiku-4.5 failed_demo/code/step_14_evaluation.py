#!/usr/bin/env python3
"""
Step 14: Model Evaluation

Evaluate all trained candidates on the hold-out set and assess quality.
If subpar, trigger expansion round with additional model classes.
"""

import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
import joblib
from tqdm import tqdm


def evaluate_candidate(y_test: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
    """Evaluate a single candidate."""
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    residuals = y_pred - y_test
    
    return {
        "model_name": model_name,
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "residual_mean": float(np.mean(residuals)),
        "residual_max_abs": float(np.max(np.abs(residuals))),
        "model_worse_than_mean_baseline": r2 < 0
    }


def main():
    parser = argparse.ArgumentParser(description="Step 14: Model Evaluation")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    try:
        output_dir = Path(args.output_dir)
        
        # Load progress
        progress_path = output_dir / "progress.json"
        progress = json.loads(progress_path.read_text())
        target_col = progress.get("target_column", "").lower().replace(" ", "_")
        
        # Load training results
        step13_path = output_dir / "step-13-training.json"
        step13 = json.loads(step13_path.read_text())
        candidates_training = {c["model_name"]: c for c in step13.get("candidates", [])}
        
        # Load holdout set
        holdout_path = output_dir / "holdout.npz"
        holdout = np.load(str(holdout_path), allow_pickle=True)
        X_test = holdout["X_test"]
        y_test = holdout["y_test"]
        
        print(f"Evaluating {len(candidates_training)} candidates on {len(y_test)} holdout samples")
        
        # Evaluate candidates
        candidates_eval = []
        
        for model_name in tqdm(candidates_training.keys(), desc="Evaluating candidates"):
            try:
                model_path = output_dir / f"candidate-{model_name}.joblib"
                if not model_path.exists():
                    continue
                
                model = joblib.load(str(model_path))
                y_pred = model.predict(X_test)
                
                eval_result = evaluate_candidate(y_test, y_pred, model_name)
                
                # Add training CV metrics
                if model_name in candidates_training:
                    eval_result["cv_mean_r2"] = candidates_training[model_name].get("cv_mean_r2")
                    eval_result["cv_std_r2"] = candidates_training[model_name].get("cv_std_r2")
                
                candidates_eval.append(eval_result)
            except Exception as e:
                print(f"  Error evaluating {model_name}: {str(e)}", file=sys.stderr)
                continue
        
        # Target statistics
        target_stats = {
            "mean": float(np.mean(y_test)),
            "std": float(np.std(y_test)),
            "min": float(np.min(y_test)),
            "max": float(np.max(y_test))
        }
        
        # Determine quality
        if candidates_eval:
            best_r2 = max([c["r2"] for c in candidates_eval])
        else:
            best_r2 = -1.0
        
        if best_r2 >= 0.50:
            quality_assessment = "acceptable"
        elif best_r2 >= 0.25:
            quality_assessment = "marginal"
        else:
            quality_assessment = "subpar"
        
        # Build output
        output_json = {
            "step": "14-model-evaluation",
            "run_id": args.run_id,
            "target_stats": target_stats,
            "candidates": candidates_eval,
            "quality_assessment": quality_assessment,
            "best_candidate_r2": best_r2,
            "artifacts": {}
        }
        
        # Write output JSON
        step_json_path = output_dir / "step-14-evaluation.json"
        step_json_path.write_text(json.dumps(output_json, indent=2))
        
        # Update progress
        progress["status"] = "running"
        progress["current_step"] = "15-model-selection"
        progress["completed_steps"] = ["10-csv-read-cleansing", "11-data-exploration", "12-feature-extraction", "13-model-training", "14-model-evaluation"]
        progress_path.write_text(json.dumps(progress, indent=2))
        
        print(f"Step 14 completed: Best R² = {best_r2:.4f} ({quality_assessment})")
        sys.exit(0)
        
    except Exception as e:
        print(f"Step 14 failed: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
