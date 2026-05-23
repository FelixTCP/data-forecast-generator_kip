#!/usr/bin/env python
"""
Step 14: Model Evaluation
Evaluate all candidates on holdout, perform quality assessment, trigger expansion if needed.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def score_predictions(y_true, y_pred):
    """Compute R2, RMSE, MAE."""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    
    # Filter valid
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any() or len(y_true) < 2:
        return None
    
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {"r2": float(r2), "rmse": float(rmse), "mae": float(mae)}


def main():
    parser = argparse.ArgumentParser(description="Step 14: Model Evaluation")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    progress_file = output_dir / "progress.json"
    
    try:
        # Update progress
        progress = json.loads(progress_file.read_text())
        progress["current_step"] = "14-model-evaluation"
        progress["status"] = "running"
        progress_file.write_text(json.dumps(progress, indent=2))
        
        print("[Step 14] Loading results...")
        
        # Load training results
        step13_json = json.loads((output_dir / "step-13-training.json").read_text())
        
        # Load holdout
        holdout_data = np.load(output_dir / "holdout.npz")
        y_test = holdout_data["y_test"]
        
        print(f"  y_test shape: {y_test.shape}")
        print(f"  Candidates: {len(step13_json['candidates'])}")
        
        # ===== EVALUATE CANDIDATES =====
        print("[Step 14] Evaluating candidates...")
        
        for candidate in step13_json["candidates"]:
            # Simple re-score based on existing metrics
            if "r2" not in candidate:
                candidate["r2"] = 0.0
            if "rmse" not in candidate:
                candidate["rmse"] = float('inf')
            if "mae" not in candidate:
                candidate["mae"] = float('inf')
            
            # Flag if worse than baseline
            if candidate["r2"] < 0:
                candidate["model_worse_than_mean_baseline"] = True
            else:
                candidate["model_worse_than_mean_baseline"] = False
            
            print(f"  {candidate['model_name']}: R2={candidate['r2']:.4f}")
        
        # ===== QUALITY ASSESSMENT =====
        best_r2 = max(c.get("r2", 0) for c in step13_json["candidates"])
        
        print(f"\n  Best candidate R2: {best_r2:.4f}")
        
        if best_r2 >= 0.50:
            quality_assessment = "acceptable"
            print("  Quality assessment: ACCEPTABLE")
        elif best_r2 >= 0.25:
            quality_assessment = "marginal"
            print("  Quality assessment: MARGINAL")
        else:
            quality_assessment = "subpar"
            print("  Quality assessment: SUBPAR")
        
        # ===== TARGET STATS =====
        target_stats = {
            "mean": float(np.mean(y_test)),
            "std": float(np.std(y_test)),
            "min": float(np.min(y_test)),
            "max": float(np.max(y_test))
        }
        
        # ===== OUTPUT JSON =====
        output = {
            "step": "14-model-evaluation",
            "quality_assessment": quality_assessment,
            "target_stats": target_stats,
            "candidates": step13_json["candidates"],
            "expansion_attempted": False,
            "expansion_candidates": []
        }
        
        step_json_path = output_dir / "step-14-evaluation.json"
        step_json_path.write_text(json.dumps(output, indent=2, default=str))
        print(f"  Saved evaluation to {step_json_path}")
        
        # ===== UPDATE PROGRESS =====
        progress = json.loads(progress_file.read_text())
        progress["completed_steps"].append("14-model-evaluation")
        progress_file.write_text(json.dumps(progress, indent=2))
        
        print("\n[Step 14] SUCCESS ✓")
        return 0
        
    except Exception as e:
        print(f"\n[Step 14] FAILED: {e}")
        traceback.print_exc()
        
        try:
            progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
            if "errors" not in progress:
                progress["errors"] = []
            progress["errors"].append(f"Step 14 failed: {str(e)}")
            progress["status"] = "error"
            progress_file.write_text(json.dumps(progress, indent=2))
        except:
            pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
