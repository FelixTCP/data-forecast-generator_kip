#!/usr/bin/env python3
"""
Step 15: Model Selection

Apply weighted scoring rule to select best model from all candidates.
Emit full ranking and explicit rationale.
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


def select_model(
    output_dir: str,
    run_id: str,
) -> dict:
    """
    Select best model using weighted scoring.
    
    Scoring rule: 50% R², 25% RMSE, 15% MAE, 10% stability
    
    Returns:
        dict: Selection output JSON
    """
    output_dir_path = Path(output_dir)
    
    # Load evaluation results
    with open(output_dir_path / "step-14-evaluation.json") as f:
        step_14_data = json.load(f)
    
    with open(output_dir_path / "step-13-training.json") as f:
        step_13_data = json.load(f)
    
    candidates = step_14_data.get("candidate_summary", [])
    quality_flag = step_14_data.get("quality_assessment", "marginal")
    
    # Determine if any viable candidate exists
    if quality_flag == "no_viable_candidate" or not candidates:
        output_json = {
            "step": "15-model-selection",
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            
            "quality_flag": "no_viable_candidate",
            "selected_model": None,
            "rationale": "No viable candidates available for selection.",
            "full_ranking": [],
            
            "context": {
                "dataset_id": run_id,
                "notes": ["No models passed quality gates"]
            }
        }
    else:
        # === COMPUTE SCORES ===
        # Normalize metrics to 0-1 scale
        
        # R² normalization: map [min_r2, 1.0] to [0, 1]
        r2_scores = [c.get("holdout_r2", 0.0) for c in candidates]
        min_r2 = min(r2_scores)
        max_r2 = max(r2_scores)
        r2_range = max_r2 - min_r2 if max_r2 != min_r2 else 1.0
        
        # RMSE normalization: lower is better, so invert
        rmse_scores = [c.get("holdout_rmse", 0.0) for c in candidates]
        min_rmse = min(rmse_scores)
        max_rmse = max(rmse_scores)
        rmse_range = max_rmse - min_rmse if max_rmse != min_rmse else 1.0
        
        # MAE normalization: lower is better, so invert
        mae_scores = [c.get("holdout_mae", 0.0) for c in candidates]
        min_mae = min(mae_scores)
        max_mae = max(mae_scores)
        mae_range = max_mae - min_mae if max_mae != min_mae else 1.0
        
        # Stability: CV std (lower is better)
        # Get from step-13 candidates
        step_13_candidates = {c["model_name"]: c for c in step_13_data.get("candidates", [])}
        
        ranked_candidates = []
        
        for cand in candidates:
            model_name = cand["model_name"]
            r2 = cand.get("holdout_r2", 0.0)
            rmse = cand.get("holdout_rmse", 0.0)
            mae = cand.get("holdout_mae", 0.0)
            
            # Normalize R²: higher is better
            r2_norm = (r2 - min_r2) / r2_range if r2_range > 0 else 0.5
            
            # Normalize RMSE: lower is better (invert)
            rmse_norm = 1.0 - ((rmse - min_rmse) / rmse_range if rmse_range > 0 else 0.5)
            
            # Normalize MAE: lower is better (invert)
            mae_norm = 1.0 - ((mae - min_mae) / mae_range if mae_range > 0 else 0.5)
            
            # Stability from step-13
            stability_norm = 0.5  # Default if not found
            if model_name in step_13_candidates:
                cv_std = step_13_candidates[model_name].get("cv_r2_std", 0.0)
                stability_norm = max(0.0, 1.0 - cv_std * 2)  # Lower std is better
            
            # Weighted score
            score = (
                0.50 * r2_norm +
                0.25 * rmse_norm +
                0.15 * mae_norm +
                0.10 * stability_norm
            )
            
            ranked_candidates.append({
                "rank": 0,  # Will be set after sorting
                "model_name": model_name,
                "holdout_r2": r2,
                "holdout_rmse": rmse,
                "holdout_mae": mae,
                "normalized_scores": {
                    "r2": float(r2_norm),
                    "rmse": float(rmse_norm),
                    "mae": float(mae_norm),
                    "stability": float(stability_norm),
                },
                "weighted_score": float(score),
            })
        
        # Sort by weighted score descending
        ranked_candidates.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        # Assign ranks
        for i, cand in enumerate(ranked_candidates):
            cand["rank"] = i + 1
        
        # Select best
        best_candidate = ranked_candidates[0]
        selected_model = best_candidate["model_name"]
        
        # Rationale
        rationale_parts = [
            f"Selected '{selected_model}' as the best model.",
            f"Weighted scoring (50% R², 25% RMSE, 15% MAE, 10% stability) yielded score {best_candidate['weighted_score']:.3f}.",
            f"Holdout performance: R²={best_candidate['holdout_r2']:.3f}, RMSE={best_candidate['holdout_rmse']:.3f}, MAE={best_candidate['holdout_mae']:.3f}.",
            f"Quality assessment: {quality_flag}.",
        ]
        
        if quality_flag in ["subpar", "subpar_after_expansion"]:
            rationale_parts.append("Note: Model quality is below standard thresholds; use with caution.")
        
        rationale = " ".join(rationale_parts)
        
        output_json = {
            "step": "15-model-selection",
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            
            "quality_flag": quality_flag,
            "selected_model": selected_model,
            "rationale": rationale,
            "full_ranking": ranked_candidates,
            
            "scoring_weights": {
                "r2": 0.50,
                "rmse": 0.25,
                "mae": 0.15,
                "stability": 0.10,
            },
            
            "context": {
                "dataset_id": run_id,
                "notes": [
                    f"Ranked {len(ranked_candidates)} candidates",
                    f"Best: {selected_model} (weighted score {best_candidate['weighted_score']:.3f})",
                ]
            }
        }
    
    # Write JSON
    step_json_path = output_dir_path / "step-15-selection.json"
    with open(step_json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    # Update progress
    progress_path = output_dir_path / "progress.json"
    with open(progress_path) as f:
        progress = json.load(f)
    
    progress["completed_steps"].append("15-model-selection")
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)
    
    selected = output_json.get("selected_model", "None")
    print(f"✓ Step 15 complete")
    print(f"  Selected model: {selected}")
    print(f"  Quality flag: {output_json.get('quality_flag', 'unknown')}")
    print(f"  Report written to: {step_json_path}")
    
    return output_json


def main():
    parser = argparse.ArgumentParser(description="Step 15: Model Selection")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    try:
        select_model(
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
        sys.exit(0)
    except Exception as e:
        print(f"✗ Step 15 failed: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
