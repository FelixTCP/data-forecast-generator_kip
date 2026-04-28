#!/usr/bin/env python3
"""
Step 15: Model Selection
Ranks candidates using weighted scoring and selects the best model.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

def load_step_output(output_dir: str, step: str) -> dict:
    """Load output JSON from a previous step."""
    path = Path(output_dir) / f"step-{step}.json"
    with open(path, 'r') as f:
        return json.load(f)

def compute_weighted_score(r2: float, rmse: float, mae: float, stability: float = 0.9) -> float:
    """
    Compute weighted score: 50% R², 25% RMSE, 15% MAE, 10% stability
    Note: RMSE and MAE are converted to negative normalized scores (lower is better).
    """
    
    # Normalize R² (higher is better, range 0-1)
    r2_score_norm = max(0, min(1, (r2 + 1) / 2))  # Shift from [-1, 1] to [0, 1]
    
    # RMSE/MAE normalization: We want lower values to be better
    # Use inverse: higher normalized score for lower error
    # Normalize to [0, 1] using 1/(1+error_value)
    rmse_norm = 1.0 / (1.0 + rmse)
    mae_norm = 1.0 / (1.0 + mae)
    
    # Stability (assume = 0.9 by default, varies minimally)
    stability_norm = min(1, stability)
    
    # Weighted sum
    score = (
        0.50 * r2_score_norm +
        0.25 * rmse_norm +
        0.15 * mae_norm +
        0.10 * stability_norm
    )
    
    return float(score)

def rank_candidates(
    candidate_evaluations: dict,
    quality_assessment: str,
    expansion_candidates: dict = None
) -> tuple[list, str]:
    """
    Rank all candidates using weighted scoring.
    
    Returns:
        (ranked_list, selected_model_name)
    """
    
    ranked = []
    
    # Score all evaluated candidates
    for model_name, eval_data in candidate_evaluations.items():
        if eval_data.get("status") not in ["evaluated", "success"]:
            # Skip failed models, but include them in ranking as ineligible
            ranked.append({
                "model": model_name,
                "r2": None,
                "rmse": None,
                "mae": None,
                "score": 0.0,
                "eligible": False,
                "reason": eval_data.get("status", "unknown")
            })
            continue
        
        r2 = eval_data.get("r2")
        rmse = eval_data.get("rmse")
        mae = eval_data.get("mae")
        
        if r2 is None or rmse is None or mae is None:
            ranked.append({
                "model": model_name,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "score": 0.0,
                "eligible": False,
                "reason": "missing_metrics"
            })
            continue
        
        score = compute_weighted_score(r2, rmse, mae)
        
        ranked.append({
            "model": model_name,
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "score": score,
            "eligible": True,
            "reason": "candidate"
        })
    
    # Score expansion candidates if available
    if expansion_candidates:
        for model_name, eval_data in expansion_candidates.items():
            if eval_data.get("status") not in ["evaluated", "success"]:
                ranked.append({
                    "model": f"expansion_{model_name}",
                    "r2": None,
                    "rmse": None,
                    "mae": None,
                    "score": 0.0,
                    "eligible": False,
                    "reason": f"expansion_{eval_data.get('status', 'unknown')}"
                })
                continue
            
            r2 = eval_data.get("r2")
            rmse = eval_data.get("rmse")
            mae = eval_data.get("mae")
            
            if r2 is None or rmse is None or mae is None:
                ranked.append({
                    "model": f"expansion_{model_name}",
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                    "score": 0.0,
                    "eligible": False,
                    "reason": "expansion_missing_metrics"
                })
                continue
            
            score = compute_weighted_score(r2, rmse, mae)
            
            ranked.append({
                "model": f"expansion_{model_name}",
                "r2": float(r2),
                "rmse": float(rmse),
                "mae": float(mae),
                "score": score,
                "eligible": True,
                "reason": "expansion_candidate"
            })
    
    # Sort by score descending
    ranked.sort(key=lambda x: x["score"], reverse=True)
    
    # Select best eligible candidate
    selected = None
    for entry in ranked:
        if entry["eligible"]:
            selected = entry["model"]
            break
    
    return ranked, selected

def main():
    parser = argparse.ArgumentParser(
        description="Step 15: Model Selection"
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
        print("[Step 15] Starting model selection...")
        
        # Load step 14 output
        step14 = load_step_output(output_dir, "14-evaluation")
        
        candidate_evaluations = step14.get("candidate_evaluations", {})
        quality_assessment = step14.get("quality_assessment")
        expansion_candidates = step14.get("expansion_candidates")
        best_r2 = step14.get("best_r2_on_holdout")
        
        print(f"[Step 15] Quality assessment: {quality_assessment}")
        print(f"[Step 15] Best R² on holdout: {best_r2:.4f}")
        
        # Rank candidates
        full_ranking, selected_model = rank_candidates(
            candidate_evaluations,
            quality_assessment,
            expansion_candidates
        )
        
        print(f"[Step 15] Ranked {len(full_ranking)} candidates")
        
        # Determine quality flag and set selected model accordingly
        quality_flag = quality_assessment
        if quality_flag == "subpar_after_expansion" and selected_model and "expansion_" in selected_model:
            # Use the expanded model
            pass
        elif quality_flag == "acceptable" or quality_flag == "marginal":
            # Use the best regular candidate
            pass
        elif quality_flag == "subpar":
            if not selected_model or (expansion_candidates and not any(e.get("status") == "evaluated" for e in expansion_candidates.values())):
                quality_flag = "no_viable_candidate"
                selected_model = None
        
        # Build rationale
        if selected_model:
            selected_entry = next((e for e in full_ranking if e["model"] == selected_model), None)
            if selected_entry:
                rationale = (
                    f"Selected {selected_model} based on weighted scoring (50% R², 25% RMSE, 15% MAE, 10% stability). "
                    f"This model achieved R²={selected_entry['r2']:.4f} on the holdout test set. "
                    f"Overall quality assessment: {quality_assessment}."
                )
            else:
                rationale = f"Selected {selected_model} as best available candidate."
        else:
            rationale = (
                "No viable candidate models met selection criteria. "
                "All candidates either failed to train or achieved poor performance. "
                "Recommend: (1) Feature engineering improvements, (2) More training data, (3) Domain expertise review."
            )
        
        # Build step output
        step_output = {
            "step": "15-model-selection",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "quality_flag": quality_flag,
            "selected_model": selected_model,
            "rationale": rationale,
            "full_ranking": full_ranking,
            "scoring_methodology": {
                "r2_weight": 0.50,
                "rmse_weight": 0.25,
                "mae_weight": 0.15,
                "stability_weight": 0.10,
                "note": "Higher score = better model. R² normalized to [0,1], RMSE/MAE converted to [0,1] via 1/(1+error)."
            },
            "notes": [
                f"Evaluated {len([e for e in full_ranking if e['eligible']])} eligible candidates",
                f"Quality assessment: {quality_assessment}",
                f"Selected model: {selected_model or 'None (no viable candidate)'}",
                f"Best holdout R²: {best_r2:.4f}"
            ]
        }
        
        # Write step output
        output_json_path = Path(output_dir) / "step-15-selection.json"
        with open(output_json_path, 'w') as f:
            json.dump(step_output, f, indent=2)
        print(f"[Step 15] Output written to: {output_json_path}")
        
        # Update progress
        progress_path = Path(output_dir) / "progress.json"
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        progress["current_step"] = "15-model-selection"
        progress["completed_steps"].append("14-model-evaluation")
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print(f"[Step 15] ✓ Completed successfully")
        print(f"[Step 15] Selected model: {selected_model}")
        sys.exit(0)
        
    except Exception as e:
        print(f"[Step 15] ✗ Failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
