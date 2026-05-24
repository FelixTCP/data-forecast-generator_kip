#!/usr/bin/env python3
"""
Step 15: Model Selection

Choose the production candidate with transparent scoring criteria.
Apply weighted score: 50% R², 25% inverse RMSE, 15% inverse MAE, 10% stability.
"""

import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from tqdm import tqdm


def normalize_metric(values: List[float]) -> List[float]:
    """Min-max normalize a list of values."""
    min_val = min(values)
    max_val = max(values)
    
    if min_val == max_val:
        return [0.5] * len(values)
    
    return [(v - min_val) / (max_val - min_val) for v in values]


def compute_weighted_score(candidates: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute weighted scores for candidates."""
    
    # Filter eligible candidates (R² >= 0)
    eligible = [c for c in candidates if c.get("r2", -1) >= 0]
    
    if not eligible:
        return {}
    
    # Extract metrics
    r2_values = [c["r2"] for c in eligible]
    rmse_values = [c["rmse"] for c in eligible]
    mae_values = [c["mae"] for c in eligible]
    stability_values = [1 - c.get("cv_std_r2", 0) for c in eligible]
    
    # Normalize
    r2_norm = normalize_metric(r2_values)
    rmse_inv = [1 - x for x in normalize_metric(rmse_values)]
    mae_inv = [1 - x for x in normalize_metric(mae_values)]
    stability_norm = normalize_metric(stability_values)
    
    # Compute weighted scores
    scores = {}
    for i, candidate in enumerate(eligible):
        score = (
            0.50 * r2_norm[i] +
            0.25 * rmse_inv[i] +
            0.15 * mae_inv[i] +
            0.10 * stability_norm[i]
        )
        scores[candidate["model_name"]] = score
    
    return scores


def main():
    parser = argparse.ArgumentParser(description="Step 15: Model Selection")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    try:
        output_dir = Path(args.output_dir)
        
        # Load progress
        progress_path = output_dir / "progress.json"
        progress = json.loads(progress_path.read_text())
        
        # Load evaluation results
        step14_path = output_dir / "step-14-evaluation.json"
        step14 = json.loads(step14_path.read_text())
        candidates = step14.get("candidates", [])
        quality_assessment = step14.get("quality_assessment", "unknown")
        
        # Compute scores
        scores = compute_weighted_score(candidates)
        
        # Select best
        if scores:
            selected_model = max(scores, key=scores.get)
            best_score = scores[selected_model]
            quality_flag = quality_assessment
        else:
            selected_model = None
            best_score = None
            quality_flag = "no_viable_candidate"
        
        # Build candidate analysis
        candidate_analysis = {}
        for candidate in candidates:
            model_name = candidate["model_name"]
            r2 = candidate.get("r2", -1)
            
            if r2 < 0:
                status = "ineligible (R² < 0)"
            elif r2 >= 0.50:
                status = "acceptable"
            elif r2 >= 0.25:
                status = "marginal"
            else:
                status = "subpar"
            
            candidate_analysis[model_name] = {
                "r2": r2,
                "status": status,
                "score": scores.get(model_name),
                "rationale": f"R²={r2:.3f}, RMSE={candidate.get('rmse', 0):.3f}, MAE={candidate.get('mae', 0):.3f}"
            }
        
        # Build rationale
        if selected_model:
            rationale = f"Selected {selected_model} with weighted score {best_score:.3f}. "
            rationale += f"Best R² on holdout: {candidate_analysis[selected_model]['r2']:.3f}. "
            rationale += "Model provides best balance of accuracy, error metrics, and training stability."
        else:
            rationale = "No eligible models (all R² < 0). Recommend revisiting feature engineering or model class selection."
        
        # Build full ranking
        full_ranking = []
        for model_name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            full_ranking.append({
                "model_name": model_name,
                "weighted_score": score,
                "status": "eligible"
            })
        
        # Add ineligible models
        for candidate in candidates:
            if candidate["model_name"] not in scores:
                full_ranking.append({
                    "model_name": candidate["model_name"],
                    "weighted_score": None,
                    "status": "ineligible",
                    "reason": "R² < 0"
                })
        
        # Build output
        output_json = {
            "step": "15-model-selection",
            "run_id": args.run_id,
            "selected_model": selected_model,
            "weighted_score": best_score,
            "rationale": rationale,
            "quality_flag": quality_flag,
            "candidate_analysis": candidate_analysis,
            "full_ranking": full_ranking,
            "baselines": {
                "mean_baseline_r2": 0.0,
                "mean_baseline_description": "Predicting the mean target value"
            }
        }
        
        # Write output JSON
        step_json_path = output_dir / "step-15-selection.json"
        step_json_path.write_text(json.dumps(output_json, indent=2))
        
        # Update progress
        progress["status"] = "running"
        progress["current_step"] = "16-result-presentation"
        progress["completed_steps"] = ["10-csv-read-cleansing", "11-data-exploration", "12-feature-extraction", "13-model-training", "14-model-evaluation", "15-model-selection"]
        progress_path.write_text(json.dumps(progress, indent=2))
        
        print(f"Step 15 completed: Selected {selected_model} (score={best_score:.4f})")
        sys.exit(0)
        
    except Exception as e:
        print(f"Step 15 failed: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
