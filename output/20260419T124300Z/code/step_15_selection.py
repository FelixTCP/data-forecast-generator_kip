#!/usr/bin/env python3
"""
Step 15: Model Selection
Select the best model using weighted scoring criteria.
"""

import json
import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Any

def select_best_model(output_dir: str, run_id: str) -> dict:
    """Select best model using weighted criteria."""
    
    results = {
        "step": "15-model-selection",
        "run_id": run_id,
        "selected_model": None,
        "weighted_score": None,
        "rationale": None,
        "quality_flag": None,
        "full_ranking": [],
    }
    
    try:
        # Load evaluation results
        eval_path = os.path.join(output_dir, "step-14-evaluation.json")
        print(f"[Step 15] Loading evaluation results from {eval_path}...")
        with open(eval_path, 'r') as f:
            eval_results = json.load(f)
        
        candidates = eval_results.get("candidates", [])
        quality_assessment = eval_results.get("quality_assessment", "unknown")
        expansion_candidates = eval_results.get("expansion_candidates", [])
        
        print(f"  ✓ Loaded {len(candidates)} candidates")
        print(f"  ✓ Quality assessment: {quality_assessment}")
        
        # Check for leakage
        if quality_assessment == "leakage_suspected":
            print("✗ Leakage detected in evaluation — halting selection")
            results["quality_flag"] = "leakage_suspected"
            results["rationale"] = "Evaluation flagged potential leakage. No model selected."
            return results
        
        # Combine candidates with expansion candidates
        all_candidates = candidates + expansion_candidates
        
        # Filter out negative R² candidates (worse than mean baseline)
        eligible_candidates = [c for c in all_candidates if c.get("r2", -1) >= 0]
        ineligible_candidates = [c for c in all_candidates if c.get("r2", -1) < 0]
        
        print(f"  ✓ Eligible candidates: {len(eligible_candidates)}")
        print(f"  ✓ Ineligible candidates: {len(ineligible_candidates)}")
        
        # Check if any eligible candidates exist
        if not eligible_candidates:
            print("✗ No eligible candidates (all have R² < 0)")
            results["quality_flag"] = "no_viable_candidate"
            results["rationale"] = "All candidates have R² < 0 (worse than mean baseline). Revisit feature engineering."
            
            # Add all to ranking for transparency
            for c in all_candidates:
                results["full_ranking"].append({
                    "model_name": c["model_name"],
                    "r2": c.get("r2"),
                    "rmse": c.get("rmse"),
                    "mae": c.get("mae"),
                    "weighted_score": None,
                    "eligible": False,
                    "reason": "R² < 0",
                })
            
            return results
        
        # Normalize metrics for scoring
        r2_values = np.array([c.get("r2", 0) for c in eligible_candidates])
        rmse_values = np.array([c.get("rmse", np.inf) for c in eligible_candidates])
        mae_values = np.array([c.get("mae", np.inf) for c in eligible_candidates])
        cv_std_values = np.array([c.get("cv_std_r2", 0) for c in eligible_candidates])
        
        # Min-max normalization
        def normalize(values):
            vmin, vmax = np.min(values), np.max(values)
            if vmax == vmin:
                return np.ones_like(values) * 0.5
            return (values - vmin) / (vmax - vmin)
        
        r2_norm = normalize(r2_values)
        rmse_norm = normalize(rmse_values)  # Lower is better, so invert
        mae_norm = normalize(mae_values)      # Lower is better, so invert
        stability_score = 1 - cv_std_values   # Lower std is better
        
        # Compute weighted scores
        # 50% R², 25% RMSE, 15% MAE, 10% stability
        weighted_scores = (
            0.50 * r2_norm +
            0.25 * (1 - rmse_norm) +
            0.15 * (1 - mae_norm) +
            0.10 * normalize(stability_score)
        )
        
        # Create ranking
        ranking = []
        for i, candidate in enumerate(eligible_candidates):
            ranking.append({
                "rank": i + 1,
                "model_name": candidate["model_name"],
                "r2": float(candidate.get("r2", 0)),
                "rmse": float(candidate.get("rmse", 0)),
                "mae": float(candidate.get("mae", 0)),
                "cv_std_r2": float(candidate.get("cv_std_r2", 0)),
                "weighted_score": float(weighted_scores[i]),
                "eligible": True,
            })
        
        # Sort by weighted score (descending) and then by complexity (ascending)
        complexity_order = {
            "ridge": 0,
            "elastic_net": 1,
            "hist_gradient_boosting": 2,
            "random_forest": 3,
            "gradient_boosting": 4,
            "svr": 5,
            "xgboost": 6,
        }
        
        ranking.sort(key=lambda x: (
            -x["weighted_score"],
            complexity_order.get(x["model_name"].replace("expansion_", ""), 999)
        ))
        
        # Re-rank
        for i, r in enumerate(ranking):
            r["rank"] = i + 1
        
        # Get winner
        winner = ranking[0]
        results["selected_model"] = winner["model_name"]
        results["weighted_score"] = winner["weighted_score"]
        
        # Determine quality flag
        best_r2 = winner["r2"]
        if best_r2 >= 0.50:
            results["quality_flag"] = "acceptable"
        elif best_r2 >= 0.25:
            results["quality_flag"] = "marginal"
        elif best_r2 >= 0:
            results["quality_flag"] = "subpar"
        
        # Generate rationale
        rationale_parts = [
            f"{winner['model_name']} achieved the highest weighted score ({winner['weighted_score']:.4f})",
            f"with R² = {winner['r2']:.4f}, RMSE = {winner['rmse']:.4f}, and MAE = {winner['mae']:.4f}.",
        ]
        
        # Add tradeoff analysis
        runner_up = ranking[1] if len(ranking) > 1 else None
        if runner_up:
            score_diff = winner["weighted_score"] - runner_up["weighted_score"]
            rationale_parts.append(
                f"It outperformed {runner_up['model_name']} by {score_diff:.4f} points."
            )
        
        if best_r2 < 0.25:
            rationale_parts.append(
                f"Note: Quality is subpar (R² < 0.25). Consider revisiting feature engineering."
            )
        
        results["rationale"] = " ".join(rationale_parts)
        
        # Full ranking including ineligible
        results["full_ranking"] = ranking
        for c in ineligible_candidates:
            results["full_ranking"].append({
                "model_name": c["model_name"],
                "r2": c.get("r2"),
                "rmse": c.get("rmse"),
                "mae": c.get("mae"),
                "weighted_score": None,
                "eligible": False,
                "reason": "R² < 0 (worse than mean baseline)",
            })
        
        print(f"\n✓ Selected model: {results['selected_model']}")
        print(f"✓ Weighted score: {results['weighted_score']:.4f}")
        print(f"✓ Quality flag: {results['quality_flag']}")
        print(f"✓ Rationale: {results['rationale']}")
        
        return results
        
    except Exception as e:
        print(f"✗ Step 15 failed: {e}", file=sys.stderr)
        raise

def main():
    parser = argparse.ArgumentParser(description="Step 15: Model Selection")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    
    args = parser.parse_args()
    
    try:
        results = select_best_model(args.output_dir, args.run_id)
        
        # Write results
        report_path = os.path.join(args.output_dir, "step-15-selection.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Wrote selection report to {report_path}")
        
        # Update progress
        progress_path = os.path.join(args.output_dir, "progress.json")
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        
        progress["current_step"] = "15-model-selection"
        progress["completed_steps"].append("15-model-selection")
        
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("✓ Step 15 completed successfully")
        sys.exit(0)
        
    except Exception as e:
        print(f"✗ Step 15 failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
