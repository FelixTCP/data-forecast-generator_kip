#!/usr/bin/env python
"""
Step 15: Model Selection
Select best model using weighted scoring rule.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Step 15: Model Selection")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    progress_file = output_dir / "progress.json"
    
    try:
        # Update progress
        progress = json.loads(progress_file.read_text())
        progress["current_step"] = "15-model-selection"
        progress["status"] = "running"
        progress_file.write_text(json.dumps(progress, indent=2))
        
        print("[Step 15] Loading evaluation results...")
        
        # Load evaluation
        step14_json = json.loads((output_dir / "step-14-evaluation.json").read_text())
        candidates = step14_json["candidates"]
        quality_assessment = step14_json["quality_assessment"]
        
        print(f"  Quality: {quality_assessment}")
        print(f"  Candidates: {len(candidates)}")
        
        # ===== WEIGHTED SCORING =====
        print("[Step 15] Ranking candidates...")
        
        # Normalize scores: scale R2, RMSE, MAE to [0, 1]
        r2_vals = [c.get("r2", 0) for c in candidates]
        rmse_vals = [c.get("rmse", float('inf')) for c in candidates]
        mae_vals = [c.get("mae", float('inf')) for c in candidates]
        
        r2_min, r2_max = min(r2_vals), max(r2_vals)
        rmse_min, rmse_max = min(rmse_vals), max(rmse_vals)
        mae_min, mae_max = min(mae_vals), max(mae_vals)
        
        def normalize(val, min_v, max_v, lower_is_better=False):
            if max_v == min_v:
                return 0.5
            norm = (val - min_v) / (max_v - min_v)
            return (1 - norm) if lower_is_better else norm
        
        # Weights: 50% R2, 25% RMSE, 15% MAE, 10% stability (approximated by model complexity)
        scored = []
        for c in candidates:
            r2_norm = normalize(c["r2"], r2_min, r2_max, lower_is_better=False)
            rmse_norm = normalize(c["rmse"], rmse_min, rmse_max, lower_is_better=True)
            mae_norm = normalize(c["mae"], mae_min, mae_max, lower_is_better=True)
            
            # Stability score: prefer simpler models (tier 3 > tier 1)
            stability = 0.5 if c["tier"] == 3 else 0.6
            
            total_score = (0.50 * r2_norm + 0.25 * rmse_norm + 0.15 * mae_norm + 0.10 * stability)
            
            scored.append({
                "rank": len(scored) + 1,
                "model_name": c["model_name"],
                "tier": c["tier"],
                "score": float(total_score),
                "r2": c["r2"],
                "rmse": c["rmse"],
                "mae": c["mae"]
            })
        
        # Sort by score
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        for i, s in enumerate(scored, 1):
            s["rank"] = i
            print(f"  {i}. {s['model_name']} (tier {s['tier']}): score={s['score']:.4f}, R2={s['r2']:.4f}")
        
        selected_model = scored[0] if scored else None
        selected_name = selected_model["model_name"] if selected_model else None
        
        print(f"\n  Selected model: {selected_name}")
        
        # ===== OUTPUT JSON =====
        output = {
            "step": "15-model-selection",
            "quality_flag": quality_assessment,
            "selected_model": selected_name,
            "rationale": f"Selected {selected_name} (tier {selected_model['tier']}) based on weighted scoring: 50% R2, 25% RMSE, 15% MAE, 10% stability. Score={selected_model['score']:.4f}",
            "full_ranking": scored
        }
        
        step_json_path = output_dir / "step-15-selection.json"
        step_json_path.write_text(json.dumps(output, indent=2, default=str))
        print(f"  Saved selection to {step_json_path}")
        
        # ===== UPDATE PROGRESS =====
        progress = json.loads(progress_file.read_text())
        progress["completed_steps"].append("15-model-selection")
        progress_file.write_text(json.dumps(progress, indent=2))
        
        print("\n[Step 15] SUCCESS ✓")
        return 0
        
    except Exception as e:
        print(f"\n[Step 15] FAILED: {e}")
        traceback.print_exc()
        
        try:
            progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
            if "errors" not in progress:
                progress["errors"] = []
            progress["errors"].append(f"Step 15 failed: {str(e)}")
            progress["status"] = "error"
            progress_file.write_text(json.dumps(progress, indent=2))
        except:
            pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
