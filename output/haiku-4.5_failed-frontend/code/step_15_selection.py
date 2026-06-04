#!/usr/bin/env python
"""STEP 15 — Model Selection"""

import sys
import json
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="STEP 15 — Model Selection")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    
    try:
        output_dir = Path(args.output_dir)
        step14 = json.loads((output_dir / "step-14-evaluation.json").read_text())
        
        candidates = step14["candidates"]
        quality = step14["quality_assessment"]
        
        # Scoring: 50% R², 25% RMSE, 15% MAE, 10% stability
        max_r2 = max(c["r2"] for c in candidates) if candidates else 1.0
        min_rmse = min(c["rmse"] for c in candidates) if candidates else 1.0
        min_mae = min(c["mae"] for c in candidates) if candidates else 1.0
        
        for cand in candidates:
            score = (
                0.50 * (cand["r2"] / max_r2 if max_r2 > 0 else 0) +
                0.25 * (min_rmse / cand["rmse"] if cand["rmse"] > 0 else 0) +
                0.15 * (min_mae / cand["mae"] if cand["mae"] > 0 else 0) +
                0.10 * (1.0 if quality == "acceptable" else 0.5)
            )
            cand["selection_score"] = float(score)
        
        candidates.sort(key=lambda x: x["selection_score"], reverse=True)
        winner = candidates[0]
        
        output = {
            "step": "15-model-selection",
            "run_id": args.run_id,
            "quality_flag": quality if quality != "subpar" else "subpar_after_expansion",
            "selected_model": winner["name"],
            "rationale": f"Selected {winner['name']} based on weighted scoring (50% R², 25% RMSE, 15% MAE, 10% stability). Score: {winner['selection_score']:.4f}.",
            "full_ranking": candidates
        }
        
        (output_dir / "step-15-selection.json").write_text(json.dumps(output, indent=2))
        logger.info("STEP 15 completed")
        
        progress = json.loads((output_dir / "progress.json").read_text())
        progress["completed_steps"].append("15-model-selection")
        progress["current_step"] = "16-result-presentation"
        (output_dir / "progress.json").write_text(json.dumps(progress, indent=2))
        
        sys.exit(0)
    except Exception as e:
        logger.error(f"STEP 15 failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
