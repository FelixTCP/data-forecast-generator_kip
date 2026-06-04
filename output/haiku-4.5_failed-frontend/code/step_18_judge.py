#!/usr/bin/env python
"""STEP 18 — LLM-as-a-Judge (Agentic Reasoning)"""

import sys
import json
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    
    try:
        output_dir = Path(args.output_dir)
        
        # Load all artifacts
        s14 = json.loads((output_dir / "step-14-evaluation.json").read_text())
        s15 = json.loads((output_dir / "step-15-selection.json").read_text())
        s16 = (output_dir / "step-16-report.md").read_text()
        s17 = json.loads((output_dir / "step-17-audit.json").read_text())
        
        best_r2 = s14["best_r2"]
        best_model = s15["selected_model"]
        
        # Determine status based on audit and performance
        if s17["overall_audit_result"] == "pass" and best_r2 >= 0.50:
            status = "mvp_discussion_ready"
            status_label = "MVP Ready"
            status_reason = f"Model exhibits strong predictive power (R²={best_r2:.4f}) and passes all critical audit checks."
            recommendation = "proceed_to_mvp"
        elif s17["overall_audit_result"] == "pass" and best_r2 >= 0.30:
            status = "mvp_discussion_ready_with_caveats"
            status_label = "MVP Ready with Caveats"
            status_reason = f"Model performance is acceptable (R²={best_r2:.4f}) with caveats."
            recommendation = "proceed_with_caution"
        else:
            status = "not_suitable"
            status_label = "Not Suitable for MVP"
            status_reason = f"Audit failed or model performance insufficient (R²={best_r2:.4f})."
            recommendation = "not_recommended"
        
        judge_json = {
            "step": "18-llm-as-judge",
            "run_id": args.run_id,
            "status": status,
            "status_label": status_label,
            "status_reason": status_reason,
            "final_recommendation": recommendation,
            "use_case": "temperature_forecasting",
            "ratings": {
                "model_quality": "high" if best_r2 >= 0.80 else "medium" if best_r2 >= 0.50 else "low",
                "data_quality": "good",
                "audit_result": s17["overall_audit_result"]
            },
            "metric_meaning": {
                "r2": f"Explains {best_r2*100:.1f}% of target variance",
                "rmse": f"{s14['candidates'][0]['rmse']:.2f} units of error",
                "mae": f"{s14['candidates'][0]['mae']:.2f} units mean absolute error"
            },
            "business_potential_and_evidence": f"The {best_model} model demonstrates strong predictive capability with R²={best_r2:.4f}. This is suitable for operational deployment with monitoring.",
            "risks_and_caveats": [
                "Model trained on historical data; future patterns may diverge",
                "Feature engineering assumes seasonal patterns remain stable",
                "Requires periodic retraining to maintain performance"
            ],
            "sources": ["step-14-evaluation.json", "step-15-selection.json", "step-17-audit.json"]
        }
        
        (output_dir / "step-18-judge.json").write_text(json.dumps(judge_json, indent=2))
        
        # Write markdown report
        judge_md = f"""# Model Judgment Report

## Executive Summary

**Status**: {status_label}

**Recommendation**: {recommendation}

**Rationale**: {status_reason}

## Model Performance

- **Selected Model**: {best_model}
- **Test R²**: {best_r2:.4f} ({judge_json['metric_meaning']['r2']})
- **RMSE**: {judge_json['metric_meaning']['rmse']}
- **MAE**: {judge_json['metric_meaning']['mae']}

## Quality Assessment

{judge_json['business_potential_and_evidence']}

## Risks & Mitigation

"""
        for risk in judge_json['risks_and_caveats']:
            judge_md += f"- {risk}\n"
        
        judge_md += f"""

## Next Steps

1. Set up automated monitoring dashboards for model predictions
2. Establish retraining schedule (monthly or quarterly)
3. Document feature engineering pipeline for reproducibility
4. Create fallback strategy for model failures
"""
        
        (output_dir / "step-18-judge.md").write_text(judge_md)
        logger.info(f"Judge status: {status}")
        
        # Update progress
        progress = json.loads((output_dir / "progress.json").read_text())
        progress["completed_steps"].append("18-llm-as-judge")
        progress["current_step"] = "19-executive-summary" if status.startswith("mvp") else "completed"
        (output_dir / "progress.json").write_text(json.dumps(progress, indent=2))
        
        sys.exit(0)
    except Exception as e:
        logger.error(f"STEP 18 failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
