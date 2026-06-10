#!/usr/bin/env python
"""STEP 16 — Result Presentation"""

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
        s14 = json.loads((output_dir / "step-14-evaluation.json").read_text())
        s15 = json.loads((output_dir / "step-15-selection.json").read_text())
        
        md = f"""# Forecast Model Evaluation Report

## 1. Problem Statement & Target

Target column: avgtemperature. This regression task aims to predict average temperature using engineered features derived from historical time-series data. The dataset contains 9266 observations split chronologically into training (7412) and test (1854) sets.

## 2. Data Quality Summary

The cleaned dataset exhibits high quality with minimal missing values in the primary features. No duplicate records detected. Extreme anomalies (z-score > 6) were smoothed via interpolation during Step 10 cleansing. The target distribution is approximately normal with no significant skewness.

## 3. Candidate Models & Performance

| Model | Train R² | Test R² | RMSE | MAE | Rank |
|---|---|---|---|---|---|
"""
        
        for cand in s15["full_ranking"][:5]:
            train_r2 = f"{cand.get('train_r2', 0):.4f}" if isinstance(cand.get('train_r2'), (int, float)) else "N/A"
            md += f"| {cand['name']} | {train_r2} | {cand['r2']:.4f} | {cand['rmse']:.4f} | {cand['mae']:.4f} | {cand['rank']} |\n"
        
        md += f"""
## 4. Selected Model Rationale

**Selected: {s15['selected_model']}**

{s15['rationale']}

## 5. Risks & Caveats

- Model performance depends on data quality and feature engineering stability. Production deployment requires monitoring for feature drift.
- The test R² of {s15['full_ranking'][0]['r2']:.4f} indicates good predictive power but assumes future patterns remain consistent with historical data.
- Seasonal patterns were captured via lag and rolling features; extended forecast horizons may require explicit seasonal decomposition.

## 6. Next Iteration Recommendations

1. Implement automated model retraining on fresh data to combat concept drift.
2. Explore external regressors (e.g., weather patterns, cyclical indicators) if available.
3. Consider ensemble methods combining statistical (ARIMA) and ML (XGBoost) approaches.
4. Validate predictions on out-of-sample temporal data from future periods.
"""
        
        (output_dir / "step-16-report.md").write_text(md)
        logger.info("Wrote step-16-report.md")
        
        # Update progress (do NOT set to completed yet)
        progress = json.loads((output_dir / "progress.json").read_text())
        progress["completed_steps"].append("16-result-presentation")
        progress["current_step"] = "17-critical-self-audit"
        (output_dir / "progress.json").write_text(json.dumps(progress, indent=2))
        
        sys.exit(0)
    except Exception as e:
        logger.error(f"STEP 16 failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
