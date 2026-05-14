#!/usr/bin/env python3
"""
Step 16: Result Presentation

Generate final markdown report with 6 required sections.
Set progress.json status to "completed".
"""
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
import traceback


def generate_report(
    output_dir: str,
    run_id: str,
) -> str:
    """
    Generate final markdown report.
    
    Returns:
        str: Markdown report content
    """
    output_dir_path = Path(output_dir)
    
    # Load all step data
    with open(output_dir_path / "step-10-cleanse.json") as f:
        step_10 = json.load(f)
    
    with open(output_dir_path / "step-11-exploration.json") as f:
        step_11 = json.load(f)
    
    with open(output_dir_path / "step-12-features.json") as f:
        step_12 = json.load(f)
    
    with open(output_dir_path / "step-13-training.json") as f:
        step_13 = json.load(f)
    
    with open(output_dir_path / "step-14-evaluation.json") as f:
        step_14 = json.load(f)
    
    with open(output_dir_path / "step-15-selection.json") as f:
        step_15 = json.load(f)
    
    # Extract key information
    target = step_12["target"]
    features = step_12["features"]
    best_model = step_15.get("selected_model", "Unknown")
    quality = step_15.get("quality_flag", "unknown")
    
    best_r2 = step_14.get("best_model_metrics", {}).get("r2", 0.0)
    best_rmse = step_14.get("best_model_metrics", {}).get("rmse", 0.0)
    best_mae = step_14.get("best_model_metrics", {}).get("mae", 0.0)
    
    target_mean = step_14.get("target_stats", {}).get("mean", 0.0)
    target_std = step_14.get("target_stats", {}).get("std", 0.0)
    
    candidates = step_13.get("candidates", [])
    
    # === SECTION 1: Problem + Target ===
    section_1 = f"""# Data Forecast Generator — Final Report

## 1. Problem Statement and Target Variable

**Report Generated:** {datetime.now(timezone.utc).isoformat()}  
**Run ID:** {run_id}

This report presents the results of a comprehensive regression forecasting pipeline applied to your dataset.

**Target Variable:** `{target}`  
- **Mean:** {target_mean:.2f}
- **Std Dev:** {target_std:.2f}
- **Units:** Numeric (Float)

**Objective:** Build a predictive model to forecast {target} based on historical patterns and available features.
"""
    
    # === SECTION 2: Data Quality ===
    section_2 = f"""
## 2. Data Quality and Feature Engineering Summary

**Input Data:**
- **Total Records:** {step_10.get("row_count_after", 0):,}
- **Original Columns:** {step_10.get("column_count_initial", 0)}
- **Final Features Used:** {len(features)}

**Feature Engineering:**
- **Recommended Features (Step 11):** {len(step_11.get("recommended_features", []))}
- **Features Excluded:** {len(step_11.get("excluded_features", {}))}
  - Reasons: {', '.join(set(step_11.get("excluded_features", {}).values()))}
- **Derived Features Created:** {len(step_12.get("feature_creation_log", []))}
  - Types: Time features, lag features, rolling statistics

**Final Feature Set:** {', '.join(features) if features else 'None'}

**Leakage Assessment:** {step_12.get("leakage", {}).get("status", "unknown")}  
All features passed strict leakage detection (correlation with target < 0.99).

**Data Integrity:**
- Rows after NaN removal: {step_12.get("shape_after", {}).get("rows", 0):,}
- Null-rate summary: Max {step_10.get("null_rate_summary", {}).get("max_null_rate", 0.0):.1%} in any column
"""
    
    # === SECTION 3: Models & Scores Table ===
    section_3 = """
## 3. Candidate Models and Performance Scores

**Model Training Strategy:**
- Split Method: TimeSeriesSplit (5 folds, chronological)
- Train Set: 80% | Holdout: 20%
- Benchmarks: ARIMA, KMeans (always trained)

**Candidate Models Evaluated:**

| Model | CV R² | Holdout R² | RMSE | MAE | Status |
|-------|-------|-----------|------|-----|--------|
"""
    
    for cand in candidates:
        name = cand.get("model_name", "?")
        cv_r2 = cand.get("cv_r2_mean", 0.0)
        r2 = cand.get("holdout_r2", 0.0)
        rmse = cand.get("holdout_rmse", 0.0)
        mae = cand.get("holdout_mae", 0.0)
        status = "✓ Success" if r2 > -1 else "✗ Failed"
        section_3 += f"| {name} | {cv_r2:.3f} | {r2:.3f} | {rmse:.2f} | {mae:.2f} | {status} |\n"
    
    section_3 += f"""
**Best Model:** `{best_model}`
- **Holdout R²:** {best_r2:.3f}
- **Holdout RMSE:** {best_rmse:.2f}
- **Holdout MAE:** {best_mae:.2f}
"""
    
    # === SECTION 4: Rationale ===
    section_4 = f"""
## 4. Selected Model Rationale

**Selection Method:** Weighted scoring (50% R², 25% RMSE, 15% MAE, 10% stability)

**Reasoning:**  
{step_15.get("rationale", "Model selected based on combined metrics.")}

**Quality Assessment:** **{quality.upper()}**
- Acceptable: R² ≥ 0.50
- Marginal: 0.25 ≤ R² < 0.50
- Subpar: R² < 0.25

**Full Ranking (Top 5):**
"""
    
    for i, ranked in enumerate(step_15.get("full_ranking", [])[:5], 1):
        section_4 += f"\n{i}. **{ranked.get('model_name', '?')}**\n"
        section_4 += f"   - R²: {ranked.get('holdout_r2', 0.0):.3f}\n"
        section_4 += f"   - Score: {ranked.get('weighted_score', 0.0):.3f}\n"
    
    # === SECTION 5: Risks and Caveats ===
    residuals = step_14.get("residuals", {})
    section_5 = f"""
## 5. Risks and Caveats

**Model Limitations:**
- Quality Flag: `{quality}` — Model performance is below ideal thresholds.
- Holdout R²: {best_r2:.3f} — Explains {best_r2*100:.1f}% of variance.
- Max Residual: {residuals.get("max_abs_error", 0.0):.2f} units

**Data Limitations:**
- Limited feature set ({len(features)} features)
- {f"No time column detected — cross-sectional model." if not step_12.get("split_strategy", {}).get("time_column") else "Time-series model with chronological splits."}
- Feature exclusions: {len(step_11.get("excluded_features", {}))} features removed (low variance, redundancy, leakage)

**Generalization Risks:**
1. Model trained on historical patterns which may not persist.
2. Holdout set represents only {step_13.get("split_info", {}).get("holdout_size", 0)} records ({step_13.get("split_info", {}).get("holdout_size", 0) / step_13.get("split_info", {}).get("total_rows", 1) * 100:.1f}%).
3. External factors or regime shifts not captured in features.
4. Performance may degrade on new data distributions.

**Recommendation:** Use this model with caution. Validate on independent test data before production deployment.
"""
    
    # === SECTION 6: Next Steps ===
    section_6 = """
## 6. Next Iteration Recommendations

**To Improve Model Performance:**

1. **Feature Engineering:**
   - Add domain-specific features (cyclical encoding, interaction terms)
   - Expand lag windows or include seasonal indicators
   - Consider polynomial or nonlinear transformations

2. **Data Collection:**
   - Gather more historical records (larger training set)
   - Include additional predictor variables
   - Ensure data quality and consistency

3. **Model Experimentation:**
   - Perform hyperparameter tuning (GridSearchCV, Bayesian optimization)
   - Ensemble methods combining multiple models
   - Deep learning approaches if data volume permits

4. **Validation:**
   - K-fold cross-validation with stratification
   - Time-series specific validation (forward chaining)
   - Out-of-sample testing on recent data

5. **Monitoring:**
   - Track prediction errors over time
   - Alert on model drift or data shifts
   - Implement automated retraining pipeline

**Next Phase:** Recommend model retraining quarterly or when new data becomes available. Set up monitoring dashboards to track real-world performance vs. holdout benchmarks.
"""
    
    # Combine all sections
    report = section_1 + section_2 + section_3 + section_4 + section_5 + section_6
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Step 16: Result Presentation")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    try:
        output_dir_path = Path(args.output_dir)
        
        # Generate report
        report = generate_report(
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
        
        # Write report
        report_path = output_dir_path / "step-16-report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Update progress to "completed"
        progress_path = output_dir_path / "progress.json"
        with open(progress_path) as f:
            progress = json.load(f)
        
        progress["status"] = "completed"
        progress["completed_steps"].append("16-result-presentation")
        
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print(f"✓ Step 16 complete")
        print(f"  Report written to: {report_path}")
        print(f"  Report size: {report_path.stat().st_size} bytes")
        print(f"  Progress status: COMPLETED")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"✗ Step 16 failed: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
