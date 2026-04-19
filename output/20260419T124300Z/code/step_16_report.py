#!/usr/bin/env python3
"""
Step 16: Result Presentation
Generate final human-readable report and mark pipeline as completed.
"""

import json
import os
import sys
import argparse
from datetime import datetime

def generate_report(output_dir: str, run_id: str, csv_path: str, target_column: str) -> str:
    """Generate final markdown report."""
    
    # Load all previous step outputs
    with open(os.path.join(output_dir, "step-00_profiler.json")) as f:
        step00 = json.load(f)
    with open(os.path.join(output_dir, "step-01-cleanse.json")) as f:
        step01 = json.load(f)
    with open(os.path.join(output_dir, "step-11-exploration.json")) as f:
        step11 = json.load(f)
    with open(os.path.join(output_dir, "step-12-features.json")) as f:
        step12 = json.load(f)
    with open(os.path.join(output_dir, "step-13-training.json")) as f:
        step13 = json.load(f)
    with open(os.path.join(output_dir, "step-14-evaluation.json")) as f:
        step14 = json.load(f)
    with open(os.path.join(output_dir, "step-15-selection.json")) as f:
        step15 = json.load(f)
    
    # Build report
    sections = []
    
    # Section 1: Problem + Selected Target
    sections.append("# Regression Forecasting Pipeline Report")
    sections.append("")
    sections.append(f"**Run ID:** {run_id}  ")
    sections.append(f"**Generated:** {datetime.utcnow().isoformat()}Z  ")
    sections.append(f"**Data Source:** {csv_path}  ")
    sections.append("")
    
    sections.append("## 1. Problem & Target Definition")
    sections.append("")
    sections.append(f"**Target Column:** `{target_column}`")
    sections.append(f"**Task:** Build a regression model to predict {target_column}")
    sections.append(f"**Dataset Size:** {step01.get('initial_row_count', 'unknown')} rows × {step01.get('initial_columns', 'unknown')} initial columns")
    sections.append("")
    
    # Section 2: Data Quality Summary
    sections.append("## 2. Data Quality Summary")
    sections.append("")
    
    target_stats = step14.get("target_stats", {})
    sections.append(f"**Target Statistics:**")
    sections.append(f"- Mean: {target_stats.get('mean', 'N/A')}")
    sections.append(f"- Std Dev: {target_stats.get('std', 'N/A')}")
    sections.append(f"- Min: {target_stats.get('min', 'N/A')}")
    sections.append(f"- Max: {target_stats.get('max', 'N/A')}")
    sections.append("")
    
    recommended_features = step11.get("recommended_features", [])
    sections.append(f"**Features Used:**")
    sections.append(f"- Initial numeric columns: {len(step11.get('numeric_columns', []))}")
    sections.append(f"- After MI filtering: {len(recommended_features)}")
    sections.append(f"- Final engineered features: {len(step12.get('features', []))}")
    sections.append("")
    
    null_summary = step01.get("null_rates", {})
    null_cols = [col for col, info in null_summary.items() if info.get("null_rate", 0) > 0]
    if null_cols:
        sections.append(f"**Data Quality Issues:**")
        sections.append(f"- Columns with nulls: {len(null_cols)}")
    else:
        sections.append(f"**Data Quality:** ✓ No null values detected")
    sections.append("")
    
    # Section 3: Candidate Models + Scores
    sections.append("## 3. Candidate Models & Evaluation Scores")
    sections.append("")
    
    candidates = step14.get("candidates", [])
    if candidates:
        sections.append("| Model | R² | RMSE | MAE | CV R² | Status |")
        sections.append("|-------|----|----|-----|------|--------|")
        for c in candidates:
            status = "❌ Below baseline" if c.get("model_worse_than_mean_baseline") else "✓"
            sections.append(
                f"| {c.get('model_name', 'N/A')} | "
                f"{c.get('r2', 0):.4f} | "
                f"{c.get('rmse', 0):.4f} | "
                f"{c.get('mae', 0):.4f} | "
                f"{c.get('cv_mean_r2', 0):.4f} | "
                f"{status} |"
            )
    sections.append("")
    
    # Add expansion candidates if any
    expansion = step14.get("expansion_candidates", [])
    if expansion:
        sections.append("**Expansion Candidates (subpar detection):**")
        sections.append("")
        for e in expansion:
            sections.append(f"- {e.get('model_name', 'N/A')}: R²={e.get('r2', 0):.4f}")
        sections.append("")
    
    # Section 4: Selected Model Rationale
    sections.append("## 4. Selected Model Rationale")
    sections.append("")
    
    selected = step15.get("selected_model")
    quality_flag = step15.get("quality_flag")
    
    if quality_flag == "no_viable_candidate":
        sections.append("⚠️ **WARNING: NO VIABLE CANDIDATE SELECTED**")
        sections.append("")
        sections.append(step15.get("rationale", "All candidates failed baseline criteria."))
        sections.append("")
    elif selected:
        sections.append(f"**Selected Model:** `{selected}`")
        sections.append(f"**Weighted Score:** {step15.get('weighted_score', 0):.4f}")
        sections.append(f"**Quality Assessment:** {quality_flag.upper()}")
        sections.append("")
        sections.append(f"**Rationale:**")
        sections.append("")
        sections.append(step15.get("rationale", ""))
        sections.append("")
    else:
        sections.append("No viable candidate selected.")
        sections.append("")
    
    # Section 5: Risks & Caveats
    sections.append("## 5. Risks & Caveats")
    sections.append("")
    
    risks = []
    
    if quality_flag and quality_flag != "acceptable":
        if quality_flag == "no_viable_candidate":
            risks.append(
                f"**🔴 CRITICAL: No viable model.** All candidates have R² < 0 or model selection failed. "
                f"This model is NOT suitable for production. Revisit data collection or feature engineering."
            )
        elif quality_flag == "subpar" or quality_flag == "subpar_after_expansion":
            risks.append(
                f"**🟠 HIGH: Model quality is poor.** R² < 0.25 indicates the model explains less than 25% "
                f"of target variance. Exercise caution before deployment."
            )
        elif quality_flag == "marginal":
            risks.append(
                f"**🟡 MEDIUM: Model quality is marginal.** R² is in [0.25, 0.50), indicating moderate predictive "
                f"power. Validate on independent test set before production use."
            )
    
    if step12.get("leakage_audit", {}).get("status") == "fail":
        risks.append(
            f"**🔴 LEAKAGE DETECTED:** Feature engineering checks flagged potential data leakage. "
            f"Review engineered features before deployment."
        )
    
    diagnosis = step14.get("expansion_diagnosis")
    if diagnosis:
        risks.append(f"**Diagnostic Notes:** {diagnosis}")
    
    if not risks:
        risks.append("✓ No major risks identified. Model may be suitable for deployment with standard validation.")
    
    for risk in risks:
        sections.append(f"- {risk}")
    sections.append("")
    
    # Section 6: Next Iteration Recommendations
    sections.append("## 6. Next Iteration Recommendations")
    sections.append("")
    
    recommendations = []
    
    if quality_flag == "no_viable_candidate":
        recommendations.append("1. **Revisit data quality:** Check for missing patterns, outliers, or measurement errors")
        recommendations.append("2. **Expand feature engineering:** Create domain-specific features or interaction terms")
        recommendations.append("3. **Assess target variable:** Verify the target is predictable from available features")
        recommendations.append("4. **Collect more data:** More rows or additional input features may help")
    elif quality_flag and "subpar" in quality_flag:
        recommendations.append("1. **Feature expansion:** Add domain-specific or derived features (e.g., seasonal, cyclical)")
        recommendations.append("2. **Target transformation:** Try log or Box-Cox transform if heavily skewed")
        recommendations.append("3. **Hyperparameter tuning:** Perform grid search for top 2-3 candidate models")
        recommendations.append("4. **Ensemble methods:** Combine multiple models for better coverage")
    elif quality_flag == "marginal":
        recommendations.append("1. **Validation testing:** Evaluate on a held-out temporal fold (if time-series data)")
        recommendations.append("2. **Feature refinement:** Iterate on lag windows and rolling window sizes")
        recommendations.append("3. **Regularization:** Experiment with different regularization strengths")
    else:  # acceptable
        recommendations.append("1. **Cross-validation:** Perform k-fold or time-series cross-validation on full dataset")
        recommendations.append("2. **Production validation:** Test on recent unseen data to confirm generalization")
        recommendations.append("3. **Monitoring:** Set up performance tracking post-deployment")
    
    recommendations.append("4. **Retraining schedule:** Plan periodic model updates as new data becomes available")
    
    for rec in recommendations:
        sections.append(rec)
    sections.append("")
    
    # Metadata
    sections.append("---")
    sections.append("")
    sections.append(f"*Pipeline Version: 1.0.0*  ")
    sections.append(f"*Framework: data-forecast-generator*  ")
    sections.append(f"*Report generated automatically by agentic pipeline*")
    
    return "\n".join(sections)

def main():
    parser = argparse.ArgumentParser(description="Step 16: Result Presentation")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--csv-path", required=True, help="Input CSV path")
    parser.add_argument("--target-column", required=True, help="Target column")
    
    args = parser.parse_args()
    
    try:
        print("[Step 16] Generating final report...")
        
        report = generate_report(args.output_dir, args.run_id, args.csv_path, args.target_column)
        
        # Write report
        report_path = os.path.join(args.output_dir, "step-16-report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Wrote final report to {report_path}")
        print(f"✓ Report size: {len(report)} bytes")
        
        # Update progress to completed
        progress_path = os.path.join(args.output_dir, "progress.json")
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        
        progress["current_step"] = "16-result-presentation"
        progress["completed_steps"].append("16-result-presentation")
        progress["status"] = "completed"
        
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("✓ Updated progress.json with status=completed")
        print("\n" + "="*60)
        print("✓ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nReport: {report_path}")
        print(f"Output directory: {args.output_dir}")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"✗ Step 16 failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
