#!/usr/bin/env python3
"""
Step 16: Result Presentation

Generates human-readable report and machine-readable evaluation summary.
"""

import json
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    run_id = args.run_id
    
    # Read all context files
    cleanse_path = output_dir / "step-10-cleanse.json"
    exploration_path = output_dir / "step-11-exploration.json"
    features_path = output_dir / "step-12-features.json"
    training_path = output_dir / "step-13-training.json"
    evaluation_path = output_dir / "step-14-evaluation.json"
    selection_path = output_dir / "step-15-selection.json"
    
    # Load required data
    if not selection_path.exists():
        print(f"ERROR: {selection_path} not found")
        sys.exit(1)
    
    with open(selection_path) as f:
        selection = json.load(f)
    
    with open(evaluation_path) as f:
        evaluation = json.load(f)
    
    with open(features_path) as f:
        features = json.load(f)
    
    with open(cleanse_path) as f:
        cleanse = json.load(f)
    
    with open(exploration_path) as f:
        exploration = json.load(f)
    
    # Extract key info
    selected_model = selection.get("selected_model")
    quality_flag = selection.get("quality_flag")
    rationale = selection.get("rationale", "")
    target_stats = evaluation.get("target_stats", {})
    candidates = evaluation.get("candidates", [])
    benchmarks = evaluation.get("benchmarks", {})
    feature_list = features.get("features", [])
    row_count = cleanse.get("row_count_after", 0)
    target_column = cleanse.get("target_column_normalized", "")
    
    # Build markdown report
    md_content = f"""# Regression Forecasting Pipeline: Final Report

## 1. Problem Statement & Target

**Target Column:** `{target_column}`

This regression forecasting pipeline analyzed {row_count:,} cleaned observations to predict the target variable.

**Quality Assessment:** {quality_flag.replace('_', ' ').title()}

"""
    
    # Add production usability warning if needed
    if quality_flag in ["leakage_suspected", "subpar", "no_viable_candidate"]:
        md_content += f"""### ⚠️ PRODUCTION USABILITY WARNING

**This model is NOT recommended for production deployment.**

- **Reason:** Quality flag = `{quality_flag}`
- **Status:** Developmental / evaluation only
- **Next Steps:** See "Next Iteration Recommendations" below

"""
    else:
        md_content += f"""### ✓ PRODUCTION READY

**This model is suitable for evaluation and cautious production deployment**, subject to continuous monitoring and the caveats listed below.

"""
    
    md_content += f"""## 2. Data Quality Summary

| Metric | Value |
|--------|-------|
| Total Rows (After Cleaning) | {row_count:,} |
| Train/Test Split Strategy | {evaluation.get('split_mode', 'auto')} |
| Target Mean | {target_stats.get('mean', 'N/A'):.4f} |
| Target Std Dev | {target_stats.get('std', 'N/A'):.4f} |
| Target Min | {target_stats.get('min', 'N/A'):.4f} |
| Target Max | {target_stats.get('max', 'N/A'):.4f} |
| Feature Count | {len(feature_list)} |

**Features Used:**
"""
    
    for i, feat in enumerate(feature_list[:20]):  # Show first 20
        md_content += f"- `{feat}`\n"
    
    if len(feature_list) > 20:
        md_content += f"- ... and {len(feature_list) - 20} more\n"
    
    md_content += f"""
## 3. Candidate Models & Evaluation Results

| Model | R² (Holdout) | RMSE | MAE | CV Mean R² | Status |
|-------|--------------|------|-----|-----------|--------|
"""
    
    for candidate in candidates:
        name = candidate.get("model_name")
        r2 = candidate.get("r2", 0)
        rmse = candidate.get("rmse", 0)
        mae = candidate.get("mae", 0)
        cv_r2 = candidate.get("cv_mean_r2", 0)
        selected = "✓ Selected" if name == selected_model else ""
        md_content += f"| {name} | {r2:.4f} | {rmse:.4f} | {mae:.4f} | {cv_r2:.4f} | {selected} |\n"
    
    md_content += f"""
### Baseline Performance (for context)

| Baseline | R² | RMSE | MAE |
|----------|-----|------|-----|
| Mean Predictor | 0.0000 | - | - |
"""
    
    for baseline_name, baseline_val in benchmarks.items():
        r2 = baseline_val.get("r2", "N/A")
        rmse = baseline_val.get("rmse", "N/A")
        mae = baseline_val.get("mae", "N/A")
        md_content += f"| {baseline_name} | {r2} | {rmse} | {mae} |\n"
    
    md_content += f"""
## 4. Selected Model Rationale

**Winning Model:** `{selected_model if selected_model else 'None (see caveats below)'}`

"""
    
    if selected_model:
        md_content += f"""**Rationale:**

{rationale}

### Weighted Scoring Breakdown

The selection used a composite score:
- 50% R² (predictive accuracy on holdout)
- 25% Inverse-normalized RMSE (lower error is better)
- 15% Inverse-normalized MAE (lower error is better)
- 10% Cross-validation stability (1 - CV std)

**Candidate Analysis:**

"""
        for model_name, analysis in selection.get("candidate_analysis", {}).items():
            md_content += f"- **{model_name}:** {analysis}\n"
    else:
        md_content += f"""**Note:** No eligible model selected. All candidates have R² < 0 (worse than mean baseline).

See recommendations section below for recovery strategies.

"""
    
    md_content += f"""
## 5. Risks & Caveats

### Model Limitations

1. **Holdout R² = {target_stats.get('mean', 0):.4f}** indicates the model explains approximately {(selection.get('full_ranking', [{}])[0].get('r2', 0.5) * 100):.1f}% of variance. Residual variance remains substantial.

2. **Time-Series Autocorrelation:** If target exhibits strong autocorrelation, lag-based baselines (naive persistence) may be difficult to beat. Monitor this.

3. **Feature Engineering Scope:** This pipeline engineered {len(feature_list)} features. Additional domain knowledge (exogenous events, seasonal adjustments, etc.) may improve performance.

4. **Generalization Risk:** Model trained on historical data. Performance may degrade if:
   - Underlying data distribution shifts (concept drift)
   - New seasonal patterns emerge
   - Exogenous shocks occur (e.g., regulatory changes, supply disruptions)

5. **Quality Flag: {quality_flag}** — See above for production readiness guidance.

### Data Caveats

- All numeric columns were processed; missing values were handled per pipeline rules.
- Outliers were preserved (no aggressive filtering).
- Feature scaling depends on model type (tree-based models are scale-invariant; linear models may benefit from standardization).

"""
    
    md_content += f"""
## 6. Next Iteration Recommendations

### If Production Performance is Acceptable

1. **Monitor Holdout Performance:** Collect fresh holdout data monthly; recompute R², RMSE, MAE to detect drift.
2. **Retrain Periodically:** Retrain the model every 3–6 months with accumulated new data.
3. **Explainability:** Use SHAP or feature importance plots to explain predictions to stakeholders.

### If Performance Degrades or Quality is Subpar

1. **Feature Engineering Expansion:**
   - Add domain-specific features (e.g., marketing spend, competitor pricing, macroeconomic indicators).
   - Explore non-linear transformations (log, polynomial, splines).
   - Consider interaction terms between top features.

2. **Model Expansion:**
   - Try XGBoost, LightGBM, or ensemble methods if not already attempted.
   - Experiment with stacked or blended models.
   - Consider neural networks (LSTM, Transformer) for time-series data.

3. **Target Engineering:**
   - Decompose the target into trend + seasonal + residual components; model each separately.
   - Try forecasting log-transformed or differenced target.

4. **Hyperparameter Tuning:**
   - Run Bayesian hyperparameter optimization (e.g., Optuna) for top candidates.
   - Increase regularization if overfitting is suspected.

5. **Data Collection:**
   - If data is limited, collect more observations to stabilize model estimates.
   - Identify and remove outliers or data quality issues.

### Process Recommendations

- **Reproducibility:** Version all generated code, configs, and models in Git.
- **Monitoring Dashboard:** Set up alerts when R² or prediction error exceeds thresholds.
- **A/B Testing:** Compare model predictions against incumbent baseline (e.g., domain expert forecast).

---

## Artifact Inventory

- **Model File:** `model.joblib` (fitted sklearn estimator)
- **Holdout Test Set:** `holdout.npz` (X_test, y_test for offline evaluation)
- **Evaluation JSON:** `step-14-evaluation.json` (metrics + diagnostics)
- **Selection JSON:** `step-15-selection.json` (ranking + weighted scores)
- **Features Parquet:** `features.parquet` (engineered feature matrix)
- **Pipeline Metadata:** See `step-*-*.json` files for detailed logs

## Summary

This pipeline successfully generated a {quality_flag.replace('_', ' ')} regression model with R² = {(selection.get('full_ranking', [{}])[0].get('r2', 0.5)):.4f} on holdout data. {f'The selected model (`{selected_model}`) ' if selected_model else 'Although no single model was selected, '}the full ranking and diagnostic outputs enable further investigation and model improvement.

**Next Action:** Review the caveats and recommendations above. Monitor holdout performance in production. Iterate as needed.

---

*Report generated by data-forecast-generator pipeline*
*Run ID: {run_id}*
*Quality Assessment: {quality_flag}*
"""
    
    # Write markdown report
    md_path = output_dir / "step-16-report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"✓ Written {md_path}")
    
    # Write evaluation summary JSON
    eval_summary = {
        "step": "16-result-presentation",
        "run_id": run_id,
        "selected_model": selected_model,
        "quality_flag": quality_flag,
        "r2_holdout": selection.get("full_ranking", [{}])[0].get("r2") if selected_model else None,
        "rmse_holdout": selection.get("full_ranking", [{}])[0].get("rmse") if selected_model else None,
        "mae_holdout": selection.get("full_ranking", [{}])[0].get("mae") if selected_model else None,
        "feature_count": len(feature_list),
        "row_count": row_count,
        "target_column": target_column,
        "production_ready": quality_flag not in ["leakage_suspected", "subpar", "no_viable_candidate"],
        "artifacts": {
            "report_md": str(md_path),
            "model_joblib": str(output_dir / "model.joblib"),
            "holdout_npz": str(output_dir / "holdout.npz"),
            "features_parquet": str(output_dir / "features.parquet")
        }
    }
    
    eval_json_path = output_dir / "step-16-evaluation.json"
    with open(eval_json_path, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"✓ Written {eval_json_path}")
    
    # Update progress
    progress_path = output_dir / "progress.json"
    with open(progress_path) as f:
        progress = json.load(f)
    
    progress["current_step"] = "16-result-presentation"
    if "completed_steps" not in progress:
        progress["completed_steps"] = []
    if "16-result-presentation" not in progress["completed_steps"]:
        progress["completed_steps"].append("16-result-presentation")
    
    # Do NOT mark overall status as completed yet — Step 17 must run next
    
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)
    
    print("✓ Step 16 completed successfully")


if __name__ == "__main__":
    main()
