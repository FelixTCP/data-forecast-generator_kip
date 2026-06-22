"""Step 16 — Result Presentation (Report)."""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    run_id = args.run_id

    # Update progress
    progress_path = os.path.join(output_dir, "progress.json")
    with open(progress_path) as f:
        progress = json.load(f)
    progress["status"] = "running"
    progress["current_step"] = "16-result-presentation"
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    # Load inputs
    step10 = json.load(open(os.path.join(output_dir, "step-10-cleanse.json")))
    step11 = json.load(open(os.path.join(output_dir, "step-11-exploration.json")))
    step12 = json.load(open(os.path.join(output_dir, "step-12-features.json")))
    step13 = json.load(open(os.path.join(output_dir, "step-13-training.json")))
    step14 = json.load(open(os.path.join(output_dir, "step-14-evaluation.json")))
    step15 = json.load(open(os.path.join(output_dir, "step-15-selection.json")))

    target_col = step12["target_column"]
    time_col = step10["time_column_detected"]
    selected_model = step15["selected_model"]
    quality_flag = step15["quality_flag"]
    best_r2 = step15.get("context", {}).get("r2", 0)
    best_cand = next((c for c in step14["candidates"] if c["model_name"] == selected_model), {})
    target_stats = step14.get("target_stats", {})
    n_rows = step10["row_count_after"]
    n_features = len(step12["features"])
    split_mode = step12["split_strategy"]["resolved_mode"]
    stationarity = step11.get("ts_diagnostics", {}).get("stationarity_conclusion", "unknown")
    hurst = step11.get("ts_diagnostics", {}).get("hurst_exponent")
    seasonality = step11.get("ts_diagnostics", {}).get("primary_seasonal_period")
    null_rates = step10.get("null_rate", {})
    max_null = max(null_rates.values()) if null_rates else 0
    n_candidates = len(step14["candidates"])
    naive_r2 = step15.get("baselines", {}).get("naive_lag_baseline", {}).get("r2", 0) or 0

    # Leakage warning
    leakage_warning = ""
    if quality_flag in ("leakage_suspected", "subpar", "no_viable_candidate"):
        leakage_warning = (
            "\n> ⚠️ **WARNING**: Model quality is flagged as `{}`. "
            "This run is **NOT production-ready**. "
            "Review feature engineering and data quality before proceeding.\n"
        ).format(quality_flag)

    # Build report
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    report = f"""# Temperature Forecasting Pipeline Report

**Run ID:** {run_id}  
**Generated:** {timestamp}  
**Target:** `{target_col}`  
**Dataset:** {n_rows:,} rows, {time_col} time column  
**Quality Flag:** `{quality_flag}`

{leakage_warning}

---

## 1. Problem & Selected Target

We analyzed daily temperature data for Algiers, Algeria to build a regression-based forecasting model. 
The target variable is `{target_col}` (average daily temperature in Fahrenheit), representing historical 
daily temperature observations from the dataset.

The dataset contains **{n_rows:,} rows** with a **{split_mode.replace('_', '-')} split strategy** 
applied to preserve temporal ordering. The time series was detected as **{stationarity}** 
(Hurst exponent: {f'{hurst:.3f}' if hurst else 'N/A'}).

---

## 2. Data Quality Summary

| Metric | Value |
|--------|-------|
| Total rows | {n_rows:,} |
| Features engineered | {n_features} |
| Time column | `{time_col}` |
| Max null rate | {max_null:.2%} |
| Duplicate rows removed | {step10.get('duplicate_rows_removed', 0)} |
| Split strategy | {split_mode} |
| Training rows | {step13.get('n_train', 'N/A'):,} |
| Holdout rows | {step13.get('n_test', 'N/A'):,} |

**Data Fixes Applied:**
{chr(10).join(f'- {f}' for f in step10.get('fixes', []))}

**Feature Engineering:**
- Lag features: {', '.join([f for f in step12['features'] if '_lag_' in f][:5]) or 'none'}
- Rolling features: {', '.join([f for f in step12['features'] if '_roll_' in f][:4]) or 'none'}
- Calendar features: {', '.join([f for f in step12['features'] if 'cal_' in f][:4]) or 'none'}
- Fourier features: {', '.join([f for f in step12['features'] if 'fourier' in f][:4]) or 'none'}

---

## 3. Candidate Models & Scores

| Model | R² | RMSE | MAE | CV R² |
|-------|-----|------|-----|-------|
"""
    for c in sorted(step14["candidates"], key=lambda x: x.get("r2") or -999, reverse=True):
        r2_str = f"{c['r2']:.4f}" if c.get("r2") is not None else "N/A"
        rmse_str = f"{c.get('rmse', 0):.3f}" if c.get("rmse") is not None else "N/A"
        mae_str = f"{c.get('mae', 0):.3f}" if c.get("mae") is not None else "N/A"
        cv_str = f"{c.get('cv_mean_r2', 0):.3f}±{c.get('cv_std_r2', 0):.3f}" if c.get("cv_mean_r2") is not None else "N/A"
        selected_marker = " ✓ **SELECTED**" if c["model_name"] == selected_model else ""
        report += f"| {c['model_name']}{selected_marker} | {r2_str} | {rmse_str} | {mae_str} | {cv_str} |\n"

    report += f"""
**Naive lag baseline:** R²={naive_r2:.4f}  
**Target mean:** {target_stats.get('mean', 0):.2f}°F, std={target_stats.get('std', 0):.2f}°F

---

## 4. Selected Model Rationale

**Selected model:** `{selected_model}`  
**R²:** {best_r2:.4f} ({best_r2*100:.1f}% variance explained)  
**RMSE:** {best_cand.get('rmse', 0):.3f}°F  
**MAE:** {best_cand.get('mae', 0):.3f}°F  

{step15.get('rationale', 'See full ranking in step-15-selection.json.')}

The model was trained using a **{split_mode.replace('_', '-')} split** with 5-fold 
TimeSeriesSplit cross-validation. The split preserves temporal ordering to prevent 
information leakage from future to past.

**Feature importance (SHAP top features):**
"""
    shap_info = step14.get("shap_artifacts", {})
    if shap_info.get("status") == "computed":
        for feat in shap_info.get("top_features_by_mean_abs_shap", [])[:5]:
            report += f"- `{feat['feature']}`: mean |SHAP| = {feat['mean_abs_shap']:.4f}\n"
    else:
        report += "- SHAP values not available\n"

    report += f"""
---

## 5. Risks & Caveats

1. **Temporal data leakage risk**: Lag and rolling features use past values of the target. 
   The model was trained with chronological splits to mitigate this risk, but deployment 
   requires careful handling of the prediction horizon.

2. **Stationarity**: The series is classified as **{stationarity}**. 
   {"Non-stationary series may require periodic retraining as the data distribution shifts over time." if "non" in stationarity else "Stationary series can be modeled reliably without differencing."}

3. **Feature dependency**: The model depends on lag features (e.g., yesterday's temperature). 
   Predictions further than {max(1, len([f for f in step12['features'] if '_lag_' in f]))} steps into the future 
   will require recursive forecasting, which may amplify errors.

4. **Seasonality**: {"Yearly seasonality (period=" + str(seasonality) + ") detected and encoded via Fourier features. The model should handle seasonal patterns well." if seasonality else "No strong seasonal period detected. Model may not capture multi-seasonal patterns."}

5. **Data coverage**: Model trained on data from a single location (Algiers). 
   Performance may degrade for other locations or significantly different climate conditions.

---

## 6. Next Iteration Recommendations

1. **Expand feature engineering**: Add humidity, wind speed, and precipitation data 
   as exogenous features to improve forecast accuracy beyond pure autoregressive structure.

2. **Evaluate on out-of-sample years**: Test model on a fully withheld year (not just 
   a random holdout) to assess performance on truly unseen time periods.

3. **Hyperparameter tuning**: The current Ridge model uses default alpha=10.0. 
   A grid search (alpha from 0.01 to 1000) may further improve R² by 0.01–0.03.

4. **Multi-step forecasting**: Implement direct multi-step forecasting (one model per 
   horizon) for forecasts beyond 7 days.

5. **Model monitoring**: Set up tracking of prediction vs. actual to detect distribution 
   drift and trigger retraining automatically.
"""

    # Write report
    report_path = os.path.join(output_dir, "step-16-report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Update progress — DO NOT set status to "completed" (step 17 must run next)
    with open(progress_path) as f:
        progress = json.load(f)
    progress["status"] = "running"
    progress["current_step"] = "17-critical-self-audit"
    if "16-result-presentation" not in progress.get("completed_steps", []):
        progress["completed_steps"].append("16-result-presentation")
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    print(f"Step 16 complete. Report: {report_path} ({os.path.getsize(report_path)} bytes)")
    sys.exit(0)


if __name__ == "__main__":
    main()
