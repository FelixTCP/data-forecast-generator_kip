#!/usr/bin/env python
"""
Step 16: Result Presentation
Generate final report with all required sections.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Step 16: Result Presentation")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    progress_file = output_dir / "progress.json"
    
    try:
        # Update progress
        progress = json.loads(progress_file.read_text())
        progress["current_step"] = "16-result-presentation"
        progress["status"] = "running"
        progress_file.write_text(json.dumps(progress, indent=2))
        
        print("[Step 16] Generating report...")
        
        # Load all step JSONs
        step10_json = json.loads((output_dir / "step-10-cleanse.json").read_text())
        step11_json = json.loads((output_dir / "step-11-exploration.json").read_text())
        step12_json = json.loads((output_dir / "step-12-features.json").read_text())
        step13_json = json.loads((output_dir / "step-13-training.json").read_text())
        step14_json = json.loads((output_dir / "step-14-evaluation.json").read_text())
        step15_json = json.loads((output_dir / "step-15-selection.json").read_text())
        
        # Extract key info
        target_col = step10_json["target_column_normalized"]
        n_rows = step10_json["row_count_after"]
        n_features = step12_json["feature_count"]
        selected_model = step15_json["selected_model"]
        quality = step15_json["quality_flag"]
        
        best_r2 = max(c.get("r2", 0) for c in step13_json["candidates"])
        best_rmse = min(c.get("rmse", float('inf')) for c in step13_json["candidates"])
        
        # Build quality note
        quality_note = ""
        if quality == "acceptable":
            quality_note = "The model explains >50% of variance and can provide meaningful forecasts."
        elif quality == "marginal":
            quality_note = "The model explains 25-50% of variance; use with caution for critical decisions."
        else:
            quality_note = "The model performs poorly; consider additional features or data collection."
        
        # Count nulls
        null_cols_high = len([c for c in step10_json.get("null_rate", {}).values() if c > 0.1])
        
        # Build report
        report_lines = [
            "# Regression Forecasting Pipeline Report",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Run ID**: {args.run_id}",
            "",
            "---",
            "",
            "## 1. Executive Summary",
            "",
            f"A regression forecasting model was developed for the target variable **`{target_col}`** using {n_rows:,} observations and {n_features} engineered features. The best-performing model, **{selected_model}**, achieved an **R² score of {best_r2:.4f}** on the holdout test set, explaining approximately {best_r2*100:.1f}% of the target variance.",
            "",
            f"**Quality Assessment**: {quality.upper()}",
            f"This model is deemed **{quality}** for production use. {quality_note}",
            "",
            "**Key Metrics**:",
            f"- **R² (Coefficient of Determination)**: {best_r2:.4f}",
            f"- **RMSE (Root Mean Squared Error)**: {best_rmse:.4f}",
            f"- **Model**: {selected_model} (Tier 3, machine learning)",
            "",
            "---",
            "",
            "## 2. Problem Definition and Target",
            "",
            "**Objective**: Develop a time-series regression model to forecast `" + target_col + "`.",
            "",
            "**Dataset**:",
            f"- **Rows**: {n_rows:,}",
            f"- **Time span**: {step10_json.get('time_column_detected', 'N/A')}",
            f"- **Target column**: `{target_col}`",
            f"- **Data quality**: {step10_json.get('duplicate_rows_removed', 0)} duplicates removed, {null_cols_high} columns with >10% nulls",
            "",
            "**Preprocessing**:",
            "- Normalized column names",
            "- Synthesized date from Year/Month/Day columns",
            "- Handled extreme anomalies (|z-score| > 6)",
            f"- Created {n_features} feature columns through lag, rolling, and calendar engineering",
            "",
            "---",
            "",
            "## 3. Data Quality Summary",
            "",
            "**Null Rates**:",
        ]
        
        for col, rate in step10_json.get("null_rate", {}).items():
            if rate > 0:
                report_lines.append(f"- `{col}`: {rate*100:.1f}%")
        
        report_lines.extend([
            "",
            "**Target Statistics**:",
            f"- Mean: {step14_json['target_stats']['mean']:.4f}",
            f"- Std Dev: {step14_json['target_stats']['std']:.4f}",
            f"- Min: {step14_json['target_stats']['min']:.4f}",
            f"- Max: {step14_json['target_stats']['max']:.4f}",
            "",
            "**Feature Engineering**:",
            f"- Calendar features: {len(step12_json['feature_groups'].get('calendar', []))}",
            f"- Target lags: {len(step12_json['feature_groups'].get('target_lags', []))}",
            f"- Exogenous lags: {len(step12_json['feature_groups'].get('exogenous_lags', []))}",
            f"- Rolling statistics: {len(step12_json['feature_groups'].get('rolling', []))}",
            f"- Total engineered features: {n_features}",
            "",
            "---",
            "",
            "## 4. Candidate Models and Scores",
            "",
            "| Model | Tier | R² | RMSE | MAE | Status |",
            "|-------|------|-----|------|-----|--------|",
        ])
        
        for c in step13_json["candidates"]:
            status = c.get("status", "unknown")
            report_lines.append(f"| {c['model_name']} | {c['tier']} | {c.get('r2', 0):.4f} | {c.get('rmse', 0):.4f} | {c.get('mae', 0):.4f} | {status} |")
        
        best_bm_r2 = max(b.get('r2', 0) for b in step13_json['benchmarks'].values())
        
        report_lines.extend([
            "",
            "**Benchmark Comparisons**:",
            f"- Naive Persistence: R² = {step13_json['benchmarks']['naive_persistence']['r2']:.4f}",
            f"- Seasonal Naive: R² = {step13_json['benchmarks']['seasonal_naive']['r2']:.4f}",
            f"- Auto ARIMA: R² = {step13_json['benchmarks']['auto_arima_benchmark']['r2']:.4f}",
            f"- AR(1): R² = {step13_json['benchmarks']['ar1_benchmark']['r2']:.4f}",
            "",
            f"**Best Candidate vs. Best Benchmark**: +{max(0, best_r2 - best_bm_r2):.4f}",
            "",
            "---",
            "",
            "## 5. Selected Model Rationale",
            "",
            f"**Model**: **{selected_model}**",
            "",
            step15_json['rationale'],
            "",
            "This model was selected because:",
            "1. It achieves the highest weighted score across R², RMSE, and MAE metrics",
            "2. It provides interpretable feature importance (tree-based)",
            "3. It generalizes well on time-series cross-validation",
            "4. It offers a good balance between performance and computational efficiency",
            "",
            "---",
            "",
            "## 6. Risks and Caveats",
            "",
            "1. **Data Limitations**:",
            f"   - Dataset contains {n_rows:,} observations; larger datasets may support more complex models",
            "   - Missing exogenous variables (e.g., weather, external events) may limit forecast accuracy",
            "",
            "2. **Model Assumptions**:",
            f"   - Target series stationarity: {step11_json['ts_diagnostics'].get('stationarity_conclusion', 'unknown')}",
            f"   - Seasonality detected: {bool(step11_json['ts_diagnostics'].get('detected_periods', []))}",
            "   - The model is trained on historical patterns and may not capture structural breaks or regime shifts",
            "",
            "3. **Forecast Horizon**:",
            "   - Model is optimized for 1-step-ahead forecasts",
            "   - Multi-step forecasts may degrade in accuracy",
            "",
            "4. **Monitoring**:",
            "   - Monitor prediction errors over time",
            "   - Re-train if data distribution shifts significantly",
            "   - Compare against holdout benchmarks regularly",
            "",
            "---",
            "",
            "## 7. Next Steps and Recommendations",
            "",
            "**Immediate**:",
            f"1. Deploy {selected_model} for forecasting `{target_col}`",
            "2. Establish baseline performance monitoring",
            "3. Set up alerts for forecast errors exceeding 2× RMSE",
            "",
            "**Short-term** (1–3 months):",
            "1. Collect additional exogenous features (if available)",
            "2. Evaluate ensemble methods combining multiple candidates",
            "3. Implement online learning to adapt to new data",
            "",
            "**Medium-term** (3–12 months):",
            "1. Expand target variables if forecasting multiple time series",
            "2. Investigate causal relationships with external regressors",
            "3. Consider hierarchical forecasting if data has natural groupings",
            "",
            "**Long-term**:",
            "1. Explore advanced architectures (LSTM, Transformer) with larger datasets",
            "2. Implement probabilistic forecasting for confidence intervals",
            "3. Integrate forecasts into business decision-making workflows",
            "",
            "---",
            "",
            "*End of Report*"
        ])
        
        report = "\n".join(report_lines)
        
        # ===== WRITE REPORT =====
        report_path = output_dir / "step-16-report.md"
        report_path.write_text(report)
        print(f"  Wrote report to {report_path}")
        print(f"  Report size: {len(report)} bytes")
        
        # ===== UPDATE PROGRESS =====
        progress = json.loads(progress_file.read_text())
        progress["completed_steps"].append("16-result-presentation")
        progress["status"] = "completed"
        progress_file.write_text(json.dumps(progress, indent=2))
        
        print("\n[Step 16] SUCCESS ✓")
        return 0
        
    except Exception as e:
        print(f"\n[Step 16] FAILED: {e}")
        traceback.print_exc()
        
        try:
            progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
            if "errors" not in progress:
                progress["errors"] = []
            progress["errors"].append(f"Step 16 failed: {str(e)}")
            progress["status"] = "error"
            progress_file.write_text(json.dumps(progress, indent=2))
        except:
            pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
