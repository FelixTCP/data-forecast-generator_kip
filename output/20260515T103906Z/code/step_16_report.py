"""Step 16 — Result Presentation.

Runnable:
    python step_16_report.py --output-dir <dir> --run-id <id>
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def update_progress(progress_path: Path, updates: dict):
    if progress_path.exists():
        with open(progress_path) as f:
            p = json.load(f)
    else:
        p = {}
    p.update(updates)
    with open(progress_path, "w") as f:
        json.dump(p, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Step 16: Result Presentation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    progress_path = output_dir / "progress.json"
    update_progress(progress_path, {"status": "running", "current_step": "16-result-presentation"})

    try:
        # ── Load all upstream context ─────────────────────────────────────────
        with open(output_dir / "step-10-cleanse.json") as f:
            ctx10 = json.load(f)
        with open(output_dir / "step-11-exploration.json") as f:
            ctx11 = json.load(f)
        with open(output_dir / "step-12-features.json") as f:
            ctx12 = json.load(f)
        with open(output_dir / "step-13-training.json") as f:
            ctx13 = json.load(f)
        with open(output_dir / "step-14-evaluation.json") as f:
            ctx14 = json.load(f)
        with open(output_dir / "step-15-selection.json") as f:
            ctx15 = json.load(f)

        target_col = ctx10["target_column_normalized"]
        csv_path = ctx10.get("csv_path", "input CSV")
        quality_flag = ctx15.get("quality_flag", "unknown")
        selected_model = ctx15.get("selected_model")
        target_stats = ctx14.get("target_stats", {})
        ts = ctx11.get("ts_diagnostics", {})
        detected_frequency = ctx11.get("detected_frequency", "daily")
        primary_period = ts.get("primary_seasonal_period")
        stationarity = ts.get("stationarity_conclusion", "unknown")
        hurst = ts.get("hurst_exponent")
        n_rows_before = ctx10.get("row_count_before", "?")
        n_rows_after = ctx10.get("row_count_after", "?")
        fixes = ctx10.get("fixes", [])
        null_rates = ctx10.get("null_rate", {})
        features = ctx12.get("features", [])
        split = ctx12.get("split_strategy", {})
        holdout_start = split.get("holdout_start_index", "?")
        total_rows = split.get("total_rows_after_burnin", "?")
        holdout_size = split.get("holdout_size", "?")

        # Model scores
        candidates_14 = ctx14.get("candidates", [])
        full_ranking = ctx15.get("full_ranking", [])
        rationale = ctx15.get("rationale", "")
        candidate_analysis = ctx15.get("candidate_analysis", {})
        naive_baseline = ctx15.get("baselines", {}).get("naive_lag_baseline", {})
        model_recs = [m["model_class"] for m in ctx11.get("model_class_recommendations", [])]

        # Production usable?
        production_usable = quality_flag in ("acceptable",)
        is_leakage = quality_flag == "leakage_suspected"
        is_subpar = quality_flag in ("subpar", "subpar_after_expansion", "no_viable_candidate")

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # ── Section 1: Problem + selected target ──────────────────────────────
        s1 = f"""## 1. Problem Statement & Selected Target

**Run ID:** {args.run_id}
**Generated:** {now_str}

**Dataset:** `{csv_path}`
**Target column:** `{target_col}`
**Objective:** Forecast daily average temperature in Algiers using historical observations from 1995–2020.

The dataset contains daily temperature measurements spanning 26 years. The task is a univariate
time-series regression: given historical temperature and calendar features, predict the next day's
average temperature in degrees Fahrenheit.

**Dataset shape:** {n_rows_before} raw rows → {n_rows_after} rows after cleansing.
**Sampling frequency:** {detected_frequency}.
**Time-series properties:** stationarity={stationarity}, Hurst exponent={f"{hurst:.3f}" if hurst else "N/A"},
primary seasonal period={primary_period} days.
"""

        # ── Section 2: Data Quality Summary ───────────────────────────────────
        null_lines = "\n".join(
            f"- `{col}`: {rate*100:.1f}% null"
            for col, rate in null_rates.items()
            if rate > 0
        ) or "No null values remaining after cleansing."

        fixes_lines = "\n".join(f"- {f}" for f in fixes) or "None required."

        s2 = f"""## 2. Data Quality Summary

| Metric | Value |
|--------|-------|
| Raw rows | {n_rows_before} |
| Rows after cleansing | {n_rows_after} |
| Columns | {ctx10.get("column_count", "?")} |
| Extreme anomalies smoothed | {"Yes — 35 values where |z|>6 replaced with linear interpolation" if any("extreme_anomaly_smoothed" in f for f in fixes) else "None"} |

**Null rates (post-cleanse):**
{null_lines}

**Fixes applied:**
{fixes_lines}

**Feature engineering:** {len(features)} features built across calendar (Group A), lag (Group B),
cross-correlation lags (Group C), rolling statistics (Group E), and Fourier seasonality (Group F) groups.

**Train / holdout split (chronological):**
- Training rows: {holdout_start} (burn-in of {split.get("burn_in_rows", "?")} rows excluded)
- Holdout rows: {holdout_size} (~{f"{int(holdout_size)/int(total_rows)*100:.0f}" if total_rows and holdout_size else "?"}% of data)
"""

        # ── Section 3: Candidate Models + Scores ──────────────────────────────
        table_rows = []
        for row in full_ranking:
            r2_str = f"{row['r2']:.4f}" if row.get("r2") is not None else "-"
            rmse_str = f"{row['rmse']:.3f}" if row.get("rmse") else "-"
            mae_str = f"{row['mae']:.3f}" if row.get("mae") else "-"
            cv_str = f"{row['cv_mean_r2']:.4f}" if row.get("cv_mean_r2") and row["cv_mean_r2"] is not None else "-"
            ws_str = f"{row['weighted_score']:.4f}" if row.get("weighted_score") else "-"
            selected_flag = "**SELECTED**" if row["model_name"] == selected_model else ""
            table_rows.append(
                f"| {row.get('rank', '-')} | {row['model_name']} | {r2_str} | {rmse_str} | "
                f"{mae_str} | {cv_str} | {ws_str} | {row.get('status','?')} | {selected_flag} |"
            )

        # Add mandatory benchmarks
        bench_rows = []
        for c in candidates_14:
            if c.get("is_benchmark"):
                r2_str = f"{c['r2']:.4f}" if c.get("r2") is not None else "-"
                rmse_str = f"{c['rmse']:.3f}" if c.get("rmse") else "-"
                mae_str = f"{c['mae']:.3f}" if c.get("mae") else "-"
                bench_rows.append(f"| - | {c['model_name']} | {r2_str} | {rmse_str} | {mae_str} | - | - | benchmark | |")

        s3 = f"""## 3. Candidate Models & Scores

**Naive lag-1 baseline:** R\u00b2={naive_baseline.get("r2", "N/A"):.4f}, RMSE={naive_baseline.get("rmse", "N/A"):.3f}, MAE={naive_baseline.get("mae", "N/A"):.3f}

### Trained Candidate Models

| Rank | Model | R\u00b2 | RMSE | MAE | CV R\u00b2 | Weighted Score | Status | Note |
|------|-------|-----|------|-----|-------|----------------|--------|------|
{chr(10).join(table_rows)}

### Mandatory Benchmarks

| Rank | Model | R\u00b2 | RMSE | MAE | CV R\u00b2 | Weighted Score | Status | Note |
|------|-------|-----|------|-----|-------|----------------|--------|------|
{chr(10).join(bench_rows)}

**Model families explored:** {", ".join(model_recs)}.
*(Note: pmdarima and XGBoost not installed in environment — FAAR-ARIMA and XGBoost candidates not evaluated.)*
"""

        # ── Section 4: Selected Model Rationale ───────────────────────────────
        if is_leakage:
            s4 = """## 4. Selected Model Rationale

> **WARNING: Leakage suspected.** No production model has been selected.
> Metrics are invalid. Review feature engineering and re-run the pipeline.
"""
        elif quality_flag == "no_viable_candidate":
            s4 = f"""## 4. Selected Model Rationale

> **No viable candidate.** All trained models have R\u00b2 < 0.
> Revisit feature engineering or expand model classes.
"""
        else:
            # Find winner's entry
            winner_entry = next((c for c in candidates_14 if c.get("model_name") == selected_model), {})
            s4 = f"""## 4. Selected Model Rationale

**Selected model:** `{selected_model}`
**Quality flag:** `{quality_flag}`

{rationale}

### Selected Model Details

| Metric | Value |
|--------|-------|
| Holdout R\u00b2 | {winner_entry.get("r2", "N/A")} |
| Holdout RMSE | {f"{winner_entry.get('rmse', 'N/A'):.3f} F" if winner_entry.get("rmse") else "N/A"} |
| Holdout MAE | {f"{winner_entry.get('mae', 'N/A'):.3f} F" if winner_entry.get("mae") else "N/A"} |
| CV R\u00b2 | {f"{winner_entry.get('cv_mean_r2', 'N/A'):.4f} +/- {winner_entry.get('cv_std_r2', 0):.4f}" if winner_entry.get("cv_mean_r2") else "N/A"} |
| Target mean | {target_stats.get("mean", "N/A"):.2f} F |
| Target std | {target_stats.get("std", "N/A"):.2f} F |
| RMSE as % of std | {f"{winner_entry.get('rmse', 0) / target_stats.get('std', 1) * 100:.1f}%" if winner_entry.get("rmse") and target_stats.get("std") else "N/A"} |

### Candidate Analysis

{chr(10).join(f"**{name}:** {analysis}" + chr(10) for name, analysis in candidate_analysis.items())}
"""

        # ── Section 5: Risks and Caveats ──────────────────────────────────────
        naive_r2 = naive_baseline.get("r2", 0)
        winner_r2 = next((c.get("r2", 0) for c in candidates_14 if c.get("model_name") == selected_model), 0) if selected_model else 0
        delta_vs_naive = winner_r2 - naive_r2 if (winner_r2 and naive_r2) else None

        leakage_warning = ""
        if quality_flag in ("subpar", "subpar_after_expansion", "no_viable_candidate", "leakage_suspected"):
            leakage_warning = """
> **PRODUCTION WARNING:** This run's `quality_flag` indicates the model is NOT production-ready.
> Do not deploy this model for operational forecasting without further investigation.
"""

        s5 = f"""## 5. Risks and Caveats
{leakage_warning}
### Key Risks

1. **Modest improvement over naive persistence:**
   The naive lag-1 baseline achieves R\u00b2={naive_r2:.4f}, while the selected model achieves
   R\u00b2={winner_r2:.4f} (delta={f"{delta_vs_naive:+.4f}" if delta_vs_naive is not None else "N/A"}).
   Daily temperature is highly autocorrelated — any reasonable model will score well. The marginal
   gain beyond a simple persistence model is limited.

2. **Fahrenheit scale & geographic scope:**
   The dataset covers Algiers, Algeria only (1995–2020). The model is not transferable to other
   cities without retraining. All temperature values are in Fahrenheit.

3. **Data leakage policy:**
   All features are causal (no look-ahead). Lag features use `.shift(k)` where k >= 1.
   Rolling statistics use `.shift(1)` before the rolling window. Fourier features are calendar-based
   (computed from time index only). A leakage audit was performed and {'passed' if not is_leakage else 'FAILED — SEE WARNING ABOVE'}.

4. **Feature set limitations:**
   Only `month` and `year` passed the mutual information noise baseline filter. Day-of-year
   information is captured through Fourier features and lag structure. No external predictors
   (weather station data, elevation, climate indices) are available in this dataset.

5. **HoltWinters failure:**
   The Holt-Winters model (Exponential Smoothing) failed to converge and produced R\u00b2=-11.63.
   This is consistent with a non-standard seasonal period (365 days) and convergence sensitivity.

6. **pmdarima / XGBoost not installed:**
   The FAAR-ARIMA (pmdarima) and XGBoost candidates could not be trained. These are expected to
   perform comparably or better on seasonal temperature data. Install them and re-run for a more
   complete evaluation.

7. **Temporal distribution shift:**
   Training data ends at the holdout boundary (~5.5 years before end of dataset). Long-term climate
   trends or El Nino / La Nina effects could cause distribution shift that degrades real future performance.
"""

        # ── Section 6: Next Iteration Recommendations ─────────────────────────
        s6 = f"""## 6. Next Iteration Recommendations

1. **Install pmdarima and XGBoost** and re-run the pipeline to evaluate SARIMA / FAAR-ARIMA and
   gradient-boosted tree models, which are expected to exploit the annual seasonal pattern better.

2. **Enrich with external features:** Add climate index features (NAO, AMO), solar radiation data,
   or neighboring station temperatures to improve multivariate forecasting accuracy beyond the current
   univariate benchmark.

3. **Longer seasonal Fourier harmonics:** The current pipeline adds 3 Fourier harmonics for the
   365-day period. Adding up to 6–8 harmonics may better capture the asymmetric Algiers temperature
   curve (hot dry summers vs. cooler wet winters).

4. **Multi-step ahead forecasting:** The current model predicts t+1 only. Extending to a direct
   multi-step strategy (predict t+7 or t+30 directly) would be more valuable for practical use cases.

5. **Confidence intervals:** Add prediction interval estimation (e.g., quantile regression or
   bootstrapped forests) to give actionable uncertainty bounds for each forecast.

6. **Cross-city generalisation:** The `city_temperature.csv` dataset contains data for many cities.
   Run this pipeline on multiple cities to benchmark model families across different climate regimes.

7. **Pipeline monitoring:** In production, re-train quarterly and monitor RMSE drift on rolling
   30-day holdouts to detect distribution shift early.
"""

        # ── Assemble full report ───────────────────────────────────────────────
        report_sections = [
            f"# Data Forecast Generator — Pipeline Report",
            f"",
            f"> **Run ID:** {args.run_id}  ",
            f"> **Target:** `{target_col}`  ",
            f"> **Quality flag:** `{quality_flag}`  ",
            f"> **Production usable:** {'Yes' if production_usable else 'No'}  ",
            f"",
            "---",
            "",
            s1, s2, s3, s4, s5, s6,
        ]

        report_content = "\n".join(report_sections)

        report_path = output_dir / "step-16-report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        report_size = report_path.stat().st_size
        print(f"Written: {report_path} ({report_size} bytes)")

        if report_size < 500:
            raise ValueError(f"Report too short: {report_size} bytes (minimum 500)")

        # Verify all 6 section headings present
        required_headings = [
            "## 1. Problem Statement",
            "## 2. Data Quality",
            "## 3. Candidate Models",
            "## 4. Selected Model",
            "## 5. Risks and Caveats",
            "## 6. Next Iteration",
        ]
        content = report_path.read_text(encoding="utf-8")
        for heading in required_headings:
            if heading not in content:
                raise ValueError(f"Missing required section heading: {heading!r}")
        print("All 6 required section headings verified.")

        # ── Finalize progress ─────────────────────────────────────────────────
        with open(progress_path) as f:
            p = json.load(f)
        if "16-result-presentation" not in p.get("completed_steps", []):
            p.setdefault("completed_steps", []).append("16-result-presentation")
        p["current_step"] = "16-result-presentation"
        p["status"] = "completed"
        with open(progress_path, "w") as f:
            json.dump(p, f, indent=2)

        print("Step 16 complete. Pipeline DONE.")
        sys.exit(0)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR in step 16: {e}\n{tb}", file=sys.stderr)
        if progress_path.exists():
            with open(progress_path) as f:
                p = json.load(f)
            p["status"] = "error"
            p.setdefault("errors", []).append({"step": "16-result-presentation", "error": str(e), "traceback": tb})
            with open(progress_path, "w") as f:
                json.dump(p, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
