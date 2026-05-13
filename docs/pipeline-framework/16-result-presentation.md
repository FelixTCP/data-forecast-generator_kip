# #16 Context Engineering: Result Presentation

## Objective

Produce user-facing outputs that synthesize the entire 6-step pipeline (steps 10–15) into actionable insights. Outputs must serve both technical stakeholders (who need model diagnostics and leakage flags) and business stakeholders (who need confidence levels and next-step recommendations).

**Critical:** If upstream quality gates indicate `leakage_suspected`, `subpar`, or `no_viable_candidate`, the report must clearly state that metrics are invalid for production forecasting and must include remediation actions.

---

## Inputs

- `step-14-evaluation.json` — full candidate evaluation + `quality_assessment` flag
- `step-15-selection.json` — selected model, weighted score, ranking, quality_flag
- `step-15-model-selection-metrics.png` — model comparison visualization
- `step-15-model-selection-report.md` — technical ranking and rationale
- `step-11-data-exploration.json` — feature recommendations, time-series characteristics
- `step-13-training.json` — training history, benchmark comparisons
- `cleaned.parquet` — final dataset from step 10
- Fitted model artifacts (`.joblib` files from step 13)

---

## Outputs

### Mandatory Outputs

1. **`evaluation.json`** — Machine-readable summary:
   ```json
   {
     "step": "16-result-presentation",
     "execution_id": "uuid",
     "timestamp": "ISO8601",
     "quality_flag": "acceptable|marginal|subpar|subpar_after_expansion|no_viable_candidate|leakage_suspected",
     "production_usable": true|false,
     "selected_model": "model_name_or_null",
     "selected_model_r2": 0.XX,
     "selected_model_rmse": XX.X,
     "selected_model_mae": XX.X,
     "dataset_summary": {
       "rows": 19735,
       "columns": 29,
       "target_column": "appliances",
       "time_column": "date",
       "time_series_detected": true
     },
     "data_quality": {
       "missing_rate": 0.0,
       "outliers_flagged": 0,
       "features_recommended": ["t6", "t1", "rh_6"],
       "features_excluded": {"rv1": "below_noise_baseline", "rv2": "redundant"}
     },
     "model_performance": {
       "best_r2": 0.XX,
       "best_rmse": XX.X,
       "best_mae": XX.X,
       "naive_baseline_r2": 0.XX,
       "arima_benchmark_r2": 0.XX,
       "kmeans_benchmark_r2": 0.XX,
       "model_beats_benchmarks": true|false
     },
     "time_series_characteristics": {
       "trend": "detected|absent",
       "seasonality": "detected|absent",
       "stationarity": "stationary|non-stationary",
       "recommended_architectures": ["SARIMA", "Prophet", "XGBoost"],
       "significant_lags": [1, 3, 6]
     },
     "quality_warnings": [
       "benchmark_warning: Best model barely beats ARIMA baseline",
       "leakage_warning: Feature X has 0.98 correlation with target at lag-0"
     ],
     "next_steps": [
       "Collect more training data to improve feature coverage",
       "Engineer cyclical features for detected seasonality"
     ]
   }
   ```

2. **`step-16-report.md`** — Human-readable technical report (see sections below)

3. **`step-16-artifacts.json`** — Artifact registry:
   ```json
   {
     "model_path": "model.joblib",
     "model_type": "GradientBoostingRegressor",
     "preprocessor_path": "preprocessor.joblib",
     "feature_names": ["t6", "t1", "rh_6", "lights", "t_out"],
     "dataset_snapshot": {
       "parquet_path": "cleaned.parquet",
       "row_count": 19735,
       "column_count": 5
     },
     "evaluation_metrics": {
       "holdout_r2": 0.52,
       "holdout_rmse": 71.3,
       "holdout_mae": 43.2,
       "cv_mean_r2": 0.51,
       "cv_std_r2": 0.08
     },
     "training_metadata": {
       "target_column": "appliances",
       "time_column": "date",
       "time_split_ratio": 0.8,
       "random_state": 42
     },
     "production_deployment": {
       "is_production_ready": false,
       "quality_gate_passed": false,
       "issues": ["marginal: R² in [0.25, 0.50)"]
     }
   }
   ```

### Optional Outputs

- **`step-16-residuals-plot.png`** — Residual distribution + time-series residuals
- **`step-16-feature-importance.png`** — Top feature contributions (if model supports)
- **`step-16-predictions-overlay.png`** — Predicted vs. actual on holdout set

---

## Report Sections (Markdown)

### 1. Executive Summary

**For business stakeholders: 2–3 sentences maximum.**

```markdown
## Executive Summary

This analysis identified **[model_name]** as the best-performing regression model for predicting 
**[target_column]** from your **[source_csv]** dataset. The model achieves **[R²]** accuracy on 
holdout data, outperforming naive statistical baselines. 

[Quality Flag Disclosure]:
- [PASS] **Production-Ready** if `quality_flag == acceptable`
- [WARNING] **Marginal Performance** if `quality_flag == marginal` — proceed with caution, collect more data
- [FAIL] **Not Recommended** if `quality_flag == subpar`, `no_viable_candidate`, or `leakage_suspected`
```

### 2. Problem Statement & Data Overview

**Include:**
- Target variable definition and units
- Data shape (rows, columns, date range)
- Key data quality issues discovered in step 10 (missing values, outliers, normalization applied)
- Time-series characteristics from step 11 (frequency, trend, seasonality, stationarity)

```markdown
## 2. Problem Statement & Data Overview

**Target Variable:** [Name] ([Units])  
**Data Source:** [CSV filename] ([Date range])  
**Dataset Size:** [Rows] observations, [Columns] features after cleaning

### Data Quality Findings (Step 10)
- Missing values: [Rate]% (fixed: [Method])
- Detected outliers: [Count] (treatment: [None/Capped/Removed])
- Column normalization: Applied min-max scaling to numeric features

### Time-Series Characteristics (Step 11)
- **Frequency:** [e.g., 10-minute intervals]
- **Stationarity:** Non-stationary (ADF p-value = [X])
- **Trend:** Strong upward trend detected
- **Seasonality:** 24-hour and 7-day cycles detected
- **Recommended architectures:** SARIMA, Prophet, XGBoost

**Client-facing summary:** [From step 11 output]
```

### 3. Feature Selection & Engineering

**Include:**
- Recommended features from step 11 (ranked by mutual information)
- Excluded features and reasons (low variance, high cardinality, leakage, redundancy)
- Lag features and rolling statistics engineered in step 12
- Any data transformations or scaling applied

```markdown
## 3. Feature Selection & Engineering (Step 12)

### Recommended Features (by Mutual Information)
| Feature | MI Score | Status |
|---------|----------|--------|
| t6 | 0.42 | [PASS] Selected |
| t1 | 0.35 | [PASS] Selected |
| rh_6 | 0.28 | [PASS] Selected |
| lights | 0.18 | [PASS] Selected |
| t_out | 0.15 | [PASS] Selected |

### Excluded Features & Reasons
| Feature | Reason | MI Score |
|---------|--------|----------|
| rv1 | Below noise baseline | 0.003 |
| rv2 | Redundant (corr=0.99 with t1) | 0.004 |
| target_copy | Leakage suspect (lag-0 corr=0.98) | — |

### Feature Engineering Applied
- Rolling mean (lag-1, window=3, 6, 24 hours)
- Lag-1, Lag-3, Lag-6 features (causal: computed after shift to prevent look-ahead)
- Hour-of-day cyclical encoding (sin/cos for 24h cycle)

**Leakage Safeguards:**
- All rolling statistics computed after `.shift(1)` to prevent forward-looking bias
- Pairwise Pearson correlation and reconstruction probe (RF R²) run to detect target proxies
```

### 4. Model Training & Candidate Comparison

**Include:**
- Mandatory benchmarks (ARIMA, k-means, naive persistence) from step 13
- Full candidate roster with CV and holdout scores
- Cross-validation stability metrics (mean R² ± std)
- Benchmark deltas (improvement over baselines)

```markdown
## 4. Model Training & Candidate Comparison (Steps 13–14)

### Mandatory Benchmarks
| Baseline | Holdout R² | Holdout RMSE | Holdout MAE | Role |
|----------|------------|--------------|-------------|------|
| Naive Lag-1 | 0.35 | 92.1 | 58.3 | Persistence baseline |
| ARIMA(p,d,q) | 0.41 | 85.2 | 52.1 | Classical TS baseline |
| K-Means (k=5) | 0.38 | 89.4 | 55.6 | Cluster centroid baseline |

### Candidate Models Trained
| Model | CV R² (mean±std) | Holdout R² | Holdout RMSE | Holdout MAE | Delta vs ARIMA | Status |
|-------|-----------------|------------|--------------|-------------|-----------------|--------|
| **Ridge** | 0.45±0.06 | 0.48 | 75.3 | 46.2 | +0.07 | [WARNING] Marginal |
| **Gradient Boosting** | 0.52±0.05 | 0.52 | 71.3 | 43.2 | +0.11 | [PASS] Selected |
| **Random Forest** | 0.49±0.08 | 0.50 | 73.1 | 44.8 | +0.09 | [PASS] Good |
| **Extra Trees** | 0.48±0.07 | 0.49 | 74.5 | 45.3 | +0.08 | [PASS] Good |
| **LightGBM** | 0.50±0.06 | 0.51 | 72.1 | 44.1 | +0.10 | [PASS] Good |

### Model Selection Criteria (Step 15)

Weighted Score = 50% R² + 25% (1 - RMSE_norm) + 15% (1 - MAE_norm) + 10% CV Stability

**Gradient Boosting** scored highest (0.78 weighted score) due to:
- Highest holdout R² (0.52) — **7 points above ARIMA baseline**
- Lower RMSE than competitors by 1.8 units
- Stable CV performance (std=0.05, <10% coefficient of variation)

### Expansion Round Result (if applicable)
*Only if step 14 quality_assessment was "subpar" — describe additional candidates trained and their results.*
```

### 5. Quality Assessment & Leakage Probe

**CRITICAL: Always include this section if quality_flag ≠ "acceptable"**

```markdown
## 5. Quality Assessment & Risk Disclosure

### Quality Assessment Result
- **Flag:** [acceptable | marginal | subpar | subpar_after_expansion | no_viable_candidate | leakage_suspected]
- **Production Usable:** [Yes | No]

### Quality Thresholds
| Condition | Threshold | Result |
|-----------|-----------|--------|
| Best R² ≥ 0.50 | Acceptable | [FAIL] FAIL (R² = [X]) |
| Best R² ∈ [0.25, 0.50) | Marginal | [PASS] PASS |
| Best R² < 0.25 | Subpar | [PASS] PASS |
| All R² < 0 | No viable | [PASS] PASS |

### Leakage Probe Results (Triggered: Yes/No)
*If R² > 0.98 or suspiciously perfect:*

| Probe | Status | Finding |
|-------|--------|---------|
| Leakage stress test (remove target-derived features) | [PASS] PASS | R² drops to 0.48 — no trivial reconstruction |
| Reconstruction RF probe | [PASS] PASS | RF R² = 0.52 (< 0.999) — no target proxy detected |

**Conclusion:** No leakage detected. Results are valid for exploratory forecasting.

### Warnings & Caveats
- [WARNING] **Benchmark Warning:** Best model barely exceeds ARIMA baseline (+0.07 R²). Consider collecting more features or data.
- [WARNING] **CV Instability:** Ridge model shows high fold variance (std=0.10). May overfit on specific time windows.
- [INFO] **Stationarity:** Target is non-stationary (ADF p-value=0.23). Differencing or trend-removal may improve future iterations.
```

### 6. Selected Model Details

**Include:**
- Model hyperparameters
- Feature list in order of importance (if available)
- Interpretability notes
- Expected error magnitude on new data

```markdown
## 6. Selected Model: Gradient Boosting Regressor

### Model Configuration
```
GradientBoostingRegressor(
  n_estimators=200,
  learning_rate=0.05,
  max_depth=5,
  min_samples_split=10,
  subsample=0.8,
  random_state=42
)
```

### Feature Importance (Top 10)
| Rank | Feature | Importance | Impact |
|------|---------|------------|--------|
| 1 | t6_lag1 | 0.28 | Previous hour temperature — strong predictor |
| 2 | hour_sin | 0.18 | Cyclical daily pattern |
| 3 | rh_6_lag3 | 0.15 | Humidity 3 hours ago |
| 4 | t1_rolling_mean_6 | 0.12 | 6-hour moving average temperature |
| 5 | lights | 0.10 | Current lighting usage |

### Performance Summary
- **Holdout R²:** 0.52 (explains 52% of target variance)
- **Holdout RMSE:** 71.3 Wh (±1 standard error)
- **Holdout MAE:** 43.2 Wh (typical error magnitude)
- **CV Stability:** R² = 0.52 ± 0.05 (low variance → good generalization)

### Interpretability
- Model relies on recent history (lag-1) and daily patterns (hour_sin).
- Ensemble of gradient boosted trees provides non-linear interactions (e.g., temperature × hour).
- Not directly interpretable per sample, but global feature importance is available above.
```

### 7. Risk & Limitations

```markdown
## 7. Risk & Limitations

### Model Assumptions
- Assumes historical patterns continue in the future (stationarity of mechanisms, not values).
- Requires regular retraining as seasonal patterns evolve.
- Performance degrades if new feature ranges appear (e.g., extreme weather).

### Residual Analysis
- Residual mean: [X] (acceptable < 1)
- Residual max absolute error: [Y] Wh
- Residuals show slight heteroscedasticity (errors larger at high target values).
- Time-series plot of residuals shows no obvious systematic drift.

### Out-of-Distribution Risks
- Model trained on [date range]. Validity beyond this range is unverified.
- Extreme weather events, structural changes (e.g., renovations) may degrade accuracy.
- New appliances or behavioral changes not seen in training data will worsen predictions.

### Known Issues from Upstream Steps
- Feature rv1 and rv2 were excluded as uninformative — consider investigating root causes.
- Non-stationarity in target may benefit from trend-removal in future iterations.
```

### 8. Deployment Readiness & Next Steps

```markdown
## 8. Deployment Readiness & Recommendations

### Production Deployment Decision
- **Status:** [[READY] Ready | [CONDITIONAL] Conditional | [NOT RECOMMENDED] Not Recommended]
- **Reason:** [Based on quality_flag]

#### If Conditional (marginal):
1. Set up automated retraining every [30/90] days to adapt to seasonal drifts.
2. Implement prediction confidence intervals (e.g., bootstrap/quantile regression) for uncertainty quantification.
3. Monitor prediction errors in production; retrain if RMSE drifts > 20%.

#### If Not Recommended (subpar / no_viable_candidate):
1. **Feature Engineering:** Collect additional domain features (e.g., occupancy, weather forecasts).
2. **Data Quality:** Investigate missing features or systematic data issues in step 10/11 output.
3. **Architecture:** Try more expressive models (neural networks, specialized time-series models).
4. **Expansion:** Re-run steps 14–15 with expanded candidate roster.

### Recommended Next Iterations

**High Priority:**
- Incorporate external weather data (temperature, humidity, solar irradiance).
- Engineer occupancy proxy features (e.g., working hours vs. weekends).
- Test adaptive forecasting (update model weights daily on recent data).

**Medium Priority:**
- Implement prediction intervals (quantile regression or conformal prediction).
- Explore hierarchical forecasting if multiple buildings/zones exist.

**Low Priority:**
- Experiment with deep learning (LSTM) if data volume grows beyond [50k] rows.
- Implement explainability dashboards (SHAP values per prediction).

### Artifact Registry

All outputs from this run are archived:
- **Model:** `[output_dir]/model.joblib` — Production model (if applicable)
- **Preprocessor:** `[output_dir]/preprocessor.joblib` — Feature scaling + lag engineering
- **Dataset:** `[output_dir]/cleaned.parquet` — Final training data snapshot
- **Report:** `[output_dir]/step-16-report.md` — This report
- **Evaluation:** `[output_dir]/evaluation.json` — Machine-readable summary
- **Plots:** `[output_dir]/step-16-*.png` — Model comparison, residuals, importance plots
```

---

## Copilot Prompt Snippet

```markdown
Implement `build_result_package(context: PipelineContext, output_dir: str)`.

INPUT:
- Step 14 evaluation.json (candidates + quality_assessment)
- Step 15 selection.json (selected model + weighted_score + quality_flag)
- Fitted model and preprocessor artifacts from step 13
- Step 11 exploration output (feature ranking, TS characteristics)
- Cleaned dataset from step 10

OUTPUT:
1. Write evaluation.json (machine-readable summary with quality_flag, production_usable, benchmarks).
2. Write step-16-report.md with all 8 sections above.
3. Generate step-16-residuals-plot.png (histogram + time-series autocorr).
4. Generate step-16-feature-importance.png (top 10 features if model supports).
5. Write step-16-artifacts.json (artifact registry).

MANDATORY LEAKAGE DISCLOSURE:
- If quality_flag == "leakage_suspected", prominently state: 
  "[WARNING] CRITICAL: Leakage detected. Metrics are INVALID. Do not deploy. Revise feature engineering (step 12)."
- If quality_flag == "subpar" or "no_viable_candidate", state:
  "[WARNING] WARNING: This model is NOT recommended for production. Expand features or data volume and re-run steps 12–15."
- If quality_flag == "marginal", state:
  "[WARNING] CAUTION: Marginal performance (R² in [0.25, 0.50)). Proceed with automated retraining and prediction uncertainty quantification."

QUALITY FLAG IMPLICATIONS:
- acceptable → production_usable = true
- marginal → production_usable = true (with caveats)
- subpar, no_viable_candidate, leakage_suspected → production_usable = false

Return paths to all generated artifacts.
```

---

## Key Conventions for Your Pipeline

### Quality Flag Definitions
| Flag | R² Threshold | Deployment | Interpretation |
|------|-------------|-----------|-----------------|
| **acceptable** | ≥ 0.50 | [PASS] Ready | Model outperforms benchmarks, acceptable for production |
| **marginal** | [0.25, 0.50) | [WARNING] Conditional | Model works but with caveats; automate retraining |
| **subpar** | [0, 0.25) | [FAIL] Not ready | Feature engineering or data expansion needed |
| **subpar_after_expansion** | [0, 0.25) after expansion | [FAIL] Not ready | Expansion round did not improve enough |
| **no_viable_candidate** | All R² < 0 | [FAIL] Not ready | All models worse than mean baseline — restart at step 11/12 |
| **leakage_suspected** | — | [FAIL] HARD STOP | Leaked features detected — revise step 12 immediately |

### Benchmark Comparisons
Always report deltas:
- `delta_r2_vs_arima` = selected_r2 - arima_r2
- `delta_r2_vs_kmeans` = selected_r2 - kmeans_r2
- `delta_r2_vs_naive` = selected_r2 - naive_r2

If deltas < 0.02 for all benchmarks → `benchmark_warning = true` in evaluation.json

### Time-Series Disclosure
When time-series characteristics are detected, include:
- Stationarity status (impacts residual interpretation)
- Significant lags (may suggest AR(p) or lag-based models)
- Seasonality (may require trend-removal or seasonal differencing)
- Multiple-series flag (indicates hierarchical forecasting opportunity)

---

## Guardrails

[PASS] **Always:**
- Include evaluation.json with full candidate ranking and quality_flag
- Include step-16-report.md with all 8 sections (even if model is not production-ready)
- Document all quality warnings and leakage probe results
- Provide next-steps recommendations tailored to the quality_flag
- Archive all artifacts with timestamps and execution IDs

[FAIL] **Never:**
- Proceed to deployment if quality_flag = leakage_suspected
- Call a model "production-ready" if quality_flag < acceptable
- Omit benchmark comparisons from the report
- Silently ignore expansion round results (if step 14 triggered expansion, report final quality_flag state)
- Present RMSE without context (always include target std for comparison)

---

## Optional Enhancements

### For Business Stakeholders
- Cost-benefit analysis: "At [RMSE] error, deployment saves approximately $X/year vs. manual estimation."
- Confidence intervals: "Predictions are 90% likely to be within ±[Y] units."

### For Operational Use
- Retraining trigger: "If holdout RMSE on recent [30-day] data exceeds [X], retrain automatically."
- Monitoring dashboard: Links to production monitoring (if available).

### For Regulators (if applicable)
- Data retention policy: Model trained on [dates]; data retention for [duration].
- Fairness assessment: Model performance across [demographic groups] (if applicable).
