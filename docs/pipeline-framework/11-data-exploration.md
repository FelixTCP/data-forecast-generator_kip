# 11 Context Engineering: Data Exploration

## Objective

Generate a decision-ready profile that critically evaluates feature quality AND performs deep time-series profiling. The output of this step directly gates what enters the feature matrix in step 12 and informs the model selection strategy in downstream steps.

## Outputs

- univariate summary stats
- target candidacy signals
- **time-series profiling** (stationarity, white noise, trend, multiple-series detection)
- **mutual information (MI) ranking** of all features vs. target
- **pairwise correlation matrix** with redundancy flags
- **near-zero variance & high cardinality flags**
- **random-baseline comparison** for MI
- **leakage risk detection**
- `recommended_features` — the filtered list step 12 should use
- `model_recommendations` — baseline architectures suggested by the TS profile
- `client_facing_summary` — a non-technical text summary of findings

## Analysis Requirements

### 1. Near-Zero Variance & Cardinality Filter
- Compute variance for every numeric column. Flag any column with variance below `1e-4` (after min-max scaling) as `low_variance`.
- Compute unique value counts for categorical/string columns. Flag those with >50 unique values as `high_cardinality`.
- Both types contribute no useful information for standard models and must be excluded unless explicitly overridden.

### 2. Leakage Risk Detection
- Per the strict non-leakage policy: identify any feature that is essentially a duplicate or direct proxy of the target at time $t$.
- If a feature has a lag-0 cross-correlation with the target > `0.98`, or an anomalously high MI score, flag it as `leakage_suspect` and add it to `excluded_features`.

#### 3. Mutual Information Ranking
- Compute `mutual_info_regression(X, y)` for all numeric features vs. the target (use `sklearn.feature_selection`).
- Also compute MI for **5 fresh random-noise columns** (standard normal, same row count) as a baseline.
- Flag any real feature whose MI score falls **at or below the average noise MI** as `below_noise_baseline`.
- Sort all features by MI descending and include the ranking in the output JSON.
- Features ranked at or below noise baseline are **not recommended** for step 12.

#### 4. Pairwise Correlation & Redundancy
- Compute the Pearson correlation matrix for all numeric features.
- For any pair with |correlation| ≥ 0.90, flag the one with the **lower MI with the target** as `redundant`.
- Redundant features are **not recommended** for step 12.

#### 5. Time-Series Profiling (when time column detected)
- **Multiple Series Check:** Detect if the dataset contains multiple independent time series (e.g., grouped by a categorical ID).
- **Stationarity:** Compute the Augmented Dickey-Fuller (ADF) test on the target column. Record if it is stationary (p < 0.05) or non-stationary.
- **White Noise Check:** Use the Ljung-Box test to flag if the target series is essentially a Random Walk / White Noise.
- **Components:** Decompose the series to flag if significant `trend` or `seasonality` is present.
- **Model Recommendations:** Based on ADF and Seasonality results, append 2-3 recommended model families (e.g., "SARIMA" for seasonal/non-stationary, "XGBoost" for highly non-linear multiple series, "Naive" for White Noise).

#### 6. Time-Series Lag Analysis
- Compute autocorrelation of the target at lags 1–24 (or up to N/4 if series is short).
- Identify lags where autocorrelation exceeds 0.1 and flag them as `significant_lags`.
- Compute cross-correlation of each feature with the target at lags 0, 1, 2, 3.
- Flag feature-lag combinations where cross-correlation exceeds 0.15 as `useful_lag_features`.
- This directly informs which lags to build in step 12.

#### 7. Client-Facing Summary
- Generate a concise, non-technical text string (`client_facing_summary`) summarizing the time-series behavior (trend/seasonality) and justifying dropped features (e.g., "Temperature is highly predictive, but sensor_v2 was identified as random noise and removed").


## Output JSON Keys

```json 
{
  "step": "11-data-exploration",
  "shape": {"rows": 19735, "columns": 29},
  "numeric_columns": ["t1", "t6", "rh_6", "lights", "t_out", "rv1", "rv2"],
  "high_cardinality": [],
  "low_variance_columns": [],
  "time_series_detected": true,
  "time_column": "date",
  "multiple_series_detected": false,
  "time_series_characteristics": {
    "trend_detected": true,
    "seasonality_detected": true,
    "stationarity": "non-stationary",
    "white_noise": false
  },
  "model_recommendations": ["SARIMA", "Prophet", "XGBoost"],
  "mi_ranking": [
    {"feature": "t6", "mi_score": 0.42, "below_noise_baseline": false},
    {"feature": "rv1", "mi_score": 0.003, "below_noise_baseline": true}
  ],
  "noise_mi_baseline": 0.005,
  "redundant_columns": ["rv2"],
  "correlation_matrix_summary": {"max_pair": ["rv1","rv2"], "max_corr": 1.0},
  "significant_lags": [1, 3, 6],
  "useful_lag_features": [{"feature": "t1", "lag": 1, "xcorr": 0.23}],
  "recommended_features": ["t6", "t1", "rh_6", "lights", "t_out"],
  "excluded_features": {"rv1": "below_noise_baseline", "rv2": "redundant", "target_copy": "leakage_suspect"},
  "target_candidates": [],
  "client_facing_summary": "Your target variable shows a strong trend and seasonal patterns. Features like t6 and t1 are highly predictive, whereas rv1 and rv2 were excluded due to being redundant or pure noise.",
  "context": {}
}
```

## Guardrails

- `recommended_features` must never be empty. If all features fail the filters, loosen the noise-baseline threshold by 50% and log a warning.
- MI computation is stochastic — set `random_state=42`.
- Log the count of features dropped at each filter stage.
- Do not silently pass `recommended_features = all_features`; every exclusion must be logged with a reason in `excluded_features`.
- Time-series checks (ADF, Ljung-Box) must handle NaN values gracefully (e.g., via imputation or dropping) before computation.
- **Polars to Pandas:** Convert the `pl.DataFrame` to `pandas` or `numpy` before passing data into `sklearn` or `statsmodels` functions.

## Copilot Prompt Snippet
```markdown
Implement `step_11_exploration.py`. The CLI receives `--output-dir` and `--run-id`. 
Read the target column name from `OUTPUT_DIR/progress.json`. Load `OUTPUT_DIR/cleaned.parquet` using `polars`, but convert to `pandas` before passing arrays to `statsmodels` and `sklearn`.
Include: near-zero variance, high cardinality check, MI ranking vs. target with random-noise baseline comparison, pairwise correlation redundancy detection (|r| >= 0.90), and a strict lag-0 leakage check (flag xcorr > 0.98).
For time series: Compute ADF stationarity test, White Noise check (Ljung-Box), Trend/Seasonality flags, and cross-correlation analysis (lags 1–24) using `statsmodels` and `scipy`.
Return the strictly formatted JSON including `recommended_features`, `time_series_characteristics`, and a generated `client_facing_summary`.
```

## Tests

- all features fail noise baseline (should loosen threshold, not return empty list)
- perfectly correlated pair (one should be flagged redundant)
- dataset consisting of pure random walk (white noise flag should trigger)
- dataset with an exact copy of the target column (leakage check should catch it)
- tiny dataset edge case (fewer than 50 rows, tests should degrade gracefully)