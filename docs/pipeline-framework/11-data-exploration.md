# 11 Context Engineering: Data Exploration

## Objective

Produce a rigorous, decision-ready time-series profile that governs every downstream modelling choice. This step runs a full statistical characterisation of the target and all features — stationarity, memory, autocorrelation structure, seasonality, noise — and translates those diagnostics directly into a **model class recommendation** that step 13 must honour. Feature quality gates (MI, redundancy, leakage) are applied here and the filtered feature list is passed to step 12.

## Outputs

- univariate summary stats for all columns
- target candidacy signals
- **full stationarity battery** (ADF + KPSS with joint interpretation)
- **ACF / PACF profiles** with significant lag extraction (up to min(48, N/4))
- **Hurst exponent** via classical R/S rescaled-range analysis
- **white-noise test** (Ljung-Box)
- **seasonal decomposition** (STL or classical additive/multiplicative)
- **mutual information (MI) ranking** vs. target with random-noise baseline
- **pairwise correlation matrix** with redundancy flags
- **near-zero variance & high cardinality flags**
- **lag-0 leakage detection**
- `recommended_features` — filtered feature list for step 12
- `ts_diagnostics` — structured stationarity / memory / seasonality profile
- `model_class_recommendations` — model families explicitly justified by diagnostics
- `acf_pacf_orders` — suggested AR order p and MA order q from ACF/PACF shape
- `client_facing_summary` — non-technical narrative

## Analysis Requirements

### 1. Near-Zero Variance & Cardinality Filter
- Compute variance for every numeric column. Flag any column with variance below `1e-4` (after min-max scaling) as `low_variance`.
- Compute unique value counts for categorical/string columns. Flag those with >50 unique values as `high_cardinality`.
- Both types are excluded from `recommended_features` unless explicitly overridden by the user.

### 2. Lag-0 Leakage Detection (Hard Gate)
- Per the strict non-leakage policy: any feature whose lag-0 Pearson correlation with the target exceeds `0.98`, or whose MI score is anomalously high relative to all other features, is flagged as `leakage_suspect` and added to `excluded_features`.
- This is a hard gate — leakage suspects must never appear in `recommended_features`.

### 3. Mutual Information Ranking
- Compute `mutual_info_regression(X, y)` for all numeric features vs. the target (`sklearn.feature_selection`, `random_state=42`).
- Compute MI for **5 fresh standard-normal noise columns** (same row count) as a random baseline.
- Flag any feature whose MI ≤ average noise MI as `below_noise_baseline`. Such features are excluded from `recommended_features`.
- Sort by MI descending and include full ranking in output.

### 4. Pairwise Correlation & Redundancy
- Compute Pearson correlation matrix for all numeric features.
- For any pair with |r| ≥ 0.90, flag the member with the **lower MI vs. target** as `redundant`. Redundant features are excluded from `recommended_features`.

---

## Time-Series Diagnostic Battery

All checks in this section operate on the **raw target series** (after null imputation or row-drop, before any feature engineering). All NaN values must be handled before any statistical test.

### 5. Stationarity Battery — ADF + KPSS with Joint Interpretation

Run **both** the Augmented Dickey-Fuller (ADF) test and the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test on the target series. Do not rely on either test alone.

Use the following joint-interpretation matrix to classify the series:

| ADF result | KPSS result | Joint conclusion | Recommended action |
|---|---|---|---|
| Reject H₀ (p < 0.05) | Fail to reject H₀ (p > 0.05) | **Stationary** | Use raw series; AR/MA/ARMA/linear models directly applicable |
| Fail to reject H₀ (p ≥ 0.05) | Reject H₀ (p ≤ 0.05) | **Non-stationary** | Apply differencing (d=1 or d=2); ARIMA, SARIMA, ETS indicated |
| Reject H₀ (p < 0.05) | Reject H₀ (p ≤ 0.05) | **Trend-stationary** | Detrend (polynomial or linear); use differenced series |
| Fail to reject H₀ (p ≥ 0.05) | Fail to reject H₀ (p > 0.05) | **Ambiguous / possibly fractionally integrated** | Compute Hurst exponent; consider fractional differencing |

Record the ADF statistic, ADF p-value, KPSS statistic, KPSS p-value, and the derived `stationarity_conclusion` string in `ts_diagnostics`.

### 6. ACF / PACF Analysis

- Compute the autocorrelation function (ACF) at lags 1 through `min(48, N/4)`.
- Compute the partial autocorrelation function (PACF) at the same lags.
- Extract `acf_significant_lags`: lags where |ACF| > 2/√N (95% confidence band).
- Extract `pacf_significant_lags`: lags where |PACF| > 2/√N.
- Derive suggested orders for classical models:
  - If ACF decays geometrically and PACF cuts off sharply at lag `p` → AR(p) structure → set `suggested_ar_order = p`
  - If PACF decays geometrically and ACF cuts off sharply at lag `q` → MA(q) structure → set `suggested_ma_order = q`
  - If both decay → ARMA(p,q) structure
  - If ACF shows a slow exponential decay pattern → differencing likely needed (consistent with non-stationarity)
- Write `acf_values`, `pacf_values`, `acf_significant_lags`, `pacf_significant_lags`, `suggested_ar_order`, `suggested_ma_order` to `ts_diagnostics`.

### 7. Hurst Exponent (R/S Rescaled-Range Analysis)

Compute the classic Hurst exponent H via rescaled-range (R/S) analysis on the target series:

1. Divide the series into non-overlapping windows of sizes in `[8, 16, 32, 64, 128, ...]` (up to N/4).
2. For each window size n: compute `R(n)/S(n)` = (range of mean-adjusted cumulative sum) / (std dev of the sub-series).
3. Fit `log(R/S)` vs. `log(n)` by OLS; the slope is the Hurst exponent H.

Interpret H:
| H range | Interpretation | Implication |
|---|---|---|
| H ∈ (0, 0.45) | Anti-persistent / mean-reverting | Short-memory; short lags sufficient; AR(1) / MA(1) plausible |
| H ∈ (0.45, 0.55) | Random walk / no persistent memory | Naive baseline competitive; ARIMA(0,1,0) as baseline |
| H ∈ (0.55, 0.75) | Mildly persistent / long memory | Moderate lag window; ARIMA with higher p; XGBoost with many lags |
| H ∈ (0.75, 1.0) | Strongly persistent / trending | Long lag window useful; SARIMA, ETS, FAAR; fractional differencing possible |

Record `hurst_exponent`, `hurst_interpretation` string, and `hurst_r2_fit` (R² of the log-log OLS fit as quality indicator) in `ts_diagnostics`.

### 8. White Noise Check (Ljung-Box)

- Run the Ljung-Box test at lags `[6, 12, 24]` (or up to `min(24, N/5)`).
- Record `ljung_box_pvalues` (dict of lag → p-value).
- If **all** tested lags have p > 0.05: set `white_noise = true`. A white-noise target cannot be forecast meaningfully — emit a prominent warning and recommend only naive/seasonal-naive benchmarks.
- If any lag has p ≤ 0.05: set `white_noise = false` — autocorrelation structure is exploitable.

### 9. Seasonal Decomposition & Period Detection

- Detect the dominant sampling frequency from the time column (e.g., `10min`, `hourly`, `daily`).
- Derive candidate seasonal periods from frequency:
  - `10min` → [6, 12, 36, 144, 1008] (1h, 2h, 6h, 24h, 1-week in 10-min ticks)
  - `hourly` → [24, 168] (day, week)
  - `daily` → [7, 30, 365]
  - `monthly` → [12]
- For each candidate period m: compute the seasonal strength statistic from STL decomposition: `Fs = max(0, 1 - Var(residual) / Var(seasonal + residual))`. Flag as significant if Fs > 0.30.
- Record `detected_periods` as a list of `{"period": m, "seasonal_strength": Fs, "significant": bool}`.
- Set `trend_strength` similarly from STL: `Ft = max(0, 1 - Var(residual) / Var(trend + residual))`. Set `trend_detected = true` if `Ft > 0.30`.
- Store `primary_seasonal_period` = the period with the highest Fs among significant ones (or null if none).

### 10. Cross-Correlation Lag Analysis

- For each feature in `recommended_features`, compute Pearson cross-correlation with the target at lags 0, 1, 2, 3, 6, 12 (where lag k means the feature value at t-k correlated with target at t).
- Lag-0 cross-correlation > 0.98 → hard-flag as `leakage_suspect` (Section 2 above).
- For lags ≥ 1: flag feature-lag pairs where |xcorr| > 0.15 as `useful_lag_features`.
- Record all useful lag pairs in `useful_lag_features` — step 12 uses this list to build concrete lag columns.

### 11. Multiple Series Detection

- If the dataset contains a categorical column with low cardinality (≤ 50 unique values) alongside the time column, check whether the time ranges per group overlap significantly. If they do, classify as `multiple_series_detected = true` and record the grouping column in `series_id_column`.
- Multiple-series data changes the recommended model class (panel models, per-series ARIMA, XGBoost with series-ID encoding).

---

## Model Class Selection Logic

After running all diagnostics, derive `model_class_recommendations` using the following decision table. **All applicable rows apply simultaneously** — list all recommended classes, not just the first match.

| Condition | Recommended Model Classes |
|---|---|
| `white_noise = true` | `Naive`, `SeasonalNaive` only — warn prominently |
| `stationarity = stationary` AND `white_noise = false` | `AR`, `MA`, `ARMA`, `ElasticNet`, `Ridge`, `XGBoost` |
| `stationarity = non-stationary` AND `seasonality_detected = false` | `ARIMA`, `ElasticNet`, `XGBoost`, `FAAR-ARIMA` |
| `stationarity = non-stationary` AND `seasonality_detected = true` | `SARIMA`, `SARIMAX`, `HoltWinters`, `FAAR-SARIMAX`, `XGBoost` |
| `stationarity = trend-stationary` | `ARIMA` (with detrend), `ETS`, `XGBoost` with differenced target |
| `hurst_exponent > 0.65` | `FAAR-ARIMA`, `FAAR-SARIMAX`, `XGBoost` with extended lag window (p up to 48) |
| `hurst_exponent < 0.45` (mean-reverting) | `AR(1)`, `ElasticNet` (short lag window), `SVR` |
| `multiple_series_detected = true` | `XGBoost` / `LightGBM` with series-ID encoding, `Factor-VAR` |
| `seasonality_detected = true` (any period) | `HoltWinters-ETS`, `TBATS` (if multiple seasonal periods), `SARIMA` |
| Multivariate features available AND `n_recommended_features ≥ 3` | `SARIMAX`, `Factor-VAR`, `FAAR-SARIMAX`, `ElasticNet`, `XGBoost` |
| Large dataset (`N > 10000`) AND many features (`k > 5`) | `XGBoost`, `LightGBM`, `HistGradientBoosting`, `FAAR-ARIMA` |

Record `model_class_recommendations` as a list of `{"model_class": str, "justification": str}` objects, each with a one-line reason derived from the diagnostics.

---

## Output JSON Schema

```json
{
  "step": "11-data-exploration",
  "shape": {"rows": 19735, "columns": 29},
  "numeric_columns": ["t1", "t6", "rh_6", "lights", "t_out", "rv1", "rv2"],
  "high_cardinality": [],
  "low_variance_columns": ["rv_noise"],
  "time_series_detected": true,
  "time_column": "date",
  "multiple_series_detected": false,
  "series_id_column": null,
  "detected_frequency": "10min",

  "ts_diagnostics": {
    "adf_statistic": -3.21,
    "adf_pvalue": 0.019,
    "kpss_statistic": 0.45,
    "kpss_pvalue": 0.041,
    "stationarity_conclusion": "non-stationary",

    "acf_values": [1.0, 0.72, 0.61, 0.54],
    "pacf_values": [1.0, 0.72, 0.11, 0.08],
    "acf_significant_lags": [1, 2, 3, 6, 12, 24],
    "pacf_significant_lags": [1, 2],
    "suggested_ar_order": 2,
    "suggested_ma_order": 0,

    "hurst_exponent": 0.71,
    "hurst_interpretation": "mildly_persistent",
    "hurst_r2_fit": 0.96,

    "ljung_box_pvalues": {"6": 0.0001, "12": 0.00003, "24": 0.00001},
    "white_noise": false,

    "trend_strength": 0.41,
    "trend_detected": true,
    "detected_periods": [
      {"period": 144, "seasonal_strength": 0.58, "significant": true},
      {"period": 1008, "seasonal_strength": 0.31, "significant": true}
    ],
    "primary_seasonal_period": 144
  },

  "model_class_recommendations": [
    {"model_class": "SARIMA", "justification": "Non-stationary target with confirmed daily seasonality (period=144)."},
    {"model_class": "SARIMAX", "justification": "Exogenous features (t1, t6) have useful cross-correlation at lags 1-3."},
    {"model_class": "HoltWinters", "justification": "Trend and seasonality both detected via STL."},
    {"model_class": "FAAR-ARIMA", "justification": "Hurst=0.71 indicates persistent memory; PCA factors can compress multivariate signal."},
    {"model_class": "XGBoost", "justification": "Large dataset (N>10k), many features, non-linear interactions likely."},
    {"model_class": "ElasticNet", "justification": "Interpretable linear baseline with collinearity-robust regularisation."}
  ],

  "acf_pacf_orders": {
    "suggested_ar_order": 2,
    "suggested_ma_order": 0,
    "suggested_d": 1,
    "suggested_seasonal_ar": 1,
    "suggested_seasonal_d": 1,
    "suggested_seasonal_ma": 1,
    "seasonal_period": 144
  },

  "mi_ranking": [
    {"feature": "t6", "mi_score": 0.42, "below_noise_baseline": false},
    {"feature": "rv1", "mi_score": 0.003, "below_noise_baseline": true}
  ],
  "noise_mi_baseline": 0.005,
  "redundant_columns": ["rv2"],
  "correlation_matrix_summary": {"max_pair": ["rv1", "rv2"], "max_corr": 1.0},
  "useful_lag_features": [
    {"feature": "t1", "lag": 1, "xcorr": 0.23},
    {"feature": "t6", "lag": 3, "xcorr": 0.19}
  ],

  "recommended_features": ["t6", "t1", "rh_6", "lights", "t_out"],
  "excluded_features": {
    "rv1": "below_noise_baseline",
    "rv2": "redundant",
    "target_copy": "leakage_suspect"
  },
  "target_candidates": [{"column": "appliances", "reason": "highest_variance_numeric"}],
  "client_facing_summary": "Energy consumption shows a strong daily rhythm (period=144 timesteps) and a mild upward trend. The series has persistent memory (Hurst=0.71), meaning recent values are good predictors of near-future values. Features like T6 and T1 (room temperatures) are highly informative. rv1 and rv2 were dropped as random noise, and sensor_copy was removed as a direct duplicate of the target.",
  "context": {}
}
```

## Guardrails

- `recommended_features` must never be empty. If all features fail the MI/correlation filters, loosen the noise-baseline threshold by 50% and log a `"threshold_relaxed"` warning before emitting the final list.
- MI computation is stochastic — set `random_state=42`.
- Log the count of features dropped at each filter stage in the output JSON under `"filter_counts"`.
- Do not silently pass `recommended_features = all_features`; every exclusion requires a reason in `excluded_features`.
- All stationarity / Ljung-Box / Hurst computations must handle NaN values gracefully (forward-fill short gaps, drop longer gaps) before computation.
- **Polars to Pandas:** Convert `pl.DataFrame` to `pandas` / `numpy` before passing into `statsmodels`, `scipy`, or `sklearn`.
- Hurst R/S analysis requires at least 64 data points. If N < 64, skip and record `"hurst_exponent": null, "hurst_skipped_reason": "insufficient_data"`.
- ADF and KPSS require at least 20 data points. If N < 20, skip and set `"stationarity_conclusion": "insufficient_data"`.
- If no time column is found, set `time_series_detected = false`, skip all TS diagnostics, and populate `ts_diagnostics` with nulls. Step 12 will build only static features.

## Copilot Prompt Snippet

```markdown
Implement `step_11_exploration.py`. CLI args: `--output-dir`, `--run-id`.
Read target column from `OUTPUT_DIR/progress.json`. Load `OUTPUT_DIR/cleaned.parquet` with polars; convert to pandas/numpy for all statistical computations.

Feature quality section:
- Near-zero variance filter (threshold 1e-4 after min-max scaling)
- High cardinality filter (> 50 unique values for categoricals)
- MI ranking via sklearn mutual_info_regression with random_state=42; 5 noise-column baseline
- Pairwise Pearson correlation redundancy (|r| >= 0.90, keep higher-MI member)
- Lag-0 leakage detection (|r| > 0.98 with target → hard-exclude)

Time-series diagnostic battery:
1. Stationarity: ADF (statsmodels adfuller) + KPSS (statsmodels kpss). Record both statistics and p-values. Derive stationarity_conclusion from joint interpretation matrix.
2. ACF/PACF: statsmodels acf() and pacf() at lags 1..min(48,N/4). Extract significant lags (|value| > 2/sqrt(N)). Derive suggested_ar_order and suggested_ma_order from shape.
3. Hurst exponent: implement R/S rescaled-range analysis. Fit log(R/S) vs log(n) by numpy.polyfit. Record H and R² of fit. Skip if N < 64.
4. Ljung-Box: statsmodels diagnostic.acorr_ljungbox at lags [6,12,24]. Set white_noise=true only if ALL p-values > 0.05.
5. STL decomposition: statsmodels STL. Compute trend_strength and seasonal_strength(s) per candidate period. Record detected_periods list and primary_seasonal_period.
6. Cross-correlation: for each recommended feature, compute scipy.stats.pearsonr at lags 1,2,3,6,12 after shifting. Record useful_lag_features (|xcorr| > 0.15).
7. Multiple series detection: check for low-cardinality categorical column co-occurring with time index; verify overlapping time ranges.

Model class selection: apply decision table to derive model_class_recommendations list (each with justification string).
Derive acf_pacf_orders (p, d, q, P, D, Q, m) for downstream steps.
Generate client_facing_summary as a non-technical paragraph.

Write step-11-exploration.json. Update progress.json current_step.
```

## Tests

- All features fail noise baseline → threshold loosened, list not empty, warning logged
- Perfectly correlated feature pair → lower-MI member flagged redundant
- Pure random walk target (AR(0)) → `white_noise=true`, ADF non-significant, Hurst ≈ 0.5
- Strongly trending series → ADF non-stationary, Hurst > 0.65, trend_detected true
- Mean-reverting series → Hurst < 0.45, `mean_reverting` interpretation
- Exact copy of target column → hard-excluded as `leakage_suspect`, never in `recommended_features`
- Tiny dataset (N < 64) → Hurst skipped gracefully, stationarity tests degrade gracefully
- Multi-period seasonal series → STL detects multiple significant periods
- Dataset with categorical panel grouping → `multiple_series_detected = true`
