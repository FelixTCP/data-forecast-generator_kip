# Data Forecast Generator — Pipeline Report

> **Run ID:** 20260515T103906Z  
> **Target:** `avgtemperature`  
> **Quality flag:** `acceptable`  
> **Production usable:** Yes  

---

## 1. Problem Statement & Selected Target

**Run ID:** 20260515T103906Z
**Generated:** 2026-05-15 11:03 UTC

**Dataset:** `artifacts\ui_uploads\algiers_temperature.csv`
**Target column:** `avgtemperature`
**Objective:** Forecast daily average temperature in Algiers using historical observations from 1995–2020.

The dataset contains daily temperature measurements spanning 26 years. The task is a univariate
time-series regression: given historical temperature and calendar features, predict the next day's
average temperature in degrees Fahrenheit.

**Dataset shape:** 9266 raw rows → 9266 rows after cleansing.
**Sampling frequency:** daily.
**Time-series properties:** stationarity=stationary, Hurst exponent=0.869,
primary seasonal period=365 days.

## 2. Data Quality Summary

| Metric | Value |
|--------|-------|
| Raw rows | 9266 |
| Rows after cleansing | 9266 |
| Columns | 8 |
| Extreme anomalies smoothed | Yes — 35 values where |z|>6 replaced with linear interpolation |

**Null rates (post-cleanse):**
No null values remaining after cleansing.

**Fixes applied:**
- normalized_column_names: ['Region', 'Country', 'State', 'City', 'Month', 'Day', 'Year', 'AvgTemperature']
- synthesized_date_column from (year, month, day)
- sorted_by_time_column: date
- extreme_anomaly_smoothed: col='avgtemperature', zscore_threshold=6, count=35
- dropped_all_null_columns: ['state']

**Feature engineering:** 70 features built across calendar (Group A), lag (Group B),
cross-correlation lags (Group C), rolling statistics (Group E), and Fourier seasonality (Group F) groups.

**Train / holdout split (chronological):**
- Training rows: 6829 (burn-in of 730 rows excluded)
- Holdout rows: 1707 (~20% of data)

## 3. Candidate Models & Scores

**Naive lag-1 baseline:** R²=0.9117, RMSE=3.157, MAE=2.300

### Trained Candidate Models

| Rank | Model | R² | RMSE | MAE | CV R² | Weighted Score | Status | Note |
|------|-------|-----|------|-----|-------|----------------|--------|------|
| 1 | ridge | 0.9277 | 2.858 | 2.083 | 0.9196 | 0.9000 | eligible | **SELECTED** |
| 2 | random_forest | 0.9246 | 2.918 | 2.134 | 0.9262 | 0.5670 | eligible |  |
| 3 | hist_gbm | 0.9243 | 2.924 | 2.144 | 0.9215 | 0.5153 | eligible |  |
| 4 | elasticnet | 0.9212 | 2.982 | 2.189 | 0.9236 | 0.0840 | eligible |  |
| None | holt_winters | -11.6327 | 37.768 | 32.421 | -3.9709 | - | ineligible |  |
| None | naive_persistence | 0.9117 | 3.158 | 2.302 | - | - | benchmark |  |
| None | seasonal_naive | 0.7042 | 5.779 | 4.432 | - | - | benchmark |  |
| None | auto_arima_benchmark | -1.5189 | 16.865 | 14.171 | - | - | benchmark |  |
| None | ar1_benchmark | 0.0099 | 10.574 | 9.118 | - | - | benchmark |  |

### Mandatory Benchmarks

| Rank | Model | R² | RMSE | MAE | CV R² | Weighted Score | Status | Note |
|------|-------|-----|------|-----|-------|----------------|--------|------|
| - | naive_persistence | 0.9117 | 3.158 | 2.302 | - | - | benchmark | |
| - | seasonal_naive | 0.7042 | 5.779 | 4.432 | - | - | benchmark | |
| - | auto_arima_benchmark | -1.5189 | 16.865 | 14.171 | - | - | benchmark | |
| - | ar1_benchmark | 0.0099 | 10.574 | 9.118 | - | - | benchmark | |

**Model families explored:** AR, ElasticNet, Ridge, XGBoost, FAAR-ARIMA, HoltWinters-ETS.
*(Note: pmdarima and XGBoost not installed in environment — FAAR-ARIMA and XGBoost candidates not evaluated.)*

## 4. Selected Model Rationale

**Selected model:** `ridge`
**Quality flag:** `acceptable`

ridge achieved the highest weighted score (0.9000) with holdout R²=0.9277, RMSE=2.858°F, MAE=2.083°F. It beats the naïve lag-1 baseline (R²=0.9117) by Δ=+0.0160. Cross-validation R²=0.9196±0.0150 indicates consistent performance across temporal folds. Note: The improvement over the naïve persistence baseline is modest. Daily temperature autocorrelation is strong — any lag-1 model performs well. The value of this model lies in better handling of seasonal transitions and feature integration.

### Selected Model Details

| Metric | Value |
|--------|-------|
| Holdout R² | 0.927665113875047 |
| Holdout RMSE | 2.858 F |
| Holdout MAE | 2.083 F |
| CV R² | 0.9196 +/- 0.0150 |
| Target mean | 64.40 F |
| Target std | 10.91 F |
| RMSE as % of std | 26.2% |

### Candidate Analysis

**elasticnet:** R²=0.9212 > 0 (beats mean baseline). Performance nearly identical to naïve lag-1 baseline (Δ=+0.0095). CV R²=0.9236 ≈ holdout R²=0.9212: good generalization. RMSE=2.98°F (27.3% of target std).

**hist_gbm:** R²=0.9243 > 0 (beats mean baseline). Beats naïve lag-1 by Δ=+0.0126 R². CV R²=0.9215 ≈ holdout R²=0.9243: good generalization. RMSE=2.92°F (26.8% of target std).

**random_forest:** R²=0.9246 > 0 (beats mean baseline). Beats naïve lag-1 by Δ=+0.0129 R². CV R²=0.9262 ≈ holdout R²=0.9246: good generalization. RMSE=2.92°F (26.7% of target std).

**ridge:** R²=0.9277 > 0 (beats mean baseline). Beats naïve lag-1 by Δ=+0.0160 R². CV R²=0.9196 ≈ holdout R²=0.9277: good generalization. RMSE=2.86°F (26.2% of target std).

**holt_winters:** R²=-11.6327 < 0 — model is worse than predicting the mean. Ineligible for selection.


## 5. Risks and Caveats

### Key Risks

1. **Modest improvement over naive persistence:**
   The naive lag-1 baseline achieves R²=0.9117, while the selected model achieves
   R²=0.9277 (delta=+0.0160).
   Daily temperature is highly autocorrelated — any reasonable model will score well. The marginal
   gain beyond a simple persistence model is limited.

2. **Fahrenheit scale & geographic scope:**
   The dataset covers Algiers, Algeria only (1995–2020). The model is not transferable to other
   cities without retraining. All temperature values are in Fahrenheit.

3. **Data leakage policy:**
   All features are causal (no look-ahead). Lag features use `.shift(k)` where k >= 1.
   Rolling statistics use `.shift(1)` before the rolling window. Fourier features are calendar-based
   (computed from time index only). A leakage audit was performed and passed.

4. **Feature set limitations:**
   Only `month` and `year` passed the mutual information noise baseline filter. Day-of-year
   information is captured through Fourier features and lag structure. No external predictors
   (weather station data, elevation, climate indices) are available in this dataset.

5. **HoltWinters failure:**
   The Holt-Winters model (Exponential Smoothing) failed to converge and produced R²=-11.63.
   This is consistent with a non-standard seasonal period (365 days) and convergence sensitivity.

6. **pmdarima / XGBoost not installed:**
   The FAAR-ARIMA (pmdarima) and XGBoost candidates could not be trained. These are expected to
   perform comparably or better on seasonal temperature data. Install them and re-run for a more
   complete evaluation.

7. **Temporal distribution shift:**
   Training data ends at the holdout boundary (~5.5 years before end of dataset). Long-term climate
   trends or El Nino / La Nina effects could cause distribution shift that degrades real future performance.

## 6. Next Iteration Recommendations

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
