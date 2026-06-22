# Temperature Forecasting Pipeline Report

**Run ID:** 20260622T174637Z  
**Generated:** 2026-06-22 18:19:24 UTC  
**Target:** `avgtemperature`  
**Dataset:** 9,265 rows, date_synthesized time column  
**Quality Flag:** `acceptable`



---

## 1. Problem & Selected Target

We analyzed daily temperature data for Algiers, Algeria to build a regression-based forecasting model. 
The target variable is `avgtemperature` (average daily temperature in Fahrenheit), representing historical 
daily temperature observations from the dataset.

The dataset contains **9,265 rows** with a **time-series split strategy** 
applied to preserve temporal ordering. The time series was detected as **stationary** 
(Hurst exponent: 0.869).

---

## 2. Data Quality Summary

| Metric | Value |
|--------|-------|
| Total rows | 9,265 |
| Features engineered | 23 |
| Time column | `date_synthesized` |
| Max null rate | 0.00% |
| Duplicate rows removed | 0 |
| Split strategy | time_series |
| Training rows | 7,412 |
| Holdout rows | 1,853 |

**Data Fixes Applied:**
- normalized_column_names
- synthesized_date_from_year_month_day
- removed_1_duplicate_date_rows
- extreme_anomaly_smoothed: col='avgtemperature', zscore_threshold=6, count=35
- final_chronological_sort_by=date_synthesized

**Feature Engineering:**
- Lag features: avgtemperature_lag_1, avgtemperature_lag_2, avgtemperature_lag_3, month_lag_1, month_lag_2
- Rolling features: avgtemperature_roll_mean_7, avgtemperature_roll_std_7, avgtemperature_roll_mean_30, avgtemperature_roll_std_30
- Calendar features: cal_month, cal_day, cal_day_of_week, cal_year
- Fourier features: fourier_sin_365_1, fourier_cos_365_1, fourier_sin_365_2, fourier_cos_365_2

---

## 3. Candidate Models & Scores

| Model | R² | RMSE | MAE | CV R² |
|-------|-----|------|-----|-------|
| ridge ✓ **SELECTED** | 0.9325 | 2.805 | 2.049 | 0.929±0.005 |
| random_forest | 0.9255 | 2.949 | 2.203 | 0.910±0.011 |
| gradient_boosting | 0.9149 | 3.150 | 2.482 | 0.888±0.040 |
| xgboost | 0.9050 | 3.328 | 2.559 | 0.865±0.032 |

**Naive lag baseline:** R²=0.9177  
**Target mean:** 65.05°F, std=10.80°F

---

## 4. Selected Model Rationale

**Selected model:** `ridge`  
**R²:** 0.9325 (93.3% variance explained)  
**RMSE:** 2.805°F  
**MAE:** 2.049°F  

ridge scored highest with a weighted score of 0.9995 (R²=0.9325, RMSE=2.81, MAE=2.05). It outperforms the naive lag baseline (R²=0.9177) and demonstrates stable cross-validation performance (CV R²=0.929). Lower-complexity models are preferred as tie-breakers to reduce overfitting risk.

The model was trained using a **time-series split** with 5-fold 
TimeSeriesSplit cross-validation. The split preserves temporal ordering to prevent 
information leakage from future to past.

**Feature importance (SHAP top features):**
- `avgtemperature_lag_1`: mean |SHAP| = 7.6363
- `avgtemperature_lag_2`: mean |SHAP| = 1.7839
- `fourier_cos_365_1`: mean |SHAP| = 1.6299
- `avgtemperature_roll_mean_30`: mean |SHAP| = 0.9972
- `avgtemperature_roll_mean_7`: mean |SHAP| = 0.6349

---

## 5. Risks & Caveats

1. **Temporal data leakage risk**: Lag and rolling features use past values of the target. 
   The model was trained with chronological splits to mitigate this risk, but deployment 
   requires careful handling of the prediction horizon.

2. **Stationarity**: The series is classified as **stationary**. 
   Stationary series can be modeled reliably without differencing.

3. **Feature dependency**: The model depends on lag features (e.g., yesterday's temperature). 
   Predictions further than 8 steps into the future 
   will require recursive forecasting, which may amplify errors.

4. **Seasonality**: Yearly seasonality (period=365) detected and encoded via Fourier features. The model should handle seasonal patterns well.

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
