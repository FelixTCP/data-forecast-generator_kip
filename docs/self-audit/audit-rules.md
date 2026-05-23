# Audit Rules and Checks

All checks use **objective metrics**, not LLM judgment. Each check returns `pass`, `marginal`, or `fail`.

## 1. Temporal Consistency Check

**Applies to:** Data with a time column detected in step 01.

**Checks:**
- **Gap Detection:** Are there unexpected breaks in the time series?
  - Expected frequency (daily, hourly, etc.) inferred from first 100 rows.
  - Flag any gap > 2× expected interval.
  - Threshold: Gap > 10% of total time span → `fail`.

- **Regularity:** Is the time series at consistent intervals?
  - Compute standard deviation of time deltas.
  - Threshold: StdDev > 10% of mean interval → `marginal`.

**Output:**
```json
{
  "temporal_consistency": {
    "status": "pass",
    "findings": [
      "no_gaps_detected",
      "regular_frequency_confirmed",
      "inferred_frequency_daily"
    ],
    "gap_count": 0,
    "max_gap_days": 0,
    "regularity_stddev_percent": 2.1
  }
}
```

---

## 2. Multi-Series Detection Check

**Applies to:** All data; especially important for financial/sensor datasets.

> **PRIORITY: Check for duplicate timestamps before other heuristics!**

**Heuristics (in order):**

1. **Duplicate Timestamps (PRIMARY SIGNAL — must be checked first)**
   - **Definition:** When `n_unique(time_column) < n_rows`, timestamps appear multiple times → definitive multi-series
   - **Scenario:** 7 entities × N time points = N×7 rows, but only N unique timestamps
   - **Action:** If duplicate timestamps present → immediately set status=`fail`, severity=`high`, regardless of other tests
   - **Finding-Text:** `"Duplicate timestamps detected: {n_unique} unique timestamps for {n_rows} rows ⇒ {n_rows / n_unique:.1f} series per timestamp"`
   - **Example:** 
     ```python
     time_col = df["date"]
     if len(time_col.unique()) < len(df):
         # Duplicate timestamps: Multi-Series
         status = "fail"
         severity = "high"
     ```

2. **Cardinality Check:** Look for high-cardinality columns (e.g., `group_id`, `entity_id`, `device_id`, `location`, `user_id`).
   - If any column has 3–100 unique values and correlates with target variance → likely multi-series.
   - High cardinality (>100) but low variance → not multi-series.

3. **Target Variance by Group:** For top 3 candidate grouping columns, split target by group and compute variance ratio.
   - If variance between groups > variance within groups → strong multi-series signal.
   - Threshold: Ratio > 2.0 → `fail` (multi-series detected, model should be per-group).

4. **Feature Redundancy Across Groups:** If a feature (e.g., `lag_1` of target) is nearly identical across all groups → leakage or weak group separation.

**Output:**
```json
{
  "multi_series_detection": {
    "status": "fail",
    "findings": [
      "duplicate_timestamps_detected_7_entities_per_timestamp",
      "inter_group_variance_high",
      "model_trained_on_mixed_entities"
    ],
    "detected_group_columns": [
      {
        "column": "entity_id",
        "unique_values": 7,
        "variance_ratio": 3.2,
        "recommendation": "train_separate_models"
      }
    ],
    "severity": "high"
  }
}
```

---

## 3. Feature-Target Alignment Check

**Applies to:** All pipelines.

**Checks (in order of priority):**

- **CRITICAL — Target Variable Leak Check:** Is the target variable (from step 10) present in the feature set?
  - If target variable appears in features → `fail`, severity=`high` (Critical leakage)
  - Finding: `"target_variable_leaked_into_features: {target_col} must be excluded_before_training"`

- **CRITICAL — Timestamp Field Check:** Is the raw timestamp/date column used as a numeric feature?
  - If time column (e.g., `date`, `timestamp`) is in features as raw data → `fail`, severity=`high` (enables perfect reconstruction)
  - Exception: Time column used only for lag calculations with `.over()` is okay
  - Finding: `"timestamp_field_leaked_as_feature: {time_col} enables_perfect_reconstruction"`

- **Monotone Index Features Check:** Are there features with names `trend_t_index`, `trend_t_index_sq`, or similar?
  - If found → `fail`, severity=`high` (monotone features = perfect KS drift = leakage)
  - Finding: `"monotone_index_feature_detected: {feature_name} causes_ks_1_0"`

- **MI Ranking Stability:** Do the top MI features from step 11 actually appear in the final feature set (step 12)?
  - Threshold: <80% of top 5 MI features retained → `marginal`.

- **Excluded Feature Justification:** Are excluded features documented with reasons?
  - If excluded list is empty → `fail` (nothing was filtered; possible under-fit).
  - If excluded list is >70% of original features → `fail` (aggressive filtering; possible over-fit).

- **Redundancy in Final Set:** Compute pairwise Pearson correlation among final features.
  - Threshold: Any pair >0.90 → `marginal`.

**Output:**
```json
{
  "feature_target_alignment": {
    "status": "fail",
    "findings": [
      "target_variable_leaked_into_features: appliances_must_be_excluded",
      "timestamp_field_leaked_as_feature: date_enables_reconstruction",
      "monotone_index_feature_detected: trend_t_index_causes_ks_1_0"
    ],
    "mi_retention_rate": 0.75,
    "correlation_max": 0.87,
    "excluded_ratio": 0.15,
    "target_variable_in_features": true,
    "timestamp_in_features": true,
    "monotone_features_found": ["trend_t_index"],
    "severity": "high"
  }
}
```

If all checks pass:
```json
{
  "feature_target_alignment": {
    "status": "pass",
    "findings": [
      "target_variable_correctly_excluded",
      "timestamp_not_used_as_raw_feature",
      "no_monotone_index_features",
      "mi_ranking_stable",
      "no_redundant_features"
    ],
    "mi_retention_rate": 0.90,
    "correlation_max": 0.82,
    "excluded_ratio": 0.25,
    "target_variable_in_features": false,
    "timestamp_in_features": false,
    "monotone_features_found": [],
    "severity": "low"
  }
}
```

---

## 4. Model Performance Baseline Check

**Applies to:** All pipelines.

**Profile-Specific R² Thresholds:**

| Detected Profile | R² Pass | R² Marginal | Rationale |
|---|---|---|---|
| `multi_series_temporal` | ≥ 0.50 | 0.25–0.50 | High difficulty; per-entity models often ≥0.65 |
| `daily_cyclical_temporal` | ≥ 0.55 | 0.30–0.55 | Moderate difficulty; clear daily patterns learnable |
| `longer_period_temporal` | ≥ 0.50 | 0.25–0.50 | Moderate-Hard; trends and longer lags needed |
| `generic_temporal` | ≥ 0.50 | 0.25–0.50 | Temporal signal weak; feature engineering critical |
| `static_regression` | ≥ 0.60 | 0.35–0.60 | Easiest case; no temporal leakage risks |

**Checks:**
- **R² vs. Profile Baseline:** Is selected model R² above the "marginal" threshold for detected profile?
  - If below → `fail`.

- **Overfitting Signal:** Is training CV R² significantly higher than holdout R²?
  - Threshold: Holdout R² < 0.8 × CV R² → `marginal`.

- **Mean Baseline Comparison:** Is R² > 0 (better than predicting target mean)?
  - If R² < 0 → `fail`.

**Output:**
```json
{
  "model_performance_baseline": {
    "status": "marginal",
    "findings": [
      "r2_below_profile_threshold",
      "holdout_worse_than_training",
      "overfitting_detected"
    ],
    "detected_profile": "multi_series_temporal",
    "r2_holdout": 0.35,
    "r2_cv_mean": 0.52,
    "r2_pass_threshold": 0.50,
    "severity": "medium"
  }
}
```

---

## 5. Data Distribution Drift Check

**Applies to:** All pipelines; critical for time-series data.

**Checks (in order):**

- **Monotone Index Features Detection (CRITICAL):** Scan feature names and values for monotone index columns.
  - Pattern: `trend_t_index`, `trend_t_index_sq`, `index`, `row_num`, `time_index`, `sequential_id`
  - Check: Are values strictly increasing (or decreasing) with no repeats?
  - If found → `fail`, severity=`high`, KS=1.0 (perfect leakage)
  - Finding: `"monotone_index_feature_detected: {feature_name} causes_ks_1_0_perfect_reconstruction"`

- **Training vs. Holdout Distribution:** Use Kolmogorov–Smirnov test on target distributions.
  - KS statistic < 0.15 → `pass`.
  - 0.15–0.25 → `marginal`.
  - > 0.25 → `fail` (strong distribution shift).

- **Feature Distribution Drift:** Check top 3 features by MI (excluding monotone index features).
  - Average KS stat across features.
  - Same thresholds as target.
  - If any feature has KS=1.0 → `fail` (perfect reconstruction/leakage detected)

- **Temporal Drift (if time-series):** Split training set into quarters; compare first vs. last quarter target distribution.
  - If drift detected: `marginal` (pipeline may need regularization or separate periods).

**Output:**
```json
{
  "data_distribution_drift": {
    "status": "fail",
    "findings": [
      "monotone_index_feature_detected: trend_t_index_causes_ks_1_0_perfect_reconstruction",
      "target_ks_stat_0.35_above_threshold_0_25",
      "feature_trend_t_index_ks_1_0_perfect_leakage"
    ],
    "target_ks_stat": 0.35,
    "feature_ks_stats": {
      "trend_t_index": 1.0,
      "feature_2": 0.18,
      "feature_3": 0.22
    },
    "feature_ks_mean": 0.47,
    "monotone_features": ["trend_t_index"],
    "temporal_drift_detected": false,
    "severity": "high"
  }
}
```

If no drift:
```json
{
  "data_distribution_drift": {
    "status": "pass",
    "findings": [
      "no_monotone_index_features",
      "training_holdout_ks_stat_0.12_acceptable",
      "no_temporal_drift_in_training",
      "feature_distributions_stable"
    ],
    "target_ks_stat": 0.12,
    "feature_ks_stats": {
      "lights": 0.08,
      "temperature": 0.11,
      "humidity": 0.09
    },
    "feature_ks_mean": 0.09,
    "monotone_features": [],
    "temporal_drift_detected": false,
    "severity": "low"
  }
}
```

---

## Check Result Summary

```json
{
  "checks": {
    "temporal_consistency": { "status": "pass", ... },
    "multi_series_detection": { "status": "fail", ... },
    "feature_target_alignment": { "status": "pass", ... },
    "model_performance_baseline": { "status": "marginal", ... },
    "data_distribution_drift": { "status": "pass", ... }
  },
  "critical_findings": [
    {
      "check": "multi_series_detection",
      "severity": "high",
      "description": "Multi-series structure detected; model not optimized for separate time series."
    }
  ],
  "overall_audit_result": "fail",
  "audit_confidence": 0.88
}
```

## Audit Result Logic

**Pass:** All checks are `pass` or `marginal`, and no high-severity critical findings.

**Fail:** ≥1 critical (high-severity) findings, or ≥2 `fail` checks.

**Remediation Required:** If audit fails, see `remediation.md` for recommended re-trigger actions.
