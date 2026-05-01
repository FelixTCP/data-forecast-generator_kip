# Remediation Actions and Re-Trigger Protocol

When the audit fails or detects high-severity issues, this document defines the remediation actions and how to re-trigger the pipeline with modified parameters.

## Quick Reference: Auto-Executable vs. Manual Actions

| Action ID | Type | Trigger | Affected Steps | Re-Trigger |
|---|---|---|---|---|
| `remove_monotonic_index_features` | `[AUTO]` | KS=1.0 drift detected | 12, 13 | Automatic |
| `improve_model_performance` | `[AUTO]` | Low holdout R² | 13, 14, 15 | Automatic |
| `extend_lag_window` | `[AUTO]` | Low TS performance + good MI | 12, 13 | Automatic |
| `add_seasonal_features` | `[AUTO]` | Strong autocorr at lag 7/24 | 12, 13 | Automatic |
| `increase_regularization` | `[AUTO]` | Overfitting detected | 13 | Automatic |
| `try_alternative_models` | `[AUTO]` | Model class under-performs | 13 | Automatic |
| `use_time_series_split` | `[AUTO]` | Strong temporal signal | 12, 13 | Automatic |
| `split_by_grouping_column` | `[AUTO]` | Multi-series detected | 12, 13, 14, 15 | Automatic — trains per-group sub-models |
| `handle_temporal_gaps` | `[MANUAL]` | Large temporal gaps | 10, 12 | User review required |
| `remove_outliers_by_isolation` | `[MANUAL]` | Distribution drift/anomalies | 10, 13 | User review required |

**Orchestrator Behavior:**
- **`[AUTO]` actions:** Orchestrator injects parameters and re-runs affected steps. Loop repeats up to `MAX_REMEDIATION_ITERATIONS = 3` until `overall_audit_result == "pass"` or limit reached.
- **CRITICAL: A `fail` audit result MUST ALWAYS trigger at least one remediation iteration.** If all collected actions are AUTO, the orchestrator executes them immediately. If only MANUAL actions remain after exhausting AUTO options, the orchestrator still writes a `remediation_required.json` and exits with code `1` to signal that the run requires human intervention before it can be marked complete.
- **`[MANUAL]` actions:** Logged in `remediation_actions`; pipeline exits with code `1` and writes `remediation_required.json` explaining required steps.

---

## Remediation Action Categories

### Category A: Data Preprocessing & Feature Engineering

**Action: `split_by_grouping_column`** `[AUTO]`
- **Triggered by:** Multi-series detection failure (multiple time series detected; model trained on mixed series).
- **Severity:** High.
- **Auto-Executable:** ✅ Yes — orchestrator automatically re-runs steps 12–17 with per-group training.
- **Description:** Re-run pipeline separately for each detected group (entity, machine, location, stock, city, etc.). Train one sub-model per group value, then ensemble predictions weighted by per-group R².
- **Affected steps:** 12 (Feature Extraction), 13 (Model Training), 14 (Evaluation), 15 (Selection).
- **Parameters injected automatically:**
  ```json
  {
    "group_column": "<auto-detected grouping column>",
    "train_separate_models": true,
    "ensemble_method": "weighted_by_r2"
  }
  ```
- **Expected improvement:** R² typically +0.2 to +0.5 per group; eliminates cross-entity contamination.
- **Implementation:**
  - Pass `--group-column=<column_name>` to step 12, 13, 14, 15.
  - Steps 12/13 loop over unique group values, fitting one sub-model each (tqdm progress bar over groups).
  - Step 14 reports per-group R² + weighted ensemble R².
  - `model.joblib` contains a dict `{group_value: fitted_model}` plus `ensemble_weights`.

---

**Action: `extend_lag_window`** `[AUTO]`
- **Triggered by:** Low model performance on time-series data despite good feature MI.
- **Severity:** Medium.
- **Description:** Increase maximum lag window (e.g., 5 → 20 days) for target and key features.
- **Affected steps:** 12 (Feature Extraction), 13 (Model Training).
- **Parameters:**
  ```json
  {
    "max_lag": 20,
    "lag_step": 1,
    "rolling_windows": [5, 10, 20]
  }
  ```
- **Expected improvement:** CV R² +0.1 to +0.3; captures longer-term dependencies.
- **Implementation:**
  - Pass `--max-lag=20` to step 12.
  - Regenerate lag features; step 13 re-trains with extended feature set.

---

**Action: `add_seasonal_features`** `[AUTO]`
- **Triggered by:** Strong autocorrelation at lag 7 (daily/weekly) or lag 24 (hourly) not reflected in CV R².
- **Severity:** Medium.
- **Description:** Add hour-of-day, day-of-week, month cyclic features and/or rolling seasonal statistics.
- **Affected steps:** 12 (Feature Extraction), 13 (Model Training).
- **Parameters:**
  ```json
  {
    "add_hour_of_day": true,
    "add_day_of_week": true,
    "add_month": true,
    "use_cyclic_encoding": true,  // sin/cos instead of one-hot
    "rolling_seasonal_windows": [7, 30]
  }
  ```
- **Expected improvement:** CV R² +0.1 to +0.2 for daily/weekly patterns.

---

### Category B: Model Selection & Regularization

**Action: `increase_regularization`** `[AUTO]`
- **Triggered by:** Holdout R² much worse than CV R² (overfitting detected).
- **Severity:** Medium.
- **Description:** Increase L1/L2 regularization (Ridge α, Lasso λ, ElasticNet mix) or use models with built-in regularization.
- **Affected steps:** 13 (Model Training).
- **Parameters:**
  ```json
  {
    "regularization_method": "ridge_cv",
    "alpha_range": [0.1, 1.0, 10.0],
    "candidates": ["ridge_strong", "elasticnet_l1_0.5"]
  }
  ```
- **Expected improvement:** Holdout R² may improve by +0.05 to +0.15; reduces overfitting.

---

**Action: `try_alternative_models`** `[AUTO]`
- **Triggered by:** Selected model (e.g., RandomForest) underperforms; heuristic suggests another class might be better.
- **Severity:** Low to Medium.
- **Description:** Train additional model types not in the default set (e.g., LightGBM, SVR, KNN for regression).
- **Affected steps:** 13 (Model Training), 14 (Evaluation), 15 (Selection).
- **Parameters:**
  ```json
  {
    "additional_candidates": ["lightgbm", "svr", "knn_weighted"],
    "skip_default_candidates": false
  }
  ```
- **Expected improvement:** May find better model for specific data characteristics; +0.05 to +0.2 R².

---

### Category C: Data Quality & Cleaning

**Action: `handle_temporal_gaps`** `[MANUAL]`
- **Triggered by:** Temporal consistency check fails; gaps > 10% of time span detected.
- **Severity:** Medium to High.
- **Description:** Document gaps; optionally interpolate or separate training windows.
- **Affected steps:** 01 (Cleansing), 12 (Feature Extraction), 13 (Model Training).
- **Parameters:**
  ```json
  {
    "gap_handling": "interpolate",  // or "separate_windows", "exclude"
    "interpolation_method": "linear",
    "max_consecutive_gaps": 10
  }
  ```
- **Expected improvement:** Cleaner training data; prevents model from learning spurious patterns in gaps.

---

---

**Action: `remove_monotonic_index_features`** `[AUTO]`
- **Triggered by:** Data Distribution Drift (Check 5) detects KS=1.000 for any feature (e.g., `trend_t_index`, `row_number`).
- **Severity:** High.
- **Auto-Executable:** ✅ Yes — Monotonic features are always non-transferable.
- **Description:** Remove monotonic integer features showing KS=1.0. These are artifacts of data ordering, not real predictors.
- **Affected steps:** 12, 13, 14, 15.
- **Parameters:**
  ```json
  {
    "exclude_features": ["trend_t_index", "trend_t_index_sq"],
    "replacement": "use_trend_elapsed_days"
  }
  ```
- **Expected improvement:** Eliminates KS=1.0 drift. Model becomes transferable. Validation more realistic.
- **Orchestrator Implementation:**
  - Pass `--exclude-features <list>` to steps 12–13.
  - Re-run affected steps.
  - Trigger re-audit (step 17).

---

**Action: `improve_model_performance`** `[AUTO]`
- **Triggered by:** Model Performance Baseline detects low R² despite good CV scores.
- **Severity:** Medium.
- **Auto-Executable:** ✅ Yes (partial).
- **Description:** Log-transform skewed targets, enable hyperparameter tuning, or expand model candidate pool.
- **Affected steps:** 12, 13, 14, 15.
- **Parameters:**
  ```json
  {
    "log_transform_target": true,
    "force_expansion_models": true
  }
  ```
- **Expected improvement:** R² typically +0.1 to +0.3.
- **Orchestrator Implementation:**
  - Inject `--log-transform-target` if target skewness > 2.0.
  - Inject `--force-expansion-models=true` for HistGradientBoosting and SVR.
  - Re-run steps 13–15.
  - Trigger re-audit.

---

**Action: `remove_outliers_by_isolation`** `[MANUAL]`
- **Triggered by:** Distribution drift suspected due to anomalies.
- **Severity:** Low–Medium.
- **Auto-Executable:** ❌ No — requires outlier threshold and action decision.
- **Description:** Use Isolation Forest or IQR to identify and flag/remove extreme outliers.
- **Affected steps:** 01 (Cleansing), 13 (Model Training).
- **Parameters:**
  ```json
  {
    "outlier_method": "isolation_forest",
    "contamination": 0.05,
    "action": "flag"  // or "remove"
  }
  ```
- **Expected improvement:** More robust model; may improve R² by 0.05–0.15 if outliers are noise.

---

### Category D: Data Splitting

**Action: `use_time_series_split`** `[AUTO]`
- **Triggered by:** Strong temporal signal; standard random split may leak future information.
- **Severity:** Medium.
- **Description:** Use TimeSeriesSplit instead of random split; ensure training always precedes test temporally.
- **Affected steps:** 12 (Feature Extraction), 13 (Model Training).
- **Parameters:**
  ```json
  {
    "split_mode": "time_series",
    "n_splits": 5
  }
  ```
- **Expected improvement:** More realistic CV R²; holdout performance more representative of real deployment.

---

## Remediation Workflow

### Step 1: Audit Detection & Recommendation
```json
{
  "overall_audit_result": "fail",
  "remediation_actions": [
    {
      "action_id": "split_by_grouping_column",
      "severity": "high",
      "description": "...",
      "suggested_parameters": {...},
      "expected_improvement": "R² likely to increase by 0.3–0.5"
    }
  ]
}
```

### Step 2: Review & Approval
- **Manual review:** Domain expert reviews audit findings and approved remediation actions.
- **Decision:** Apply remediation, skip it, or apply subset of actions.

### Step 3: Generate New RUN_ID & Parameters
```bash
NEW_RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
NEW_PARAMS="--split-mode=time_series --max-lag=20 --group-column=<detected_grouping_column>"
```

### Step 4: Re-Trigger Orchestrator
```bash
python orchestrator.py \
  --csv-path <path_to_data.csv> \
  --target-column <target_column_name> \
  --output-dir output/${NEW_RUN_ID} \
  --run-id ${NEW_RUN_ID} \
  --code-dir output/${NEW_RUN_ID}/code \
  ${NEW_PARAMS}
  # NO --resume flag; start fresh with new parameters
```

### Step 5: Re-Audit
- Run audit again after pipeline completes.
- Compare metrics to prior run.
- Document improvement (or lack thereof) in audit report.

---

## Re-Trigger Parameters Map

| Remediation Action | CLI Parameter | Step(s) Affected | Default Value | New Value |
|--------------------|---------------|-----------------|---------------|-----------|
| split_by_grouping_column | `--group-column` | 12,13,14,15 | None | `<detected_grouping_column>` |
| extend_lag_window | `--max-lag` | 12,13 | 5 | 20–40 |
| add_seasonal_features | `--seasonal-features` | 12,13 | false | true |
| increase_regularization | `--regularization=ridge_cv` | 13 | N/A | `ridge_cv` with high α |
| try_alternative_models | `--extra-models` | 13,14,15 | empty | `lightgbm,svr,histgradient` |
| use_time_series_split | `--split-mode` | 12,13 | `auto` | `time_series` |

---

## Audit-to-Remediation Mapping

| Audit Finding | Recommended Actions | Priority |
|---------------|-------------------|----------|
| Multi-series detected; model mixed | `split_by_grouping_column` | High |
| Low R² on time-series data | `extend_lag_window` + `add_seasonal_features` | High |
| Strong seasonal pattern not captured | `add_seasonal_features` + `extend_lag_window` | High |
| Holdout R² << CV R² (overfitting) | `increase_regularization` | Medium |
| Large temporal gaps detected | `handle_temporal_gaps` | Medium |
| Distribution drift high | `remove_outliers_by_isolation` + `use_time_series_split` | Medium |
| Generic data; low feature MI | `try_alternative_models` | Low |

---

## Tracking Remediation Attempts

Each re-trigger should be recorded in a **remediation log** (optional but recommended):

```json
{
  "original_run_id": "20260424T120000Z",
  "remediation_attempts": [
    {
      "attempt": 1,
      "new_run_id": "20260424T130000Z",
      "actions_applied": ["split_by_grouping_column"],
      "parameters": { "group_column": "<detected_column>" },
      "result_r2_before": 0.35,
      "result_r2_after": 0.72,
      "audit_passed": true,
      "notes": "Per-group models significantly improved performance."
    }
  ]
}
```

This allows tracking of improvement trajectory and decision history.
