# Schritt 12 — Feature Extraction & Model Preselection

## Überblick

Generate `step_12_features.py` — Complete, immediately executable Python CLI script.

| Feld | Wert |
|---|---|
| **Dateiname** | `step_12_features.py` |
| **CLI** | `python step_12_features.py --output-dir <dir> --run-id <id> [--split-mode auto\|random\|time_series] [--exclude-features feat1,feat2]` |
| **Input** | `OUTPUT_DIR/cleaned.parquet`, `step-11-exploration.json`, `step-10-cleanse.json` |
| **Output** | `features.parquet`, `step-12-features.json` |
| **Exit Codes** | 0=Success, 1=Error, 2=Leakage Detected |

---

## MANDATORY CHECKLIST

✅ **BEFORE ANY CODE:**
- [ ] **Target Variable**: Filtered AFTER feature selection; if found in final features → `sys.exit(2)`
- [ ] **Timestamp Field**: NOT used as raw numeric feature; only for lag/trend calculation
- [ ] **CLI `--exclude-features`**: Comma-separated list for orchestrator remediation
- [ ] **`tqdm` import**: All feature engineering loops use `from tqdm import tqdm`
- [ ] **Multi-Series Lags**: Use `.shift(n).over(group_col)` (no global `.shift()` for multi-series)
- [ ] **Rolling Causality**: `.shift(1)` BEFORE every `.rolling_*()` — prevents look-ahead
- [ ] **Leakage Exit 2**: Pearson |r| > 0.98 OR RandomForest R² > 0.999 → `sys.exit(2)`, no artifacts
- [ ] **Zero-Variance Removal (Step L)**: After engineering, before leakage check
- [ ] **Feature Scaling (Step K)**: StandardScaler (linear/SARIMA), MinMaxScaler (LSTM), none (trees)
- [ ] **All 13 Functions**: Z, A–L implemented as separate functions, called in order from `main()`
- [ ] **Minimum Features**: `len(final_features) >= 2` after cleanup, else `sys.exit(1)`

---

## Input Contract (from Step 11)

**Required fields in `step-11-exploration.json`:**
```python
{
    "recommended_features": ["T6", "T1", "RH_6", ...],
    "excluded_features": {"rv1": "reason", ...},
    "time_column": "date",
    "multiple_series_detected": bool,
    "group_column": str | null,
    "significant_lags": [1, 3, 6],
    "useful_lag_features": [{"feature": "T1", "lag": 1, "xcorr": 0.23}, ...],
    "time_series_characteristics": {
        "trend_detected": bool,
        "seasonality_detected": bool,
        "white_noise": bool
    }
}
```

---

## Output Contract

**`step-12-features.json` MUST contain:**
```json
{
  "step": "12-feature-extraction",
  "features": ["lag1", "lag3", "rolling_mean_7", ...],
  "features_excluded": {"feature_name": "reason", ...},
  "target_column": "appliances",
  "split_strategy": {"resolved_mode": "time_series"},
  "leakage": {
    "status": "pass|fail",
    "leakage_candidates": [],
    "threshold": 0.98,
    "reconstruction_probe_r2": null
  },
  "scaling_metadata": {"scaler_used": "StandardScaler|MinMaxScaler|None", ...},
  "artifacts": {"features_parquet": "path/features.parquet"}
}
```

**`features.parquet` contains:**
- All engineered features + target column
- No timestamp, no target column duplicates

---

## Implementation: 13 Mandatory Functions

### Z — `auto_detect_target_column(df, numeric_cols, explicit_target=None)`
- If `explicit_target`: validate in `numeric_cols`, return it
- Else: return highest-variance column
- Return: `(target_col_name, {"method": "explicit"|"highest_variance", "score": float})`

### A — `compute_lag_mutual_information(df, target_col, max_lag=12)` (FAST)
- Use `sklearn.feature_selection.mutual_info_regression`
- **Optimization**: max_lag=12 (not 48); MI is stable after lag 12 for most series
- Return: `pl.DataFrame[lag, mutual_information]` sorted descending

### B — `find_best_lags(df, target_col, max_lag=12, top_n=3)` (FAST)
- Combine ACF + PACF + MI; use statsmodels
- **Optimization**: max_lag=12 (fast ACF); return only TOP-3 lags by magnitude
- Return: `{"best_lags_acf": [...], "best_lags_mi": [...], "recommended_lags": [...]}`
- Lag cap: max 3 target lags (not 6) for speed

### C — `detect_seasonality(df, target_col, time_col)`
- STL decomposition + FFT + ACF peaks
- Return: `{"has_seasonality": bool, "dominant_period": int, "strength": float}`
- If detected: generate Fourier features `sin(2πkt/period)`, `cos(...)`

### D — `analyze_target_distribution(df, target_col)`
- Metrics: mean, std, skewness, kurtosis, CV
- Assess tree-model suitability (outliers < 5%, CV < 1.0)
- Return: `{"tree_model_suitable": "yes"|"no", "cv": float, "skewness": float}`

### E — `compute_state_space_embedding(series, embedding_dim=3)`
- Auto-delay selection (first MI local minimum)
- Embedding matrix: `[x(t), x(t-τ), x(t-2τ), ...]`
- Return: `{"embedding_matrix": np.ndarray, "chosen_delay": int}`

### F — `create_strata_features(df, time_col, target_col)`
- Hour-of-day, day-of-week, month, season (as applicable)
- ANOVA F-test usefulness check (p < 0.05)
- Return: `{"strata_features": {...}, "active_strata": [...]}`

### G — `engineer_timeseries_features(df, target_col, time_col, lags, rolling_windows)` (SELECTIVE)
- **Lag features**: Only for TOP-10 features (by MI from step 11) → `shift(lag)` for recommended_lags only
- **Rolling**: Only for TOP-10 features → `.shift(1).rolling(w).mean()`, `.std()` only (skip min/max/range for speed)
- **Windows**: Only [7, 30] (skip 14 for speed)
- **Differences**: `shift(1).diff(1)` only (skip diff(2))
- **Trend**: `t_elapsed_days` only (skip t_index, t_index_sq — monotone-index risks)
- **Calendar**: hour, day_of_week, month (skip quarter, is_weekend)
- **For other features**: Calendar + Trend only (no lags/rolling)
- **Return**: `(feature_df, metadata_dict)`

### H — `preselect_models(feature_matrix, analysis_data, best_lags)`
- Evaluate model types (XGBoost, SARIMA, LSTM, Ridge, Prophet, RF)
- Return: `{"top_recommendation": str, "top_3": [str, ...], "reasoning": {...}}`

### I — `add_features_for_models(feature_matrix, target_col, recommended_models, analysis_data)`
- Add model-specific features (ARIMA: diffs, LSTM: embedding, trees: interactions)
- Return: `(extended_feature_matrix, newly_added_features_list)`

### J — `detect_feature_leakage(feature_matrix, target_col, threshold=0.98)` (FAST)
- **Step 1**: Pearson |r| ≥ 0.98 with target → candidates
- **Step 2**: RandomForestRegressor(n_estimators=3, max_depth=3, random_state=42) on candidates; R² > 0.999 → leakage confirmed
  - **Optimization**: n_estimators=3 (not 10), max_depth=3 (shallow, fast)
- **Exempt**: Correct lag features (e.g., `target_lag_1` is allowed)
- **Return**: `{"status": "pass"|"fail", "leakage_candidates": [...], "probe_r2": float|null}`
- **On Fail**: `sys.exit(2)`, NO artifacts written

### K — `remove_zero_variance_features(feature_matrix, variance_threshold=1e-10)`
- Remove columns where `std() ≤ sqrt(threshold)`
- Return: `(cleaned_matrix, {"removed_feature": "zero_variance", ...})`
- Fail if < 2 features remain: `sys.exit(1)`

### L — `compute_scaling_metadata(feature_matrix, target_col, recommended_models, output_dir)` (FAST)
- **Linear/SARIMA/Prophet**: StandardScaler
- **LSTM/Temporal CNN**: MinMaxScaler (0..1)
- **Trees/GB**: No scaling
- **Binary features** (0/1 only): NEVER scale
- **Target**: NEVER scale
- **Optimization**: Only scale if top_recommendation is Linear/SARIMA/LSTM; skip for trees (most common)
- **Output**: Write `features_scaled.parquet` + save scaler to `scaler.joblib` (only if scaling applied)
- Return: `(scaled_matrix, {"scaler_used": str, "features_scaled": [...]})`

### `main()` — CLI Entry Point
```python
def main():
    # 1. Parse args: --output-dir, --run-id, --split-mode, --exclude-features
    # 2. Load inputs: cleaned.parquet, step-11-exploration.json
    # 3. Call functions Z → L in order
    # 4. Build output JSON
    # 5. Write features.parquet + step-12-features.json
    # 6. Update progress.json
    # 7. Return 0 on success, 1 on error, 2 on leakage
```

---

## Critical Implementation Rules

### Leakage Prevention (Highest Priority)
1. **Target variable removal**: Filter target column AFTER feature selection. If found in final features → `sys.exit(2)`
2. **Timestamp field**: Use ONLY for lag/trend calculation; never as raw numeric feature
3. **Lag cap**: Max 3 target lags (ACF-sorted; optimized for speed)
4. **Multi-series causality**: For multi-series data, use `.shift(n).over(group_col)` (never global `.shift()`)
5. **Rolling causality**: ALWAYS `.shift(1)` before `.rolling_*()`
6. **Pearson threshold**: 0.98 (NOT 0.99)
7. **Reconstruction probe**: RandomForest (n_estimators=3, max_depth=3) R² > 0.999 confirms leakage
8. **Monotone index**: FORBIDDEN — use `trend_elapsed_days` instead

### Performance Rules (OPTIMIZED)
- **Selective Feature Engineering**: Top-10 features get lags/rolling; others get calendar+trend only
- **Rolling Windows**: [7, 30] only (fast; skip 14, 60, 90)
- **Rolling Stats**: mean + std only (skip min, max, range for speed)
- **ACF Lag**: Max 12 (not 24 or N/4) for faster computation
- **RF Leakage Probe**: Shallow (max_depth=3, n_estimators=3) for speed
- **Scaling**: Only apply if recommended model is Linear/SARIMA/LSTM (skip for trees)
- **Fourier Features**: Only when seasonality_strength > 0.3 (skip weak seasonality)

### Feature Engineering Standards
- **All loops use `tqdm`** for progress tracking
- **Causal design**: No future data in any feature
- **Stationarity**: Apply diff(1) only if detected non-stationary
- **Zero-variance check**: BEFORE leakage detection, remove variance ≤ 1e-10

### Artifact Validation (Orchestrator Gate)
- `features` list is non-empty
- `features_excluded` documents all dropped columns
- `split_strategy.resolved_mode` is `"random"` or `"time_series"`
- `features.parquet` exists on disk
- No feature in `features` appears in `step-11["excluded_features"]`

---

## Execution Checklist (FAST VERSION)

- [ ] All 13 functions (Z–L) implemented and called from `main()` in order
- [ ] `argparse`: `--output-dir`, `--run-id`, `--split-mode`, `--exclude-features`
- [ ] Inputs: `cleaned.parquet`, `step-10-cleanse.json`, `step-11-exploration.json`
- [ ] Outputs: `features.parquet`, `step-12-features.json`, `scaler.joblib` (if scaled)
- [ ] **FAST OPTIMIZATIONS ACTIVE:**
  - [ ] Lags A/B: max_lag=12, top_n=3 (not 48/6)
  - [ ] Rolling G: only 2 windows [7, 30], mean+std only (no min/max/range)
  - [ ] Lags/Rolling G: only for TOP-10 features; others get calendar+trend only
  - [ ] Leakage J: RF with n_estimators=3, max_depth=3 (not 10, no limit)
  - [ ] Scaling L: only if needed (skip for trees, most common)
- [ ] `features_excluded` as dict (not list): `{feature_name: reason}`
- [ ] Leakage detected → `sys.exit(2)`, NO artifacts
- [ ] Fewer than 2 features → `sys.exit(1)`
- [ ] `progress.json` updated with `completed_steps` on success
- [ ] Exit code 0 on success
- [ ] `tqdm` on all loops
- [ ] No print() — only logging/JSON output

---

## Final Output Structure

```
output/<RUN_ID>/
├── step-12-features.json       # Complete audit trail
├── features.parquet            # Feature matrix + target
├── features_scaled.parquet     # Scaled features (if scaling applied; usually NOT)
├── scaler.joblib               # Persisted scaler object (if scaling applied)
└── progress.json               # Updated with 12-feature-extraction
```

## Performance Notes

**Expected Runtime** (with optimizations):
- Small datasets (< 5K rows): ~1-2 min
- Medium datasets (5-20K rows): ~3-5 min
- Large datasets (> 20K rows): ~5-8 min

**Why Fast?**
- Lags/Rolling only for TOP-10 features (80% fewer operations)
- RF leakage probe is shallow (max_depth=3, n_estimators=3)
- Scaling skipped for tree-based models (most common)
- ACF limited to lag 12 (fast, sufficient)