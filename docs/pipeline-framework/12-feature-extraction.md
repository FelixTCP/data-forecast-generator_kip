# Step 12 — Feature Extraction (Time-Series Focused)

**Script**: `CODE_DIR/step_12_features.py`  
**Input**: `OUTPUT_DIR/cleaned.parquet` + `OUTPUT_DIR/step-11-exploration.json`  
**Artifacts**: `OUTPUT_DIR/features.parquet`, `OUTPUT_DIR/step-12-features.json`, `OUTPUT_DIR/leakage_audit.json`

```
[10] csv_read_cleansing → [11] data_exploration → [12] feature_extraction → [13] model_training → ...
```

---

## Feature Philosophy

| Principle | Rule |
|---|---|
| Causal Rolling | `.shift(1)` applied **before** every `rolling_*` operation — prevents any look-ahead |
| Leakage → Hard Fail | `RuntimeError` if leakage probe fires — no artifact is written, pipeline stops |
| Leakage Probe | Pairwise Pearson correlation (|r| >= 0.98) **and** RF reconstruction probe (R² > 0.95) |
| Minimum Features | Fewer than 2 features after cleanup → `ValueError` |
| Step-11 Driven | All feature decisions (lags, diffs, Fourier periods) are read from step-11 diagnostics — no hardcoding |
| No Pandas Leakage | Convert polars → numpy only at the point of scikit-learn calls; keep pipeline logic in polars |

---

## Input Contract (from Step 11)

Step 12 reads the following fields from `step-11-exploration.json`:

| Field | Used for |
|---|---|
| `time_column` | Extracting calendar features |
| `detected_frequency` | Deriving default lag / Fourier periods |
| `ts_diagnostics.stationarity_conclusion` | Deciding whether differencing features are needed |
| `ts_diagnostics.acf_significant_lags` | ACF-driven lag feature set |
| `ts_diagnostics.pacf_significant_lags` | PACF-driven lag feature set |
| `ts_diagnostics.suggested_ar_order` | Upper bound for AR lag window |
| `ts_diagnostics.primary_seasonal_period` | Seasonal lag and Fourier period |
| `ts_diagnostics.detected_periods` | All significant seasonal periods for multi-period Fourier |
| `ts_diagnostics.hurst_exponent` | Extended lag window when H > 0.65 |
| `ts_diagnostics.trend_detected` | Whether to add differencing features |
| `ts_diagnostics.white_noise` | Short-circuit: skip complex features if true |
| `ts_diagnostics.multiple_series_detected` | Whether to add series-ID encoding |
| `ts_diagnostics.series_id_column` | Column for panel encoding |
| `recommended_features` | Exogenous feature columns to include |
| `useful_lag_features` | Feature-lag pairs with significant cross-correlation |
| `model_class_recommendations` | Which model families to prepare feature subsets for |

---

## Feature Engineering Groups

All groups below are applied unless the `white_noise=true` short-circuit fires.  
If `white_noise=true`, only Group A (calendar) and Group B lag-1 are built — complex engineered features are useless for a white-noise target.

### Group A — Calendar Features (always built when time column exists)

Extract from the parsed datetime column. These are known in advance — never shifted:

| Feature | Description |
|---|---|
| `hour_of_day` | 0–23 (hourly or sub-hourly data) |
| `day_of_week` | 0 (Mon) – 6 (Sun) |
| `day_of_month` | 1–31 |
| `month` | 1–12 |
| `quarter` | 1–4 |
| `week_of_year` | ISO week 1–53 |
| `is_weekend` | 1 if day_of_week >= 5, else 0 |
| `is_month_start` | 1 if day_of_month = 1 |
| `is_month_end` | 1 if last day of month |

Only include granularities matching the data frequency. For 10-min data include `hour_of_day`; for daily data skip it.

### Group B — Lag Features (ACF/PACF-driven)

Build target lags from `acf_significant_lags` and `pacf_significant_lags` in step 11.

- Always include the **union** of ACF and PACF significant lags.
- Always include lag 1 (minimum first-order AR term).
- If `hurst_exponent > 0.65`: extend lag window to `min(primary_seasonal_period * 2, 96)`.
- If `primary_seasonal_period` not null: always include lags `[m, 2*m]` where m = primary_seasonal_period.
- Cap total target lag features at **30** (keep highest-ACF lags) to prevent dimensionality explosion.

```python
# CORRECT: shift k steps to get lag k
df = df.with_columns(pl.col(target).shift(k).alias(f"y_lag_{k}"))
```

### Group C — Exogenous Feature Lags (cross-correlation driven)

For each entry in `useful_lag_features` from step 11 (|xcorr| > 0.15):
- Create `{feature}_lag_{k}` using `.shift(k)`.
- Never use a raw exogenous feature at lag 0 unless it is a forward-known variable (e.g., calendar, external schedule).

### Group D — Differencing Features (stationarity-driven)

Apply only when `stationarity_conclusion` is `non-stationary` or `trend-stationary`:

| Feature | Formula | When to add |
|---|---|---|
| `y_diff_1` | `y(t-1) - y(t-2)` = `.shift(1).diff(1)` | Always when non-stationary |
| `y_diff_seasonal` | `y(t-1) - y(t-1-m)` = `.shift(1).diff(m)` | When `seasonality_detected=true` and m not null |

**Rule**: diff is always applied to the already-shifted series to preserve causality.

### Group E — Rolling Statistics (always with prior shift)

Rolling statistics over the **lagged target** `.shift(1)` to prevent look-ahead.

Default windows: `[primary_seasonal_period // 2, primary_seasonal_period, primary_seasonal_period * 2]`  
Fallback if no seasonal period: `[6, 12, 24]`.

| Feature | Implementation |
|---|---|
| `rolling_mean_{w}` | `pl.col(target).shift(1).rolling_mean(w)` |
| `rolling_std_{w}` | `pl.col(target).shift(1).rolling_std(w)` |
| `rolling_min_{w}` | `pl.col(target).shift(1).rolling_min(w)` |
| `rolling_max_{w}` | `pl.col(target).shift(1).rolling_max(w)` |
| `ewm_span_{m}` | Exponentially weighted mean, span = primary_seasonal_period |

### Group F — Fourier Features (seasonality-driven)

For every **significant** period in `detected_periods` (seasonal_strength > 0.30):
- Build K harmonic pairs where K = `min(3, period // 4)`.
- `fourier_sin_{m}_{k} = sin(2π·k·t / m)` where `t` is the integer position within the cycle (0..m-1).
- `fourier_cos_{m}_{k} = cos(2π·k·t / m)`.
- Fourier features represent calendar structure at time t — no shift needed.

```python
t_index = np.arange(len(df)) % period
fourier_sin = np.sin(2 * np.pi * k * t_index / period)
fourier_cos = np.cos(2 * np.pi * k * t_index / period)
```

### Group G — PCA Factor Components (for FAAR models)

Built only when `model_class_recommendations` contains `FAAR-ARIMA`, `FAAR-SARIMAX`, or `Factor-VAR`.

1. Collect all exogenous feature columns (from `recommended_features` after their lags are built).
2. Standardise with `StandardScaler` — fit on training portion only.
3. Apply `PCA(n_components=k)` where k captures ≥ 95% of variance (cap at `n_recommended_features`).
4. Store as `pca_factor_1`, `pca_factor_2`, ..., `pca_factor_k`.
5. Serialise fitted scaler + PCA to `OUTPUT_DIR/pca_preprocessor.joblib`.
6. Write `pca_n_components`, `pca_explained_variance_ratio`, `pca_loadings_dict` to output JSON.

**Critical**: PCA is fitted on **training rows only** (index < `holdout_start_index`) and applied to the full series.

---

## Leakage Detection (Hard Fail)

Executed after all feature groups are built, **before any artifact is written**.

### Probe 1 — Pearson Correlation Gate
For every feature column `f`: compute `|pearsonr(f, y)|`.  
If any `|r| >= 0.98` → flag that column as `leakage_suspect`.

### Probe 2 — RF Reconstruction Probe
If Probe 1 finds suspects: fit `RandomForestRegressor(n_estimators=50, random_state=42, oob_score=True)` on suspect columns vs. target (training rows only). If OOB R² > 0.95 → leakage confirmed.

**On confirmed leakage**:
- Write `leakage_audit.json` with `"status": "fail"` + full diagnostics.
- Raise `RuntimeError("Leakage detected — see leakage_audit.json")`.
- **Do not write** `features.parquet` or `step-12-features.json`.

**Probe 1 fires, Probe 2 R² ≤ 0.95**: write `leakage_audit.json` with `"status": "warn"`, proceed.

**No suspects**: write `leakage_audit.json` with `"status": "pass"`.

---

## NaN Handling Before Artifact Write (mandatory)

After all feature groups are built and before leakage probes run, apply the following NaN cleanup **in this exact order**:

```python
# 1. Drop rows where the TARGET column is NaN — these are un-trainable
df = df.drop_nulls(subset=[target_column])

# 2. Drop leading NaN rows introduced by lag/rolling features
#    (rows at the START of the series where all lag features are null)
#    Identify the first row where the maximum-lag column is non-null:
max_lag_col = max(lag_feature_cols, key=lambda c: int(c.split("_")[-1]))  # e.g. y_lag_144
first_valid = df[max_lag_col].is_not_null().arg_true()[0]
df = df.slice(first_valid)

# 3. For any remaining NaN in feature columns (e.g. from missing exogenous data),
#    forward-fill then backward-fill with polars:
df = df.with_columns([
    pl.col(c).forward_fill().backward_fill()
    for c in feature_cols if df[c].null_count() > 0
])

# 4. Assert: after cleanup, NO NaN must remain in feature columns
assert df.select(pl.all().is_null().any()).to_numpy().any() == False, \
    "features.parquet still contains NaN after cleanup"
```

> **⛔ CRITICAL**: `features.parquet` must contain ZERO NaN values in feature columns before it is written. NaN in features causes ALL sklearn models to crash with `ValueError: Input X contains NaN`. The SimpleImputer in the step-13 pipeline is a last-resort safety net — step 12 is the authoritative NaN-free guarantee.

Record `"rows_dropped_by_lags"` and `"rows_with_forward_fill"` in the output JSON.

---

## Split Boundary

Step 12 computes the holdout boundary once and records it so step 13 uses the **identical split**:

- Holdout = last chronological 20% of rows (integer index, rounded down).
- Record `holdout_start_index`, `holdout_start_timestamp`, `train_row_count`, `holdout_row_count`.
- Write to `step-12-features.json` under `split_info`.

---

## Output JSON Schema

```json
{
  "step": "12-feature-extraction",
  "target_column": "appliances",
  "feature_names": ["y_lag_1", "y_lag_2", "y_lag_144", "hour_of_day", "day_of_week",
                    "fourier_sin_144_1", "fourier_cos_144_1", "rolling_mean_72",
                    "pca_factor_1", "pca_factor_2"],
  "feature_count": 42,
  "rows_dropped_by_lags": 144,
  "final_row_count": 19591,

  "split_info": {
    "holdout_start_index": 15673,
    "holdout_start_timestamp": "2016-04-15T00:00:00",
    "train_row_count": 15673,
    "holdout_row_count": 3918,
    "split_strategy": "last_20pct_chronological"
  },

  "feature_groups": {
    "calendar": ["hour_of_day", "day_of_week", "is_weekend"],
    "target_lags": ["y_lag_1", "y_lag_2", "y_lag_144"],
    "exogenous_lags": ["t1_lag_1", "t6_lag_3"],
    "differencing": ["y_diff_1", "y_diff_144"],
    "rolling": ["rolling_mean_72", "rolling_std_72", "rolling_mean_144"],
    "fourier": ["fourier_sin_144_1", "fourier_cos_144_1"],
    "pca_factors": ["pca_factor_1", "pca_factor_2"]
  },

  "pca_info": {
    "n_components": 2,
    "explained_variance_ratio": [0.61, 0.18],
    "cumulative_variance": 0.79,
    "pca_preprocessor_path": "OUTPUT_DIR/pca_preprocessor.joblib"
  },

  "leakage_audit": {
    "status": "pass",
    "probe1_suspects": [],
    "probe2_r2": null,
    "threshold": 0.98
  },

  "features_excluded": {
    "rv1": "excluded_by_step_11_noise_baseline",
    "rv2": "excluded_by_step_11_redundant"
  },

  "artifacts": {
    "features_parquet": "OUTPUT_DIR/features.parquet",
    "leakage_audit_json": "OUTPUT_DIR/leakage_audit.json"
  }
}
```

`features.parquet` contains all feature columns plus the target column, chronologically ordered, with NaN rows from lag-creation dropped from the **start** of the series only.

---

## CLI Contract

```bash
python step_12_features.py \
  --output-dir OUTPUT_DIR \
  --run-id RUN_ID \
  [--target-column TARGET_COLUMN]
```

Reads `TARGET_COLUMN` from `progress.json` if not supplied via CLI.  
Reads `step-11-exploration.json` for all TS diagnostics.  
Reads `cleaned.parquet` as input data.

---

## Implementation Checklist

- [ ] All feature groups A–G implemented
- [ ] `.shift(1)` applied before every rolling call — verified by code review
- [ ] Differencing logic reads `stationarity_conclusion` from step-11 JSON — not hardcoded
- [ ] Lag selection reads `acf_significant_lags` + `pacf_significant_lags` from step-11 JSON
- [ ] Fourier periods read from `detected_periods` list — supports multi-period
- [ ] PCA fitted on training portion only; scaler+PCA saved to `pca_preprocessor.joblib`
- [ ] Leakage probes run before any artifact is written
- [ ] `leakage_audit.json` always written (pass, warn, or fail)
- [ ] `split_info` written to JSON for step 13 consumption
- [ ] `features.parquet` written via `feature_df.write_parquet(parquet_path)`
- [ ] `output_dir` created with `mkdir(parents=True, exist_ok=True)`
- [ ] Logging via `logging`, no bare `print()` statements
- [ ] `execution_id` via `str(uuid.uuid4())[:8]`; `runtime_seconds` via `time.time()`
- [ ] All JSON values serialisable (`default=str` as fallback)
- [ ] `progress.json` updated at start and on successful completion

---

## Tests

- `white_noise=true`: only calendar + lag-1 present; no differencing, rolling, or Fourier features
- Non-stationary input: `y_diff_1` and `y_diff_seasonal` present in feature names
- Stationary input: no differencing features created
- Fourier: sin/cos values correct at t=0, t=m/4, t=m/2 for a known period m
- Rolling: `rolling_mean_w` values match manually computed `.shift(1).rolling_mean(w)`
- Leakage Probe 1: feature with |r| >= 0.98 → `leakage_audit.json` `"status": "fail"`, RuntimeError raised
- Leakage Probe 1 fires but RF OOB R² <= 0.95 → `"status": "warn"`, no RuntimeError
- PCA: components present when FAAR in `model_class_recommendations`; absent when not
- Split info: `holdout_start_index` = floor(0.8 * total_rows)
- Lag cap: ACF returns > 30 significant lags → feature list capped at 30 highest-ACF lags
