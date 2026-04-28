# Step 12 — Feature Extraction & Model Preselection

**Module**: `src/data_forecast_generator/pipeline/feature_extraction.py`  
**Input**: Output from `data_exploration.py` (Step 11)  
**Output**: Feature Matrix + Model Recommendations → `model_training.py` (Step 13)  
**Artifacts**: `outputs/step-12/features.parquet`, `outputs/step-12/step-12-features.json`

```
[10] csv_read_cleansing → [11] data_exploration → [12] feature_extraction → [13] model_training → ...
```

---

## Feature Philosophy

| Principle | Rule |
|---|---|
| Causal Rolling | `.shift(1)` before every `rolling_*` calculation — prevents look-ahead |
| Leakage → Hard Fail | `RuntimeError` if `\|r\| ≥ threshold` — no artifact is written |
| Leakage Probe | Pairwise Pearson correlation **and** reconstruction probe (RF R² > 0.999) |
| Minimum Features | Fewer than 2 features after cleanup → `ValueError` |

---

## Input Contract (Output from Step 11)

The input for `feature_extraction.py` is the direct output from `data_exploration.py`:

```python
exploration_output = {
    # === METADATA ===
    "step": "11-data-exploration",
    "metadata": {
        "execution_id": "expl-2024-04-09-123",
        "module_name": "data_exploration",
        "timestamp_created": "2024-04-09T12:34:56Z",
        "source_file": "appliances_energy.csv",
    },

    # === SHAPE & COLUMN INFO ===
    "shape": {"rows": 19735, "columns": 29},
    "numeric_columns": ["Appliances", "lights", "T1", "RH_1", "T_out", ...],
    "high_cardinality": [],
    "low_variance_columns": [],

    # === RAW DATA ===
    "data": {
        "df": "pl.DataFrame (in-memory)",
        "time_col": "date",
        "numeric_cols": ["Appliances", "lights", "T1", "RH_1", "T_out", ...],
        "categorical_cols": [],
        "target_column": null
    },

    # === EXPLORATION RESULTS ===
    "exploration": {
        "frequency": "10min",                                  # Detected time series frequency
        "stationarity_adf_pvalue": 0.23,                       # ADF test p-value
        "missing_fraction": 0.0,                               # Fraction of missing values (before cleaning)
        "column_stats": {                                      # Descriptive statistics per column
            "Appliances": {"mean": 93.7, "std": 102.3, "min": 0, "max": 2942},
            "lights": {"mean": 4.2, "std": 8.4, "min": 0, "max": 176},
        },
    },

    # === ANALYSIS RESULTS ===
    "analysis": {
        "mi_ranking": [                                        # Mutual Information ranking (descending)
            {"feature": "T6", "mi_score": 0.42, "below_noise_baseline": false},
            {"feature": "T1", "mi_score": 0.35, "below_noise_baseline": false},
        ],
        "noise_mi_baseline": 0.005,                            # Average MI of noise columns
        "redundant_columns": ["rv2"],                          # Redundant features (high correlation)
        "correlation_matrix_summary": {
            "max_pair": ["rv1", "rv2"],
            "max_corr": 1.0
        },
        "recommended_features": ["T6", "T1", "RH_6", "lights", "T_out"],  # Recommended features for Step 12
        "excluded_features": {
            "rv1": "below_noise_baseline",
            "rv2": "redundant",
            "target_copy": "leakage_suspect"
        }
    },

    # === TIME SERIES SPECIFIC ANALYSIS ===
    "time_series_detected": true,
    "time_column": "date",
    "multiple_series_detected": false,
    "time_series_characteristics": {
        "trend_detected": true,                                # Trend present?
        "seasonality_detected": true,                          # Seasonality present?
        "stationarity": "non-stationary",                      # ADF test result
        "white_noise": false                                   # Is the series pure noise?
    },
    "model_recommendations": ["SARIMA", "Prophet", "XGBoost"],  # Recommended model architectures
    "significant_lags": [1, 3, 6],                             # Lags with significant autocorrelation
    "useful_lag_features": [                                   # Features with significant cross-correlation
        {"feature": "T1", "lag": 1, "xcorr": 0.23},
        {"feature": "RH_6", "lag": 3, "xcorr": 0.18},
    ],

    # === CLIENT-FACING SUMMARY ===
    "client_facing_summary": "Your target variable shows a strong trend and seasonal patterns. Features like T6 and T1 are highly predictive, whereas rv1 and rv2 were excluded due to being redundant or pure noise.",

    # === ERRORS & WARNINGS ===
    "errors": [],
    "warnings": [],
}
```

**Contract Guarantees:**
- Input is the **exact output from Step 11** (no transformation)
- `df` is a polars.DataFrame with no NaN in the target variable
- `numeric_cols` and `categorical_cols` cover all columns
- `recommended_features` is never empty (or error is raised in Step 11)
- `analysis.excluded_features` explains every excluded feature
- `time_series_characteristics`, `model_recommendations`, `significant_lags` are always present (even if `time_series_detected=false`)

---

## Output Contract (for Training Module)

```python
feature_output = {
    # === METADATA ===
    "metadata": {
        "execution_id": "feat-2024-04-09-456",           # uuid4()[:8]
        "module_name": "feature_extraction",
        "timestamp_created": "2024-04-09T14:23:45Z",
        "runtime_seconds": 12.34,
    },

    # === INPUT REFERENCE ===
    "input_ref": {
        "exploration_execution_id": "expl-2024-04-09-234",
        "source_file": "energy_demand.csv",
    },

    # === TARGET COLUMN ===
    "target_info": {
        "target_column": "energy",
        "detection_method": "highest_variance",           # "explicit" if passed via config
        "detection_info": {
            "candidates": ["energy", "temperature"],
            "scores": {"energy": 0.95, "temperature": 0.42},
            "reason": "energy has highest variance (0.95)",
        },
    },

    # === FEATURES ===
    "features": {
        "feature_matrix": pl.DataFrame,                  # Shape: (n_rows, n_features), numeric only, incl. target
        "target_series": pl.Series,                      # Target variable (in-memory for next module)
        "parquet_path": "outputs/step-12/features.parquet",  # Path to saved feature matrix
        "feature_names": ["y_lag_1", "y_lag_24", "hour", "is_weekend"],
        "feature_count": 47,
        "rows_dropped_by_lags": 50,
        "final_row_count": 8650,
        "adaptive_features_added": ["y_diff_1", "embedding_matrices"],
    },

    # === ARTIFACTS ===
    "artifacts": {
        "features_parquet": "outputs/step-12/features.parquet",  # Feature matrix (numeric, incl. target)
        "audit_json": "outputs/step-12/step-12-features.json",   # Complete audit trail
    },

    # === LEAKAGE ===
    "leakage": {
        "status": "pass",                # "pass" | "fail" — on "fail" RuntimeError is raised
        "leakage_candidates": [],        # Feature names with |r| >= threshold
        "correlations": {"y_lag_0": 1.0, "y_lag_1": 0.87},
        "threshold": 0.98,
        "reconstruction_probe_r2": None, # RF R² of leakage probe (None if no candidate)
    },

    # === SCALING METADATA ===
    "scaling_metadata": {
        "scaling_required": True,
        "per_model": {
            "Gradient Boosting": {"scaler": None,            "features": []},
            "SARIMA":            {"scaler": "StandardScaler", "features": ["y_lag_1", "hour"]},
        },
        "never_scale": ["is_weekend", "hour_of_day"],
    },

    # === ANALYSIS RESULTS ===
    "analysis": {
        "best_lags": [1, 2, 3, 24, 48],
        "seasonality": {
            "strength": 0.72,
            "label": "strong",
            "dominant_period": 24,
        },
        "target_distribution": {
            "mean": 23.1,
            "std": 14.2,
            "cv": 0.61,
            "tree_model_suitable": "conditional",
        },
        "strata": {
            "active": ["hour_of_day", "day_of_week"],
            "variance_ratios": {"hour": 0.45, "day": 0.32},
        },
    },

    # === MODEL RECOMMENDATION ===
    "model_recommendations": {
        "recommendations": [
            {
                "model_type": str,
                "suitability": "suitable" | "somewhat suitable" | "not suitable",
                "score": float,                          # 0.0–1.0
                "reasoning": str,
                "required_features": list[str],
                "required_features_missing": list[str],
                "why_good_fit": str,
            },
            # ...
        ],
        "top_recommendation": str,
        "top_3_recommendations": list[str],
        "adaptive_features_needed": dict[str, list[str]],
    },

    # === ERRORS & WARNINGS ===
    "errors": [],
    "warnings": [
        "Time series was shortened by 50 rows due to lag",
        "State Space Embedding: time series < 1 year, skipped",
    ],
}
```

---

## Entry Function

```python
# import json, logging, time, uuid
# from datetime import datetime, timezone
# from pathlib import Path
# import polars as pl

def run_analysis(
    exploration_output: dict,  # Output from Step 11 (see Input Contract above)
    config: dict | None = None,
    # config keys: max_lag=48, embedding_dim=3, use_state_space=True,
    #              rolling_windows=[7,14,30], output_dir="outputs/step-12",
    #              target_column=None  # optional: explicit target column, overrides auto-detection
) -> dict:
    """
    **Input**: exploration_output from Step 11
      - exploration_output["data"]["df"] is used as input DataFrame
      - exploration_output["data"]["numeric_cols"] defines features
      - exploration_output["analysis"]["recommended_features"] guides feature selection
      - exploration_output["time_series"]["significant_lags"] guides lag creation
      - exploration_output["time_series"]["time_series_characteristics"] guides feature strategy
      - exploration_output["time_series"]["model_recommendations"] determines feature subsets

    **Output**: Feature extraction output (see Output Contract)
      - Writes features.parquet and step-12-features.json (only if leakage.status="pass")
      - feature_matrix is pl.DataFrame with all engineered features + target
      - model_recommendations lists which features are used for which model

    **Error Handling**:
      - ValueError for invalid input (too few features, empty DataFrame, etc.)
      - RuntimeError when leakage is detected — no artifact is written
    """
```

---

## Feature Engineering Strategy (based on TS Characteristics)

This strategy adapts feature creation to the results of Step 11:

| TS Characteristic | Recommendation | Feature Action |
|---|---|---|
| **Trend Detected** | SARIMA, Prophet, XGBoost | Differencing features: `diff_1`, `diff_7`, `diff_365` |
| **Seasonality Detected** | SARIMA, Prophet, Prophet-QR | Seasonal Lags: `y_lag_24`, `y_lag_168` (12h, 1W for hourly data) |
| **Stationary** | Gradient Boosting, Linear Regression | No differencing needed; raw lags suffice |
| **Non-Stationary** | SARIMA, Prophet, ETS | Differencing features MUST be created |
| **White Noise Detected** | Naive, Exponential Smoothing | Only trivial lag features; complex engineered features are useless |
| **Multiple Series** | XGBoost with grouping, LightGBM | Categorical features for series ID + cross-series features |

### Concrete Implementation:

```python
def create_adaptive_features(
    exploration_output: dict,
    df: pl.DataFrame,
    target_col: str,
) -> pl.DataFrame:
    """
    Creates features adaptively based on time_series_characteristics.
    
    Procedure:
    1. Read exploration_output["time_series"]["time_series_characteristics"]
    2. Read exploration_output["time_series"]["significant_lags"]
    3. For each recommended model type in model_recommendations:
       - Create feature subset for this model
    4. Union all subsets in final feature_matrix
    
    Example:
        if time_series_characteristics["trend_detected"]:
            features["diff_1"] = df[target_col].diff(1)
            features["diff_7"] = df[target_col].diff(7)
        
        if time_series_characteristics["seasonality_detected"]:
            # Detect frequency from exploration_output["exploration"]["frequency"]
            seasonal_lag = infer_seasonal_lag(frequency)
            for lag in significant_lags:
                if lag % seasonal_lag == 0:
                    features[f"y_lag_{lag}"] = df[target_col].shift(lag)
    """
```

---

## Function Specifications

### Z — Target Column Detection

```python
def auto_detect_target_column(
    df: pl.DataFrame,
    time_col: str,
    numeric_cols: list[str],
    heuristic: str = "highest_variance",
    explicit_target: str | None = None,
) -> tuple[str, dict]:
    """
    Returns explicit target column if provided (after validation against numeric_cols).
    Otherwise: automatic selection from numeric columns.

    Heuristics (only for auto-detection):
    - "highest_variance": Column with highest variance
    - "most_correlated_with_self": Highest autocorrelation

    Exclusions (only for auto-detection): time_col, constant columns (variance ≈ 0)

    Returns:
        (target_column_name, detection_info)
        detection_info: {"method": "explicit"|"highest_variance"|..., "candidates", "scores", "reason"}

    Raises:
        ValueError: If explicit_target not in numeric_cols
    """
```

---

### A — Mutual Information

```python
def compute_lag_mutual_information(
    df: pl.DataFrame,
    target_col: str,
    max_lag: int = 48,
    n_neighbors: int = 5,
) -> pl.DataFrame:
    """
    Computes MI between y(t) and y(t-lag) for lag=1..max_lag.

    Uses: sklearn.feature_selection.mutual_info_regression

    Returns:
        pl.DataFrame with columns [lag, mutual_information], sorted by MI desc
    """
```

---

### B — Best Lags (ACF + PACF + MI)

```python
def find_best_lags(
    df: pl.DataFrame,
    target_col: str,
    max_lag: int = 48,
    top_n: int = 10,
    method: str = "combined",
) -> dict:
    """
    Combines ACF, PACF, and MI for lag selection.

    - statsmodels.tsa.stattools.acf / pacf
    - 95% confidence bounds: ±1.96 / √n

    Returns:
        {
            "best_lags_acf": [...],
            "best_lags_pacf": [...],
            "best_lags_mi": [...],
            "recommended_lags": [...],
            "acf_values": pl.DataFrame,
            "pacf_values": pl.DataFrame,
        }
    """
```

---

### C — Seasonality Detection

```python
def detect_seasonality(
    df: pl.DataFrame,
    target_col: str,
    time_col: str,
    candidate_periods: list[int] | None = None,
) -> dict:
    """
    STL decomposition + FFT + ACF peaks.

    Classification:
    - < 0.2  → "no clear seasonality"
    - 0.2–0.5 → "weak seasonality"
    - > 0.5  → "strong seasonality"

    When seasonality is detected: generate Fourier features
        sin(2π·k·t/period), cos(2π·k·t/period) for k=1,2

    Returns:
        {"seasonality_strength", "seasonality_label", "dominant_period",
         "seasonal_features", ...}
    """
```

---

### D — Target Distribution

```python
def analyze_target_distribution(
    df: pl.DataFrame,
    target_col: str,
) -> dict:
    """
    Analyzes y distribution for model selection.

    Metrics: min, max, mean, std, skewness, kurtosis, IQR, CV

    Tree-model heuristic:
    - Outlier fraction < 5%: positive
    - CV < 1.0: positive
    - CV > 3.0: negative
    - |skew| < 1: positive

    Returns:
        {"min", "max", "mean", "std", "cv",
         "tree_model_suitable": "yes"|"conditional"|"not suitable",
         "tree_model_reasoning": str, ...}
    """
```

---

### E — State-Space Embedding

```python
def compute_state_space_embedding(
    series: pl.Series,
    embedding_dim: int = 3,
    delay: int | None = None,
    max_delay_search: int = 20,
) -> dict:
    """
    Delay embedding with automatic delay selection (Takens' theorem).

    Auto-delay: first local minimum of MI curve
    Embedding matrix: [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]

    Returns:
        {"embedding_matrix": np.ndarray,  # shape (n_valid, embedding_dim)
         "chosen_delay": int,
         "embedding_features": pl.DataFrame,
         "notes": str}
    """
```

---

### F — Strata Features

```python
def create_strata_features(
    df: pl.DataFrame,
    time_col: str,
    target_col: str,
) -> dict:
    """
    Time-based stratification with ANOVA F-test.

    Activation based on frequency & data length:
    - hour_of_day  → if freq ≤ 1h
    - day_of_week  → always if timestamp present
    - month        → if data ≥ 60 days
    - season       → if data ≥ 180 days

    Usefulness check: ANOVA F-test, p < 0.05 → useful

    Returns:
        {"strata_features", "active_strata", "strata_usefulness",
         "variance_by_strata"}
    """
```

---

### G — Feature Engineering

```python
def engineer_timeseries_features(
    df: pl.DataFrame,
    target_col: str,
    time_col: str,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    include_calendar: bool = True,
    include_trend: bool = True,
    include_differences: bool = True,
) -> tuple[pl.DataFrame, dict]:
    """
    Creates the complete feature matrix.

    Features:
    - Lag:       y_lag_{k}
    - Rolling:   y_rolling_mean_{w}, _std, _min, _max, _range
                 MANDATORY: `.shift(1).rolling_*(w)` — no look-ahead!
                 Violation → feature is excluded + logged in warnings.
    - Diff:      y_diff_1, y_diff_2, y_pct_change_1
    - Trend:     t_index, t_index_squared, t_elapsed_days
    - Calendar:  hour, day_of_week, month, quarter, year, is_weekend

    NaN handling: Drop first N rows (= max lag / window).
    Fewer than 2 features after cleanup → ValueError.

    Returns:
        (feature_df, feature_metadata_dict)
    """
```

---

### H — Model Preselection

```python
def preselect_models(
    feature_matrix: pl.DataFrame,
    analysis_data: dict,
    best_lags: list[int],
) -> dict:
    """
    Evaluates model types based on actually available features.

    Model types & typical feature requirements:

    1. Gradient Boosting (XGBoost, LightGBM)
       ✓ Lags, rolling, calendar, trend (tabular)
    2. SARIMA
       ✓ y_diff_1/2 (stationarization), dominant seasonal period
       ✗ Too many exogenous regressors problematic
    3. LSTM / Temporal CNN
       ? embedding_matrices / sequences (possibly adaptive generation)
       ✓ Long time series (> 500 points)
    4. Linear Models (Ridge, Lasso)
       ✗ High MI with small ACF → nonlinear → unsuitable
    5. State-Space (Kalman, Prophet)
       ✓ Trend + seasonality features, tolerates missing values
    6. Tree-Based (RF, ExtraTrees)
       ✓ All features; risk of overfitting with very many features

    Returns:
        {
            "recommendations": [{
                "model_type", "suitability", "score",
                "reasoning", "required_features",
                "required_features_missing", "why_good_fit"
            }, ...],
            "top_recommendation": str,
            "top_3_recommendations": list[str],
            "adaptive_features_needed": dict[str, list[str]],
        }
    """
```

---

### I — Adaptive Feature Engineering

```python
def add_features_for_models(
    feature_matrix: pl.DataFrame,
    target_col: str,
    recommended_models: list[str],
    analysis_data: dict,
) -> tuple[pl.DataFrame, list[str]]:
    """
    Adds missing, model-specific features afterwards.

    Examples:
    - For ARIMA/SARIMA: y_diff_1, y_diff_2 (stationarization)
    - For LSTM/Temporal CNN: state-space embedding columns
    - For tree-based: interaction features between top lags
    - For state-space: explicit trend and seasonality features

    Returns:
        (feature_matrix_extended, newly_added_feature_names)
    """
```

---

### J — Leakage Detection

```python
def detect_feature_leakage(
    feature_matrix: pl.DataFrame,
    target_col: str,
    threshold: float = 0.98,
) -> dict:
    """
    Checks if features unnaturally strongly anticipate the target column.

    Method:
    1. Pairwise Pearson correlation: |r| ≥ threshold → leakage candidate
    2. Reconstruction probe: RandomForestRegressor (n_estimators=10) on leakage candidates;
       R² > 0.999 → leakage confirmed
    - Special rule: exclude target_col itself and correct lag features (y_lag_k)

    Returns:
        {
            "status": "pass" | "fail",
            "leakage_candidates": list[str],         # Feature names with |r| >= threshold
            "correlations": dict[str, float],        # {feature: r} for all features
            "threshold": float,
            "reconstruction_probe_r2": float | None, # RF R² of probe (None if no candidate)
        }

    Raises:
        ValueError: If target_col not in feature_matrix
        RuntimeError: If status == "fail" (passed on by run_analysis)
    """
```

When `status == "fail"`, `run_analysis` raises a `RuntimeError` — no artifact is written.

---

### K — Feature Scaling Metadata

```python
def compute_scaling_metadata(
    feature_matrix: pl.DataFrame,
    target_col: str,
    recommended_models: list[str],
) -> dict:
    """
    Determines which features must be scaled for which model types.
    Scaling itself occurs in Step 13; this dict is the instruction for it.

    Rules:
    - Linear Models / SARIMA / State-Space: all numeric features → StandardScaler
    - Gradient Boosting / Tree-based / RF:  no scaler needed
    - LSTM / Temporal CNN:                  all features → MinMaxScaler (0..1)
    - Binary features (0/1):               never scale

    Returns:
        {
            "scaling_required": bool,
            "per_model": {
                "Gradient Boosting": {"scaler": None,            "features": []},
                "SARIMA":            {"scaler": "StandardScaler", "features": ["y_lag_1", ...]},
                "LSTM":              {"scaler": "MinMaxScaler",   "features": ["y_lag_1", ...]},
                # ...
            },
            "never_scale": list[str],   # Binary features (is_weekend, etc.)
        }
    """
```

---

## Implementation Checklist

- [ ] All functions (Z, A–K) fully implemented
- [ ] `import json, logging, time, uuid` and `from pathlib import Path` present
- [ ] No `argparse`, no CLI code, no `print()`
- [ ] Input validation with meaningful `ValueError`
- [ ] `execution_id` via `uuid.uuid4()[:8]`
- [ ] `runtime_seconds` via `time.time()`
- [ ] `features.parquet` written via `feature_df.write_parquet(parquet_path)`
- [ ] `step-12-features.json` contains complete audit trail (no `pl.DataFrame`/`pl.Series` directly — only serializable types; `default=str` as fallback)
- [ ] `output_dir` created with `mkdir(parents=True, exist_ok=True)`
- [ ] Return dict contains `artifacts` with `features_parquet` and `audit_json` as path strings
- [ ] Return dict contains `features.feature_matrix` as `pl.DataFrame` (in-memory for next module)
- [ ] Output dictionary matches the Output Contract exactly
- [ ] All functions stateless (no global variables)
- [ ] Type hints complete
- [ ] `leakage` block in output with `status: "pass"|"fail"`; on `"fail"` → `RuntimeError`, no artifact
- [ ] Leakage probe: RandomForest R² > 0.999 counts as confirmed leakage
- [ ] Rolling features: `.shift(1)` before each rolling calculation; violation → feature excluded
- [ ] `scaling_metadata` block in output; binary features in `never_scale`
- [ ] Tests under `tests/test_feature_extraction.py`

---

## 🤖 AGENT IMPLEMENTATION DIRECTIVE

**ATTENTION: This section is for agent implementation. ALL 14 functions MUST be present.**

### Required 14 Functions (ALL IN ONE FILE)

**KEEP PRAGMATIC: 5-15 lines per function, not over-engineered**

1. **Z — auto_detect_target_column()**
   - Input: `df`, `numeric_cols`, `explicit_target=None`
   - Output: `(target_col: str, detection_info: dict)`
   - Logic: If `explicit_target` → validate against `numeric_cols`. Else → highest variance

2. **A — compute_lag_mutual_information()**
   - Input: `df`, `target_col`, `max_lag=48`
   - Output: `pl.DataFrame` with columns `[lag, mutual_information]`
   - Use: `sklearn.feature_selection.mutual_info_regression`

3. **B — find_best_lags()**
   - Input: `df`, `target_col`, `max_lag=48`, `top_n=10`
   - Output: `dict` with `best_lags`, `acf_values`, `pacf_values`
   - Use: ACF + PACF + MI combined

4. **C — detect_seasonality()**
   - Input: `df`, `target_col`, `frequency_str="10min"`
   - Output: `dict` with `has_seasonality`, `period`, `strength`
   - Simple: Detect periodic pattern in data

5. **D — create_differencing_features()**
   - Input: `df`, `target_col`
   - Output: `dict[str, pl.Series]` with `diff_1`, `diff_7`, possibly `diff_365`
   - Logic: Only if non-stationary detected

6. **E — create_temporal_features()**
   - Input: `df`, `time_col`
   - Output: `dict[str, pl.Series]` with `hour_of_day`, `day_of_week`, `month`, `quarter`
   - Logic: Extract from datetime column

7. **F — detect_leakage()**
   - Input: `df`, `target_col`, `threshold=0.98`
   - Output: `dict` with `leakage_status: "pass"|"fail"`, `candidates: list[str]`
   - Logic: Pearson correlation > threshold with target → leakage suspicion

8. **G — create_adaptive_features()**
   - Input: `df`, `target_col`, `exploration_output`
   - Output: `dict[str, pl.Series]` with adapted features
   - Logic: IF trend_detected → diff. IF seasonality → seasonal_lags. IF stationary → raw features.

9. **H — create_lag_features()**
   - Input: `df`, `target_col`, `lags: list[int]`
   - Output: `dict[str, pl.Series]` with `lag_1`, `lag_3`, etc.
   - Logic: `df[col].shift(lag)` for each lag

10. **I — create_rolling_features()**
    - Input: `df`, `target_col`, `windows: list[int]`
    - Output: `dict[str, pl.Series]` with `rolling_mean_X`
    - Logic: `.shift(1)` FIRST, then `.rolling(window).mean()`

11. **J — validate_feature_count()**
    - Input: `features: dict`
    - Output: `None` (or raise `ValueError`)
    - Logic: `len(features) >= 2` or `ValueError: "Too few features"`

12. **K — generate_model_subsets()**
    - Input: `feature_dict`, `model_recommendations: list[str]`, `analysis_data`
    - Output: `dict[model_name: list[feature_names]]`
    - Logic: Per model type → feature subset (e.g., SARIMA needs `diff_*`, GB needs all)

13. **L — create_state_space_embedding()**
    - Input: `df`, `target_col`, `embedding_dim=3`
    - Output: `dict[str, pl.Series]` with embedded features
    - Logic: Simple: If time series → create `[y_t-2, y_t-1, y_t]` as embedding

14. **M — serialize_artifacts()**
    - Input: `feature_df`, `output_json`, `output_dir`
    - Output: `(features_parquet_path: str, audit_json_path: str)`
    - Logic: Write `.parquet` and `.json` to `output_dir`

### Audit Trail (MUST be in output JSON)

```python
"audit_trail": {
    "Z_auto_detect_target": True,
    "A_compute_lag_mi": True,
    "B_find_best_lags": True,
    "C_detect_seasonality": True,
    "D_differencing": True,
    "E_temporal_features": True,
    "F_detect_leakage": True,
    "G_adaptive_features": True,
    "H_lag_features": True,
    "I_rolling_features": True,
    "J_validate_features": True,
    "K_model_subsets": True,
    "L_state_space": True,
    "M_serialize": True,
}
```

### Validation Rule for Agent

✅ **SUCCESS**: If ALL these conditions are met:
- File `step_12_features.py` contains ALL 14 functions (grep `^def `)
- Audit trail shows all 14 as `True`
- Leakage detection is implemented → RuntimeError on leakage_status="fail"
- Features are used from `exploration_output` fields (trend, seasonality, model_recommendations)
- Minimum 2 features after validation

❌ **FAIL**: If:
- Fewer than 14 functions in code
- Audit trail incomplete
- Functions are stubs without logic
- exploration_output fields are NOT used
