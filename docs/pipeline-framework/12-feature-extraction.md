# Schritt 12 â€” Feature Extraction & Model Preselection

## Code Generator Instructions

> This file is an LLM prompt. Generate `step_12_features.py` — a complete, immediately executable Python CLI script.

| Feld | Wert |
|---|---|
| **Dateiname** | `step_12_features.py` |
| **Step-ID** | `12-feature-extraction` |
| **CLI** | `python step_12_features.py --output-dir <dir> --run-id <id> [--split-mode auto|random|time_series] [--exclude-features feat1,feat2]` |

---

## MANDATORY CHECKLIST — COMPLETE BEFORE WRITING ANY CODE

> **The agent MUST explicitly go through this checklist during PHASE-1 reasoning and answer each point with YES/NO before writing a single line of code. Missing points = invalid implementation.**
- [ ] **Target Variable Removal**: Target column is filtered AFTER feature selection; if still present → `sys.exit(2)`
- [ ] **Timestamp Control**: Timestamp (date/timestamp) is NOT included as a raw numeric feature in the final feature list
- [ ] **CLI `--exclude-features`**: Argument implemented as comma-separated list (for orchestrator remediation after Step 17)
- [ ] **`tqdm`**: `from tqdm import tqdm` imported and all feature engineering loops (lags, cross-corr, rolling) run through `tqdm(...)`
- [ ] **Multi-Series Lags**: `if multiple_series_detected: pl.col(...).shift(n).over(group_col)` — NO global `.shift()` for multi-series
- [ ] **Rolling Features Causal**: `.shift(1)` BEFORE every `.rolling_mean()` calculation — prevents look-ahead leakage
- [ ] **Exit Code 2 on Leakage**: If |r| > 0.98, `sys.exit(2)` is called and NO features.parquet is written
- [ ] **Reconstruction Probe**: After Pearson check, additionally RandomForest R² > 0.999 as a second leakage test
- [ ] **Mandatory Steps Z-K**: All 11 functions (Z, A, B, C, D, E, F, G, H, I, J, K) are implemented as separate functions and called in this order from `main()`
- [ ] **`features_excluded` as Dict**: Every excluded feature with reason (not as a list)
- [ ] **Minimum Features Check**: `if len(final_features) < 2: sys.exit(1)` after leakage cleanup
- [ ] **Monotone Index Features Forbidden**: `trend_t_index`, `trend_t_index_sq` MUST NOT be generated; use `trend_elapsed_days` instead

---

### Required Inputs

| Source | Path |
|---|---|
| Previous Step (11) | `OUTPUT_DIR/step-11-exploration.json` |
| Previous Step (10) | `OUTPUT_DIR/step-10-cleanse.json` |
| Data | `OUTPUT_DIR/cleaned.parquet` |

### Required Outputs

| File | Content |
|---|---|
| `OUTPUT_DIR/step-12-features.json` | JSON with `"step": "12-feature-extraction"`, `features`, `features_excluded`, `split_strategy`, `artifacts` |
| `OUTPUT_DIR/features.parquet` | Feature matrix (all features + target variable) |

### Required Fields in step-12-features.json

`json

{

  "step": "12-feature-extraction",
  "features": ["close_lag1", "close_lag2", "volume"],
  "features_excluded": {"trend_t_index": "monotonic_index_ks_drift"},
  "target_column": "close",
  "split_strategy": {"resolved_mode": "time_series"},
  "artifacts": {"features_parquet": "output/RUN_ID/features.parquet"}
}

`

### Critical Implementation Rules

**LEAKAGE PROTECTION — HIGHEST PRIORITY:**

0. **TARGET VARIABLE MUST BE REMOVED:** The target variable (e.g. `appliances`, `close`, etc.) must NEVER appear in the final features, even if it is in `recommended_features`. After feature selection: filter all features where `feature_name == target_column_normalized`. If the target variable is found in features → `sys.exit(2)` with message "Target variable leaked into features"

0b. **TIMESTAMP FIELD CONTROL:** The timestamp field (e.g. `date`, `timestamp`) MUST NOT be used as a raw numeric feature (it would enable perfect reconstruction). It may ONLY be used for:

   - Time-based lag calculation (`.over(time_col)`)

   - TimeSeriesSplit determination

   - Trend features (`trend_elapsed_days`, not `trend_t_index`)

   - If it was included anyway → remove from features before saving parquet

1. Start exclusively with `step-11-exploration.json["recommended_features"]` — never add features that were excluded in Step 11

2. Lag features ONLY for combinations from `step-11["useful_lag_features"]` + target lags from `step-11["significant_lags"]`

3. **Multi-Series Requirement (MANDATORY):** When `step-11["multiple_series_detected"] == true`, every lag and rolling feature MUST be computed **per group**: `pl.col("...").shift(n).over(group_column)`. Global `.shift(n)` without `.over()` is **forbidden** for multi-series — it causes cross-series data leaks. The script MUST contain an `assert group_col is not None` check before creating lags in multi-series mode.

4. **Lag Cap:** Use at most **6 target lags**, even if `step-11["significant_lags"]` contains more. Select by ACF magnitude (largest autocorrelation first).

5. **Leakage Guard (MANDATORY Exit Code 2):** 
   - **Step A:** Before ML checks: remove target variable and timestamp from features (see point 0 above)
   - **Step B:** Pearson |r| > 0.98 MUST trigger `sys.exit(2)` — no features.parquet is written
   - **Step C:** Additionally: RandomForest reconstruction probe (R² > 0.999 = leakage)
   - **Step D:** Check for monotone index features (`trend_t_index`, `trend_t_index_sq`) — these ARE leakage and must be removed
   - The threshold is 0.98, NOT 0.99.

6. Create all features with `pl.col().shift(n).over(group_column)` (multi-series) or `pl.col().shift(n)` (single-series) (causal, no look-ahead). Rolling ALWAYS preceded by `.shift(1)`.

7. NO monotonically increasing index features (`trend_t_index`, `trend_t_index_sq`) — these produce KS=1.0; use `trend_elapsed_days` instead

8. **`tqdm` is MANDATORY** (`from tqdm import tqdm`) for all feature engineering loops — the agent must not write loops without tqdm

9. **`--exclude-features` CLI argument (MANDATORY):** `parser.add_argument("--exclude-features", default="")` — comma-separated list of features to be forcibly excluded. Injected by the orchestrator after Step-17 remediation.

10. **Mandatory Steps Z-K** are separate Python functions — no inline code in `main()` for this logic

---

**Script**: `CODE_DIR/step_12_features.py`  
**AusfÃ¼hrung**: `python step_12_features.py --output-dir OUTPUT_DIR --run-id RUN_ID [--split-mode auto|random|time_series]`  
**Input**: `OUTPUT_DIR/cleaned.parquet`, `OUTPUT_DIR/step-10-cleanse.json`, `OUTPUT_DIR/step-11-exploration.json`  
**Output**: `OUTPUT_DIR/features.parquet`, `OUTPUT_DIR/step-12-features.json`  
**Exit-Codes**: `0` = Erfolg, `1` = Fehler, `2` = Leakage erkannt

```
[10] csv_read_cleansing â†’ [11] data_exploration â†’ [12] feature_extraction â†’ [13] model_training â†’ ...

```

> **WICHTIG FÃœR AGENTEN**: Dieses Dokument beschreibt ein **CLI-Script**, kein importierbares Modul.
> Es gibt KEINE `run_analysis()`-Einstiegsfunktion.
> Der Einstiegspunkt ist `main()` in `if __name__ == "__main__": sys.exit(main())`.
> Alle Funktionen Zâ€“K sind eigenstÃ¤ndige Hilfsfunktionen, die von `main()` aufgerufen werden.

---

## Feature Philosophy

| Principle | Rule |

|---|---|
| Kausales Rolling | `.shift(1)` vor jeder `rolling_*`-Berechnung â€” verhindert Look-ahead |
| Leakage â†’ Hard Fail | `RuntimeError` wenn `\|r\| â‰¥ threshold` â€” kein Artefakt wird geschrieben |
| Leakage-Probe | Paarweise Pearson-Korrelation **und** Rekonstruktions-Probe (RF RÂ² > 0.999) |
| Mindest-Features | Weniger als 2 Features nach Bereinigung â†’ `ValueError` |

---

## Input Contract (Output from Step 11)

Das Input fÃ¼r `feature_extraction.py` ist der direkte Output von `data_exploration.py`:

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
        "frequency": "10min",                                  # Detected time-series frequency
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

        "recommended_features": ["T6", "T1", "RH_6", "lights", "T_out"],  # Empfohlene Features fÃ¼r Step 12

        "excluded_features": {
            "rv1": "below_noise_baseline",
            "rv2": "redundant",
            "target_copy": "leakage_suspect"
        }
    },

    # === TIME-SERIES SPECIFIC ANALYSIS ===

    "time_series_detected": true,
    "time_column": "date",
    "multiple_series_detected": false,
    "group_column": null,   # Name of the grouping column (e.g. "stock_name_encoded") -- REQUIRED FIELD for Step 12
    "n_series": 1,          # Number of unique groups -- REQUIRED FIELD for Step 12
    "time_series_characteristics": {
        "trend_detected": true,                                # Trend present?
        "seasonality_detected": true,                          # SaisonalitÃ¤t vorhanden?
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
- `df` is polars.DataFrame with no NaN in the target variable
- `numeric_cols` and `categorical_cols` cover all columns
- `recommended_features` is never empty (or an error is thrown in Step 11)
- `analysis.excluded_features` erklÃ¤rt jeden ausgeschlossenen Feature
- `time_series_characteristics`, `model_recommendations`, `significant_lags` are always present (even when `time_series_detected=false`)

---

## Output-Vertrag (fÃ¼r Training-Modul)

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
        "detection_method": "highest_variance",           # "explicit" wenn via config Ã¼bergeben
        "detection_info": {
            "candidates": ["energy", "temperature"],
            "scores": {"energy": 0.95, "temperature": 0.42},
            "reason": "energy hat hÃ¶chste Varianz (0.95)",
        },
    },

    # === FEATURES ===

    "features": {

        "feature_matrix": pl.DataFrame,                  # Shape: (n_rows, n_features), numeric only, incl. target
        "target_series": pl.Series,                      # Zielvariable (in-memory fÃ¼r nÃ¤chstes Modul)
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
        "audit_json": "outputs/step-12/step-12-features.json",   # VollstÃ¤ndiger Audit Trail
    },

    # === LEAKAGE ===

    "leakage": {
        "status": "pass",                # "pass" | "fail" â€” bei "fail" wird RuntimeError raised
        "leakage_candidates": [],        # Feature names with |r| >= threshold
        "correlations": {"y_lag_0": 1.0, "y_lag_1": 0.87},
        "threshold": 0.98,
        "reconstruction_probe_r2": None, # RF RÂ² der Leakage-Probe (None wenn kein Kandidat)
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
            "tree_model_suitable": "conditionally",
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
                "suitability": "suitable" | "rather suitable" | "rather not",
                "score": float,                          # 0.0â€“1.0
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
        "Zeitreihe wurde um 50 Zeilen wegen Lag gekÃ¼rzt",
        "State Space Embedding: time series < 1 year, skipped",
    ],
}

```

---

## CLI Entry Point (`main()`)
> **ZWINGEND**: Das Script MUSS als `python step_12_features.py --output-dir ... --run-id ...` lauffÃ¤hig sein.
> Kein `run_analysis()`, keine Modul-API. Einstiegspunkt ist ausschlieÃŸlich `main()`.

## Entry Function

```python

# import json, logging, time, uuid
# from datetime import datetime, timezone
# from pathlib import Path
# import polars as pl

def run_analysis(
    exploration_output: dict,  # Output von Schritt 11 (siehe Input-Vertrag oben)
    config: dict | None = None,
    # config keys: max_lag=48, embedding_dim=3, use_state_space=True,
    #              rolling_windows=[7,14,30], output_dir="outputs/step-12",
    #              target_column=None  # optional: explizite Zielspalte, überspringt Auto-Detection

) -> dict:

    """
    **Input**: exploration_output from Step 11
      - exploration_output["data"]["df"] is used as the input DataFrame
      - exploration_output["data"]["numeric_cols"] defines the features
      - exploration_output["analysis"]["recommended_features"] guides feature selection
      - exploration_output["time_series"]["significant_lags"] guides lag creation
      - exploration_output["time_series"]["time_series_characteristics"] guides feature strategy
      - exploration_output["time_series"]["model_recommendations"] determines feature subsets

    **Output**: Feature extraction output (see Output Contract)
      - Writes features.parquet and step-12-features.json (only when leakage.status="pass")
      - feature_matrix is pl.DataFrame with all engineered features + target
      - model_recommendations lists which features are used for which model

    **Error Handling**:
      - ValueError on invalid input (too few features, empty DataFrame, etc.)
      - RuntimeError when leakage is detected — no artifact is written
    """

```

---

## Feature Engineering Strategy (Based on TS Characteristics)

This strategy adapts feature creation to the results of Step 11:
| TS Characteristic | Recommendation | Feature Action |

|---|---|---|
| **Trend detected** | SARIMA, Prophet, XGBoost | Differencing features: `diff_1`, `diff_7`, `diff_365` |
| **Seasonality detected** | SARIMA, Prophet, Prophet-QR | Seasonal lags: `y_lag_24`, `y_lag_168` (12h, 1W for hourly data) |
| **Stationary** | Gradient Boosting, Linear Regression | No differencing needed; raw lags sufficient |
| **Non-stationary** | SARIMA, Prophet, ETS | Differencing features MUST be created |
| **White Noise detected** | Naive, Exponential Smoothing | Only trivial lag features; complex engineered features add nothing |
| **Multiple Series** | XGBoost with grouping, LightGBM | Categorical features for series ID + cross-series features |

### Konkrete Implementierung:

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
    4. Union of all subsets into final feature_matrix

    Example:

        if time_series_characteristics["trend_detected"]:
            features["diff_1"] = df[target_col].diff(1)
            features["diff_7"] = df[target_col].diff(7)

        if time_series_characteristics["seasonality_detected"]:

            # Erkenne Frequenz aus exploration_output["exploration"]["frequency"]
            seasonal_lag = infer_seasonal_lag(frequency)

            for lag in significant_lags:
                if lag % seasonal_lag == 0:
                    features[f"y_lag_{lag}"] = df[target_col].shift(lag)
    """

```

---

## Funktions-Spezifikationen

### Z — Zielspalten-Bestimmung

```python

def auto_detect_target_column(

    df: pl.DataFrame,
    time_col: str,
    numeric_cols: list[str],
    heuristic: str = "highest_variance",
    explicit_target: str | None = None,

) -> tuple[str, dict]:

    """
    Returns the explicit target column if provided (after validation against numeric_cols).

    Otherwise: automatic selection from numeric columns.

    Heuristics (only for auto-detection):
    - "highest_variance": Column with highest variance
    - "most_correlated_with_self": highest autocorrelation

    Exclusions (only for auto-detection): time_col, constant columns (variance ≈ 0)

    Returns:
        (target_column_name, detection_info)
        detection_info: {"method": "explicit"|"highest_variance"|..., "candidates", "scores", "reason"}

    Raises:
        ValueError: When explicit_target is not in numeric_cols
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
    Nutzt: sklearn.feature_selection.mutual_info_regression

    Returns:
        pl.DataFrame with columns [lag, mutual_information], sorted by MI desc
    """

```

---

### B — Beste Lags (ACF + PACF + MI)

```python

def find_best_lags(

    df: pl.DataFrame,
    target_col: str,
    max_lag: int = 48,
    top_n: int = 10,
    method: str = "combined",

) -> dict:

    """

    Combines ACF, PACF and MI for lag selection.
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

### C — Saisonalitätserkennung

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

    If seasonality detected: generate Fourier features
        sin(2π·k·t/period), cos(2π·k·t/period) für k=1,2
    Returns:
        {"seasonality_strength", "seasonality_label", "dominant_period",
         "seasonal_features", ...}
    """

```

---

### D — Zielverteilung

```python

def analyze_target_distribution(

    df: pl.DataFrame,
    target_col: str,

) -> dict:

    """
    Analyzes the y distribution for model selection.

    Metrics: min, max, mean, std, skewness, kurtosis, IQR, CV
    Tree model heuristic:
    - Outlier fraction < 5%: positive
    - CV < 1.0: positivee
    - CV > 3.0: negativee
    - |skew| < 1: positivee

    Returns:
        {"min", "max", "mean", "std", "cv",
         "tree_model_suitable": "yes"|"conditionally"|"rather not",
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

    Delay embedding with automatic delay selection (Takens theorem).
    Auto-delay: first local minimum of the MI curve
    Embedding matrix: [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]
    Returns:

        {"embedding_matrix": np.ndarray,  # shape (n_valid, embedding_dim)
         "chosen_delay": int,
         "embedding_features": pl.DataFrame,
         "notes": str}
    """

```

---

### F — Strata-Features

```python

def create_strata_features(

    df: pl.DataFrame,
    time_col: str,
    target_col: str,

) -> dict:

    """
    Time-based stratification with ANOVA F-test.

    Activation depending on frequency & data length:
    - hour_of_day  → when freq ≤ 1h
    - day_of_week  → always when timestamp present
    - month        → when data ≥ 60 days
    - season       → when data ≥ 180 days
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
    NaN handling: Drop the first N rows (= max lag / window).
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

    Evaluates model types based on the actually available features.
    Model types & typical feature requirements:
    1. Gradient Boosting (XGBoost, LightGBM)
       ✓ Lags, Rolling, Calendar, Trend (tabular)
    2. SARIMA
       ✓ y_diff_1/2 (stationarization), dominant seasonal period
       ✗ Too many exogenous regressors problematic
    3. LSTM / Temporal CNN
       ? embedding_matrices / sequences (create adaptively if needed)
       ✓ Long time series (> 500 points)
    4. Linear Models (Ridge, Lasso)
       ✗ High MI with small ACF → non-linear → not suitable
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
    Supplements missing model-specific features retroactively.

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

### J — Leakage-Erkennung

```python

def detect_feature_leakage(
    feature_matrix: pl.DataFrame,
    target_col: str,
    threshold: float = 0.98,

) -> dict:

    """
    Checks whether features unaturally anticipate the target column.

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
            "reconstruction_probe_r2": float | None, # R² of the RF probe (None if no candidate)
        }

    Raises:
        ValueError: When target_col is not in feature_matrix
        RuntimeError: When status == "fail" (propagated by run_analysis)
    """

```
When `status == "fail"`, `run_analysis` raises a `RuntimeError` — no artifact is written.
---

### K — Feature-Scaling-Metadaten

```python

def compute_scaling_metadata(
    feature_matrix: pl.DataFrame,
    target_col: str,
    recommended_models: list[str],

) -> dict:

    """
    Determines which features need to be scaled for which model types.
    Scaling itself is performed in Step 13; this dict is the instruction for it.
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
- [ ] Input validation with descriptive `ValueError`
- [ ] `execution_id` via `uuid.uuid4()[:8]`
- [ ] `runtime_seconds` via `time.time()`
- [ ] `features.parquet` is written via `feature_df.write_parquet(parquet_path)`
- [ ] `step-12-features.json` contains complete audit trail (no `pl.DataFrame`/`pl.Series` directly — only serializable types; `default=str` as fallback)
- [ ] `output_dir` is created with `mkdir(parents=True, exist_ok=True)`
- [ ] Return dict contains `artifacts` with `features_parquet` and `audit_json` as path strings
- [ ] Return dict contains `features.feature_matrix` as `pl.DataFrame` (in-memory for next module)
- [ ] Output dictionary matches exactly the Output Contract
- [ ] All functions stateless (no global variables)
- [ ] Type hints complete
- [ ] `leakage` block in output with `status: "pass"|"fail"`; on `"fail"` → `RuntimeError`, no artifact
- [ ] Leakage probe: RandomForest R² > 0.999 counts as confirmed leakage
- [ ] Rolling features: `.shift(1)` before every rolling calculation; violation → exclude feature
- [ ] `scaling_metadata` block in output; binary features in `never_scale`
- [ ] Tests under `tests/test_feature_extraction.py`

---

## AGENT IMPLEMENTATION DIRECTIVE

**NOTE: This section is for the agent to implement. ALL 14 functions MUST be present.**

### Required 14 Functions (ALL IN ONE FILE)

**KEEP IT PRAGMATIC: 5-15 lines per function, not over-engineered**

1. **Z — auto_detect_target_column()**
   - Input: `df`, `numeric_cols`, `explicit_target=None`
   - Output: `(target_col: str, detection_info: dict)`
   - Logic: If `explicit_target` → validate against `numeric_cols`. Otherwise → highest variance

2. **A — compute_lag_mutual_information()**
   - Input: `df`, `target_col`, `max_lag=48`
   - Output: `pl.DataFrame` mit Spalten `[lag, mutual_information]`
   - Uses: `sklearn.feature_selection.mutual_info_regression`

3. **B — find_best_lags()**
   - Input: `df`, `target_col`, `max_lag=48`, `top_n=10`
   - Output: `dict` mit `best_lags`, `acf_values`, `pacf_values`
   - Uses: ACF + PACF + MI combined

4. **C — detect_seasonality()**
   - Input: `df`, `target_col`, `frequency_str="10min"`
   - Output: `dict` mit `has_seasonality`, `period`, `strength`
   - Simple: detect periodic pattern in data

5. **D — create_differencing_features()**
   - Input: `df`, `target_col`
   - Output: `dict[str, pl.Series]` mit `diff_1`, `diff_7`, ggf. `diff_365`
   - Logic: Only when non-stationary detected

6. **E — create_temporal_features()**
   - Input: `df`, `time_col`
   - Output: `dict[str, pl.Series]` mit `hour_of_day`, `day_of_week`, `month`, `quarter`

   - Logic: Extract from datetime column
7. **F — detect_leakage()**
   - Input: `df`, `target_col`, `threshold=0.98`
   - Output: `dict` mit `leakage_status: "pass"|"fail"`, `candidates: list[str]`
   - Logic: Pearson correlation > threshold with target → leakage suspected

8. **G — create_adaptive_features()**
   - Input: `df`, `target_col`, `exploration_output`
   - Output: `dict[str, pl.Series]` mit adaptierten Features
   - Logic: IF trend_detected → diff. IF seasonality → seasonal_lags. IF stationary → raw features.

9. **H — create_lag_features()**
   - Input: `df`, `target_col`, `lags: list[int]`
   - Output: `dict[str, pl.Series]` mit `lag_1`, `lag_3`, etc.
   - Logic: `df[col].shift(lag)` for each lag

10. **I — create_rolling_features()*
    - Input: `df`, `target_col`, `windows: list[int]`
    - Output: `dict[str, pl.Series]` mit `rolling_mean_X`
    - Logic: `.shift(1)` FIRST, then `.rolling(window).mean()`

11. **J — validate_feature_count()**
    - Input: `features: dict`
    - Output: `None` (oder raise `ValueError`)
    - Logic: `len(features) >= 2` or `ValueError: "Too few features"`

12. **K — generate_model_subsets()**
    - Input: `feature_dict`, `model_recommendations: list[str]`, `analysis_data`
    - Output: `dict[model_name: list[feature_names]]`
    - Logic: Per model type → subset of features (e.g. SARIMA needs `diff_*`, GB needs everything)

13. **L — create_state_space_embedding()**
    - Input: `df`, `target_col`, `embedding_dim=3`
    - Output: `dict[str, pl.Series]` mit embedded Features
    - Logic: Simple: if time series → create `[y_t-2, y_t-1, y_t]` as embedding

14. **M — serialize_artifacts()**
    - Input: `feature_df`, `output_json`, `output_dir`
    - Output: `(features_parquet_path: str, audit_json_path: str)`
    - Logic: Write `.parquet` and `.json` to `output_dir`

### Audit Trail (MUSS im Output-JSON sein)

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

## Required Content of `step-12-features.json`

Das Output-JSON MUSS folgende Schlüssel enthalten (Validation Gate in `orchestrator.py`):
```json
{

  "step": "12-feature-extraction",
  "features": ["...", "..."],
  "features_excluded": {"col_name": "reason", ...},
  "split_strategy": {"resolved_mode": "time_series"},
  "artifacts": {
    "features_parquet": "output/<RUN_ID>/features.parquet"
  },
  "analysis": {
    "best_lags": [...],
    "seasonality": {"has_seasonality": true, "dominant_period": 24, "seasonality_strength": 0.7},
    "target_distribution": {"tree_model_suitable": "yes", "skewness": 0.3},
    "strata": {"active": ["hour_of_day", "day_of_week"]},
    "state_space_embedding": {"chosen_delay": 2, "notes": "..."}
  },
  "leakage": {
    "status": "pass",
    "leakage_candidates": [],
    "threshold": 0.98,
    "reconstruction_probe_r2": null
  },
  "scaling_metadata": {...},
  "model_recommendations": {"top_3_recommendations": ["GradientBoosting", "RandomForest", "Ridge"]},
  "shape": {"rows": 19720, "features": 42}
}

```

**Validation Gate** (blocks Step 13 when not fulfilled):
- `features` is a non-empty list
- `features_excluded` exists (audit trail)
- `split_strategy.resolved_mode` is `"random"` or `"time_series"`
- `artifacts.features_parquet` exists on disk
- No feature in `features` appears in `step-11-exploration.json["excluded_features"]`

---

## Implementation Checklist

**All 12 mandatory functions** (grep `^def ` in `step_12_features.py` must return all 12):

- [ ] `auto_detect_target_column()` — Step Z
- [ ] `compute_lag_mutual_information()` — Step A
- [ ] `find_best_lags()` — Step B
- [ ] `detect_seasonality()` — Step C
- [ ] `analyze_target_distribution()` — Step D
- [ ] `compute_state_space_embedding()` — Step E
- [ ] `create_strata_features()` — Step F
- [ ] `engineer_timeseries_features()` — Step G
- [ ] `preselect_models()` — Step H
- [ ] `add_features_for_models()` — Step I
- [ ] `detect_feature_leakage()` — Step J
- [ ] `compute_scaling_metadata()` — Step K
- [ ] `main()` — CLI entry point

**Feature Engineering Rules** (all mandatory):

- [ ] Rolling: `.shift(1)` ALWAYS before `rolling_mean/std/min/max/range`
- [ ] Diff: `.shift(1).diff(n)` ALWAYS, NEVER `.diff(n)` directly
- [ ] Target lags: `shift(lag)` for each lag in `best_lags`
- [ ] All 5 rolling variants: mean, std, min, max, range
- [ ] Calendar features when `time_col` present: year, month, day_of_week, hour, is_weekend
- [ ] Trend features: t_index, t_index_sq, (t_elapsed_days when time_col present)
- [ ] Fourier features when seasonality detected: sin/cos for k=1,2

**Leakage Rules** (mandatory):

- [ ] Pearson |r| ≥ 0.98 against target → candidate
- [ ] RF R² > 0.999 on candidates → leakage confirmed → `sys.exit(2)`
- [ ] Correct lag features (`{target}_lag{k}`) are exempt from leakage check
- [ ] On leakage: do NOT write any artifact

**Output Rules**:

- [ ] `features_excluded` documents every dropped column with reason
- [ ] `step-12-features.json` contains all required keys (see above)
- [ ] `features.parquet` contains all features + target (no other columns)
- [ ] `progress.json` is updated with `completed_steps` on success
- [ ] Exit code `0` on success, `1` on error, `2` on leakage

---

## Validation Gate (After Executing Step 12)

These checks run in `orchestrator.py` before starting Step 13:

```python

def gate_12(output_dir):
    j = json.loads((output_dir / "step-12-features.json").read_text())
    assert j.get("step") == "12-feature-extraction"
    assert len(j.get("features", [])) > 0, "features list is empty"
    assert "features_excluded" in j, "features_excluded missing"
    assert j["split_strategy"]["resolved_mode"] in ("random", "time_series")
    assert Path(j["artifacts"]["features_parquet"]).exists()

    

    # Leakage-Guard: Kein ausgeschlossenes Feature darf in features auftauchen
    step11 = json.loads((output_dir / "step-11-exploration.json").read_text())
    excluded_in_11 = set(step11.get("excluded_features", []))
    for f in j["features"]:
        assert f not in excluded_in_11, f"Leakage guard: {f} was excluded in step 11 but appears in step 12 features"

```