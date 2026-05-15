# Step 13 — Model Training (Time-Series Focused)

## Objective

Train a full suite of time-series candidate models — classical univariate, factor-augmented, and multivariate ML — against mandatory benchmarks. Every candidate is evaluated on the **same chronological holdout** defined in step 12. Model selection is driven by the diagnostics from step 11: stationarity, ACF/PACF order hints, Hurst exponent, seasonal periods, and recommended model classes.

## Inputs

- `OUTPUT_DIR/features.parquet` — feature matrix (all groups A–G from step 12, includes target)
- `OUTPUT_DIR/step-12-features.json` — split boundary, feature group metadata, PCA info
- `OUTPUT_DIR/step-11-exploration.json` — TS diagnostics driving model selection
- `OUTPUT_DIR/pca_preprocessor.joblib` — fitted StandardScaler + PCA (for FAAR models)
- `OUTPUT_DIR/leakage_audit.json` — must be `"status": "pass"` or `"warn"`; halt if `"fail"`

## Outputs

- `OUTPUT_DIR/candidate-{name}.joblib` — serialised estimator per candidate
- `OUTPUT_DIR/model.joblib` — best candidate copy
- `OUTPUT_DIR/holdout.npz` — `X_test`, `y_test` arrays for step 14
- `OUTPUT_DIR/step-13-training.json` — full training history

---

## Pre-Training Leakage Gate

Before training any model:
1. Load `leakage_audit.json`. If `"status": "fail"` → raise `RuntimeError` immediately, do not train.
2. For every column in `X_train`, compute Pearson correlation with `y_train`. Any column with `|r| > 0.98` → raise `RuntimeError("Leaked feature detected at training time")`.

---

## Mandatory Benchmarks — ALWAYS run first, NEVER skip

> **⛔ HARD RULE**: All four benchmarks below are UNCONDITIONAL. They run on every invocation regardless of dataset size, model recommendations, or any other condition. There is no flag, config option, or code path that legitimately bypasses them. A step-13 script that omits any benchmark is INCORRECT and must be regenerated.

These four baselines define the performance floor that every candidate must beat.

| Benchmark | Role | Implementation |
|---|---|---|
| **`naive_persistence`** | Absolute minimum — predict last known value | `ŷ(t) = y(t-1)`, applied on holdout only; no training required |
| **`seasonal_naive`** | Seasonal minimum — repeat last season | `ŷ(t) = y(t-m)` where m = `primary_seasonal_period` from step 11; use m=1 if no seasonality detected |
| **`auto_arima_benchmark`** | Classical univariate statistical baseline | `pmdarima.auto_arima(y_train, seasonal=True, m=m)` — auto-selects p,d,q,P,D,Q by AIC; fallback to `statsmodels ARIMA(1,1,1)` if pmdarima unavailable |
| **`ar1_benchmark`** | Autoregressive baseline — captures serial correlation | `statsmodels.tsa.ar_model.AutoReg(endog=y_train, lags=1).fit()`, then `.predict(start, end)` on the holdout horizon |

### Benchmark Execution Pseudocode (implement exactly like this)

```python
# ── STEP A: always run, no conditions, no early exit ──────────────────────
benchmarks = {}

# 1. naive_persistence — ŷ(t) = y(t-1) on holdout
y_pred_naive = np.concatenate([[y_train[-1]], y_test[:-1]])
benchmarks["naive_persistence"] = _score_benchmark(y_test, y_pred_naive)

# 2. seasonal_naive — ŷ(t) = y(t-m), fallback to persistence if m=1 or None
m = primary_seasonal_period or 1
y_hist = np.concatenate([y_train, y_test])  # full series for lookback
y_pred_seasonal = np.array([y_hist[i - m] for i in range(len(y_train), len(y_hist))])
benchmarks["seasonal_naive"] = _score_benchmark(y_test, y_pred_seasonal)
benchmarks["seasonal_naive"]["seasonal_period"] = m

# 3. auto_arima_benchmark — wrap in try/except, fall back to ARIMA(1,1,1)
try:
    import pmdarima
    bm_model = pmdarima.auto_arima(y_train, seasonal=True, m=m,
                                    max_p=4, max_q=4, suppress_warnings=True,
                                    error_action="ignore")
except Exception:
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    bm_model = _ARIMA(y_train, order=(1, 1, 1)).fit()
y_pred_arima = bm_model.predict(n_periods=len(y_test)) \
    if hasattr(bm_model, "predict") else bm_model.forecast(steps=len(y_test))
benchmarks["auto_arima_benchmark"] = _score_benchmark(y_test, y_pred_arima)

# 4. ar1_benchmark — AutoReg(lags=1) trained on y_train, forecasts holdout
from statsmodels.tsa.ar_model import AutoReg as _AutoReg
try:
    ar1_model = _AutoReg(y_train, lags=1, old_names=False).fit()
    # predict over the holdout index range
    start_idx = len(y_train)
    end_idx   = len(y_train) + len(y_test) - 1
    # NOTE: AutoReg.predict() returns a numpy array, NOT a pandas Series.
    # Do NOT call .values on it — wrap with np.asarray() instead.
    y_pred_ar1 = np.asarray(ar1_model.predict(start=start_idx, end=end_idx), dtype=float)
except Exception as _e:
    # fallback: persistence
    y_pred_ar1 = np.concatenate([[y_train[-1]], y_test[:-1]])
benchmarks["ar1_benchmark"] = _score_benchmark(y_test, y_pred_ar1)
benchmarks["ar1_benchmark"]["lags"] = 1

# ── STEP B: record holdout predictions for all four benchmarks ────────────
benchmark_predictions = {
    "naive_persistence": y_pred_naive,
    "seasonal_naive": y_pred_seasonal,
    "auto_arima_benchmark": np.asarray(y_pred_arima, dtype=float),
    "ar1_benchmark": np.asarray(y_pred_ar1, dtype=float),
}
```

**Benchmark warning rule**: if the best candidate's holdout R² does not exceed **both** `auto_arima_benchmark` and `ar1_benchmark` by at least `0.02` → set `"benchmark_warning": true` in output.

---

## Model Class Roster — Driven by Step 11

The agent **must** read `model_class_recommendations` from `step-11-exploration.json` and instantiate every recommended class that is available. Do not skip recommended classes silently — record any skip with reason in `skipped_models`.

### Tier 1 — Classical Univariate Statistical Models

These models train on the **univariate target series only** (no exogenous features). They use the suggested orders from `acf_pacf_orders` in step 11.

| Model | Condition | Implementation | Order hints |
|---|---|---|---|
| `AR(p)` | `suggested_ar_order >= 1` AND `suggested_ma_order == 0` | `statsmodels.tsa.ar_model.AutoReg(lags=p)` | p = `suggested_ar_order` |
| `MA(q)` | `suggested_ma_order >= 1` AND `suggested_ar_order == 0` | `statsmodels.tsa.arima.ARIMA(order=(0,0,q))` | q = `suggested_ma_order` |
| `ARIMA(p,d,q)` | Always (if pmdarima available) | `pmdarima.auto_arima(y, d=d, max_p=8, max_q=4, information_criterion='aic')` | d from `stationarity_conclusion`; p,q auto-selected |
| `SARIMA(p,d,q)(P,D,Q,m)` | `seasonality_detected=true` | `pmdarima.auto_arima(y, seasonal=True, m=m)` | m = `primary_seasonal_period` |
| `SARIMAX` | `seasonality_detected=true` AND `n_recommended_features >= 2` | `statsmodels.tsa.statespace.SARIMAX(y, exog=X_exog, ...)` | Use top-3 exogenous features from MI ranking |
| `HoltWinters (ETS)` | `trend_detected=true` OR `seasonality_detected=true` | `statsmodels.tsa.holtwinters.ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=m)` | m = primary period |
| `TBATS` | Multiple significant seasonal periods (len(detected_periods) >= 2) | `sktime.forecasting.tbats.TBATS` or statsmodels fallback | periods = all significant periods |

**Important**: classical univariate models are fitted on `y_train` only (the target column from the training rows). They generate a forecast on the holdout horizon. For models with exogenous variables (SARIMAX), also supply `X_exog_test` on the holdout.

### Tier 2 — Factor-Augmented Models (FAAR Family)

These models require PCA factors from `pca_preprocessor.joblib`. Only instantiate when `pca_factors` group is non-empty in step-12-features.json.

| Model | Description | Implementation |
|---|---|---|
| `FAAR-ARIMA` | PCA → ARIMA: k factor components passed as exogenous regressors to ARIMA | `pmdarima.auto_arima(y_train, exogenous=pca_train, seasonal=False)` |
| `FAAR-SARIMAX` | PCA → SARIMAX: factors + seasonal structure | `statsmodels.SARIMAX(y_train, exog=pca_train, order=(p,d,q), seasonal_order=(P,D,Q,m))` |
| `Factor-VAR` | Vector AutoRegression on [y, pca_factor_1, ..., pca_factor_k]: models multivariate dynamics | `statsmodels.tsa.vector_ar.var_model.VAR(endog=[y_train, f1_train, f2_train]).fit(ic='aic')` then extract y forecast |

**FAAR pipeline**:
1. Load PCA factors from `features.parquet` columns named `pca_factor_*`.
2. Split into `pca_train` (rows < holdout_start_index) and `pca_test` (rows >= holdout_start_index).
3. Pass `pca_train` as exogenous to the statistical model.
4. Use `pca_test` to generate the holdout forecast.

### Tier 3 — Multivariate ML Models

These models use the full feature matrix X (all groups A–G except `pca_factors`) as tabular features. They are trained with `TimeSeriesSplit` cross-validation.

| Model | Condition | Library | Param search |
|---|---|---|---|
| `ElasticNet` | Always — interpretable regularised linear baseline | `sklearn.linear_model.ElasticNet` | `GridSearchCV(alpha=[0.001,0.01,0.1,1.0], l1_ratio=[0.2,0.5,0.8])` |
| `Ridge` | Always — fast collinearity-robust baseline | `sklearn.linear_model.Ridge` | `GridSearchCV(alpha=[0.1,1.0,10.0,100.0])` |
| `XGBoost` | `N > 2000` OR `seasonality_detected=true` | `xgboost.XGBRegressor` | `RandomizedSearchCV(n_iter=30)` over n_estimators, max_depth, learning_rate, subsample |
| `LightGBM` | `N > 2000` AND `n_features > 10` | `lightgbm.LGBMRegressor` | `RandomizedSearchCV(n_iter=30)` |
| `RandomForest` | General purpose | `sklearn.ensemble.RandomForestRegressor` | `RandomizedSearchCV(n_iter=20)` over n_estimators, max_depth, min_samples_split |
| `HistGradientBoosting` | `N > 5000` OR high feature count | `sklearn.ensemble.HistGradientBoostingRegressor` | `RandomizedSearchCV(n_iter=20)` |
| `SVR(rbf)` | `N < 20000` AND `hurst < 0.45` (mean-reverting smooth signal) | `sklearn.svm.SVR(kernel='rbf')` | `GridSearchCV(C=[0.1,1,10], epsilon=[0.01,0.1,1.0])` |

All ML models are wrapped in a `sklearn.pipeline.Pipeline` that **always** includes a median imputer as the first step. This is mandatory — lag features inherently produce NaN at the start of the series and the imputer must handle them:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # handles NaN from lags/rolling — REQUIRED
    ("scaler", StandardScaler()),
    ("model", <estimator>),
])
```

> **⛔ DO NOT omit the imputer step.** Lag and rolling features produce NaN for the first N rows. Without the imputer, every sklearn model will raise `ValueError: Input X contains NaN` and fail. SimpleImputer with `strategy="median"` is safe and non-leaking when fitted on the training fold only (which GridSearchCV/RandomizedSearchCV handle automatically).

Scaling is applied to all features. The pipeline is cached with `joblib.Memory(cache_dir)`.

### Tier 4 — Creative Hybrid Models

| Model | Description | Implementation |
|---|---|---|
| `ARIMA_XGB_residual` | Fit ARIMA on training series; train XGBoost on ARIMA residuals + lag features; final forecast = ARIMA forecast + XGBoost residual forecast | Step 1: fit `auto_arima(y_train)`; Step 2: compute `residuals_train = y_train - arima_fitted`; Step 3: fit `XGBRegressor` on `[X_train, residuals_train]`; Step 4: holdout forecast = arima_forecast + xgb_forecast |
| `ElasticNet_ARIMA_stack` | Stacked ensemble: ARIMA and ElasticNet forecasts as meta-features; ElasticNet as meta-learner | Step 1: generate cross-val predictions from ARIMA and ElasticNet via `TimeSeriesSplit`; Step 2: fit `ElasticNet` on stacked OOF predictions; Step 3: predict on holdout using meta-learner |

These hybrid models should only be instantiated when at least one of ARIMA and ElasticNet has already been successfully trained (they re-use fitted objects).

---

## Split & CV Rules

- **Always** use `TimeSeriesSplit(n_splits=5)`. Any random or shuffle-based split is forbidden and must raise a `RuntimeError`.
- Holdout = rows with index >= `step-12-features.json["split_info"]["holdout_start_index"]`. This is fixed and must **not** be recalculated.
- Apply a purge gap: skip the `gap = N_train // (n_splits * 2)` rows immediately before each validation fold (avoids boundary leakage from rolling features).
- ARIMA-family models do not use TimeSeriesSplit CV — they are evaluated directly on the holdout horizon. Record `cv_r2_mean = null` and `cv_r2_std = null` for these.
- Cache sklearn pipeline transformations with `joblib.Memory(location=OUTPUT_DIR/cache)`.
- Fixed `random_state=42` on all models and splits.

---

## Training Order

1. Leakage gate (hard fail if triggered)
2. Mandatory benchmarks: `naive_persistence` → `seasonal_naive` → `auto_arima_benchmark` → `kmeans_benchmark`
3. Tier 1 classical models (AR, MA, ARIMA, SARIMA, SARIMAX, HoltWinters, TBATS) — in order of complexity
4. Tier 2 factor models (FAAR-ARIMA, FAAR-SARIMAX, Factor-VAR) — requires PCA factors
5. Tier 3 ML models (ElasticNet, Ridge, XGBoost, LightGBM, RF, HistGBM, SVR)
6. Tier 4 hybrid models (ARIMA+XGBoost residual, stacked) — requires Tiers 1 and 3 to complete first

Update `progress.json` after each model finishes. If a candidate raises an exception, log it in `skipped_models` with `"reason": "<error message>"` and continue to the next candidate — the pipeline must not halt on a single model failure.

---

## Guardrails

- **Hard-fail** before training if leakage audit is `"fail"` or if any X column has `|r(X, y)| > 0.98`.
- **TimeSeriesSplit only** — any code path that would produce random splits must raise.
- CV fold variance check: if `cv_r2_std / |cv_r2_mean| > 0.3` for any candidate, log a `"high_cv_variance_warning"` for that candidate — this signals potential overfitting or leakage.
- A candidate that raises during training or evaluation must be recorded as `"status": "failed"` with the full exception message. The pipeline continues.
- Do not select the best model in this step. Step 15 selects. Step 13 only ranks by holdout R² for `best_model_name` identification (used by step 14 to know which joblib to load as primary).
- Compute and log `delta_r2_vs_auto_arima` and `delta_r2_vs_ar1` for every non-benchmark candidate.
- If the dataset is detected as `multiple_series_detected=true` from step 11: ML models must receive the `series_id_column` encoded as an integer label feature. Classical univariate models should be trained per-series and their individual forecasts aggregated.

---

## Model Artifact Rules

- Every successfully trained candidate: `candidate-{name}.joblib` via `joblib.dump(fitted_estimator, path)`.
- `model.joblib` = copy of the candidate with the highest holdout R² (not necessarily the "final" model — step 15 makes that decision). Must be loadable by `joblib.load()` and expose `.predict(X)` or `.forecast(steps)`.
- Classical statistical models (ARIMA, SARIMAX, HoltWinters, etc.) do not expose a standard sklearn `.predict(X)` interface. Wrap them in a thin adapter:

```python
class StatsmodelsAdapter:
    def __init__(self, fitted_model, model_type: str, n_steps: int):
        self.model = fitted_model
        self.model_type = model_type
        self.n_steps = n_steps

    def predict(self, X=None):
        # For univariate models: X is ignored, forecast n_steps ahead
        return self.model.forecast(steps=self.n_steps)
```

- `holdout.npz`: `np.savez(path, X_test=X_test, y_test=y_test)` — arrays reusable by step 14 without step 13's script.
- **`forecast_comparison.npz`**: save holdout predictions for **every successfully trained model** (benchmarks + all candidates). Format:
  ```python
  all_preds = {"y_test": y_test}  # always include ground truth
  # add each benchmark
  for name, y_pred in benchmark_predictions.items():
      all_preds[name] = np.asarray(y_pred, dtype=float)
  # add each successful candidate
  for c in candidates:
      if c["status"] == "success":
          all_preds[c["model_name"]] = np.asarray(c["holdout_predictions"], dtype=float)
  np.savez(OUTPUT_DIR / "forecast_comparison.npz", **all_preds)
  ```
  This file powers the interactive forecast comparison plot in the Streamlit UI. It is **mandatory** and must always be written as long as at least one model succeeded.
- `code_audit.json`: record Python file path and SHA-256 hash of `step_13_training.py`.

---

## Output JSON Schema

```json
{
  "step": "13-model-training",
  "run_id": "20260501T120000Z",
  "split_mode": "time_series_chronological",
  "n_splits": 5,
  "random_state": 42,
  "holdout_start_index": 15673,

  "benchmarks": {
    "naive_persistence":    {"holdout_r2": 0.41, "holdout_rmse": 74.2, "holdout_mae": 48.1},
    "seasonal_naive":       {"holdout_r2": 0.52, "holdout_rmse": 64.8, "holdout_mae": 41.3, "seasonal_period": 144},
    "auto_arima_benchmark": {"holdout_r2": 0.69, "holdout_rmse": 52.4, "holdout_mae": 34.2,
                             "arima_order": [2, 1, 1], "seasonal_order": [1, 1, 1, 144]},
    "ar1_benchmark":         {"holdout_r2": 0.58, "holdout_rmse": 60.7, "holdout_mae": 38.2, "lags": 1}
  },

  "candidates": [
    {
      "model_name": "SARIMA",
      "tier": 1,
      "status": "success",
      "order": [2, 1, 1],
      "seasonal_order": [1, 1, 1, 144],
      "cv_r2_mean": null,
      "cv_r2_std": null,
      "holdout_r2": 0.73,
      "holdout_rmse": 48.9,
      "holdout_mae": 31.2,
      "delta_r2_vs_auto_arima": 0.04,
      "delta_r2_vs_ar1": 0.15,
      "fit_time_sec": 22.1,
      "artifact": "OUTPUT_DIR/candidate-SARIMA.joblib"
    },
    {
      "model_name": "FAAR-ARIMA",
      "tier": 2,
      "status": "success",
      "n_pca_factors": 2,
      "arima_order": [2, 1, 1],
      "cv_r2_mean": null,
      "cv_r2_std": null,
      "holdout_r2": 0.77,
      "holdout_rmse": 45.1,
      "holdout_mae": 28.7,
      "delta_r2_vs_auto_arima": 0.08,
      "delta_r2_vs_ar1": 0.19,
      "fit_time_sec": 18.4,
      "artifact": "OUTPUT_DIR/candidate-FAAR-ARIMA.joblib"
    },
    {
      "model_name": "XGBoost",
      "tier": 3,
      "status": "success",
      "best_params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05},
      "cv_r2_mean": 0.81,
      "cv_r2_std": 0.04,
      "holdout_r2": 0.79,
      "holdout_rmse": 42.3,
      "holdout_mae": 26.1,
      "delta_r2_vs_auto_arima": 0.10,
      "delta_r2_vs_ar1": 0.21,
      "fit_time_sec": 47.3,
      "high_cv_variance_warning": false,
      "artifact": "OUTPUT_DIR/candidate-XGBoost.joblib"
    },
    {
      "model_name": "ARIMA_XGB_residual",
      "tier": 4,
      "status": "success",
      "base_arima_r2": 0.69,
      "residual_xgb_r2": 0.31,
      "holdout_r2": 0.82,
      "holdout_rmse": 39.8,
      "holdout_mae": 24.5,
      "delta_r2_vs_auto_arima": 0.13,
      "delta_r2_vs_ar1": 0.24,
      "fit_time_sec": 55.2,
      "artifact": "OUTPUT_DIR/candidate-ARIMA_XGB_residual.joblib"
    }
  ],

  "best_model_name": "ARIMA_XGB_residual",
  "benchmark_warning": false,
  "skipped_models": [
    {"name": "TBATS", "reason": "sktime not installed"},
    {"name": "Factor-VAR", "reason": "insufficient PCA factors (n_components < 2)"}
  ]
}
```

## Progress JSON Update Contract

```json
{
  "current_step": "13-model-training",
  "current_model": "XGBoost",
  "completed_models": ["naive_persistence", "seasonal_naive", "auto_arima_benchmark",
                       "ar1_benchmark", "AR", "ARIMA", "SARIMA", "FAAR-ARIMA",
                       "ElasticNet", "Ridge"],
  "model_history": [
    {"model_name": "ARIMA", "holdout_r2": 0.70, "fit_time_sec": 8.1},
    {"model_name": "ElasticNet", "holdout_r2": 0.64, "fit_time_sec": 1.2}
  ],
  "model_progress": 0.65
}
```

---

## CLI Contract

```bash
python step_13_training.py \
  --output-dir OUTPUT_DIR \
  --run-id RUN_ID \
  [--target-column TARGET_COLUMN]  # fallback: read from progress.json
```

---

## Test Matrix

- `TimeSeriesSplit` always used; any random split raises `RuntimeError`.
- All four benchmarks always present in output.
- `naive_persistence` holdout computed correctly: `ŷ(t) = y(t-1)` on holdout only.
- `seasonal_naive` output is null when `primary_seasonal_period` is null.
- `auto_arima_benchmark` uses pmdarima when available; falls back to `ARIMA(1,1,1)`.
- `ar1_benchmark` uses `statsmodels.AutoReg(lags=1)`; falls back to naive persistence on failure.
- FAAR models skip gracefully when `pca_factors` group is empty.
- Tier 4 hybrids skip gracefully when prerequisite Tier 1/3 models failed.
- `StatsmodelsAdapter.predict()` returns correct holdout-length array.
- `holdout.npz` loadable in a fresh Python process without step 13 script in scope.
- `benchmark_warning=true` when best holdout R² does not exceed both `auto_arima_benchmark` and `ar1_benchmark` by at least 0.02.
- High CV variance warning triggered when `cv_r2_std / |cv_r2_mean| > 0.3`.
- Failed candidate logged as `"status": "failed"`; pipeline continues to next candidate.
- Leakage gate: feature with `|r(X,y)| > 0.98` raises before any model is trained.
