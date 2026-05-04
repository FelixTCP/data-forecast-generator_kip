# #13 Context Engineering: Model Training

## Objective

Train a rich set of candidate models against two mandatory benchmarks. Every candidate is evaluated on the same chronological holdout. Log full training history and output serialized artifacts for step-14.

## Inputs

- `X`, `y` — features and target, chronologically ordered, leakage-free
- `preprocessor` — fitted sklearn-compatible transformer
- `training_config` — `n_splits`, `search`, `cache_dir`, `random_state`, `output_dir`

## Outputs

- `candidate-{name}.joblib` — serialized estimator per candidate
- `step-13-training.json` — full history: params, CV scores (r2/rmse/mae), holdout scores, fit time, benchmark deltas
- `progress.json` updated throughout training (see contract below)

---

## Mandatory Benchmarks — ALWAYS run, NEVER skip

Two benchmarks are required on every run regardless of data characteristics. They establish the performance floor and define what "improvement" means.

| Benchmark | Role | Implementation |
|---|---|---|
| **`arima_benchmark`** | Classical statistical time-series baseline | `pmdarima.auto_arima` (auto-selects p,d,q via AIC); fall back to `statsmodels.tsa.arima.model.ARIMA(order=(1,1,1))` if pmdarima unavailable |
| **`kmeans_benchmark`** | Cluster-centroid pattern baseline | Fit `KMeans(n_clusters=k)` on lag features; predict by mapping each test sample to its nearest cluster centroid's mean target value from the training set |

If the winning candidate does not outperform **both** benchmarks on holdout R² by at least 0.02, emit a `"benchmark_warning"` in the output JSON.

A naive persistence baseline (`ŷ_t = y_{t-1}`) is also always logged (but does not count as a candidate).

---

## Candidate Model Menu — Agent Selects ≥ 3 Beyond Benchmarks

The agent MUST be creative and data-driven. Use step-11 exploration output (autocorrelation, seasonality strength, dataset size, feature count) to select the most promising candidates. Select at least 3 from this menu:

| Model Family | When to prioritize | Library |
|---|---|---|
| **Ridge / Lasso** | Always — cheap, interpretable linear baseline | scikit-learn |
| **Gradient Boosting** | Non-linear interactions, tabular features | `xgboost` or `lightgbm`; fallback: `sklearn.GradientBoostingRegressor` |
| **Random Forest** | General-purpose, high variance data | scikit-learn |
| **SVR** | Smooth target, medium dataset (< 50 k rows) | scikit-learn |
| **MLP / Neural Net** | Large dataset (> 5 k rows), many features | `sklearn.neural_network.MLPRegressor` or PyTorch |
| **Extra Trees** | High-dimensional feature space | scikit-learn |
| **SARIMA / Holt-Winters** | Strong confirmed seasonality from step-11 | statsmodels |
| **LightGBM / XGBoost** | Best default for structured time-series tabular data | xgboost / lightgbm |

Skip a family only if its library is unavailable; record every skipped family with reason in `skipped_models`.

---

## Split & CV Rules

- **Always** use `TimeSeriesSplit(n_splits=5)` — random splits are forbidden and must raise.
- Holdout = last chronological 20 % of data, fixed before any model sees it.
- Apply a purge gap of at least 1 fold-width between train and validation folds to avoid boundary leakage.
- ARIMA and k-means benchmarks use the **same holdout window** for a fair comparison.
- Cache sklearn pipeline transformations with `joblib.Memory`.
- Fixed `random_state` on all models and splits.

---

## Guardrails

- **Hard-fail** if any leaked feature is detected in `X` (target rolling stats computed on the full series; future-dated regressors).
- Track `r2`, `rmse`, `mae` for every candidate on both CV folds and holdout.
- Warn if CV fold variance `(std / |mean|) > 0.3` for any candidate — high variance signals overfitting or leakage.
- Compute and log `delta_r2_vs_arima` and `delta_r2_vs_kmeans` for every candidate.
- Do not select a best model based on CV alone — holdout R² is the tiebreaker.

---

## Agent Prompt

```
Implement step 13: model training.

MANDATORY:
- Always train `arima_benchmark` (pmdarima.auto_arima or statsmodels fallback) and `kmeans_benchmark` (KMeans on lag features, predict by nearest-centroid mean target) before any other candidate.
- Always log naive persistence baseline (ŷ_t = y_{t-1}) on the holdout window.

CREATIVE CANDIDATE SELECTION:
- Inspect step-11 output. Based on autocorrelation structure, seasonality strength, dataset size, and feature count, select at least 3 candidate families from: Ridge/Lasso, GradientBoosting (xgboost/lightgbm), RandomForest, SVR, MLPRegressor, ExtraTrees, SARIMA/Holt-Winters, LightGBM/XGBoost.
- Be precise: justify each selection in a one-line comment in progress.json.
- If a library is missing, skip it and record the reason in `skipped_models`.

SPLIT: TimeSeriesSplit(n_splits=5) only. Random splits are forbidden.
HOLDOUT: last 20% chronologically, fixed before training.
SEARCH: RandomizedSearchCV for non-trivial param spaces; GridSearchCV for small spaces.
CACHE: joblib.Memory for sklearn pipeline step caching.

OUTPUT per candidate in step-13-training.json:
- model_name, best_params, cv_r2_mean, cv_r2_std, holdout_r2, holdout_rmse, holdout_mae, delta_r2_vs_arima, delta_r2_vs_kmeans, fit_time_sec

LEAKAGE GATE: hard-fail if any forbidden leaked feature is present in X.
BENCHMARK WARNING: if best candidate holdout_r2 - arima holdout_r2 < 0.02 OR best candidate holdout_r2 - kmeans holdout_r2 < 0.02, set benchmark_warning=true.

Update OUTPUT_DIR/progress.json after each model completes.
```

---

## step-13-training.json Schema

```json
{
  "run_id": "2026-04-03T15:00:00Z",
  "split_mode": "time_series",
  "random_state": 42,
  "benchmarks": {
    "naive_persistence": { "holdout_r2": 0.41, "holdout_rmse": 24.6, "holdout_mae": 18.1 },
    "arima_benchmark":   { "holdout_r2": 0.69, "holdout_rmse": 12.4, "holdout_mae": 9.2 },
    "kmeans_benchmark":  { "holdout_r2": 0.55, "holdout_rmse": 18.1, "holdout_mae": 13.5 }
  },
  "candidates": [
    {
      "model_name": "gradient_boosting",
      "best_params": { "model__n_estimators": 200, "model__max_depth": 5 },
      "cv_r2_mean": 0.88,
      "cv_r2_std": 0.03,
      "holdout_r2": 0.85,
      "holdout_rmse": 7.1,
      "holdout_mae": 5.4,
      "delta_r2_vs_arima": 0.16,
      "delta_r2_vs_kmeans": 0.30,
      "fit_time_sec": 18.3
    }
  ],
  "best_model_name": "gradient_boosting",
  "benchmark_warning": false,
  "skipped_models": [{ "name": "torch_lstm", "reason": "torch not installed" }]
}
```

## progress.json Contract

```json
{
  "current_step": "13-model-training",
  "current_model": "gradient_boosting",
  "completed_models": ["arima_benchmark", "kmeans_benchmark", "ridge"],
  "model_history": [
    { "model_name": "arima_benchmark", "holdout_r2": 0.69, "fit_time_sec": 4.2 },
    { "model_name": "kmeans_benchmark", "holdout_r2": 0.55, "fit_time_sec": 0.8 }
  ]
}
```

---

## Test Matrix

- `TimeSeriesSplit` always used; random split raises immediately.
- Both `arima_benchmark` and `kmeans_benchmark` always present in output regardless of config.
- `naive_persistence` always logged.
- Deterministic results with fixed `random_state`.
- Hard-fail on leaked features in input.
- `delta_r2_vs_arima` and `delta_r2_vs_kmeans` computed for every candidate.
- `benchmark_warning` correctly set when improvement margin < 0.02.
- `skipped_models` populated when library is absent.
- Failed candidates logged with `"status": "failed"` and error message; pipeline continues.
