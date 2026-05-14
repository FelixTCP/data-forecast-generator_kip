# 14 Context Engineering: Model Evaluation

## Objective

Evaluate all trained candidates on the hold-out set, produce comparison-ready metrics, and conduct deep time-series diagnostic checks (residual analysis, horizon degradation). **Critically assess whether the results are acceptable**. If they are subpar or fail baseline checks, emit a structured diagnosis and trigger an expansion round (more model classes, adjusted features) rather than silently accepting poor predictions.

## Outputs

- classical metrics (R², RMSE, MAE, MAPE, sMAPE)

- CV metrics (`cv_mean_r2`, `cv_std_r2`) and target statistics

- **computational metrics** (`inference_time_ms`)

- **baseline comparison (MASE & Naive R²/RMSE/MAE)**

- **systematic bias detection** (Mean Error / Over- vs. Under-Forecasting)

- **residual diagnostics** (White Noise check, Normality, Autocorrelation)

- **horizon-based degradation analysis**

- **directional accuracy**

- **uncertainty/coverage check** (if probabilistic bounds are supported)

- explorative evaluation / model-specific insights (feature importance, coefficients)

- `quality_assessment` — strict status enum (`acceptable`, `marginal`, `subpar`, `leakage_suspected`)

- `quality_assessment_text` — human-readable diagnostic text

- `expansion_diagnosis` & `expansion_candidates` (if triggered)

- `leakage_probe` details (if triggered)

## Analysis Requirements

**1. Error Metrics, Bias & Computational Cost**

- Compute RMSE, MAE, MAPE, sMAPE, and R² on the holdout set for all candidates. (Omit MAPE if target has zeros).

- Compute `cv_mean_r2`, `cv_std_r2` from training cross-validation.

- **Systematic Bias:** Calculate the Mean Error (ME = mean(y_true - y_pred)). A highly positive or negative ME indicates systematic under- or over-forecasting.

- **Computational Cost:** Log the time it takes to predict on the holdout set (`inference_time_ms`). 

- Log the target's mean, std, min, and max alongside metrics so human reviewers can contextualise RMSE.

**2. Baseline Comparison (The Reality Check)**

- Generate a "Naive Forecast" (e.g., $y_{pred}(t) = y_{true}(t-1)$).

- Compute naive baseline metrics (`naive_baseline_r2`, `naive_baseline_rmse`, `naive_baseline_mae`) and the Mean Absolute Scaled Error (MASE).

- **Rule:** A model with MASE $\ge$ 1.0 provides no value over a naive guess and must be heavily penalised.

**3. Residual & Horizon Diagnostics (Time-Series specific)**

- **Ljung-Box Test:** Check for remaining autocorrelation in the residuals (lags 1–10). A p-value < 0.05 indicates missed systematic patterns.

- **Normality Test:** Check if residuals are normally distributed (Shapiro-Wilk).

- **Horizon Degradation:** Calculate MAE for the first 25% of the horizon vs. the last 25%. Log the degradation percentage.

- **Directional Accuracy:** Compute the percentage of times the model correctly predicted the trend direction ($y_t > y_{t-1}$).

**4. Uncertainty & Coverage (If Supported)**

If a candidate model outputs prediction intervals (e.g., lower and upper bounds for a 90% confidence interval):

- Calculate the **Prediction Interval Coverage Probability (PICP):** The percentage of true holdout values that actually fell within the predicted bounds.

- Flag models as `overconfident` if PICP is severely below the expected level (e.g., < 80% for a 90% interval).

**5. Quality Threshold Assessment**

After computing metrics, evaluate the result set against these thresholds:

| Condition | Status | Action |
|-----------|--------|---------|
|Best candidate R² $\ge$ 0.50 AND MASE < 1.0|`acceptable`|Proceed to step 15.|
|Best candidate R² in [0.25, 0.50) OR MASE $\ge$ 1.0|`marginal`|Log a warning, proceed to step 15 but flag in report.|
|Best candidate R² < 0.25|`subpar`|Trigger Expansion Round.|
|R² < 0|n/a|Flag candidate as `"model_worse_than_mean_baseline": true`.|

**6. Suspiciously-Perfect Score Protocol (Mandatory)**

When any candidate achieves unusually perfect performance (e.g., R² > 0.98), run an explicit leakage stress test before accepting the result:

1) Re-evaluate after removing all target-derived engineered features.

2) Re-evaluate after removing all rolling target features.

3) Run a linear reconstruction probe on target-derived features only.

4) If any probe indicates trivial target reconstruction, set `quality_assessment` to `"leakage_suspected"`, write diagnostics, and **halt progression** to step 15.

**7. Expansion Round (Triggered when quality_assessment = "subpar")**

When the best R² < 0.25, do not proceed to step 15. Instead:

1) Update `progress.json` with `"status": "expansion_required"`.

2) Diagnose the likely cause using this checklist (log findings in `"expansion_diagnosis"`):

    - Are candidate RMSE values much larger than the target's standard deviation? If yes, models are not learning.

    - Is training CV R² also near zero? If yes, the feature set is uninformative — go back to step 11/12.

    - Is training CV R² decent but holdout R² poor? If yes, the models are overfitting — review the split and lag feature construction.

    - Is the target highly skewed or heavy-tailed? If yes, recommend log-transform of target.

3) Propose and train an **expansion set** of additional candidates:

    - `ElasticNet` (handles collinear features better than Ridge)

    - `HistGradientBoostingRegressor` (handles mixed features, built-in missing value support)

    - `SVR(kernel='rbf')` (non-linear, good for smaller datasets)

    - If diagnosis suggests feature issues: re-run step 12 with a relaxed MI threshold before training expansion candidates.

4) Evaluate the expansion candidates under identical conditions.

5) If the best expanded candidate achieves R² $\ge$ 0.25: update JSON, set `quality_assessment` to `"marginal"` or `"acceptable"`, and proceed to step 15.

6) If still below threshold: proceed to step 15, but set `"quality_assessment"`: `"subpar_after_expansion"`.

**8. Explorative / Model-Specific Insights**

- For Tree-based models (e.g., XGBoost, LightGBM, HistGradientBoosting): Extract and log the top 5 most important features.

- For Linear/Statistical models (e.g., Ridge, ElasticNet): Extract and log the top 5 most significant coefficients.

- Gracefully handle models that do not natively support feature importance (return `null` for these models).

## Output JSON Keys
```
{
  "step": "14-model-evaluation",
  "target_stats": {"mean": 97.7, "std": 102.5, "min": 10, "max": 1080},
  "naive_baseline": {"r2": 0.05, "rmse": 99.2, "mae": 60.1},
  "candidates": [
    {
      "model_name": "xgboost_v1",
      "computational_cost": {
        "fit_time_sec": 4.2,
        "inference_time_ms": 12.5
      },
      "metrics": {
        "r2": 0.88,
        "rmse": 12.4,
        "mae": 9.2,
        "smape": 0.045,
        "mase": 0.65,
        "mean_error": -1.2,
        "directional_accuracy": 0.72
      },
      "cv_metrics": {
        "cv_mean_r2": 0.85,
        "cv_std_r2": 0.04
      },
      "residual_diagnostics": {
        "residual_max_abs": 45.1,
        "ljung_box_p_value": 0.12,
        "residuals_normal": true
      },
      "horizon_degradation": {
        "short_term_mae": 8.0,
        "long_term_mae": 11.5,
        "degradation_pct": 43.7
      },
      "probabilistic_evaluation": {
        "supports_intervals": true,
        "picp_90": 0.88,
        "overconfident_flag": false
      },
      "model_insights": {
        "top_features": ["t1_lag_24", "temperature", "day_of_week"]
      },
      "model_worse_than_mean_baseline": false
    }
  ],
  "quality_assessment": "subpar",
  "quality_assessment_text": "All initial models failed to capture the signal effectively, with XGBoost performing best but below the R2 0.25 threshold.",
  "expansion_diagnosis": "Training CV R² ≈ 0 for tree models — feature set likely uninformative for non-linear models after time-series split. Recommend expanding to ElasticNet and HistGradientBoosting.",
  "expansion_candidates": [],
  "leakage_probe": {
    "triggered": false,
    "status": "pass",
    "details": []
  },
  "context": {}
}
```

## Guardrails

- A model with R² < 0 is **worse than predicting the mean**. It must never be selected as the final model. Do not normalise negative R² into a ranked weight.

- Same holdout split as used in training — never re-split for evaluation.

- If candidate score is dramatically above naive baseline and above expected realism range, require leakage probe even when thresholds classify as acceptable.

- **quality_assessment = leakage_suspected** is a hard stop; no final model selection.

- Ensure correct chronologies when calculating Directional Accuracy (comparing $t$ to $t-1$).

## Copilot Prompt Snippet
```
Implement `CODE_DIR/step_14_evaluation.py`. The CLI receives `--output-dir` and `--run-id`.
Load trained models from `OUTPUT_DIR/candidate-*.joblib` and holdout from `OUTPUT_DIR/holdout.npz`.
For each candidate: compute computational cost, classical metrics (R², RMSE, MAE, sMAPE), Bias (Mean Error), and MASE.
Compute naive baseline metrics for context. Conduct TS diagnostics (Ljung-Box, horizon degradation, directional accuracy) and PICP (if probabilistic bounds exist).
Extract model-specific insights (feature importance / top coefficients) where supported.
If suspiciously-perfect scores appear (R2 > 0.98), run leakage stress tests and fail with `leakage_suspected` when triggered.
Compare best R² against quality thresholds. If subpar: diagnose, update `progress.json` status to `expansion_required`, train expansion candidates, re-evaluate.
Output the highly structured `OUTPUT_DIR/step-14-evaluation.json`.
```

## Tests

- model performs worse than mean baseline (R² < 0 should trigger `model_worse_than_mean_baseline` flag)

- perfect predictions (ensure MAPE/sMAPE don't divide by zero; must trigger `leakage_probe`)

- residuals show strong autocorrelation (Ljung-Box p < 0.05)

- systematic bias (model consistently under-predicts, Mean Error should be distinct from MAE)

- R² < 0.25 triggers expansion diagnosis logic and evaluates ElasticNet/HistGradientBoosting