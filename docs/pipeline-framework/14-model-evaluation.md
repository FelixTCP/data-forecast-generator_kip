# #14 Context Engineering: Model Evaluation

## Objective

Evaluate all trained candidates on the hold-out set, produce comparison-ready metrics, and **critically assess whether the results are acceptable**. If they are not, emit a structured diagnosis and trigger an expansion round (more model classes, adjusted features) rather than silently accepting poor predictions.

## Required Metrics (per candidate)

- `r2` — coefficient of determination
- `rmse` — root mean squared error
- `mae` — mean absolute error
- `cv_mean_r2`, `cv_std_r2` — from training cross-validation
- `residual_mean`, `residual_max_abs` — basic residual summary
- `mape` — only when target has no zeros; otherwise omit and log the skip
- `naive_baseline_r2`, `naive_baseline_rmse`, `naive_baseline_mae` for `y_hat_t = y_{t-1}` on holdout

## Quality Threshold Assessment

After computing metrics, evaluate the result set against these thresholds:

| Condition | Status |
|---|---|
| Best candidate R² ≥ 0.50 | `acceptable` |
| Best candidate R² in [0.25, 0.50) | `marginal` — log a warning, proceed to step 15 but flag in report |
| Best candidate R² < 0.25 | `subpar` — trigger expansion round (see below) |
| Any candidate has R² < 0 | Always log as `"model_worse_than_mean_baseline": true` for that candidate; a model with negative R² has negative value and must be called out explicitly |

The quality threshold result must be written to `step-14-evaluation.json` under `"quality_assessment"`.

## Suspiciously-Perfect Score Protocol (Mandatory)

When any candidate achieves unusually perfect performance (for example R2 > 0.98 on real-world energy-use data), run an explicit leakage stress test before accepting the result:

1. Re-evaluate after removing all target-derived engineered features.
2. Re-evaluate after removing all rolling target features.
3. Run a linear reconstruction probe on target-derived features only.
4. If any probe indicates trivial target reconstruction, set `quality_assessment = "leakage_suspected"`, write diagnostics, and halt progression to step 15.

## Expansion Round (triggered when quality_assessment = "subpar")

When the best R² < 0.25, do **not** proceed to step 15. Instead:

1. Write `step-14-evaluation.json` with the current results and `"quality_assessment": "subpar"`.
2. Update `progress.json` with `"status": "expansion_required"`.
3. Diagnose the likely cause using this checklist (log findings in `"expansion_diagnosis"`):
   - Are candidate RMSE values much larger than the target's standard deviation? If yes, models are not learning.
   - Is training CV R² also near zero? If yes, the feature set is uninformative — go back to step 11/12.
   - Is training CV R² decent but holdout R² poor? If yes, the models are overfitting — review the split and lag feature construction.
   - Is the target highly skewed or heavy-tailed? If yes, recommend log-transform of target.
4. Propose and train an **expansion set** of additional candidates:
   - `ElasticNet` (handles collinear features better than Ridge)
   - `HistGradientBoostingRegressor` (handles mixed features, built-in missing value support)
   - `SVR(kernel='rbf')` (non-linear, good for smaller datasets)
   - If the diagnosis suggests feature issues: re-run step 12 with a relaxed MI threshold before training expansion candidates.
5. Evaluate the expansion candidates under identical conditions.
6. If the best expanded candidate achieves R² ≥ 0.25: update `step-14-evaluation.json` with all results, set `quality_assessment` to `"marginal"` or `"acceptable"`, and proceed to step 15.
7. If still below threshold after expansion: proceed to step 15 anyway, but set `"quality_assessment": "subpar_after_expansion"`. Step 16 must prominently flag this.

## Guardrails

- A model with R² < 0 is **worse than predicting the mean**. It must never be selected as the final model.
- Do not normalise negative R² into a ranked weight — a negative-R² model scores zero in the selection step.
- Same holdout split as used in training — never re-split for evaluation.
- Log the target's mean, std, and range alongside metrics so human reviewers can contextualise RMSE.
- If candidate score is dramatically above naive baseline and above expected realism range, require leakage probe even when thresholds classify as acceptable.
- `quality_assessment = leakage_suspected` is a hard stop; no final model selection.

## Output JSON Keys

```json
{
  "step": "14-model-evaluation",
  "target_stats": {"mean": 97.7, "std": 102.5, "min": 10, "max": 1080},
  "candidates": [
    {
      "model_name": "ridge",
      "r2": 0.11,
      "rmse": 85.7,
      "mae": 49.9,
      "cv_mean_r2": 0.09,
      "cv_std_r2": 0.10,
      "residual_mean": 0.2,
      "residual_max_abs": 820.0,
      "model_worse_than_mean_baseline": false
    }
  ],
  "quality_assessment": "subpar",
  "expansion_diagnosis": "Training CV R² ≈ 0 for tree models — feature set likely uninformative for non-linear models after time-series split. Recommend expanding to ElasticNet and HistGradientBoosting.",
  "expansion_candidates": [...],
  "leakage_probe": {
    "triggered": true,
    "status": "pass",
    "details": []
  },
  "context": {...}
}
```

## Copilot Prompt Snippet

```markdown
Implement `evaluate_candidates(candidates, holdout_path, output_dir) -> dict`.
For each candidate: compute R², RMSE, MAE, residual summary.
Compute naive baseline metrics for context.
If suspiciously-perfect scores appear, run leakage stress tests and fail with leakage_suspected when triggered.
Compare best R² against quality thresholds. If subpar: diagnose, train expansion candidates, evaluate, update results.
Compute and save SHAP values for the best candidate.
Write step-14-evaluation.json with quality_assessment field.
```

## SHAP Value Computation (Required for Streamlit Best Model View)

After all candidates are evaluated and the best model is identified (but only if `quality_assessment != "leakage_suspected"`), compute SHAP values for the **best candidate** on the holdout test set and save them for the Streamlit UI.

### Explainer Selection by Model Type

| Model type | Explainer | Notes |
|---|---|---|
| Tree-based (XGBoost, LightGBM, RandomForest, GradientBoosting, ExtraTrees, HistGBM) | `shap.TreeExplainer(model)` | Fast, exact |
| Linear (Ridge, Lasso, ElasticNet) | `shap.LinearExplainer(model, X_train_sample)` | Use min(500, N_train) background samples |
| Other sklearn (SVR, MLP) | `shap.KernelExplainer(model.predict, shap.sample(X_train, 100))` | Slow — use min(200, N_test) test samples |
| Statistical models (`StatsmodelsAdapter`) | Skip — not differentiable | Set `shap_skipped_reason: "statistical_model"` |

For pipeline-wrapped models, extract the final estimator step before computing SHAP.

### Computation Rules

1. Use min(500, N_test) samples from `X_test` for SHAP computation (avoids OOM on large holdout sets). Sample **chronologically** from the start of the holdout (not randomly).
2. Use `shap.Explanation` objects when available (shap ≥ 0.41) for portability.
3. Convert feature matrix to numpy before calling explainer (polars DataFrames are not accepted by shap).
4. Load `feature_names` from `step-12-features.json["feature_names"]`.

### Artifacts to Write

- **`OUTPUT_DIR/shap_values.npz`** — numpy archive:
  ```
  shap_values      : float32 array, shape (n_samples, n_features)
  base_values      : float32 array, shape (n_samples,) — per-sample expected value
  expected_value   : scalar float — global expected value (mean base_value)
  X_test_sample    : float32 array, shape (n_samples, n_features) — the input rows used
  feature_names    : object array of str, shape (n_features,)
  ```
- **`shap_artifacts`** key in `step-14-evaluation.json`:
  ```json
  "shap_artifacts": {
    "status": "computed",
    "model_name": "XGBoost",
    "explainer_type": "TreeExplainer",
    "n_samples_used": 500,
    "shap_values_path": "OUTPUT_DIR/shap_values.npz",
    "top_features_by_mean_abs_shap": [
      {"feature": "y_lag_1", "mean_abs_shap": 12.4},
      {"feature": "hour_of_day", "mean_abs_shap": 8.1}
    ]
  }
  ```
  If SHAP is skipped: `"status": "skipped"` and `"shap_skipped_reason": "<reason>"`.

### Guardrails

- Wrap the entire SHAP computation in a try/except. If SHAP raises (e.g., library not installed), set `"status": "failed"` with `"shap_error": "<message>"` — do not halt the pipeline.
- `shap_values.npz` must only be written if SHAP computation succeeds completely.
- Do not compute SHAP when `quality_assessment = "leakage_suspected"` — metrics are invalid.
