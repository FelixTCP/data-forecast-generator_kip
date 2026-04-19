# #12 Context Engineering: Feature Extraction

## Objective

Build a leakage-safe, critically filtered feature matrix. This step is **not** "take all numeric columns" — it is a deliberate selection step guided directly by the analysis in step 11. Every feature included must have evidence of predictive signal. Every feature excluded must have a logged reason.

## Inputs

- `cleaned.parquet` from step 10
- `step-11-exploration.json` — critically: `recommended_features`, `significant_lags`, `useful_lag_features`, `excluded_features`
- `TARGET_COLUMN` and `SPLIT_MODE`

## Outputs

- `features.parquet` — feature matrix (numeric only, target included)
- `step-12-features.json` with full audit trail

## Feature Construction Rules

### Rule 1 — Start from `recommended_features`
Only features listed in `step-11-exploration.json["recommended_features"]` are candidates for the feature matrix. Do **not** add any feature that was excluded in step 11 unless it is the target column itself.

### Rule 2 — Time Features (only when time column detected)
Create date decomposition features: `year`, `month`, `day_of_week`, `hour`.
These are always included when a time column exists.

### Rule 3 — Lag Features (only from `useful_lag_features`)
Only create lag features for feature-lag combinations listed in `step-11["useful_lag_features"]` (cross-correlation > 0.15). Do **not** create lags for all features blindly.
Additionally, create lag features for the target itself only at lags listed in `step-11["significant_lags"]`.

### Rule 4 — Rolling Window Features
Create causal rolling features only. This means shift first, then roll:

- Allowed: `target.shift(1).rolling_mean(N)`, `target.shift(1).rolling_std(N)`
- Forbidden: `target.rolling_mean(N)`, `target.rolling_std(N)` (these include target at time t and leak label information)

If no significant lags, skip rolling features.

### Rule 5 — Drop After Construction
After constructing lag/rolling features, rows with NaN introduced by shifting must be dropped. Log the row count lost.

### Rule 6 — Final Numeric Filter
After all construction: keep only columns with numeric dtype. Log any non-numeric column that was dropped.

### Rule 7 — Leakage Guard
Assert that no feature column has |Pearson correlation with target| > 0.99 on the full dataset. If any does, flag it as potential leakage and exclude it with a log entry.

### Rule 8 — Algebraic Leakage Guard (Mandatory)
In addition to pairwise correlation, test for reconstruction-style leakage:

- For every target-derived feature pair/triple, run a linear probe `y ~ features` on train only and evaluate on test.
- If a tiny subset of engineered features achieves near-perfect performance (R2 > 0.995 or RMSE < 1e-6 in target units), flag leakage.
- If leakage is flagged, fail step 12 with actionable diagnostics and do not write a "successful" features artifact.

## Output JSON Keys

```json
{
  "step": "12-feature-extraction",
  "features": ["t6", "t1", "rh_6", "lights", "t_out", "target_lag_1", "t1_lag_1", "hour"],
  "features_excluded": {
    "rv1": "below_noise_baseline (step 11)",
    "rv2": "redundant with rv1 (step 11)"
  },
  "created_features": [
    {"name": "target_lag_1", "reason": "significant_lag=1 from step 11"},
    {"name": "t1_lag_1", "reason": "useful_lag_feature: t1 at lag=1, xcorr=0.23"},
    {"name": "hour", "reason": "time decomposition"}
  ],
  "rows_dropped_by_lag": 3,
  "leakage_flags": [],
  "leakage_audit": {
    "status": "pass",
    "checks": ["pairwise_corr", "linear_probe_reconstruction"],
    "details": []
  },
  "split_strategy": {
    "requested_mode": "auto",
    "resolved_mode": "time_series",
    "time_column": "date",
    "random_state": 42
  },
  "artifacts": {"features_parquet": "output/.../features.parquet"},
  "context": {...}
}
```

## Guardrails

- If `recommended_features` from step 11 is empty or missing, raise a `ValueError` with a message pointing to step 11 output — do not fall back to all numeric columns.
- If no lag features are created (no significant lags), log this explicitly: "No lag features created — no significant autocorrelation detected."
- Minimum viable feature count: if fewer than 2 features survive all filters, raise an error and report which rules caused the exclusions.
- Any feature name matching `target_rolling_` must be explicitly verified as shift-before-roll; otherwise exclude and flag as leakage.
- Persist `leakage_audit.json` to `OUTPUT_DIR` and include summary in `step-12-features.json`.

## Copilot Prompt Snippet

```markdown
Implement `build_features(df, target_column, step11_output: dict, time_column: str | None, split_mode: str)`.
Start from step11_output["recommended_features"] only.
Create lag features strictly from step11_output["useful_lag_features"] and step11_output["significant_lags"].
Create rolling features using shift-before-roll only (causal).
Document every inclusion and exclusion. Enforce leakage guard. Output feature matrix and full audit trail.
```

## Tests

- recommended_features is empty (should raise, not fall back)
- no significant lags (no lag features created, logged)
- leakage column present (cross-corr 0.999 with target — should be excluded)
- deterministic feature order
- forbidden target rolling feature (without shift) should hard-fail
- algebraic reconstruction test detects near-perfect leakage and hard-fails
