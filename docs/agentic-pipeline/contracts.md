# Agentic Pipeline Contracts

## Non-Negotiable Leakage Policy

This pipeline is for real forecasting. Any target leakage invalidates the run.

- Feature engineering must be causal: at timestamp t, features may only use information available at or before t-1.
- Forbidden features include (but are not limited to):
  - direct target copies/transforms at t (`target`, `target_scaled`, `target_diff_0`)
  - rolling statistics that include current target value (`rolling_mean(target, N)` without a prior shift)
  - any algebraic combination that can reconstruct target exactly or near-exactly
- Required target-derived features must be lagged-only, e.g. `target_lag_k` where k >= 1.
- If leakage is suspected, the pipeline must fail with explicit diagnostics and must not mark status as completed.

## Global Runtime Inputs
- `CSV_FILE`: "/data/appliances_energy_prediction.csv"
- `RUN_ID`: create unique run id (timestamp in format YYYYMMDDTHHMMZ)
- `OUTPUT_DIR`: "/$RUN_ID/"
- `ARTIFACTS_DIR`: "/$RUN_ID/artifacts"
- `CODE_DIR`: directory for generated step Python files (default: `/$OUTPUT_DIR/code/`)

## Code Organisation Contract

**One Python file per pipeline step. No monolithic scripts.**

## Model Artifact Portability Rules
- `model.joblib` must be loadable via `joblib.load(...)` in a fresh Python process.
- The loaded object must expose `.predict(X)` directly (not wrapped in a plain dict).
- Do not pickle classes defined under `__main__`; use importable sklearn/sklearn-compatible estimators only.
- `holdout.npz` must contain `X_test` and `y_test` arrays reusable by step 14 without access to step 13's script.