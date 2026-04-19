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

## Source-of-Truth Rule

- `docs/agentic-pipeline/*` defines runtime execution contracts (resume, file layout, step script invariants).
- `docs/pipeline-framework/*` defines canonical per-step behavior (metrics, thresholds, feature/model rules).
- If guidance conflicts, prefer `docs/pipeline-framework/*` for step logic and `contracts.md` for orchestration/runtime behavior.

## Global Runtime Inputs

- `CSV_PATH`: absolute or repo-relative input CSV (required)
- `TARGET_COLUMN`: regression target column (required)
- `RUN_ID`: unique run id (UTC timestamp)
- `OUTPUT_DIR`: run artifact directory (recommended: `output/<RUN_ID>/`)
- `CODE_DIR`: generated step script directory (default: `OUTPUT_DIR/code/`)
- `SPLIT_MODE`: `auto|random|time_series` (default: `auto`)
- `CONTINUE_MODE`: `true|false` (default: `false`)

## Code Organisation Contract

**One Python file per pipeline step. No monolithic scripts.**

| Step                      | Script file                       | Output JSON                |
| ------------------------- | --------------------------------- | -------------------------- |
| 10 — CSV Read & Cleansing | `CODE_DIR/step_10_cleanse.py`     | `step-10-cleanse.json`     |
| 11 — Data Exploration     | `CODE_DIR/step_11_exploration.py` | `step-11-exploration.json` |
| 12 — Feature Extraction   | `CODE_DIR/step_12_features.py`    | `step-12-features.json`    |
| 13 — Model Training       | `CODE_DIR/step_13_training.py`    | `step-13-training.json`    |
| 14 — Model Evaluation     | `CODE_DIR/step_14_evaluation.py`  | `step-14-evaluation.json`  |
| 15 — Model Selection      | `CODE_DIR/step_15_selection.py`   | `step-15-selection.json`   |
| 16 — Result Presentation  | `CODE_DIR/step_16_report.py`      | `step-16-report.md`        |
| Orchestrator              | `CODE_DIR/orchestrator.py`        | —                          |

## Step Script Contract

Each step script must:

1. Read its inputs from `OUTPUT_DIR` (prior step JSONs / parquet files) — not from hardcoded paths.
2. Write its output JSON to `OUTPUT_DIR` and update `OUTPUT_DIR/progress.json`.
3. Exit with code `0` on success, non-zero on any unhandled failure.
4. Be idempotent: running the same step twice with the same inputs produces identical outputs.
5. Be independently runnable without requiring the previous step's script to be in scope.
6. Enforce leakage checks and fail fast when triggered. A run with leakage cannot proceed to production selection.

## Resume / Skip Contract

A step is considered complete (and may be skipped) when all are true:

1. `CONTINUE_MODE=true` (or orchestrator `--resume`).
2. The step output file exists.
3. The output is valid JSON with the expected `"step"` value.

If any of these checks fail, the step must be re-run.

## Required Files per Run

- `OUTPUT_DIR/progress.json`
- `OUTPUT_DIR/cleaned.parquet`
- `OUTPUT_DIR/features.parquet`
- `OUTPUT_DIR/holdout.npz`
- `OUTPUT_DIR/step-10-cleanse.json`
- `OUTPUT_DIR/step-11-exploration.json`
- `OUTPUT_DIR/step-12-features.json`
- `OUTPUT_DIR/step-13-training.json`
- `OUTPUT_DIR/model.joblib` (best fitted model)
- `OUTPUT_DIR/candidate-*.joblib` (all trained candidates)
- `OUTPUT_DIR/step-14-evaluation.json`
- `OUTPUT_DIR/step-15-selection.json`
- `OUTPUT_DIR/step-16-report.md`
- `OUTPUT_DIR/code_audit.json` (Python file inventory + hashes per step)
- `OUTPUT_DIR/leakage_audit.json` (explicit leakage diagnostics and pass/fail decision)

## Progress Schema

```json
{
  "run_id": "20260404T000000Z",
  "csv_path": "data/file.csv",
  "target_column": "appliances",
  "status": "running",
  "current_step": "13-model-training",
  "completed_steps": [
    "10-csv-read-cleansing",
    "11-data-exploration",
    "12-feature-extraction"
  ],
  "errors": []
}
```

## Model Artifact Portability Rules

- `model.joblib` must be loadable via `joblib.load(...)` in a fresh Python process.
- The loaded object must expose `.predict(X)` directly (not wrapped in a plain dict).
- Do not pickle classes defined under `__main__`; use importable sklearn/sklearn-compatible estimators only.
- `holdout.npz` must contain `X_test` and `y_test` arrays reusable by step 14 without access to step 13's script.
