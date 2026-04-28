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
- `CONTINUE_MODE`: `true|false` (default: `false`)

## Code Generation Contract

**Every step script must be written from scratch for every run. Copying or recycling scripts from previous runs is strictly forbidden.**

- The agent MUST write each `step_NN_name.py` using `create_file` or equivalent file-creation tooling — never by copying from a prior run directory.
- Using shell commands such as `Copy-Item`, `cp`, `shutil.copy`, or equivalent to bring scripts from `output/<OLD_RUN_ID>/code/` into a new run is **prohibited**.
- This rule applies to ALL step scripts including `step_17_audit.py` and `orchestrator.py`.
- If the agent cannot generate a script (e.g., missing spec), the pipeline must halt and report the gap — it must not fall back to copying.
- The code-audit SHA256 hashes in `code_audit.json` must reflect freshly written files; recycled files will produce duplicate hashes across runs and can be detected.

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
| 17 — Critical Self-Audit  | `CODE_DIR/step_17_audit.py`       | `step-17-audit.json`       |
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

A step is considered complete (and may be skipped) when ALL are true:

1. The orchestrator is invoked with `--resume` flag (or `CONTINUE_MODE=true`).
2. The step output file exists.
3. The output is valid JSON with the expected `"step"` value.

**Critical:** If `--resume` is NOT set, ALL steps must be re-executed from scratch, even if prior step outputs exist in `OUTPUT_DIR`. Old code and artifacts from previous runs must never be reused.

If any of the three checks above fail, the step must be re-run (re-execution is always safe and produces identical outputs).

## Remediation Loop Contract

**The pipeline is not considered complete until the Self-Audit (Step 17) passes or `MAX_REMEDIATION_ITERATIONS = 3` remediation rounds have been completed.**

### Procedure

1. After all 9 steps complete, the orchestrator reads `step-17-audit.json`.
2. `overall_audit_result == "pass"` → Pipeline is immediately finalized.
3. `overall_audit_result == "fail"`:
   - Auto-executable `remediation_actions` are collected (see `17-critical-self-audit.md`).
   - `OUTPUT_DIR/remediation_config.json` is updated with new excluded features or parameter flags.
   - Output files of all steps from the earliest affected step through Step 17 are deleted.
   - These steps are re-executed in order — Step 12 receives the `--exclude-features` parameter.
   - `step-17-audit.json` is re-read. Loop runs for at most 3 iterations.
4. At the end, `progress.json` receives the `final_audit_result` field (`"pass"` or `"fail"`).

### Not auto-executable

`split_by_grouping_column` requires per-group training and is only logged — it does not trigger a restart.

### State File: `remediation_config.json`

```json
{
  "iteration": 1,
  "applied_actions": ["remove_monotonic_index_features"],
  "exclude_features": ["trend_elapsed_days"],
  "force_expansion_models": false
}
```

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
- `OUTPUT_DIR/step-17-audit.json` (objective audit results and optional remediation actions)
- `OUTPUT_DIR/code_audit.json` (Python file inventory + hashes per step)
- `OUTPUT_DIR/leakage_audit.json` (explicit leakage diagnostics and pass/fail decision)

## Progress Schema

```json
{
  "run_id": "20260410T120000Z",
  "csv_path": "data/file.csv",
  "target_column": "appliances",
  "status": "running",
  "current_step": "13-model-training",
  "current_model": "ridge",
  "completed_models": ["ridge", "random_forest"],
  "model_progress": 0.5,
  "completed_steps": ["10-csv-read-cleansing", "11-data-exploration", "12-feature-extraction"],
  "errors": []
}
```

## Abort / Cleanup Contract

If a run is aborted (manually interrupted, Ctrl+C, keyboard interrupt, or any unhandled exception):

1. The orchestrator or calling script **must delete the entire `OUTPUT_DIR` directory** associated with that `RUN_ID`.
2. Do not leave partial artifacts or partial progress state on disk.
3. Rationale: partial outputs are invalid and will cause confusion on retry; cleanup ensures a clean slate.

Implementation:
- The orchestrator should use a try/finally block to catch interrupts.
- On KeyboardInterrupt or SystemExit, delete `OUTPUT_DIR` before re-raising.
- Document this behavior clearly in orchestrator.py comments.

## Model Artifact Portability Rules

- `model.joblib` must be loadable via `joblib.load(...)` in a fresh Python process.
- The loaded object must expose `.predict(X)` directly (not wrapped in a plain dict).
- Do not pickle classes defined under `__main__`; use importable sklearn/sklearn-compatible estimators only.
- `holdout.npz` must contain `X_test` and `y_test` arrays reusable by step 14 without access to step 13's script.

## Critical Self-Audit Contract (Step 17)

### When Audit Runs

- **Automatically after** step 16 (Result Presentation) completes successfully.
- **Before** model is considered finalized or ready for deployment.
- Audit does **not** block pipeline status; it is informational and provides optional remediation guidance.

### Audit Inputs

Step 17 reads all prior step outputs:
- `step-01-cleanse.json` (data quality, time column)
- `step-11-exploration.json` (MI ranking, excluded features)
- `step-12-features.json` (final features, feature exclusions)
- `step-13-training.json` (CV scores)
- `step-14-evaluation.json` (holdout metrics, quality assessment)
- `step-15-selection.json` (selected model, rationale)
- `cleaned.parquet` (access to raw data for profile detection and distribution checks)
- `model.joblib` (fitted model)
- `holdout.npz` (test set)

### Audit Outputs

- **File:** `step-17-audit.json`
- **Fields:**
  - `detected_profile`: Data type classification (e.g., `stocks_multi_series`, `energy_multi_temporal`, `generic_regression`)
  - `checks`: Results of five objective checks (temporal consistency, multi-series detection, feature-target alignment, model performance baseline, distribution drift)
  - `critical_findings`: List of high-severity issues (if any)
  - `overall_audit_result`: `"pass"` or `"fail"`
  - `remediation_actions`: List of suggested re-trigger actions (if audit fails)

See `docs/self-audit/overview.md` for full schema.

### Audit Decision Logic

**Audit Passes** if:
- All five checks return `"pass"` or `"marginal"`.
- No high-severity critical findings.

**Audit Fails** if:
- ≥1 critical (high-severity) findings, OR
- ≥2 checks return `"fail"`.

**On Failure:** Remediation actions are logged. User may optionally re-trigger pipeline with suggested parameters (see `docs/self-audit/remediation.md`).

### Re-Trigger Protocol

If audit detects issues, remediation actions are logged in `step-17-audit.json`:

```json
{
  "remediation_actions": [
    {
      "action_id": "split_by_grouping_column",
      "description": "Split by symbol and train separate models",
      "affected_steps": ["12-feature-extraction", "13-model-training", "14-model-evaluation", "15-model-selection"],
      "suggested_parameters": {"group_column": "symbol"},
      "expected_improvement": "R² likely +0.3 to +0.5"
    }
  ]
}
```

To apply remediation:
1. Generate new `RUN_ID` (e.g., `20260424T130000Z`).
2. Extract `suggested_parameters` from remediation action.
3. Re-invoke orchestrator with new RUN_ID **and without `--resume`**:
   ```bash
   python orchestrator.py \
     --csv-path data/stocks.csv \
     --target-column close \
     --output-dir output/${NEW_RUN_ID} \
     --run-id ${NEW_RUN_ID} \
     --code-dir output/${NEW_RUN_ID}/code \
     --group-column symbol \
     --max-lag 20
   ```
4. Audit runs again after new pipeline completes.

**Important:** Always use a new `RUN_ID` for remediation runs; do NOT re-use the old one.

See `docs/self-audit/remediation.md` for full remediation protocol and action definitions.
