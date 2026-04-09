# Agentic Pipeline Contracts

This repo is instruction-first. Runtime code should be generated/adapted by Copilot CLI per run.

## Global Runtime Inputs
- `CSV_PATH`: absolute or repo-relative input CSV
- `TARGET_COLUMN`: regression target
- `OUTPUT_DIR`: run artifact directory
- `RUN_ID`: unique run id (timestamp)
- `SPLIT_MODE`: `auto|random|time_series`
- `CODE_DIR`: hashed code workspace for generated python code
- `COMMAND_HASH`: stable fingerprint for code workspace reuse
- `CONTINUE_MODE`: `true|false` for reusing previous workspace

## Step Contract
Each step must:
1. Read previous step outputs from `OUTPUT_DIR`.
2. Write machine-readable JSON output and update `OUTPUT_DIR/progress.json`.
3. Be idempotent for same `RUN_ID` when possible.
4. Write generated runnable python files under `CODE_DIR` (not scattered in repo).

## Required Files per run
- `OUTPUT_DIR/progress.json`
- `OUTPUT_DIR/step-10-cleanse.json`
- `OUTPUT_DIR/step-11-exploration.json`
- `OUTPUT_DIR/step-12-features.json`
- `OUTPUT_DIR/step-13-training.json`
- `OUTPUT_DIR/model.joblib` (serialized selected/best model)
- `OUTPUT_DIR/step-14-evaluation.json`
- `OUTPUT_DIR/step-15-selection.json`
- `OUTPUT_DIR/step-16-report.md`
- `OUTPUT_DIR/code_audit.json` (python file inventory + hashes per step)

## Progress Schema
```json
{
  "run_id": "20260404T000000Z",
  "csv_path": "data/file.csv",
  "target_column": "appliances",
  "status": "running",
  "current_step": "13-model-training",
  "completed_steps": ["10-csv-read-cleansing"],
  "errors": []
}
```

## Model Artifact Portability Rules
- `model.joblib` must be loadable via `joblib.load(...)` in a fresh Python process.
- The loaded object (or wrapped object under common keys) must provide `.predict(...)`.
- Avoid pickling classes defined under `__main__`; prefer importable sklearn/sklearn-compatible estimators.

## Code Audit & Continue Rules
- Generated step code must be persisted under hashed `CODE_DIR`.
- Each step should be auditable via file hashes (recorded in `code_audit.json`).
- With `--continue`, the runner reuses prior `CODE_DIR` contents for same `COMMAND_HASH` as starting point.
