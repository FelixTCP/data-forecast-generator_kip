# Agentic Pipeline Contracts

This repo is instruction-first. Runtime code is generated/adapted by Copilot per run.

## Global Runtime Inputs
- `CSV_PATH`: absolute or repo-relative input CSV
- `TARGET_COLUMN`: regression target
- `OUTPUT_DIR`: run artifact directory
- `RUN_ID`: unique run id (timestamp)
- `SPLIT_MODE`: `auto|random|time_series`
- `CODE_DIR`: directory for generated step Python files (default: `OUTPUT_DIR/code/`)
- `CONTINUE_MODE`: `true|false` — when true, skip steps whose output JSON already exists

## Code Organisation Contract

**One Python file per pipeline step. No monolithic scripts.**

| Step | Script file | Output JSON |
|------|-------------|-------------|
| 10 — CSV Read & Cleansing | `CODE_DIR/step_10_cleanse.py` | `step-10-cleanse.json` |
| 11 — Data Exploration | `CODE_DIR/step_11_exploration.py` | `step-11-exploration.json` |
| 12 — Feature Extraction | `CODE_DIR/step_12_features.py` | `step-12-features.json` |
| 13 — Model Training | `CODE_DIR/step_13_training.py` | `step-13-training.json` |
| 14 — Model Evaluation | `CODE_DIR/step_14_evaluation.py` | `step-14-evaluation.json` |
| 15 — Model Selection | `CODE_DIR/step_15_selection.py` | `step-15-selection.json` |
| 16 — Result Presentation | `CODE_DIR/step_16_report.py` | `step-16-report.md` |
| Orchestrator | `CODE_DIR/orchestrator.py` | — |

The orchestrator calls each step script as a subprocess. It must support `--resume` to skip any step whose output JSON already exists with a valid `"step"` key.

## Step Script Contract

Each step script must:
1. Accept `--output-dir` and `--run-id` as CLI arguments minimum. Step 10 also requires `--csv-path`. Step 12 also requires `--target-column` and `--split-mode`.
2. Read its inputs from `OUTPUT_DIR` (prior step JSONs / parquet files) — not from hardcoded paths.
3. Write its output JSON to `OUTPUT_DIR` and update `OUTPUT_DIR/progress.json`.
4. Exit with code `0` on success, non-zero on any unhandled failure.
5. Be idempotent: running the same step twice with the same inputs produces identical outputs.
6. Be independently runnable without requiring the previous step's script to be in scope.

## Resume / Skip Contract

A step is considered **complete** and may be skipped if:
- Its output JSON exists at `OUTPUT_DIR/<step-output-filename>`
- The JSON is parseable and contains a `"step"` key matching the expected step name
- `CONTINUE_MODE` is `true` (or `--resume` is passed to orchestrator)

A step must be re-run if its output JSON is absent, corrupt, or the step is forced via `--force-step=<NN>`.

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

## Progress Schema
```json
{
  "run_id": "20260404T000000Z",
  "csv_path": "data/file.csv",
  "target_column": "appliances",
  "status": "running",
  "current_step": "13-model-training",
  "completed_steps": ["10-csv-read-cleansing", "11-data-exploration", "12-feature-extraction"],
  "errors": []
}
```

## Model Artifact Portability Rules
- `model.joblib` must be loadable via `joblib.load(...)` in a fresh Python process.
- The loaded object must expose `.predict(X)` directly (not wrapped in a plain dict).
- Do not pickle classes defined under `__main__`; use importable sklearn/sklearn-compatible estimators only.
- `holdout.npz` must contain `X_test` and `y_test` arrays reusable by step 14 without access to step 13's script.

## Code Audit Rules
- After each step script is written and executed successfully, record its path and SHA-256 hash in `code_audit.json`.
- Format: `{ "step_10_cleanse.py": "<hash>", "step_11_exploration.py": "<hash>", ... }`
