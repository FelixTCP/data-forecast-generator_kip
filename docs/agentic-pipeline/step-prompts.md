# Step Prompts (Runtime)

These prompts guide per-step execution. The orchestrator injects runtime variables.

Each step follows a three-phase protocol:
1. **Reason** — read the spec, list inputs/outputs, identify failure modes
2. **Code** — write `CODE_DIR/step_NN_name.py` as a standalone CLI script
3. **Validate** — run the script, check the gate, only proceed on pass

---

## 10-csv-read-cleansing

```markdown
PHASE 1 — REASON (complete before writing code):
- Read docs/pipeline-framework/10-csv-read-cleansing.md.
- Inputs: {{CSV_PATH}} (raw CSV file).
- Outputs needed by step 11: cleaned.parquet, step-10-cleanse.json with keys:
  row_count_after, target_column_normalized, null_rate, inferred_dtypes, time_column_detected, artifacts.
- Failure modes to handle: file not found; target column missing after normalization;
  numeric columns that loaded as strings (scientific notation, leading spaces); all-null columns.
- State your normalization approach and coercion strategy before writing code.

PHASE 2 — CODE:
Write CODE_DIR/step_10_cleanse.py as a standalone CLI script.
CLI args: --csv-path, --target-column, --output-dir, --run-id.
Use polars only. Try_parse_dates=True.
Normalize column names: strip → lowercase → replace spaces/hyphens with underscores.
Attempt numeric coercion for string columns (handle scientific notation, leading/trailing spaces).
Detect time column: dtype is Date/Datetime OR column name contains "date" or "time".
Write cleaned.parquet and step-10-cleanse.json to {{OUTPUT_DIR}}.
Update progress.json at start (status=running) and end (status=completed for this step).
Exit 0 on success, 1 on failure.

PHASE 3 — VALIDATE (run after execution):
Check: step-10-cleanse.json exists | row_count_after > 0 |
target_column_normalized present | cleaned.parquet exists on disk.
If any check fails: diagnose from the traceback, fix the script, re-run. Max 3 attempts.
```

---

## 11-data-exploration

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/11-data-exploration.md.
- Inputs: OUTPUT_DIR/cleaned.parquet, OUTPUT_DIR/step-10-cleanse.json (for time_column, target_column_normalized).
- Outputs needed by step 12: step-11-exploration.json with keys:
  numeric_columns, high_cardinality, cardinality, numeric_summary, correlation_preview,
  target_candidates, time_series_detected, time_column.
- Failure modes: all-categorical dataset (numeric_columns empty); tiny dataset (<10 rows);
  target column not in numeric columns.

PHASE 2 — CODE:
Write CODE_DIR/step_11_exploration.py as a standalone CLI script.
CLI args: --output-dir, --run-id.
Read step-10-cleanse.json and cleaned.parquet from {{OUTPUT_DIR}}.
Compute pairwise correlations for first 8 numeric columns only (performance guard).
Rank target candidates by (low null_rate, high std).
Write step-11-exploration.json to {{OUTPUT_DIR}} and update progress.json.

PHASE 3 — VALIDATE:
Check: step-11-exploration.json exists | numeric_columns non-empty |
target_candidates has ≥1 entry | shape.rows > 10.
If numeric_columns is empty: log a clear error — do not proceed with an empty feature set.
```

---

## 12-feature-extraction

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/12-feature-extraction.md.
- Inputs: OUTPUT_DIR/cleaned.parquet, step-11-exploration.json (time_column, numeric_columns).
- Resolved TARGET_COLUMN: {{TARGET_COLUMN}} (use normalized form from step-10-cleanse.json).
- Outputs needed by step 13: features.parquet (numeric feature columns + target), step-12-features.json with keys:
  features (list of feature column names), created_features, split_strategy, artifacts.
- Failure modes: target column missing from parquet; time features created but time_column is None;
  all features dropped after numeric filter (check for this explicitly before writing parquet).

PHASE 2 — CODE:
Write CODE_DIR/step_12_features.py as a standalone CLI script.
CLI args: --target-column, --split-mode, --output-dir, --run-id.
If time_column detected: add year, month, day-of-week, hour; add lag-1 and rolling-3-mean of target.
Only keep numeric columns in final feature set (exclude target from features list).
GUARD: assert len(features) > 0 before writing — raise ValueError with a clear message if empty.
Document every created feature with a brief reason string.
Write features.parquet and step-12-features.json to {{OUTPUT_DIR}}.

PHASE 3 — VALIDATE:
Check: step-12-features.json exists | features list non-empty |
split_strategy.resolved_mode in {random, time_series} | features.parquet exists on disk.
```

---

## 13-model-training

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/13-model-training.md.
- Inputs: OUTPUT_DIR/features.parquet, step-12-features.json (features list, split_strategy, target_column).
- Outputs needed by steps 14–15: model.joblib, holdout.npz, candidate-*.joblib, step-13-training.json
  with per-candidate: model_name, cv_mean_r2, cv_std_r2, fit_time_sec, hyperparameters.
- Critical failure modes to reason about before coding:
  1. Target column missing from features.parquet → read feature list from step-12-features.json explicitly.
  2. Test set is empty after split → this happened in prior run; add assertion on X_test.shape[0] > 0 before predict.
  3. String/object columns surviving to sklearn → force cast to float64; fail fast with column name in error.
  4. NaN in target → drop NaN target rows, log count; assert y_train.shape[0] >= 10.
  5. model.joblib pickling __main__ class → use only sklearn Pipeline with importable estimators.
- State explicitly: how you will split, which CV strategy, which candidates, and how you select best.

PHASE 2 — CODE:
Write CODE_DIR/step_13_training.py as a standalone CLI script.
CLI args: --split-mode, --output-dir, --run-id, and --target-column (read from step-12 JSON if not provided).
BEFORE training: print shapes of X_train, X_test, y_train, y_test — fail fast if any is zero.
Use TimeSeriesSplit(n_splits=5) if split_strategy.resolved_mode == "time_series", else KFold(n_splits=5).
Candidates: Ridge, RandomForestRegressor, GradientBoostingRegressor. Try XGBoost only with try/except ImportError.
For each candidate: wrap in Pipeline([imputer, scaler, estimator]) — do not define custom classes.
Save each fitted pipeline as candidate-<name>.joblib.
Save holdout arrays as holdout.npz (keys: X_test, y_test, feature_names).
Save best candidate (by cv_mean_r2) as model.joblib.

PHASE 3 — VALIDATE:
Check: step-13-training.json exists | model.joblib exists |
load test passes (python -c "import joblib; m=joblib.load('...'); print(type(m))") |
holdout.npz exists | at least one candidate has finite r2.
If load test raises: the model was likely defined under __main__ — refactor to use only sklearn Pipeline.
```

---

## 14-model-evaluation

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/14-model-evaluation.md.
- Inputs: OUTPUT_DIR/holdout.npz, OUTPUT_DIR/candidate-*.joblib, step-13-training.json.
- Outputs needed by step 15: step-14-evaluation.json with per-candidate r2, rmse, mae, residual_note.
- Failure modes: candidate joblib missing; holdout arrays shape mismatch vs candidate's expected features;
  divide-by-zero in MAPE if target contains zeros.

PHASE 2 — CODE:
Write CODE_DIR/step_14_evaluation.py as a standalone CLI script.
CLI args: --output-dir, --run-id.
Load holdout.npz; load all candidate-*.joblib from OUTPUT_DIR.
Use feature_names from holdout.npz to ensure column alignment.
Compute R², RMSE, MAE for each candidate. Add residual note: mean residual and max absolute error.
Skip MAPE if target has zeros; log the skip as a note.

PHASE 3 — VALIDATE:
Check: step-14-evaluation.json exists | every candidate entry has r2, rmse, mae as finite numbers.
```

---

## 15-model-selection

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/15-model-selection.md for the weighted scoring rule.
- Inputs: step-14-evaluation.json (per-candidate metrics).
- Outputs needed by step 16: step-15-selection.json with selected_model, rationale, full_ranking.
- Failure modes: only one candidate (still apply scoring, state trivial rationale);
  all candidates have near-identical scores (flag this in rationale).

PHASE 2 — CODE:
Write CODE_DIR/step_15_selection.py as a standalone CLI script.
CLI args: --output-dir, --run-id.
Apply weighted score: 50% normalised R², 25% inv-normalised RMSE, 15% inv-normalised MAE, 10% (1-cv_std).
Tie-break: prefer simpler model (Ridge > GBM > RF).
Write full ranking table and explicit rationale for the winner.

PHASE 3 — VALIDATE:
Check: step-15-selection.json exists | selected_model non-empty | rationale non-empty string.
```

---

## 16-result-presentation

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/16-result-presentation.md.
- Inputs: step-10 through step-15 JSONs, step-16-report.md (to write).
- The report must be readable by a non-technical stakeholder — avoid jargon, use a metrics table.
- Failure modes: missing input JSON for a section (load with fallback, note the gap in report).

PHASE 2 — CODE:
Write CODE_DIR/step_16_report.py as a standalone CLI script.
CLI args: --output-dir, --run-id.
Write step-16-report.md with exactly these 6 sections (use ## headings):
  1. Problem + selected target
  2. Data quality summary
  3. Candidate models + scores (markdown table with R², RMSE, MAE)
  4. Selected model rationale
  5. Risks and caveats
  6. Next iteration recommendations
Set progress.json status to "completed" and add this step to completed_steps.

PHASE 3 — VALIDATE:
Check: step-16-report.md exists and is ≥500 bytes |
progress.json has status="completed" |
report contains all 6 section headings.
```
