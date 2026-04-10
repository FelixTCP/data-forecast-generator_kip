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
  numeric_columns, low_variance_columns, mi_ranking, noise_mi_baseline, redundant_columns,
  significant_lags, useful_lag_features, recommended_features, excluded_features,
  target_candidates, time_series_detected, time_column.
- Failure modes: all features below noise baseline (loosen threshold 50%, log warning, do not return empty);
  all-categorical dataset; tiny dataset (<50 rows, skip lag analysis);
  target column not in numeric columns (error).

PHASE 2 — CODE:
Write CODE_DIR/step_11_exploration.py as a standalone CLI script.
CLI args: --output-dir, --run-id.
Read step-10-cleanse.json and cleaned.parquet from OUTPUT_DIR.

Near-zero variance filter:
  Compute variance for every numeric column. Flag scaled variance < 1e-4 as low_variance.

Mutual information ranking:
  Compute mutual_info_regression(X_numeric, y, random_state=42) for all numeric features vs target.
  Generate 5 random-noise columns (np.random.default_rng(42).standard_normal) and compute their MI.
  Set noise_mi_baseline = average MI of the 5 noise columns.
  Flag any real feature with MI <= noise_mi_baseline as below_noise_baseline.
  Sort features by MI descending.

Redundancy filter:
  Compute Pearson correlation matrix. For any pair |r| >= 0.90, flag the lower-MI one as redundant.

Lag analysis (only if time_column detected):
  Compute autocorrelation of target at lags 1 to min(24, N//4).
  Flag lags where |autocorr| > 0.10 as significant_lags.
  Compute cross-correlation of each recommended feature with target at lags 0,1,2,3.
  Flag feature+lag combinations with |xcorr| > 0.15 as useful_lag_features.

Build recommended_features: exclude low_variance, below_noise_baseline, and redundant columns.
If recommended_features is empty: loosen noise baseline by 50%, retry; log warning.
Write step-11-exploration.json and update progress.json.

PHASE 3 — VALIDATE:
Check: step-11-exploration.json exists | numeric_columns non-empty |
mi_ranking non-empty | recommended_features non-empty | noise_mi_baseline is a finite float |
target_candidates has >= 1 entry | shape.rows > 10.
If recommended_features is empty after threshold loosening: halt and report — do not proceed.
```

---

## 12-feature-extraction

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/12-feature-extraction.md.
- Inputs: OUTPUT_DIR/cleaned.parquet, step-11-exploration.json.
  Critically read: recommended_features, significant_lags, useful_lag_features, excluded_features.
- Resolved TARGET_COLUMN from step-10-cleanse.json["target_column_normalized"].
- Outputs needed by step 13: features.parquet (feature columns + target), step-12-features.json with keys:
  features, features_excluded, created_features, rows_dropped_by_lag, leakage_flags,
  split_strategy, artifacts.
- Failure modes:
  1. recommended_features empty or missing → raise ValueError; do NOT fall back to all numeric columns.
  2. All lag features would be dropped → log "no lag features created", proceed with base features.
  3. Leakage flagged (|r| > 0.99 with target) → exclude the column, log it.
  4. Non-causal target rolling feature included (target at t used in predictor at t) → hard fail with leakage error.
  5. Algebraic leakage/reconstruction detected from target-derived features → hard fail with leakage error.
  6. Fewer than 2 features after all filters → raise ValueError with full audit trail.
- State which features you plan to include and exactly which lags, before writing any code.

PHASE 2 — CODE:
Write CODE_DIR/step_12_features.py as a standalone CLI script.
CLI args: --target-column, --split-mode, --output-dir, --run-id.
Start from recommended_features only — never re-include excluded_features.
Add time decomposition (year, month, day_of_week, hour) if time_column detected.
Add target lag features only at significant_lags. Add feature lag features only at useful_lag_features.
Add rolling mean and std for target only at top-2 significant lag window sizes (skip if no significant lags),
  but enforce causal construction: target.shift(1).rolling_mean/std(window).
  Never use target.rolling_* directly because it leaks the label at time t.
Drop NaN rows introduced by lags; log count.
Leakage guard: compute Pearson r between every feature and target on full dataset;
  exclude any with |r| > 0.99 and log it.
Algebraic leakage guard: run a reconstruction probe with target-derived features only.
  If R2 > 0.995 or RMSE ~ 0 on holdout, fail step and write leakage diagnostics.
Assert len(features) >= 2 before writing parquet.
Persist OUTPUT_DIR/leakage_audit.json.
Write features.parquet and step-12-features.json to OUTPUT_DIR.

PHASE 3 — VALIDATE:
Check: step-12-features.json exists | features list non-empty |
features_excluded key exists | split_strategy.resolved_mode in {random, time_series} |
features.parquet exists on disk |
no feature in features appears in step-11["excluded_features"] (spot-check 3 excluded features).
Check: leakage_audit.json exists and status == "pass".
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
- Inputs: OUTPUT_DIR/holdout.npz, all OUTPUT_DIR/candidate-*.joblib, step-13-training.json.
- Outputs needed by step 15 (and possibly expansion training): step-14-evaluation.json with
  per-candidate r2/rmse/mae, model_worse_than_mean_baseline flag, quality_assessment,
  target_stats, expansion_diagnosis (if subpar), expansion_candidates (if run).
- Quality thresholds: best R² >= 0.50 = acceptable; [0.25, 0.50) = marginal; < 0.25 = subpar.
- Suspicious-score protocol: if any candidate gets R² > 0.98, run leakage stress tests before accepting.
- Failure modes: holdout arrays shape mismatch vs model's expected features (use feature_names from holdout.npz);
  divide-by-zero in MAPE; expansion training fails (catch and log, do not abort evaluation).

PHASE 2 — CODE:
Write CODE_DIR/step_14_evaluation.py as a standalone CLI script.
CLI args: --output-dir, --run-id.
Load holdout.npz; load all candidate-*.joblib.
Use feature_names from holdout.npz to align columns.
Compute R², RMSE, MAE, residual_mean, residual_max_abs for each candidate.
Compute naive baseline metrics (y_hat_t = y_{t-1}) on the same holdout.
Flag any candidate with R² < 0 as model_worse_than_mean_baseline=true.
Compute target_stats (mean, std, min, max).
Determine quality_assessment based on best candidate R².

IF any candidate has R² > 0.98:
  Run leakage stress tests:
    - re-score after removing target-derived engineered features,
    - re-score after removing target rolling features,
    - run linear reconstruction probe on target-derived features only.
  If leakage is indicated, set quality_assessment="leakage_suspected",
    write leakage diagnostics, update progress status=error, and exit non-zero.

IF quality_assessment is "subpar" (best R² < 0.25):
  Diagnose: compare training CV R² (from step-13-training.json) vs holdout R².
    - CV R² ≈ 0 => feature set uninformative for these model families. Log recommendation to revisit step 11/12.
    - CV R² decent but holdout poor => overfitting, likely from lag features or too-small test window.
    - target std >> rmse baseline => models converging to mean predictor.
  Write expansion_diagnosis string.
  Train expansion candidates: ElasticNet, HistGradientBoostingRegressor, SVR(kernel='rbf').
    Use same Pipeline([imputer, scaler, estimator]) pattern. Use same holdout split.
  Evaluate expansion candidates under identical conditions.
  Update quality_assessment if expansion improves best R² above 0.25.
  If still subpar: set quality_assessment = "subpar_after_expansion".

Update progress.json (status=expansion_required if subpar, else running).
Write step-14-evaluation.json.

PHASE 3 — VALIDATE:
Check: step-14-evaluation.json exists | every candidate entry has finite r2, rmse, mae |
quality_assessment key present | target_stats key present |
IF quality_assessment is subpar: expansion_diagnosis is non-empty AND expansion_candidates present.
Do NOT proceed to step 15 if quality_assessment is "subpar" and no expansion was attempted.
Do NOT proceed to step 15 if quality_assessment is "leakage_suspected".
```

---

## 15-model-selection

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/15-model-selection.md.
- Inputs: step-14-evaluation.json (per-candidate metrics including quality_assessment).
- Outputs needed by step 16: step-15-selection.json with selected_model, weighted_score,
  rationale, full_ranking, quality_flag.
- Pre-selection filter: exclude any candidate with R² < 0 from scoring.
- Failure modes: all candidates have R² < 0 (halt, do not name a winner);
  only one eligible candidate (apply scoring, state trivial rationale, do NOT skip step).

PHASE 2 — CODE:
Write CODE_DIR/step_15_selection.py as a standalone CLI script.
CLI args: --output-dir, --run-id.
Filter ineligible candidates (R² < 0) from scoring; mark them in full_ranking as ineligible.
If no eligible candidates remain: write step-15-selection.json with quality_flag="no_viable_candidate",
  selected_model=null, and a clear message; update progress.json; exit 1.
Apply weighted score to eligible candidates:
  50% normalised R², 25% inv-normalised RMSE, 15% inv-normalised MAE, 10% (1 - cv_std_r2).
Tie-break: Ridge > ElasticNet > HistGBM > RF > GBM > SVR > XGBoost.
Write step-15-selection.json with quality_flag derived from best candidate R².

PHASE 3 — VALIDATE:
Check: step-15-selection.json exists | quality_flag present |
IF quality_flag != "no_viable_candidate": selected_model non-empty AND rationale non-empty |
full_ranking includes all candidates (eligible and ineligible).
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
