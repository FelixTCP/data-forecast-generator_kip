# Step Prompts (Runtime)

These prompts are runtime wrappers. The canonical step requirements live in
`docs/pipeline-framework/01-...16-...md`.

## Execution Protocol for Every Step

**CODE GENERATION RULE (applies to ALL steps, no exceptions):**
- Each step script must be written fresh using file-creation tooling.
- Never copy, reference, or paste from scripts in `output/<OLD_RUN_ID>/code/` or any other prior run directory.
- If you find yourself writing `Copy-Item`, `cp`, `shutil.copy`, or opening a previous run's `.py` file to paste from it — STOP. Write the script from scratch based on the spec in `docs/pipeline-framework/`.

**RESUME CHECK (before PHASE 1):**
- Read `OUTPUT_DIR/progress.json`.
- If orchestrator called with `--resume` AND the step is in `completed_steps` AND the step output JSON is valid: log "Step NN already complete, resuming from existing output" and skip to the next step.
- Otherwise, proceed to PHASE 1 below.

Use this protocol only if resuming from existing state:
1. **Reason** — read the corresponding step spec, confirm inputs/outputs/failure modes
2. **Code** — implement `CODE_DIR/step_NN_name.py` as standalone CLI script
3. **Validate** — run the step and enforce gate checks before continuing

**Abort / Cleanup:**
- If any step fails after 3 retry attempts, or if the run is manually interrupted: delete the entire `OUTPUT_DIR` directory.
- Do not leave partial artifacts or progress state on disk.

---

## 00-pre_exploration

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/00-pre_exploration.md.
- Confirm required outputs for step 01:
  - OUTPUT_DIR/step-00_profiler.json
  - OUTPUT_DIR/step-00_data_profile_report.md

PHASE 2 — CODE:
- Write CODE_DIR/step_00_pre_exploration.py FROM SCRATCH using create_file. Do not copy from previous runs.
- CLI args: --csv-path, --output-dir, --run-id.
- Implement all behavior from step 00 spec exactly.

PHASE 3 — VALIDATE:
- step-00_profiler.json exists.
- step-00_data_profile_report.md exists.
```

---

## 01-csv-read-cleansing

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/01-csv-read-cleansing.md.
- Confirm required outputs for step 11:
  - OUTPUT_DIR/cleaned.parquet
  - OUTPUT_DIR/step-01-cleanse.json with target normalization, null/dtype report, time column detection.

PHASE 2 — CODE:
- Write CODE_DIR/step_01_cleanse.py FROM SCRATCH using create_file. Do not copy from previous runs.
- CLI args: --csv-path, --target-column, --output-dir, --run-id.
- Implement all behavior from step 01 spec exactly.

PHASE 3 — VALIDATE:
- step-01-cleanse.json exists and reports row_count_after > 0.
- target_column_normalized present.
- cleaned.parquet exists.
```

---

## 11-data-exploration

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/11-data-exploration.md.
- Confirm required outputs for step 12:
  - OUTPUT_DIR/step-11-exploration.json including MI ranking, noise baseline,
    recommended_features, excluded_features, lag signals.

PHASE 2 — CODE:
- Write CODE_DIR/step_11_exploration.py FROM SCRATCH using create_file. Do not copy from previous runs.
- CLI args: --output-dir, --run-id.
- Implement all behavior from step 11 spec exactly.

PHASE 3 — VALIDATE:
- step-11-exploration.json exists.
- numeric_columns non-empty.
- mi_ranking non-empty.
- recommended_features non-empty.
```

---

## 12-feature-extraction

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/12-feature-extraction.md.
- Confirm required outputs for step 13:
  - OUTPUT_DIR/features.parquet
  - OUTPUT_DIR/step-12-features.json
  - OUTPUT_DIR/leakage_audit.json

PHASE 2 — CODE:
- Write CODE_DIR/step_12_features.py FROM SCRATCH using create_file. Do not copy from previous runs.
- CLI args: --target-column, --split-mode, --output-dir, --run-id.
- Implement all behavior from step 12 spec exactly.

PHASE 3 — VALIDATE:
- step-12-features.json exists with non-empty features.
- features.parquet exists.
- leakage_audit.json exists with pass status.
```

---

## 13-model-training

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/13-model-training.md.
- Confirm required outputs for step 14:
  - OUTPUT_DIR/model.joblib
  - OUTPUT_DIR/candidate-*.joblib
  - OUTPUT_DIR/holdout.npz
  - OUTPUT_DIR/step-13-training.json

PHASE 2 — CODE:
- Write CODE_DIR/step_13_training.py FROM SCRATCH using create_file. Do not copy from previous runs.
- CLI args: --split-mode, --output-dir, --run-id, [--target-column].
- Implement all behavior from step 13 spec exactly.

PHASE 3 — VALIDATE:
- step-13-training.json exists.
- model.joblib exists and load-test passes.
- holdout.npz exists.
```

---

## 14-model-evaluation

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/14-model-evaluation.md.
- Confirm required outputs for step 15:
  - OUTPUT_DIR/step-14-evaluation.json with candidate metrics,
    quality_assessment, baseline comparison, and any expansion diagnostics.

PHASE 2 — CODE:
- Write CODE_DIR/step_14_evaluation.py FROM SCRATCH using create_file. Do not copy from previous runs.
- CLI args: --output-dir, --run-id.
- Implement all behavior from step 14 spec exactly.

PHASE 3 — VALIDATE:
- step-14-evaluation.json exists.
- each candidate has finite r2/rmse/mae.
- quality_assessment present.
- if quality is subpar, expansion diagnostics/candidates are present.
```

---

## 15-model-selection

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/15-model-selection.md.
- Confirm required outputs for step 16:
  - OUTPUT_DIR/step-15-selection.json with selected_model,
    weighted_score, rationale, full_ranking, quality_flag.

PHASE 2 — CODE:
- Write CODE_DIR/step_15_selection.py FROM SCRATCH using create_file. Do not copy from previous runs.
- CLI args: --output-dir, --run-id.
- Implement all behavior from step 15 spec exactly.

PHASE 3 — VALIDATE:
- step-15-selection.json exists.
- quality_flag present.
- if viable candidate exists: selected_model and rationale present.
- full_ranking includes eligible and ineligible candidates.
```

---

## 16-result-presentation

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/16-result-presentation.md.
- Confirm required output:
  - OUTPUT_DIR/step-16-report.md with required 6 sections.

PHASE 2 — CODE:
- Write CODE_DIR/step_16_report.py FROM SCRATCH using create_file. Do not copy from previous runs.
- CLI args: --output-dir, --run-id.
- Implement all behavior from step 16 spec exactly.

PHASE 3 — VALIDATE:
- step-16-report.md exists and includes all required headings.
- progress.json status is completed.
```

---

## 17-critical-self-audit

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/17-critical-self-audit.md (canonical spec for all thresholds, schemas, and rules).
- Understand the five audit checks: temporal_consistency, multi_series_detection, feature_target_alignment, model_performance_baseline, data_distribution_drift.
- Confirm required outputs:
  - OUTPUT_DIR/step-17-audit.json with audit results and remediation actions.

PHASE 2 — CODE:
- Write CODE_DIR/step_17_audit.py FROM SCRATCH using create_file. Do not copy from previous runs.
- CLI args: --output-dir, --run-id.
- Inputs: step-01-cleanse.json, step-11-exploration.json, step-12-features.json,
  step-13-training.json, step-14-evaluation.json, cleaned.parquet, features.parquet, holdout.npz.
- Implement all five audit checks with EXACT thresholds from the spec:
    - temporal_consistency: gap > 10% OR stddev > 10% → fail/high
    - multi_series_detection: variance_ratio > 2.0 → fail/high; 1.5–2.0 → marginal/medium
    - feature_target_alignment: retention < 60% OR excluded > 70% → fail/high
    - model_performance_baseline: profile-dependent R² thresholds (see spec table)
    - data_distribution_drift: KS ≥ 0.25 → fail/high; KS=1.000 → monotonic index rule
- Status values: ONLY "pass", "marginal", "fail" — "warning" is FORBIDDEN.
- overall_audit_result = "fail" if ANY check.status=="fail" OR check.severity=="high".
- critical_findings must be non-empty when overall_audit_result=="fail".
- remediation_actions: list of objects with action_id, severity, description, affected_steps,
  suggested_parameters, expected_improvement.
- Use tqdm for the 5-check loop.
- NO external API calls; all checks use objective metrics only.

PHASE 3 — VALIDATE:
- step-17-audit.json exists and is valid JSON.
- Contains "step": "17-critical-self-audit".
- overall_audit_result is "pass" or "fail" (no other values).
- All five checks present in "checks" dict with status, findings, severity, confidence.
- No check has status="warning".
- If overall_audit_result=="fail": critical_findings is non-empty.
- remediation_actions is a list of objects (not flat strings).
```
