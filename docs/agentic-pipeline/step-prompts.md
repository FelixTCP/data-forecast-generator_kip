# Step Prompts (Runtime)

These prompts are runtime wrappers. The canonical step requirements live in
`docs/pipeline-framework/01-...16-...md`.

Use this execution protocol for every step:
1. **Reason** — read the corresponding step spec, confirm inputs/outputs/failure modes
2. **Code** — implement `CODE_DIR/step_NN_name.py` as standalone CLI script
3. **Validate** — run the step and enforce gate checks before continuing

---

## 00-pre_exploration

```markdown
PHASE 1 — REASON:
- Read docs/pipeline-framework/00-pre_exploration.md.
- Confirm required outputs for step 01:
  - OUTPUT_DIR/step-00_profiler.json
  - OUTPUT_DIR/step-00_data_profile_report.md

PHASE 2 — CODE:
- Write CODE_DIR/step_00_pre_exploration.py.
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
- Write CODE_DIR/step_01_cleanse.py.
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
- Write CODE_DIR/step_11_exploration.py.
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
- Write CODE_DIR/step_12_features.py.
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
- Write CODE_DIR/step_13_training.py.
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
- Write CODE_DIR/step_14_evaluation.py.
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
- Write CODE_DIR/step_15_selection.py.
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
- Write CODE_DIR/step_16_report.py.
- CLI args: --output-dir, --run-id.
- Implement all behavior from step 16 spec exactly.

PHASE 3 — VALIDATE:
- step-16-report.md exists and includes all required headings.
- progress.json status is completed.
```
