# Step Prompts (Runtime)

The orchestrator injects variables into these prompts.

## 10-csv-read-cleansing
```markdown
Run step 10 (CSV Read/Cleansing).
Input CSV: {{CSV_PATH}}
Output file: {{OUTPUT_DIR}}/step-10-cleanse.json
Also update {{OUTPUT_DIR}}/progress.json.
Write generated code under {{CODE_DIR}}.
Use polars only.
```

## 11-data-exploration
```markdown
Run step 11 (Data Exploration).
Use outputs from step 10.
Write {{OUTPUT_DIR}}/step-11-exploration.json and update progress.
Write generated code under {{CODE_DIR}}.
```

## 12-feature-extraction
```markdown
Run step 12 (Feature Extraction).
Use target column: {{TARGET_COLUMN}}
Use split mode hint: {{SPLIT_MODE}}
Write {{OUTPUT_DIR}}/step-12-features.json and update progress.
Write generated code under {{CODE_DIR}}.
```

## 13-model-training
```markdown
Run step 13 (Model Training) with deep context engineering.
Use split mode: {{SPLIT_MODE}}.
If date/time exists and split mode is auto, use chronological split.
Write {{OUTPUT_DIR}}/step-13-training.json and update progress.
Include per-model metrics and chosen hyperparameters.
Persist the selected/best fitted model as {{OUTPUT_DIR}}/model.joblib using joblib.
`model.joblib` must contain a fitted estimator or pipeline object with a `.predict(...)` method (not metadata-only dict).
Do NOT serialize custom classes defined in notebook/script `__main__`; use importable sklearn (or sklearn-compatible) classes to avoid pickle load errors.
Write generated code under {{CODE_DIR}}.
```

## 14-model-evaluation
```markdown
Run step 14 (Model Evaluation).
Write {{OUTPUT_DIR}}/step-14-evaluation.json and update progress.
Write generated code under {{CODE_DIR}}.
```

## 15-model-selection
```markdown
Run step 15 (Model Selection).
Select final model with explicit rationale.
Write {{OUTPUT_DIR}}/step-15-selection.json and update progress.
Write generated code under {{CODE_DIR}}.
```

## 16-result-presentation
```markdown
Run step 16 (Result Presentation).
Write a human summary to {{OUTPUT_DIR}}/step-16-report.md and update progress.
Set progress status to completed.
Write generated code under {{CODE_DIR}}.
```
