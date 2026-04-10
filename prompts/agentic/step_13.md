Run step 13 (Model Training).
Use split mode: {{SPLIT_MODE}}.
If date/time exists and split mode is auto, use chronological split.
Write {{OUTPUT_DIR}}/step-13-training.json and update progress.
Persist selected/best fitted model as {{OUTPUT_DIR}}/model.joblib.
`model.joblib` must be loadable and expose .predict(...).
No custom classes under __main__; use importable sklearn-compatible classes only.
Write generated code under {{CODE_DIR}}.
