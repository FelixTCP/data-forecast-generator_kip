# #14 Context Engineering: Model Evaluation

## Objective

Evaluate trained candidates consistently and produce comparison-ready metrics.

## Required Metrics

- `r2`
- `rmse`
- `mae`
- optional: `mape` (only when target supports it)

## Copilot Prompt Snippet

```markdown
Implement `evaluate_model(model, X_test, y_test) -> dict` and `evaluate_candidates(...) -> list[dict]`.
Include residual summary and metric sanity checks.
```

## Guardrails

- Same holdout split policy as training.
- Explicit warning when metric assumptions are violated.
