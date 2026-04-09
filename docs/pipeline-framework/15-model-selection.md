# #15 Context Engineering: Model Selection

## Objective

Choose the production candidate with transparent criteria (not only highest R²).

## Selection Rule (MVP)

Use weighted score:

- 50% normalized `r2`
- 25% inverse normalized `rmse`
- 15% inverse normalized `mae`
- 10% stability bonus (`1 - cv_std`)

## Copilot Prompt Snippet

```markdown
Implement `select_best_model(candidate_reports: list[dict], weights: dict | None = None) -> dict`.
Return selected model plus full ranking rationale.
```

## Guardrails

- Tie-breaker: lower complexity model first.
- Always emit full ranking table for traceability.
