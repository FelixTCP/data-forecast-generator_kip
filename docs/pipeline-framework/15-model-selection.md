# #15 Context Engineering: Model Selection

## Objective

Choose the production candidate with transparent criteria. Do not blindly pick the highest R² when all candidates are poor.

## Pre-Selection Filter

Before scoring, remove any candidate with R² < 0 from consideration. A model worse than a mean-predictor has negative predictive value. If all candidates have R² < 0, the selection step must **not** name a winner — it must halt with a clear message: "All candidates are below mean-baseline. Revisit feature engineering (step 12) or expand model classes (step 14 expansion)."

If `step-14-evaluation.json` sets `quality_assessment = "leakage_suspected"`, selection must hard-stop immediately and must not pick any winner.

## Selection Rule (MVP)

Apply weighted score only to candidates with R² ≥ 0:

- 50% normalized `r2`
- 25% inverse normalized `rmse`
- 15% inverse normalized `mae`
- 10% stability bonus (`1 - cv_std_r2`)

Normalize each metric across the competing candidates (min-max). For RMSE and MAE, invert after normalization (1 - normalized value) so lower is better.

## Output Fields

- `selected_model` — name of winning candidate
- `weighted_score` — the computed score for the winner
- `rationale` — at least 2 sentences: why this model scored highest and what its key tradeoffs are
- `full_ranking` — table of all candidates with their scores (include R²<0 candidates as `ineligible`)
- `quality_flag` — one of: `acceptable` (R² ≥ 0.50), `marginal` (R² in [0.25, 0.50)), `subpar` (best R² < 0.25 but ≥ 0), `no_viable_candidate` (all R² < 0)
- `quality_flag` — one of: `acceptable` (R² ≥ 0.50), `marginal` (R² in [0.25, 0.50)), `subpar` (best R² < 0.25 but ≥ 0), `no_viable_candidate` (all R² < 0), `leakage_suspected` (evaluation invalidated)

## Guardrails

- Tie-breaker: lower complexity model first (Ridge < ElasticNet < HistGBM < RF < GBM < SVR < XGBoost).
- Always emit full ranking table for traceability — including models marked ineligible.
- If `quality_flag` is `subpar` or `no_viable_candidate`, step 16 must prominently display this.
- If `quality_flag` is `leakage_suspected`, output only diagnostics and terminate with non-zero exit.

## Copilot Prompt Snippet

```markdown
Implement `select_best_model(candidate_reports: list[dict]) -> dict`.
Abort if evaluation indicates leakage_suspected.
Filter out R²<0 candidates as ineligible before scoring.
Apply weighted score to remaining candidates.
Return selected_model, weighted_score, rationale (≥2 sentences), full_ranking, quality_flag.
Halt with a clear error message if no eligible candidates remain.
```
