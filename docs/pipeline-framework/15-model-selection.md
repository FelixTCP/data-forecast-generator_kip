# #15 Context Engineering: Model Selection

## Objective

Choose the production candidate with transparent criteria. Do not blindly pick the highest R² when all candidates are poor.

Step 15 assumes that `step-14-evaluation.json` already exists and contains the full model evaluation result set. It does not re-evaluate models and does not re-split data.

## Baseline Policy

Step 15 must explicitly document the baselines used to interpret the model scores:

- Mean baseline: always document that R² = 0 represents predicting the mean target value.
- Naive lag baseline: when Step 14 provides `naive_baseline_r2`, `naive_baseline_rmse`, and `naive_baseline_mae`, copy those values into the selection output and compare each candidate against them in `candidate_analysis`.

## Pre-Selection Filter

Before scoring, remove any candidate with R² < 0 from consideration. A model worse than a mean-predictor has negative predictive value.

If all candidates have R² < 0, the selection step must **not** name a winner. It must set `selected_model = null`, `weighted_score = null`, and `quality_flag = "no_viable_candidate"`, while still writing `step-15-selection.json`, `step-15-model-selection-report.md`, and `step-15-model-selection-metrics.png` when candidates exist. The rationale must include the message: "All candidates are below mean-baseline. Revisit feature engineering (step 12) or expand model classes (step 14 expansion)."

If `step-14-evaluation.json` sets `quality_assessment = "leakage_suspected"`, selection must hard-stop immediately and must not pick any winner. It must output only diagnostics or error status for Step 15 and terminate with a non-zero exit code.

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
- `baselines` — mean baseline and, when present, naive lag baseline metrics from Step 14
- `candidate_analysis` — short technical explanation for why each candidate performed well or poorly, covering performance relative to the mean baseline, performance relative to the naive lag baseline when available, error magnitude, CV stability, and a likely technical reason for the result
- `full_ranking` — table of all candidates with their scores (include R²<0 candidates as `ineligible`)
- `quality_flag` — one of: `acceptable` (R² ≥ 0.50), `marginal` (R² in [0.25, 0.50)), `subpar` (best R² < 0.25 but ≥ 0), `subpar_after_expansion` (Step 14 expansion still stayed below threshold but produced at least one non-negative R² candidate), `no_viable_candidate` (all R² < 0), `leakage_suspected` (evaluation invalidated)
- `artifacts.selection_report_md` — path to the technical Markdown report
- `artifacts.selection_metrics_png` — path to the model comparison PNG plot

## Guardrails

- Tie-breaker: lower complexity model first (Ridge < ElasticNet < HistGBM < RF < GBM < SVR < XGBoost).
- Always emit full ranking table for traceability — including models marked ineligible.
- Always emit a technical Markdown report with baseline summary, ranking table, model analysis, and rationale.
- Always emit at least one PNG plot comparing candidate metrics when candidates exist.
- If `quality_flag` is `subpar`, `subpar_after_expansion`, or `no_viable_candidate`, step 16 must prominently display this.
- If `quality_flag` is `leakage_suspected`, output only diagnostics and terminate with non-zero exit.

## Copilot Prompt Snippet

```markdown
Implement `select_best_model(candidate_reports: list[dict]) -> dict`.
Abort if evaluation indicates leakage_suspected.
Filter out R²<0 candidates as ineligible before scoring.
Apply weighted score to remaining candidates.
Document mean and naive lag baselines from Step 14.
Explain why each candidate works well or poorly based on mean-baseline status, naive-baseline delta when available, error magnitude, CV stability, and likely technical cause.
Return selected_model, weighted_score, rationale (≥2 sentences), full_ranking, quality_flag, baselines, candidate_analysis, and artifact paths.
Write step-15-model-selection-report.md with Markdown tables.
Write step-15-model-selection-metrics.png with matplotlib.
If no eligible candidates remain, set selected_model=null and quality_flag=no_viable_candidate, still write JSON/report/PNG, and include the below-mean-baseline message in the rationale.
```