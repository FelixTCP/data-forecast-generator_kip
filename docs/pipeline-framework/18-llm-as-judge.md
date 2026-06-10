# Step 18 — LLM-as-a-Judge

## Purpose

Step 18 creates the final Judge assessment for a completed forecast run.

The Judge translates technical forecast artifacts into a concise, honest MVP-readiness and use-case assessment. It is not a model scoreboard and not a deployment approval.

The assessment must work for any dataset with a forecast target. It must explain relevant metrics in relation to the selected target, the evaluation context, and the plausible use case.

The Judge helps decide whether the forecast result is suitable for MVP discussion, stakeholder review, planning, monitoring, or further validation.

---

## Execution Model

Step 18 is executed after Step 17.

It must write exactly:

```text
output/<RUN_ID>/step-18-judge.json
output/<RUN_ID>/step-18-judge.md
```

Do not create additional scripts, sub-agents, folders, frameworks, renderer modules, or extra output files.

The frontend handles layout and rendering.

The Judge output should support:

```text
Final Recommendation
Use Case
Forecastability
Use Case Potential
Business Potential
Business Value Evidence
Metric Meaning for This Use Case
Business Potential and Evidence
Sources
```

---

## Input Sources

Use only artifacts from the current `RUN_ID`.

Use these files when available:

```text
output/<RUN_ID>/progress.json
output/<RUN_ID>/step-11-exploration.json
output/<RUN_ID>/step-12-features.json
output/<RUN_ID>/step-14-evaluation.json
output/<RUN_ID>/step-15-selection.json
output/<RUN_ID>/step-16-report.md
output/<RUN_ID>/step-17-audit.json
```

Use artifacts selectively and only when relevant.

Use `step-16-report.md` as context, but cross-check customer-facing statements against evaluation, selection, and audit artifacts.

If artifacts conflict, prefer evaluation, selection, and audit evidence over narrative report claims.

Do not mix runs. Do not invent missing information.

If relevant information is missing, inconsistent, or unclear, continue with available evidence and choose a conservative assessment.

---

## Core Logic

The Judge separates four dimensions:

| Dimension | Meaning |
| --- | --- |
| Forecastability | Whether the selected target appears meaningfully predictable from available data and evaluation results. |
| Use Case Potential | Whether the forecast can plausibly support planning, monitoring, forecasting, regression, or risk-screening. |
| Business Potential | Whether the result is worth discussing as a possible MVP direction. |
| Business Value Evidence | Whether actual business impact, ROI, operational value, production usefulness, or stakeholder validation is evidenced. |

Rules:

- A strong model can support business potential.
- A strong model does not prove business value.
- A strong R² does not prove ROI.
- Low RMSE or MAE does not prove production readiness.
- Forecasting performance does not prove optimization potential.
- Predictive association does not prove causality.
- If business KPIs, cost data, ROI data, operational tolerance, domain validation, external validation, monitoring evidence, or production validation are missing, `business_value_evidence` must be `low` or `unclear`.

---

## Status Values

Use only these machine-readable status values:

```text
mvp_discussion_ready
mvp_discussion_ready_with_caveats
needs_validation_before_mvp_discussion
not_mvp_ready
no_reliable_forecast_use_case_supported
```

Use only these customer-facing labels:

```text
MVP discussion ready
MVP discussion ready with caveats
Needs validation before MVP discussion
Not MVP-ready
No reliable forecast use case supported
```

Status rules:

- Use `mvp_discussion_ready` only when forecast quality, audit result, use-case plausibility, and business evidence are all strong.
- Use `mvp_discussion_ready_with_caveats` when the forecast result is promising but business value, operational tolerance, external validation, or production readiness is not evidenced.
- Use `needs_validation_before_mvp_discussion` when the result may become useful but important evidence is missing, unclear, unstable, or insufficiently interpreted.
- Use `not_mvp_ready` when the model result is too weak, unstable, poorly supported, or unsafe to present as an MVP result.
- Use `no_reliable_forecast_use_case_supported` when no trustworthy forecast use case can be derived.
- If `business_value_evidence` is `low` or `unclear`, final status must be at most `mvp_discussion_ready_with_caveats`.

---

## Forbidden Claims

Do not use these claims or equivalent wording:

```text
approved
approved for deployment
production-ready
ready for rollout
guaranteed accuracy
guaranteed business impact
proven ROI
proven cost savings
excellent performance
```

Do not claim production suitability unless production validation, monitoring, drift handling, deployment requirements, and operational safeguards are documented.

Do not claim business impact unless business KPIs, cost data, operational evidence, or stakeholder validation are documented.

Do not describe RMSE, MAE, or any error metric as a guaranteed prediction bound.

Do not claim that a metric proves future performance outside the evaluation setup.

---

## Metric Interpretation

The Judge must explain every metric it mentions.

Each metric explanation must include:

```text
actual value
plain-language meaning
relation to the target or use case
```

Use the target unit if available. If unknown, use `target units`.

Use `null` in JSON for missing metric values.

Do not invent units or metric values.

### R²

If available, explain how much observed target variance the selected model explains on the evaluation split.

State whether the value supports a strong, moderate, weak, negative, or unclear predictive signal.

If R² is negative, explain that the model performs worse than predicting the average target value.

### RMSE and MAE

If available, explain RMSE and MAE as error measures on the evaluation split.

State actual value and unit.

Clarify that they are not guaranteed maximum errors.

If target scale is available, interpret whether the error appears small, moderate, or large relative to observed target range or variability.

### Baseline and Target Scale

If baseline metrics are available, compare the selected model against them.

If baseline metrics are missing, state that baseline superiority cannot be claimed.

Use target mean, standard deviation, minimum, maximum, range, or distribution information when available to interpret RMSE and MAE.

If target scale is missing, state that practical error size cannot be fully judged without target scale or decision thresholds.

### Features and Context

Use feature evidence only when available.

Temporal, lag, seasonal, calendar, or contextual features may support forecasting, planning, monitoring, or risk-screening.

Do not claim optimization unless controllable features, interventions, or operational levers are documented.

Do not claim causality unless causal evidence is documented.

---

## Use Case Inference

Infer the most plausible use case from:

```text
target column
time information
feature evidence
selected model
evaluation results
selection rationale
report content
audit result
```

Use only these use-case types:

```text
forecasting
planning
monitoring
regression_analysis
anomaly_or_risk_screening
unclear
```

Do not invent a city, industry, customer goal, business process, ROI story, production workflow, climate strategy, or operational intervention unless documented.

The use-case title must be concise and target-related.

The use-case description must contain two to four sentences.

The decision context must contain one to two sentences and explain what kind of decision the forecast could plausibly support.

Evidence strength must be `high`, `medium`, `low`, or `unclear`.

---

## Ratings

Use only:

```text
high
medium
low
unclear
```

Each rating must include:

```text
rating
headline
explanation
```

Apply ratings as follows:

- `forecastability`: rate how meaningfully the target can be predicted from current evidence.
- `use_case_potential`: rate how clearly the artifacts support a stakeholder-facing forecast use case.
- `business_potential`: rate whether the result is worth discussing as an MVP direction.
- `business_value_evidence`: rate whether business value is already evidenced.

Business potential may be higher than business value evidence, but the explanation must clearly separate potential from proof.

Rate `business_value_evidence` as `high` only when artifacts document business KPIs, cost evidence, ROI evidence, operational thresholds, domain validation, production-relevant evidence, or stakeholder validation.

---

## Required JSON Output

Write:

```text
output/<RUN_ID>/step-18-judge.json
```

The JSON must include these top-level fields:

```text
run_id
status
status_label
status_reason
final_recommendation
use_case
ratings
metric_meaning
business_potential_and_evidence
risks_and_caveats
sources
```

Required structure:

```json
{
  "run_id": "...",
  "status": "...",
  "status_label": "...",
  "status_reason": "...",
  "final_recommendation": {
    "label": "...",
    "summary": "...",
    "strongest_supporting_reason": "...",
    "main_caveat": "..."
  },
  "use_case": {
    "type": "forecasting | planning | monitoring | regression_analysis | anomaly_or_risk_screening | unclear",
    "title": "...",
    "description": "...",
    "decision_context": "...",
    "evidence_strength": "high | medium | low | unclear"
  },
  "ratings": {
    "forecastability": {
      "rating": "high | medium | low | unclear",
      "headline": "...",
      "explanation": "..."
    },
    "use_case_potential": {
      "rating": "high | medium | low | unclear",
      "headline": "...",
      "explanation": "..."
    },
    "business_potential": {
      "rating": "high | medium | low | unclear",
      "headline": "...",
      "explanation": "..."
    },
    "business_value_evidence": {
      "rating": "high | medium | low | unclear",
      "headline": "...",
      "explanation": "..."
    }
  },
  "metric_meaning": {
    "r2": {
      "value": null,
      "meaning": "..."
    },
    "rmse": {
      "value": null,
      "unit": "target units",
      "meaning": "..."
    },
    "mae": {
      "value": null,
      "unit": "target units",
      "meaning": "..."
    },
    "baseline": {
      "available": false,
      "meaning": "..."
    },
    "target_scale": {
      "available": false,
      "meaning": "..."
    }
  },
  "business_potential_and_evidence": {
    "supported_discussion_points": ["..."],
    "evidence_limits": ["..."]
  },
  "risks_and_caveats": ["..."],
  "sources": ["..."]
}
```

---

## Required Markdown Output

Write:

```text
output/<RUN_ID>/step-18-judge.md
```

Use exactly these major sections:

```markdown
# Step 18 — LLM-as-a-Judge

## Final Recommendation

## Use Case

## Assessment Scores

## Metric Meaning for This Use Case

## Business Potential and Evidence

## Sources
```

Do not add extra major sections.

Do not use third-level headings.

Keep the Markdown concise and decision-oriented.

Section requirements:

- `Final Recommendation`: status label, short summary, strongest supporting reason, main caveat.
- `Use Case`: use-case title, type, short description, decision context, evidence strength.
- `Assessment Scores`: exactly Forecastability, Use Case Potential, Business Potential, Business Value Evidence.
- `Metric Meaning for This Use Case`: only available and relevant metrics; include actual value and practical meaning.
- `Business Potential and Evidence`: two concise lists: `Supported discussion points` and `Evidence limits`.
- `Sources`: only artifacts actually used, listed as filenames.

---

## Final Quality Check

Before writing outputs, verify:

```text
The report is concise and decision-oriented.
The final recommendation is conservative and evidence-based.
The use case is derived from current run artifacts.
Metrics are explained with actual values when available.
Metric explanations describe meaning, not only values.
RMSE and MAE are not described as guaranteed bounds.
Business potential is separated from business value evidence.
Missing evidence is stated neutrally and only when relevant.
No ROI, production readiness, causality, optimization, or guaranteed accuracy is invented.
Sources belong only to the current run.
The Markdown uses only the required six major sections.
The JSON contains all required frontend-friendly fields.
```
