# Critical Self-Audit System

## Purpose

The Critical Self-Audit is a **post-pipeline objective evaluation** that checks whether the regression model and feature engineering are appropriate for the given data type and problem characteristics. It is not an LLM-as-Judge system; it is a structured external audit using measurable indicators.

**Goal:** Detect when the pipeline has missed critical data characteristics (e.g., multi-series structure, temporal anomalies, distribution shifts) and recommend remediation actions, including possible re-triggering of earlier steps with modified parameters.

## When Audit Runs

- **Always after** step 16 (Result Presentation) completes successfully.
- **Before** final model export or deployment recommendations.
- If audit **fails**, suggest remediation and optionally re-trigger pipeline with new parameters.
- If audit **passes**, model is approved for production consideration.

## Audit Inputs

The audit reads outputs from all completed steps:
- `step-01-cleanse.json` (data quality, row counts, time column info)
- `step-11-exploration.json` (MI ranking, excluded features, lag signals)
- `step-12-features.json` (features used, exclusions, leakage audit)
- `step-13-training.json` (candidate models, CV scores)
- `step-14-evaluation.json` (metrics, quality assessment, target stats)
- `step-15-selection.json` (selected model, ranking, rationale)
- `cleaned.parquet` (access to actual data for distribution checks)
- `model.joblib` (the fitted model)
- `holdout.npz` (test set)

## Audit Output

**File:** `step-17-audit.json`

```json
{
  "step": "17-critical-self-audit",
  "run_id": "20260424T120000Z",
  "audit_timestamp": "2026-04-24T12:00:00Z",
  "data_profile": {
    "detected_profile": "multi_series_temporal",
    "confidence": 0.92,
    "characteristics": ["temporal_data", "multiple_entities", "duplicate_timestamps"]
  },
  "checks": {
    "temporal_consistency": {
      "status": "pass",
      "findings": ["no_gaps_detected", "regular_frequency_confirmed"]
    },
    "multi_series_detection": {
      "status": "fail",
      "findings": ["detected_7_distinct_entities", "model_trained_on_mixed_entities"],
      "severity": "high"
    },
    "feature_target_alignment": {
      "status": "pass",
      "findings": ["mi_ranking_stable", "no_redundant_features"]
    },
    "model_performance_baseline": {
      "status": "marginal",
      "findings": ["R²_below_profile_threshold", "holdout_worse_than_training"],
      "severity": "medium"
    },
    "data_distribution_drift": {
      "status": "pass",
      "findings": ["training_holdout_ks_stat_0.12_acceptable"]
    }
  },
  "overall_audit_result": "fail",
  "audit_confidence": 0.88,
  "remediation_actions": [
    {
      "action_id": "split_by_grouping_column",
      "severity": "high",
      "description": "Split multi-series data by grouping column and train separate models per entity.",
      "affected_steps": ["12-feature-extraction", "13-model-training", "14-model-evaluation", "15-model-selection"],
      "suggested_parameters": {
        "group_column": "<detected_grouping_column>",
        "train_separate_models": true
      },
      "expected_improvement": "R² likely to increase by 0.3–0.5 when models are series-specific"
    },
    {
      "action_id": "lag_optimization",
      "severity": "medium",
      "description": "Extend lag window from 5 to 20 days to capture longer-term dependencies.",
      "affected_steps": ["12-feature-extraction", "13-model-training"],
      "suggested_parameters": {
        "max_lag": 20,
        "rolling_windows": [5, 10, 20]
      },
      "expected_improvement": "Cross-validation R² may improve by 0.1–0.2"
    }
  ],
  "next_steps": [
    "Review multi-series detection finding with domain expert.",
    "If confirmed, re-trigger pipeline with group_column='<detected_column>' and train separate models per entity.",
    "Re-run audit after remediation."
  ],
  "notes": "High-confidence detection of multi-series structure. Current model is under-specified for this data type."
}
```

## Audit Decision Logic

### Pass
- All critical checks pass.
- No high-severity findings.
- Model confidence ≥ 0.50 for the detected data profile.

### Fail (Remediation Required)
- One or more critical checks fail.
- One or more high-severity findings.
- Suggested remediation actions must be reviewed and optionally applied.

## Data Profiles

The audit system recognizes and adapts to different data characteristics. See `data-type-profiles.md` for full definitions:

1. **Multi-Series Temporal Data** (`multi_series_temporal`)
   - Multiple entities (stocks, machines, locations, devices, etc.) over time
   - Duplicate timestamps; high variance between entities
   - Highest difficulty; requires per-entity modeling

2. **Daily-Cyclical Temporal Data** (`daily_cyclical_temporal`)
   - Single entity with repeating daily/weekly patterns
   - Strong autocorrelation at short lags (1–7)
   - Moderate difficulty; patterns learnable

3. **Longer-Period Temporal Data** (`longer_period_temporal`)
   - Single entity with multi-week/month/quarterly cycles
   - Weak short-lag autocorrelation; strong longer-lag acf
   - Moderate-Hard difficulty; trend handling required

4. **Generic Temporal Data** (`generic_temporal`)
   - Temporal column present but weak predictive signal
   - No strong autocorrelation pattern
   - Moderate difficulty; feature engineering critical

5. **Static Regression** (`static_regression`)
   - No temporal structure; cross-sectional data
   - Standard regression; no temporal leakage concerns
   - Easiest difficulty; depends on feature quality
   - See `data-type-profiles.md`

## Integration with Pipeline

See `docs/agentic-pipeline/` for:
- `contracts.md` — Audit step contract and re-trigger protocol
- `setup-prompt.md` — When and how to invoke audit
- `step-prompts.md` — Step 17 (Critical Self-Audit) prompt

See `docs/pipeline-framework/17-critical-self-audit.md` for:
- Full audit algorithm and implementation details
- Validation gates
