# Step 17 — Critical Self-Audit

## Purpose

The Critical Self-Audit is an **objective, post-pipeline evaluation** that detects whether the regression model and feature engineering are appropriate for the given data. 

**Output:** `step-17-audit.json` with audit results and remediation recommendations.

---

## Code Generator Instructions

> **This file is an LLM prompt.** Generate `step_17_audit.py` — a complete, immediately executable Python CLI script.

| Feld | Wert |
|---|---|
| **Dateiname** | `step_17_audit.py` |
| **Step-ID** | `17-critical-self-audit` |
| **CLI** | `python step_17_audit.py --output-dir <dir> --run-id <id>` |
| **Inputs** | steps 10–14 JSON, parquet files, model.joblib, holdout.npz |
| **Output** | `step-17-audit.json` with checks, findings, remediation |

---

## READ THIS FIRST

**These 3 master files contain all specifications:**

1. **`docs/self-audit/audit-rules.md`** ✅
   - Detailed metrics and thresholds for all 5 checks
   - Profile-dependent R² thresholds
   - KS thresholds: < 0.40 = pass, 0.40–0.80 = marginal, ≥ 0.80 = fail (relaxed for weak signal data)
   - Duplicate timestamps, variance ratio, monotone indices

2. **`docs/self-audit/data-type-profiles.md`** ✅
   - 5 profiles: multi_series_temporal, daily_cyclical_temporal, longer_period_temporal, generic_temporal, static_regression
   - Profile detection heuristics
   - Typical problems per profile

3. **`docs/self-audit/remediation.md`** ✅
   - 10+ concrete remediation actions
   - Auto-executable vs. manual actions
   - Re-trigger logic

**Implement the checks exactly as specified there. Deviations = errors.**

---

## Execution Protocol

### Phase 1: Data Profile Detection
1. Read `step-10-cleanse.json` and `cleaned.parquet`
2. Apply heuristics from `docs/self-audit/data-type-profiles.md`
3. Output: `data_profile` object with `detected_profile`, `confidence`, `characteristics`

### Phase 2: Run 5 Audit Checks (with tqdm)
For each check (in order):
1. **Temporal Consistency** — See `docs/self-audit/audit-rules.md` § Check 1
2. **Multi-Series Detection** — See § Check 2 (**READ: Duplicate timestamps first!**)
3. **Feature-Target Alignment** — See § Check 3
4. **Model Performance Baseline** — See § Check 4 (profile-dependent R² thresholds; NOTE: never return "fail" due to low R² alone — only marginal/pass)
5. **Data Distribution Drift** — See § Check 5 (KS statistics; NOTE: only return "fail" if KS > 0.95 or monotone features detected)

Each check outputs:
```json
{
  "status": "pass" | "marginal" | "fail",
  "findings": ["list of findings"],
  "severity": "low" | "medium" | "high",
  "confidence": 0.0
}
```

### Phase 3: Identify Critical Findings
- Critical finding triggered ONLY if: (target_variable_in_features == true) OR (timestamp_in_features == true) OR (monotone_features_detected == true) 
- **NO critical finding should be generated simply for low R² or moderate KS drift**
- Each must have: `check`, `status`, `severity`, `description`
- **Note:** Low R² is acceptable for weak-signal datasets. Marginal performance with clean features = PASS overall

### Phase 4: Map to Remediation Actions (CRITICAL: ALWAYS GENERATE)

**This phase is MANDATORY.** Even if `overall_audit_result == "fail"`, you MUST generate at least one remediation action per failed/high-severity check.

Use `docs/self-audit/remediation.md` as the master reference. For each failed or high-severity check:

**Check → Remediation Action Mapping (CRITICAL: USE EXACT action_ids FROM remediation.md):**

| Failed/High-Severity Check | Required Remediation Action(s) | Affected Steps | Suggested Parameters |
|---|---|---|---|
| `temporal_consistency` = "fail" (gaps > 10%) | `handle_temporal_gaps` **[action_id MUST match remediation.md exactly]** | [10, 12] | `{"gap_handling": "interpolate"}` |
| `multi_series_detection` = "fail" OR "marginal" | `split_by_grouping_column` **[action_id MUST match remediation.md exactly]** | [12, 13, 14, 15] | `{"group_column": "<detected_column>"}` |
| `feature_target_alignment` = "fail" (excluded_ratio > 0.5) | `extend_lag_window` **[action_id MUST match remediation.md exactly]** | [12, 13] | `{"max_lag": 20}` |
| `model_performance_baseline` = "fail" (R² < 0.10) | `increase_regularization` + `try_alternative_models` **[action_ids MUST match remediation.md exactly]** | [13, 14, 15] | `{"regularization_method": "ridge_cv"}` |
| `model_performance_baseline` = "marginal" (0.10 ≤ R² < 0.30) + overfitting | `increase_regularization` **[action_id MUST match remediation.md exactly]** | [13] | `{"alpha_range": [0.1, 1.0, 10.0]}` |
| `data_distribution_drift` = "fail" (KS ≥ 0.25) | `remove_monotonic_index_features` (if KS=1.0) OR `add_seasonal_features` (if autocorr detected) **[action_ids MUST match remediation.md exactly]** | [12, 13] | Check specific findings for parameters |
| `data_distribution_drift` + monotonic features found | `remove_monotonic_index_features` **[action_id MUST match remediation.md exactly]** | [12, 13] | `{}` |

**Implementation Logic (in code):**

⚠️ **CRITICAL:** Use the EXACT action_ids listed below. Do NOT invent new action_ids or rename these. The Agent's orchestrator matches these IDs against `docs/self-audit/remediation.md` to classify as [AUTO] or [MANUAL].

```python
# ✅ CORRECT action_ids (from remediation.md):
remediation_actions = []

# Check 1: temporal_consistency
if checks["temporal_consistency"]["status"] == "fail":
    remediation_actions.append({
        "action_id": "handle_temporal_gaps",  # ✅ EXACT from remediation.md
        "severity": "high",
        "description": "Interpolate or separate training windows to handle temporal gaps",
        "affected_steps": ["10", "12"],  # step numbers as strings
        "suggested_parameters": {"gap_handling": "interpolate", "interpolation_method": "linear"},
        "expected_improvement": "Cleaner training data; prevents spurious patterns in gaps"
    })

# Check 2: multi_series_detection
if checks["multi_series_detection"]["status"] in ["fail", "marginal"]:
    group_col = checks["multi_series_detection"].get("potential_group_columns", ["(auto)"])[0]
    remediation_actions.append({
        "action_id": "split_by_grouping_column",  # ✅ EXACT from remediation.md
        "severity": "high",
        "description": f"Train separate models per group ({group_col}); ensemble predictions",
        "affected_steps": ["12", "13", "14", "15"],
        "suggested_parameters": {"group_column": group_col, "train_separate_models": True, "ensemble_method": "weighted_by_r2"},
        "expected_improvement": "R² +0.2 to +0.5 per group; eliminates cross-entity contamination"
    })

# Check 3: feature_target_alignment
if checks["feature_target_alignment"]["status"] == "fail":
    remediation_actions.append({
        "action_id": "extend_lag_window",  # ✅ EXACT from remediation.md
        "severity": "medium",
        "description": "Increase lag window for time-series features",
        "affected_steps": ["12", "13"],
        "suggested_parameters": {"max_lag": 20, "lag_step": 1},
        "expected_improvement": "CV R² +0.1 to +0.3; captures longer-term dependencies"
    })

# Check 4: model_performance_baseline
if checks["model_performance_baseline"]["status"] == "fail":
    remediation_actions.append({
        "action_id": "increase_regularization",  # ✅ EXACT from remediation.md
        "severity": "high",
        "description": "Strengthen L1/L2 regularization to reduce overfitting",
        "affected_steps": ["13"],
        "suggested_parameters": {"regularization_method": "ridge_cv", "alpha_range": [0.1, 1.0, 10.0]},
        "expected_improvement": "Holdout R² +0.05 to +0.15; better generalization"
    })
    remediation_actions.append({
        "action_id": "try_alternative_models",  # ✅ EXACT from remediation.md
        "severity": "medium",
        "description": "Train additional model types (LightGBM, SVR)",
        "affected_steps": ["13", "14", "15"],
        "suggested_parameters": {"candidates": ["lightgbm", "svr"]},
        "expected_improvement": "R² +0.1 to +0.3; may capture non-linear patterns"
    })

# Check 5: data_distribution_drift
if checks["data_distribution_drift"]["status"] == "fail":
    if ks_statistic >= 1.0:
        remediation_actions.append({
            "action_id": "remove_monotonic_index_features",  # ✅ EXACT from remediation.md
            "severity": "high",
            "description": "Remove constant-slope index features (row number, sequence ID, etc.)",
            "affected_steps": ["12", "13"],
            "suggested_parameters": {},
            "expected_improvement": "Distribution becomes valid; eliminates spurious correlation with time index"
        })
    elif autocorr_lag_7 > 0.5 or autocorr_lag_24 > 0.5:
        remediation_actions.append({
            "action_id": "add_seasonal_features",  # ✅ EXACT from remediation.md
            "severity": "medium",
            "description": "Add day-of-week, hour-of-day, month and rolling seasonal statistics",
            "affected_steps": ["12", "13"],
            "suggested_parameters": {
                "add_day_of_week": True,
                "add_hour_of_day": True,
                "add_month": True,
                "rolling_seasonal_windows": [7, 30]
            },
            "expected_improvement": "R² +0.1 to +0.2; captures seasonal patterns"
        })
```

**CRITICAL VALIDATION:**
- Before returning remediation_actions, verify that every `action_id` in the array **exists** in `docs/self-audit/remediation.md` as a row in the Quick Reference table.
- Do NOT invent action_ids like `"increase_lag_features"`, `"improve_model"`, `"fix_overfitting"`.
- If a check fails but no matching action exists in remediation.md, **log a WARNING** and skip that check's remediation (do not fabricate).

        "suggested_parameters": {"candidates": ["lightgbm", "svr"]},
        "expected_improvement": "May discover better model class; R² +0.1 to +0.3"
    })
elif checks["model_performance_baseline"]["status"] == "marginal" and checks["model_performance_baseline"].get("overfitting_detected"):
    remediation_actions.append({
        "action_id": "increase_regularization",  # ✅ EXACT from remediation.md
        "severity": "medium",
        "description": "Improve generalization by increasing regularization",
        "affected_steps": ["13"],
        "suggested_parameters": {"regularization_method": "ridge_cv", "alpha_range": [0.01, 0.1, 1.0]},
        "expected_improvement": "Holdout R² +0.05 to +0.1"
    })

# Check 4b: model_performance_baseline — Low R² MUST generate actions
if checks["model_performance_baseline"]["status"] == "fail":
    # R² < 0.01 or very weak model performance
    r2_value = checks["model_performance_baseline"].get("best_r2", 0)
    if r2_value < 0.10:
        # Strategy 1: Try alternative models
        remediation_actions.append({
            "action_id": "try_alternative_models",  # ✅ EXACT from remediation.md
            "severity": "high",
            "description": "Current models severely underperform (R² < 0.10). Try LightGBM, SVR, and other non-linear models",
            "affected_steps": ["13", "14", "15"],
            "suggested_parameters": {"additional_candidates": ["lightgbm", "svr", "xgboost"]},
            "expected_improvement": "May discover better model class; R² +0.1 to +0.3"
        })
        # Strategy 2: Extend lag window
        remediation_actions.append({
            "action_id": "extend_lag_window",  # ✅ EXACT from remediation.md
            "severity": "medium",
            "description": "Extend temporal feature window to capture longer-term dependencies",
            "affected_steps": ["12", "13"],
            "suggested_parameters": {"max_lag": 20, "lag_step": 1, "rolling_windows": [5, 10, 20]},
            "expected_improvement": "CV R² +0.1 to +0.3; captures longer-term temporal patterns"
        })

# Check 5: data_distribution_drift
if checks["data_distribution_drift"]["status"] in ["fail", "marginal"]:
    # ✅ CRITICAL: Always generate removal action if high-drift features detected
    high_drift_features = checks["data_distribution_drift"].get("high_drift_features", [])
    
    if high_drift_features:
        # Any feature with KS ≥ 0.80 must be removed before model is valid
        remediation_actions.append({
            "action_id": "remove_monotonic_index_features",  # ✅ EXACT from remediation.md
            "severity": "high",
            "description": f"Remove {len(high_drift_features)} high-drift features with severe train-test distribution shift (KS ≥ 0.80): {', '.join(high_drift_features[:5])}{'...' if len(high_drift_features) > 5 else ''}",
            "affected_steps": ["12", "13", "14", "15"],
            "suggested_parameters": {"high_drift_features_to_drop": high_drift_features},
            "expected_improvement": "Eliminates severe data leakage; model becomes transferable and realistic"
        })
    
    # If no drift features but still marginal, suggest seasonal features
    if not high_drift_features and checks["data_distribution_drift"]["status"] == "marginal":
        remediation_actions.append({
            "action_id": "add_seasonal_features",  # ✅ EXACT from remediation.md
            "severity": "medium",
            "description": "Add seasonal/cyclical features to capture moderate data drift patterns",
            "affected_steps": ["12", "13"],
            "suggested_parameters": {"add_hour_of_day": True, "add_day_of_week": True, "use_cyclic_encoding": True},
            "expected_improvement": "CV R² +0.1 to +0.2 for seasonal patterns"
        })
```

**Next Steps (to communicate to user/orchestrator):**
- If all actions are `[AUTO]` (in `remediation.md`), pipeline will auto-restart.
- If any action is `[MANUAL]`, flag in `next_steps` that user review is required.

### Phase 5: Determine Overall Result
```python
overall_audit_result = "fail" if (
    any(check.status == "fail") or any(check.severity == "high")
) else "pass"

audit_confidence = (count of "pass" checks) / 5
```

---

## MANDATORY CHECKLIST (JSON Output)

### Top-Level Fields (ALL MANDATORY)
- `"step": "17-critical-self-audit"`
- `"run_id": "<RUN_ID>"`
- `"audit_timestamp"`: ISO-8601 (with Z suffix)
- `"overall_audit_result"`: "pass" or "fail" only
- `"audit_confidence"`: float [0.0, 1.0] = n_pass / 5
- `"data_profile"`: `{"detected_profile": "...", "confidence": 0.x, "characteristics": [...]}`
- `"checks"`: object with 5 keys (see below)
- `"critical_findings"`: list (empty if result=="pass", non-empty if result=="fail")
- `"remediation_actions"`: list of objects
- `"next_steps"`: list of strings
- `"notes"`: string

### Every Check Must Have
- `"status"`: "pass", "marginal", or "fail" (NEVER "warning"!)
- `"findings"`: list of strings
- `"severity"`: "low", "medium", or "high"
- `"confidence"`: float [0.0, 1.0]

### Check-Specific Fields (Mandatory)
- Check 2: `"potential_group_columns"` (list)
- Check 3: `"target_variable_in_features"` (bool), `"timestamp_in_features"` (bool), `"monotone_features_found"` (list), `"mi_retention_rate"` (float), `"correlation_max"` (float), `"excluded_ratio"` (float)
- Check 4: `"best_r2"` (float), `"profile"` (string)
- Check 5: `"feature_ks_stats"` (dict: feature → KS value), `"monotone_features"` (list), `"drifted_features"` (list), `"target_ks_stat"` (float)

### Critical Findings Format
```json
{
  "check": "check_name",
  "status": "fail",
  "severity": "high",
  "description": "concrete description (no generic text)"
}
```

### Remediation Actions Format
```json
{
  "action_id": "action_from_remediation_md",
  "severity": "low|medium|high",
  "description": "what will be done",
  "affected_steps": ["list", "of", "step", "ids"],
  "suggested_parameters": {},
  "expected_improvement": "description"
}
```

---

## Multi-Run and Remediation Iteration Protocol

**CRITICAL:** This step may be executed **multiple times** by the orchestrator during the remediation loop.

### First Run (Initial Audit)
1. Orchestrator runs Step 17 for the first time (after Step 16)
2. Generates full audit with all 5 checks, data profile, critical findings
3. If `overall_audit_result == "pass"`: Pipeline is complete
4. If `overall_audit_result == "fail"`: Orchestrator initiates remediation loop (see `.github/agents/Single Agent Pipeline.agent.md` for details)

### Subsequent Runs (Remediation Loop Iterations)
1. Orchestrator executes remediation actions (steps 10–16 are re-run with new parameters)
2. Orchestrator runs Step 17 **again** to audit the remediated pipeline
3. **Key difference:** `progress.json` now contains `"remediation_iteration": N`
4. **Your job:** Generate a fresh audit using the remediated outputs
   - Read the latest step JSON files (which are from the remediation iteration)
   - Evaluate all 5 checks against remediated data
   - Generate new remediation_actions if audit still fails
   - **Do NOT carry over** remediation_actions from previous iterations
5. Report the new `overall_audit_result` in the JSON output

### What Changes Between Iterations
- **Feature set:** May be different (Step 12 re-run with new parameters)
- **Model:** May be different (Step 13 re-run, possibly different candidates)
- **Metrics:** Holdout R², RMSE, MAE may improve (Goal!)
- **Audit checks:** May pass/fail differently based on remediated data

### What Stays the Same
- `run_id` (same run across all iterations)
- `data_profile` (base data hasn't changed, only pipeline)
- Step 10 output (unless remediation explicitly re-cleansed)

### Iteration Termination Criteria
- **Success:** `overall_audit_result == "pass"` → Orchestrator halts remediation, marks pipeline as complete
- **Max iterations reached:** 3 remediation iterations attempted → Orchestrator halts, marks as "failed_with_manual_actions_required"
- **No more [AUTO] actions:** All remaining actions are `[MANUAL]` → Orchestrator halts, escalates to user

### How You Know You're in a Remediation Iteration
- `progress.json` has `"remediation_iteration": N` where N ≥ 1
- Latest step JSON files are newer (modification time) than previous iterations
- Feature set, model type, or metrics differ from initial audit

---

## Implementation Guidelines

1. **Use `polars`** for all data I/O
2. **Use `scipy.stats.ks_2samp()`** for KS statistics  
3. **Use `tqdm`** to wrap the 5-check loop
4. **KS Statistics:** Compute for ALL features — even low-drift ones. Report all in `ks_stats` dict.
5. **Check 2 Priority:** Duplicate timestamps check (n_unique(time_col) < n_rows) **before** variance ratio
6. **Monotonic Features:** KS=1.000 flags as "fail" with dedicated finding
7. **Status Enum (Strict):** Only "pass", "marginal", "fail" — never "warning"
8. **Load Steps Correctly:** step-10-cleanse.json (or step-01-cleanse.json), steps 11–14, all parquet/joblib files
9. **Error Handling:** If input missing/corrupt, exit code 1 with clear error message
10. **JSON Serialization:** Use `_NumpyEncoder` for numpy types. Round floats to 4 decimals.

---

## Example Minimal Output

```json
{
  "step": "17-critical-self-audit",
  "run_id": "20260426T112847Z",
  "audit_timestamp": "2026-04-26T11:28:47Z",
  "data_profile": {
    "detected_profile": "daily_cyclical_temporal",
    "confidence": 0.95,
    "characteristics": ["temporal_data", "single_series", "diurnal_pattern"]
  },
  "checks": {
    "temporal_consistency": {
      "status": "pass",
      "findings": ["regular_10min_frequency", "no_gaps"],
      "severity": "low",
      "confidence": 1.0
    },
    "multi_series_detection": {
      "status": "pass",
      "findings": ["single_time_series"],
      "severity": "low",
      "confidence": 1.0,
      "potential_group_columns": []
    },
    "feature_target_alignment": {
      "status": "pass",
      "findings": ["100%_mi_retention"],
      "severity": "low",
      "confidence": 0.9
    },
    "model_performance_baseline": {
      "status": "pass",
      "findings": ["R²=0.68_exceeds_threshold"],
      "severity": "low",
      "confidence": 0.95,
      "best_r2": 0.6795,
      "profile": "daily_cyclical_temporal"
    },
    "data_distribution_drift": {
      "status": "fail",
      "findings": ["9_features_KS_≥_0.25"],
      "severity": "high",
      "confidence": 0.95,
      "ks_stats": {"t1": 0.28, "t2": 0.31},
      "drifted_features": ["t1", "t2", "t4", "t6", "t8", "t9", "rh_6", "rh_out", "tdewpoint"]
    }
  },
  "critical_findings": [
    {
      "check": "data_distribution_drift",
      "status": "fail",
      "severity": "high",
      "description": "Multiple features show KS ≥ 0.25 drift. Example: seasonal shift between training and test periods, or systematic distribution change."
    }
  ],
  "remediation_actions": [
    {
      "action_id": "add_seasonal_features",
      "severity": "high",
      "description": "Add seasonal/cyclical features (hour-of-day, day-of-week) to capture the detected distribution shift patterns",
      "affected_steps": ["12", "13"],
      "suggested_parameters": {
        "add_hour_of_day": true,
        "add_day_of_week": true,
        "use_cyclic_encoding": true,
        "rolling_seasonal_windows": [7, 30]
      },
      "expected_improvement": "CV R² +0.1 to +0.2 by modeling seasonal variation; better alignment with holdout"
    }
  ],
  "next_steps": [
    "AUTOMATIC: Restart pipeline from Step 12 with new seasonal features",
    "Re-run Step 17 audit to verify improvement",
    "If still failing after 3 iterations: Consider rolling retrain window or data stratification"
  ],
  "notes": "4/5 checks pass. Drift is real (seasonal), not a data quality issue. Suggested remediation is AUTO-executable.",
  "overall_audit_result": "fail",
  "audit_confidence": 0.8
}
```

---

## Notes

- This is a **pure generator prompt** — keep implementation code minimal.
- All algorithm details in `docs/self-audit/*.md` — reference them, don't duplicate.
- Status values are **strict**: Only "pass", "marginal", "fail". No "warning" ever.
- Validation gates (from `.agent.md`) apply: all fields mandatory, proper types, no empty dicts where objects required.


---

## Post-Execution: Remediation Restart

When `overall_audit_result == "fail"`, the pipeline is **not complete**. You MUST immediately execute the PHASE 4 — REMEDIATE procedure defined in `docs/agentic-pipeline/step-prompts.md` (step 17 section).

### Action-to-restart-step mapping

| action_id | Type | Restart from step |
|---|---|---|
| `remove_monotonic_index_features` | AUTO | 12 |
| `extend_lag_window` | AUTO | 12 |
| `add_seasonal_features` | AUTO | 12 |
| `use_time_series_split` | AUTO | 12 |
| `split_by_grouping_column` | AUTO | 12 |
| `improve_model_performance` | AUTO | 13 |
| `increase_regularization` | AUTO | 13 |
| `try_alternative_models` | AUTO | 13 |
| `handle_temporal_gaps` | MANUAL | — write `remediation_required.json` |
| `remove_outliers_by_isolation` | MANUAL | — write `remediation_required.json` |

- Full action specs: `docs/self-audit/remediation.md`.
- The pipeline is not finalized until `overall_audit_result == "pass"` or 3 remediation iterations have been exhausted.
- After 3 failed iterations: write `remediation_required.json`, set `status = "remediation_required"`, exit code 1.
