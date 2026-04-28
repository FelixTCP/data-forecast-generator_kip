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
   - KS thresholds: < 0.10 = pass, 0.10–0.25 = marginal, ≥ 0.25 = fail
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
4. **Model Performance Baseline** — See § Check 4 (profile-dependent R² thresholds)
5. **Data Distribution Drift** — See § Check 5 (KS statistics)

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
- Critical finding triggered if: status == "fail" OR severity == "high"
- Each must have: `check`, `status`, `severity`, `description`
- Must NOT be empty if `overall_audit_result == "fail"`

### Phase 4: Map to Remediation Actions
From `docs/self-audit/remediation.md`:
- Each action: `action_id`, `severity`, `description`, `affected_steps`, `suggested_parameters`, `expected_improvement`
- If drift detected in grouping column, **must** include `split_by_grouping_column` action

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
  "remediation_actions": [],
  "next_steps": ["Monitor seasonal drift in production", "Consider rolling retrain window"],
  "notes": "4/5 checks pass. Drift is real (seasonal), not a code issue.",
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
