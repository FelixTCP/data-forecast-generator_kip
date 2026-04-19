# #11 Context Engineering: Data Exploration

## Objective

Generate a decision-ready profile that critically evaluates which features are actually worth engineering and which should be dropped before step 12 ever sees them. The output of this step directly gates what enters the feature matrix.

## Outputs

- univariate summary stats
- target candidacy signals
- time-series suitability flags and autocorrelation profile
- **mutual information (MI) ranking** of all features vs. target
- **pairwise correlation matrix** with redundancy flags
- **near-zero variance flags**
- **random-baseline comparison** for MI
- leakage risk hints
- `recommended_features` — the filtered list step 12 should use as its starting point

## Analysis Requirements

### 1. Near-Zero Variance Filter
- Compute variance for every numeric column.
- Flag any column with variance below `1e-4` (after min-max scaling) as `low_variance`.
- These columns contribute no information and must be excluded unless explicitly overridden.

### 2. Mutual Information Ranking
- Compute `mutual_info_regression(X, y)` for all numeric features vs. the target (use `sklearn.feature_selection`).
- Also compute MI for **5 fresh random-noise columns** (standard normal, same row count) as a baseline.
- Flag any real feature whose MI score falls **at or below the average noise MI** as `below_noise_baseline`.
- Sort all features by MI descending and include the ranking in the output JSON.
- Features ranked at or below noise baseline are **not recommended** for step 12.

### 3. Pairwise Correlation & Redundancy
- Compute the Pearson correlation matrix for all numeric features.
- For any pair with |correlation| ≥ 0.90, flag the one with the **lower MI with the target** as `redundant`.
- Redundant features are **not recommended** for step 12.

### 4. Time-Series Lag Analysis (only when time column detected)
- Compute autocorrelation of the target at lags 1–24 (or up to N/4 if series is short).
- Identify lags where autocorrelation exceeds 0.1 and flag them as `significant_lags`.
- Compute cross-correlation of each feature with the target at lags 0, 1, 2, 3.
- Flag feature-lag combinations where cross-correlation exceeds 0.15 as `useful_lag_features`.
- This directly informs which lags to build in step 12; do not create lags for features with no cross-correlation signal.

### 5. Target Candidacy
- Rank numeric columns by (low null rate, high variance, high MI with other columns).
- Provide top-5 candidates; highlight the user-supplied `TARGET_COLUMN` explicitly.

## Output JSON Keys

```json
{
  "step": "11-data-exploration",
  "shape": {"rows": 19735, "columns": 29},
  "numeric_columns": [...],
  "high_cardinality": [...],
  "low_variance_columns": [...],
  "mi_ranking": [
    {"feature": "t6", "mi_score": 0.42, "below_noise_baseline": false},
    {"feature": "rv1", "mi_score": 0.003, "below_noise_baseline": true}
  ],
  "noise_mi_baseline": 0.005,
  "redundant_columns": ["rv2"],
  "correlation_matrix_summary": {"max_pair": ["rv1","rv2"], "max_corr": 1.0},
  "significant_lags": [1, 3, 6],
  "useful_lag_features": [{"feature": "t1", "lag": 1, "xcorr": 0.23}],
  "recommended_features": ["t6", "t1", "rh_6", "lights", "t_out"],
  "excluded_features": {"rv1": "below_noise_baseline", "rv2": "redundant"},
  "target_candidates": [...],
  "time_series_detected": true,
  "time_column": "date",
  "context": {...}
}
```

## Guardrails

- `recommended_features` must never be empty. If all features fail the filters, loosen the noise-baseline threshold by 50% and log a warning.
- MI computation is stochastic — set `random_state=42`.
- Log the count of features dropped at each filter stage.
- Do not silently pass `recommended_features = all_features`; every exclusion must be logged with a reason.

## Copilot Prompt Snippet

```markdown
Implement `explore_data(df: pl.DataFrame, target: str, time_column: str | None) -> dict`.
Include: near-zero variance filter, MI ranking vs. target with random-noise baseline comparison,
pairwise correlation redundancy detection (|r| >= 0.90), lag autocorrelation and cross-correlation
analysis (lags 1–24) when time_column is present. Return `recommended_features` list.
```

## Tests

- all features fail noise baseline (should loosen threshold, not return empty list)
- perfectly correlated pair (one should be flagged redundant)
- all-categorical dataset
- tiny dataset edge case (fewer than 50 rows)
