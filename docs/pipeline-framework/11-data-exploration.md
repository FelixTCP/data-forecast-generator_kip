# #11 Context Engineering: Data Exploration

## Objective

Generate a compact, decision-ready profile for forecasting and feature planning.

## Outputs

- univariate summary stats
- target candidacy signals
- time-series suitability flags
- leakage risk hints

## Copilot Prompt Snippet

```markdown
Implement `explore_data(df: pl.DataFrame, target_hint: str | None) -> dict`.
Include summary stats, correlation preview, cardinality scan, and likely target suggestions.
```

## Code Skeleton

```python
import polars as pl


def explore_data(df: pl.DataFrame, target_hint: str | None = None) -> dict:
    numeric_cols = [c for c, dt in zip(df.columns, df.dtypes, strict=False) if dt.is_numeric()]
    profile = {
        "shape": (df.height, df.width),
        "numeric_columns": numeric_cols,
        "high_cardinality": [
            c for c in df.columns if df[c].n_unique() > max(100, int(0.5 * df.height))
        ],
    }
    return profile
```

## Tests

- target hint given vs missing
- all-categorical dataset
- tiny dataset edge case
