# #10 Context Engineering: CSV Read / Cleansing

## Objective

Load customer CSV robustly and produce a typed, clean `polars.DataFrame` with tracked issues.

## Inputs

- CSV path
- optional schema hints

## Outputs

- `clean_df`
- quality report (`missingness`, invalid casts, duplicates, outliers)
- updated `PipelineContext.notes`

## Guardrails

- Fail fast on unreadable files.
- Do not silently drop rows/columns without logging count + reason.
- Preserve original column names in metadata even if normalized.

## Copilot Prompt Snippet

```markdown
Implement `load_and_clean_csv(csv_path: str, config: dict) -> tuple[pl.DataFrame, dict]`.
Return a `quality_report` with null-rate per column, inferred dtypes, duplicate rows, and applied fixes.
Use only `polars` operations.
```

## Code Skeleton

```python
import polars as pl


def load_and_clean_csv(csv_path: str, config: dict) -> tuple[pl.DataFrame, dict]:
    df = pl.read_csv(csv_path, try_parse_dates=True)

    quality_report = {
        "row_count_before": df.height,
        "column_count": df.width,
        "null_rate": {
            c: float(df.select(pl.col(c).is_null().mean()).item()) for c in df.columns
        },
        "fixes": [],
    }

    # Example: normalize column names
    normalized = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if normalized != df.columns:
        quality_report["fixes"].append("normalized_column_names")
        df = df.rename(dict(zip(df.columns, normalized, strict=False)))

    return df, quality_report
```

## Tests

- malformed csv
- mixed dtypes
- high missingness column
- duplicate rows present
