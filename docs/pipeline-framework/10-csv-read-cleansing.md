# #10 Context Engineering: CSV Read / Cleansing

## Objective

Load customer CSV robustly and produce a typed, clean `polars.DataFrame` exported to a Parquet file, alongside a tracked issues report.

## Inputs

- CSV path
- optional schema hints

## Outputs

- `$OUTPUT_DIR/cleaned.parquet`: The exported cleaned dataset
- `$OUTPUT_DIR/step-10-cleanse.json`: Structured cleansing report for downstream steps
- `clean_df`: Evaluation result (`polars.DataFrame`)
- quality report (`missingness`, invalid casts, duplicates, outliers, applied fixes)
- updated `PipelineContext.notes`

## Guardrails

- Use strictly `polars`. No pandas.
- Use defensive programming (`strict=False` when casting, handle nulls explicitly, define an explicit `schema_overrides` if necessary).
- Fail fast on unreadable files.
- Do not silently drop rows/columns without logging initial count, final count, and reason.
- Preserve original column names in metadata even if normalized.

## Copilot Prompt Snippet

```markdown
Implement `load_and_clean_csv(csv_path: str, config: dict, output_path: str) -> tuple[pl.DataFrame, dict]`.
Apply robust schema inference and defensive cleansing defaults so the step is self-contained.
Use only the `polars` Lazy API (`pl.scan_csv()`), executing `.collect()` only before returning/writing the Parquet file to `output_path`.
Return a `quality_report` with null-rate per column, inferred dtypes, duplicate rows, and applied fixes.
Write a `pytest`-compatible test file to `$CODE_DIR/tests/test_10_ingest.py`.
```

## Code Skeleton

```python
import polars as pl
import os

def load_and_clean_csv(csv_path: str, config: dict, output_path: str) -> tuple[pl.DataFrame, dict]:
    # Lazy Evaluation Pipeline
    lf = pl.scan_csv(csv_path, try_parse_dates=True)
    
    quality_report = {
        "fixes": [],
    }

    # Example: normalize column names
    initial_columns = lf.columns
    normalized = [c.strip().lower().replace(" ", "_") for c in initial_columns]
    
    if normalized != initial_columns:
        quality_report["fixes"].append("normalized_column_names")
        lf = lf.rename(dict(zip(initial_columns, normalized)))

    # Evaluate the graph
    df = lf.collect()
    
    # Run stats logging after evaluation
    quality_report.update({
        "row_count_final": df.height,
        "column_count": df.width,
        "null_rate": {
            c: float(df.select(pl.col(c).is_null().mean()).item()) for c in df.columns
        }
    })
    
    # Write to Parquet output
    df.write_parquet(output_path)
    
    print(f"Final logged rows: {df.height}")
    print(f"Final logged schema: {df.schema}")

    return df, quality_report
```

## Tests

- Validate the output Parquet file exists
- Validate the schema matches expectations (e.g., Dates are actually datetime, not strings)
- Explicit explicit null handling applies correctly
- malformed csv
- mixed dtypes
- high missingness column
- duplicate rows present
