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
- Preserve original column names in metadata even if renamed.
- **Extreme anomaly smoothing is mandatory**: After sorting by time, scan the target column (and all other numeric columns) for values with an **absolute z-score above 6** (i.e. $|z| > 6$). These are statistically impossible readings — not real extremes but corrupted / sentinel / instrument-error values. Null them out and replace with **linear interpolation** (`interpolate()`), falling back to `forward_fill()` / `backward_fill()` for boundary nulls. Log each replacement in `fixes` including the threshold, count, and affected column.
- **Chronological sort is the LAST operation before `write_parquet` — no exceptions.** `df.unique()` does not preserve row order; any interpolation, fill, or feature step can also shuffle rows. The very last line before writing must be `df = df.sort(time_col)`. Sort by the detected time column; if a synthetic date was constructed from year/month/day columns, sort by that synthetic column. If no time column can be identified at all, raise a `RuntimeError` with a descriptive message — a non-chronological parquet will silently corrupt every downstream step.
- **CRITICAL**: when synthesizing a date from year/month/day integer columns, use `pl.date(pl.col("year"), pl.col("month"), pl.col("day"))` — **never** `str.pad_left()` which does not exist in polars; use `pl.date()` directly.
## Copilot Prompt Snippet

```markdown
Implement `load_and_clean_csv(csv_path: str, config: dict, output_path: str) -> tuple[pl.DataFrame, dict]`.
Apply robust schema inference and defensive cleansing defaults so the step is self-contained.
Use only the `polars` Lazy API (`pl.scan_csv()`), executing `.collect()` only before returning/writing the Parquet file to `output_path`.
Return a `quality_report` with null-rate per column, inferred dtypes, duplicate rows, and applied fixes.

After sorting by time, scan the target column (and all other numeric columns) for **extreme anomalies** defined
as values whose absolute z-score exceeds 6 (`|z| > 6`). These are not real extremes — they are corrupted,
sentinel, or instrument-error readings. Null them out and replace with linear interpolation
(`pl.col(col).interpolate().forward_fill().backward_fill()`). Log every replacement in the `fixes` list as
`"extreme_anomaly_smoothed: col='<col>', zscore_threshold=6, count=<N>"`.

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
    
    # ── MANDATORY FINAL SORT ─────────────────────────────────────────────
    # cleaned.parquet MUST be in strict chronological order.
    # This sort happens LAST, after all dedup / interpolation / fill steps,
    # so no earlier operation can silently undo the ordering.
    if time_col is None:
        raise RuntimeError(
            "No time column detected — cannot guarantee chronological order. "
            "Aborting step 10 to prevent silent corruption of downstream steps."
        )
    df = df.sort(time_col)
    quality_report["sorted_by"] = time_col
    quality_report["fixes"].append(f"final_chronological_sort_by={time_col}")

    # Write to Parquet output
    df.write_parquet(output_path)

    print(f"Final logged rows: {df.height}")
    print(f"Final logged schema: {df.schema}")
    print(f"Parquet written in chronological order by '{time_col}'.")

    return df, quality_report
```

## Outlier Detection (required for Streamlit EDA view)

After collecting the cleaned DataFrame, compute per-column outlier statistics for every **numeric** column. Write results to the `"outliers"` key of the quality report.

Use IQR-based detection:
- `Q1 = column.quantile(0.25)`, `Q3 = column.quantile(0.75)`, `IQR = Q3 - Q1`
- `lower_bound = Q1 - 1.5 * IQR`, `upper_bound = Q3 + 1.5 * IQR`
- Outlier rows: values below `lower_bound` OR above `upper_bound`
- Store the **first 200 row indices** of outlier rows (for UI scatter plots)

Also compute z-score outlier count (|z| > 3) for each column.

```json
"outliers": {
  "appliances": {
    "iqr_outlier_count": 312,
    "zscore_outlier_count": 289,
    "iqr_lower_bound": -75.0,
    "iqr_upper_bound": 325.0,
    "outlier_fraction": 0.016,
    "outlier_indices_sample": [4, 17, 88, 204, 371]
  },
  "lights": {
    "iqr_outlier_count": 0,
    "zscore_outlier_count": 5,
    "iqr_lower_bound": 0.0,
    "iqr_upper_bound": 0.0,
    "outlier_fraction": 0.0,
    "outlier_indices_sample": []
  }
}
```

## Complete Output JSON Schema

```json
{
  "step": "10-csv-read-cleansing",
  "row_count_initial": 19737,
  "row_count_after": 19735,
  "column_count": 29,
  "target_column_normalized": "appliances",
  "time_column_detected": "date",
  "null_rate": {
    "appliances": 0.0,
    "lights": 0.0,
    "t1": 0.001
  },
  "duplicate_rows_removed": 2,
  "inferred_dtypes": {
    "date": "Datetime",
    "appliances": "Float64",
    "lights": "Float64"
  },
  "outliers": {
    "appliances": {
      "iqr_outlier_count": 312,
      "zscore_outlier_count": 289,
      "iqr_lower_bound": -75.0,
      "iqr_upper_bound": 325.0,
      "outlier_fraction": 0.016,
      "outlier_indices_sample": [4, 17, 88]
    }
  },
  "sorted_by": "date",
  "fixes": ["normalized_column_names", "removed_duplicates", "final_chronological_sort_by=date"],
  "artifacts": {
    "cleaned_parquet": "OUTPUT_DIR/cleaned.parquet"
  }
}
```

## Tests

- Validate the output Parquet file exists
- Validate the schema matches expectations (e.g., Dates are actually datetime, not strings)
- Explicit null handling applies correctly
- malformed csv
- mixed dtypes
- high missingness column
- duplicate rows present
- outlier detection: column with known extreme values produces correct `iqr_outlier_count`
- column with no outliers produces `iqr_outlier_count: 0`
- **chronological order**: load a shuffled CSV and verify `cleaned.parquet` rows are in strict ascending time order (assert `df["date"].is_sorted()` returns `True`)
- **no time column**: verify the step raises `RuntimeError` when no time/date column can be detected
- `sorted_by` key present in output JSON and matches the detected time column name
