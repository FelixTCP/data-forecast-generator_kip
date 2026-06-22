"""Step 10 — CSV Read & Cleansing."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import polars as pl


def update_progress(output_dir: str, step: str, status: str, run_id: str, csv_path: str, target_col: str, extra: dict = None):
    progress_path = os.path.join(output_dir, "progress.json")
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
    else:
        progress = {"run_id": run_id, "csv_path": csv_path, "target_column": target_col,
                    "completed_steps": [], "errors": []}
    progress["status"] = status
    progress["current_step"] = step
    if extra:
        progress.update(extra)
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--target-column", required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    run_id = args.run_id
    csv_path = args.csv_path
    target_col_raw = args.target_column

    os.makedirs(output_dir, exist_ok=True)
    update_progress(output_dir, "10-csv-read-cleansing", "running", run_id, csv_path, target_col_raw)

    fixes = []
    quality_report = {"fixes": fixes}

    # --- Load CSV ---
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    lf = pl.scan_csv(csv_path, try_parse_dates=True, ignore_errors=False)
    original_cols = lf.columns
    row_count_initial = lf.select(pl.len()).collect().item()

    # Normalize column names
    normalized_cols = [normalize_col(c) for c in original_cols]
    rename_map = {o: n for o, n in zip(original_cols, normalized_cols) if o != n}
    if rename_map:
        lf = lf.rename(rename_map)
        fixes.append("normalized_column_names")

    df = lf.collect()

    # Normalize target column name
    target_col = normalize_col(target_col_raw)
    if target_col not in df.columns:
        print(f"ERROR: Target column '{target_col}' not found. Available: {df.columns}", file=sys.stderr)
        sys.exit(1)

    # Remove duplicates
    row_before_dedup = df.height
    df = df.unique()
    dup_removed = row_before_dedup - df.height
    if dup_removed > 0:
        fixes.append(f"removed_{dup_removed}_duplicate_rows")

    # Detect time column
    time_col = None
    # Check for datetime dtype columns
    for col in df.columns:
        if df[col].dtype in (pl.Datetime, pl.Date):
            time_col = col
            break

    # If no datetime found, try to synthesize from year/month/day columns
    if time_col is None:
        has_year = "year" in df.columns
        has_month = "month" in df.columns
        has_day = "day" in df.columns
        if has_year and has_month and has_day:
            df = df.with_columns(
                pl.date(pl.col("year"), pl.col("month"), pl.col("day")).alias("date_synthesized")
            )
            time_col = "date_synthesized"
            fixes.append("synthesized_date_from_year_month_day")
        else:
            # Try column names containing date/time
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    time_col = col
                    break

    if time_col is None:
        raise RuntimeError(
            "No time column detected — cannot guarantee chronological order. "
            "Aborting step 10 to prevent silent corruption of downstream steps."
        )

    # Deduplicate on time column if synthesized (to prevent duplicate dates)
    before_date_dedup = df.height
    df = df.unique(subset=[time_col], keep="last")
    date_dedup_removed = before_date_dedup - df.height
    if date_dedup_removed > 0:
        fixes.append(f"removed_{date_dedup_removed}_duplicate_date_rows")

    # Attempt numeric coercion for string columns that look numeric
    for col in df.columns:
        if df[col].dtype == pl.Utf8 and col != time_col:
            try:
                coerced = df[col].cast(pl.Float64, strict=False)
                null_before = df[col].is_null().sum()
                null_after = coerced.is_null().sum()
                if null_after <= null_before + 2:
                    df = df.with_columns(coerced)
                    fixes.append(f"coerced_{col}_to_float64")
            except Exception:
                pass

    # Identify numeric columns
    numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)]

    # Sort by time (required before anomaly smoothing)
    df = df.sort(time_col)

    # Extreme anomaly smoothing: |z-score| > 6
    for col in numeric_cols:
        if col == time_col:
            continue
        series = df[col].cast(pl.Float64)
        mean_val = series.mean()
        std_val = series.std()
        if std_val is None or std_val == 0:
            continue
        mask = ((series - mean_val).abs() / std_val) > 6
        count = mask.sum()
        if count > 0:
            # Null out extremes, then interpolate
            df = df.with_columns(
                pl.when(mask).then(None).otherwise(series).alias(col)
            )
            df = df.with_columns(
                pl.col(col).interpolate().forward_fill().backward_fill()
            )
            fixes.append(f"extreme_anomaly_smoothed: col='{col}', zscore_threshold=6, count={count}")

    # Outlier detection (IQR and z-score)
    outliers = {}
    for col in numeric_cols:
        if col == time_col:
            continue
        series = df[col].cast(pl.Float64).drop_nulls()
        if series.len() == 0:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_mask = (df[col].cast(pl.Float64) < lower) | (df[col].cast(pl.Float64) > upper)
        iqr_count = iqr_mask.sum()
        mean_v = series.mean()
        std_v = series.std()
        if std_v and std_v > 0:
            z_mask = ((df[col].cast(pl.Float64) - mean_v).abs() / std_v) > 3
            z_count = z_mask.sum()
        else:
            z_count = 0
        indices = [i for i, v in enumerate(iqr_mask.to_list()) if v][:200]
        outliers[col] = {
            "iqr_outlier_count": int(iqr_count),
            "zscore_outlier_count": int(z_count),
            "iqr_lower_bound": float(lower),
            "iqr_upper_bound": float(upper),
            "outlier_fraction": round(float(iqr_count) / df.height, 4) if df.height > 0 else 0.0,
            "outlier_indices_sample": indices,
        }

    # MANDATORY FINAL SORT
    df = df.sort(time_col)
    quality_report["sorted_by"] = time_col
    fixes.append(f"final_chronological_sort_by={time_col}")

    # Write parquet
    parquet_path = os.path.join(output_dir, "cleaned.parquet")
    df.write_parquet(parquet_path)

    # Compute null rates
    null_rate = {c: float(df[c].is_null().mean()) for c in df.columns}

    # Inferred dtypes
    inferred_dtypes = {c: str(df[c].dtype) for c in df.columns}

    row_count_after = df.height

    result = {
        "step": "10-csv-read-cleansing",
        "row_count_initial": row_count_initial,
        "row_count_after": row_count_after,
        "column_count": df.width,
        "target_column_normalized": target_col,
        "time_column_detected": time_col,
        "null_rate": null_rate,
        "duplicate_rows_removed": dup_removed,
        "inferred_dtypes": inferred_dtypes,
        "outliers": outliers,
        "sorted_by": time_col,
        "fixes": fixes,
        "artifacts": {
            "cleaned_parquet": parquet_path
        },
        "context": {
            "target_column": target_col,
            "time_column": time_col,
        }
    }

    out_json = os.path.join(output_dir, "step-10-cleanse.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    # Update progress
    progress_path = os.path.join(output_dir, "progress.json")
    with open(progress_path) as f:
        progress = json.load(f)
    progress["status"] = "running"
    progress["current_step"] = "11-data-exploration"
    if "10-csv-read-cleansing" not in progress.get("completed_steps", []):
        progress.setdefault("completed_steps", []).append("10-csv-read-cleansing")
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    print(f"Step 10 complete. Rows: {row_count_after}, Time col: {time_col}, Target: {target_col}")
    print(f"Parquet: {parquet_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
