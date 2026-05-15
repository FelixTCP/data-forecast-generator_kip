"""Step 10 — CSV Read & Cleansing.

Runnable:
    python step_10_cleanse.py --csv-path <path> --target-column <col> \
                              --output-dir <dir> --run-id <id>
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import polars as pl
import numpy as np

OUTPUT_DIR = None
RUN_ID = None


def update_progress(status: str, current_step: str = "10-csv-read-cleansing"):
    path = Path(OUTPUT_DIR) / "progress.json"
    existing = {}
    if path.exists():
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update({"status": status, "current_step": current_step})
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def load_and_clean_csv(csv_path: str, target_column: str, output_dir: str, run_id: str) -> dict:
    fixes = []

    # ── 1. Load ──────────────────────────────────────────────────────────────
    lf = pl.scan_csv(csv_path, try_parse_dates=True, null_values=["", "NA", "N/A", "null", "NULL"])
    raw_columns = lf.schema.names()
    row_count_before = lf.select(pl.len()).collect().item()
    print(f"Loaded {row_count_before} rows, columns: {raw_columns}")

    # ── 2. Normalize column names ─────────────────────────────────────────────
    normalized = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in raw_columns]
    rename_map = {old: new for old, new in zip(raw_columns, normalized) if old != new}
    if rename_map:
        lf = lf.rename(rename_map)
        fixes.append(f"normalized_column_names: {list(rename_map.keys())}")
        print(f"Renamed columns: {rename_map}")

    target_col_normalized = target_column.strip().lower().replace(" ", "_").replace("-", "_")
    if target_col_normalized not in normalized:
        raise ValueError(
            f"Target column '{target_column}' (normalized: '{target_col_normalized}') "
            f"not found in columns: {normalized}"
        )

    # ── 3. Collect for mutation operations ────────────────────────────────────
    df = lf.collect()

    # ── 4. Detect & synthesize time column ───────────────────────────────────
    time_column = None
    detected_frequency = None

    # Check for existing datetime column
    for col in df.columns:
        if df[col].dtype in (pl.Date, pl.Datetime):
            time_column = col
            break

    # Check for date/time in name
    if time_column is None:
        for col in df.columns:
            if any(kw in col.lower() for kw in ("date", "time", "timestamp")):
                if df[col].dtype in (pl.Date, pl.Datetime, pl.Utf8, pl.String):
                    time_column = col
                    break

    # Synthesize from year/month/day integer columns
    if time_column is None:
        cols_lower = {c.lower(): c for c in df.columns}
        year_col = cols_lower.get("year")
        month_col = cols_lower.get("month")
        day_col = cols_lower.get("day")

        if year_col and month_col and day_col:
            print(f"Synthesizing date from columns: {year_col}, {month_col}, {day_col}")
            df = df.with_columns(
                pl.date(pl.col(year_col), pl.col(month_col), pl.col(day_col)).alias("date")
            )
            time_column = "date"
            fixes.append(f"synthesized_date_column from ({year_col}, {month_col}, {day_col})")

    if time_column is None:
        raise RuntimeError(
            "No time column detected. Need a datetime/date column or year+month+day integer columns."
        )

    print(f"Time column: {time_column}")

    # ── 5. Remove nulls in target ─────────────────────────────────────────────
    null_before = df[target_col_normalized].is_null().sum()
    if null_before > 0:
        df = df.filter(pl.col(target_col_normalized).is_not_null())
        fixes.append(f"dropped_null_target_rows: count={null_before}")
        print(f"Dropped {null_before} null target rows")

    # ── 6. Remove duplicates ──────────────────────────────────────────────────
    n_before_dedup = len(df)
    df = df.unique()
    n_dropped_dedup = n_before_dedup - len(df)
    if n_dropped_dedup > 0:
        fixes.append(f"dropped_duplicate_rows: count={n_dropped_dedup}")
        print(f"Dropped {n_dropped_dedup} duplicate rows")

    # ── 7. Chronological sort (MANDATORY after dedup) ─────────────────────────
    df = df.sort(time_column)
    fixes.append(f"sorted_by_time_column: {time_column}")

    # ── 8. Detect sampling frequency ─────────────────────────────────────────
    time_series = df[time_column]
    if time_series.dtype == pl.Date:
        diffs = (
            df.with_columns(pl.col(time_column).cast(pl.Int32).alias("_t"))
            .select(pl.col("_t").diff().drop_nulls())["_t"]
            .to_list()
        )
    else:
        diffs = (
            df.with_columns(pl.col(time_column).cast(pl.Int64).alias("_t"))
            .select(pl.col("_t").diff().drop_nulls())["_t"]
            .to_list()
        )

    if diffs:
        median_diff = float(np.median([d for d in diffs if d > 0])) if any(d > 0 for d in diffs) else 1.0
        if isinstance(df[time_column][0], type(None)):
            detected_frequency = "daily"
        else:
            dtype = df[time_column].dtype
            if dtype == pl.Date:
                if median_diff <= 1.5:
                    detected_frequency = "daily"
                elif median_diff <= 7.5:
                    detected_frequency = "weekly"
                elif median_diff <= 31.5:
                    detected_frequency = "monthly"
                else:
                    detected_frequency = "yearly"
            else:
                # datetime diff in microseconds
                if median_diff <= 60_000_000:
                    detected_frequency = "minutely"
                elif median_diff <= 3600_000_000:
                    detected_frequency = "hourly"
                elif median_diff <= 86400_000_000:
                    detected_frequency = "daily"
                else:
                    detected_frequency = "daily"
    else:
        detected_frequency = "daily"

    print(f"Detected frequency: {detected_frequency}")

    # ── 9. Extreme anomaly smoothing (|z| > 6) — mandatory ───────────────────
    numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8, pl.UInt32, pl.UInt64, pl.UInt16, pl.UInt8)
                    and c != time_column and c not in ("year", "month", "day")]

    for col in numeric_cols:
        col_vals = df[col].cast(pl.Float64)
        mean_val = col_vals.mean()
        std_val = col_vals.std()
        if std_val is None or std_val == 0:
            continue
        z_scores = ((col_vals - mean_val) / std_val).abs()
        extreme_mask = z_scores > 6
        extreme_count = extreme_mask.sum()
        if extreme_count > 0:
            # Null out extreme values, then interpolate
            df = df.with_columns(
                pl.when(extreme_mask)
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
            df = df.with_columns(
                pl.col(col).cast(pl.Float64)
                .interpolate()
                .forward_fill()
                .backward_fill()
                .alias(col)
            )
            fixes.append(
                f"extreme_anomaly_smoothed: col='{col}', zscore_threshold=6, count={int(extreme_count)}"
            )
            print(f"Smoothed {int(extreme_count)} extreme anomalies in '{col}'")

    # ── 10. Numeric coercion for string columns ───────────────────────────────
    for col in df.columns:
        if df[col].dtype in (pl.Utf8, pl.String) and col not in ("region", "country", "state", "city"):
            try:
                coerced = df[col].cast(pl.Float64, strict=False)
                null_before_coerce = df[col].is_null().sum()
                null_after_coerce = coerced.is_null().sum()
                if null_after_coerce <= null_before_coerce + int(0.05 * len(df)):
                    df = df.with_columns(coerced.alias(col))
                    fixes.append(f"coerced_string_to_numeric: col='{col}'")
            except Exception:
                pass

    # ── 11. Drop constant/all-null columns ────────────────────────────────────
    cols_to_drop = []
    for col in df.columns:
        if col in (time_column, target_col_normalized):
            continue
        null_frac = df[col].is_null().mean()
        if null_frac == 1.0:
            cols_to_drop.append(col)
    if cols_to_drop:
        df = df.drop(cols_to_drop)
        fixes.append(f"dropped_all_null_columns: {cols_to_drop}")
        print(f"Dropped all-null columns: {cols_to_drop}")

    # ── 12. Outlier stats (for Streamlit EDA) ────────────────────────────────
    outliers = {}
    final_numeric = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8, pl.UInt32, pl.UInt64, pl.UInt16, pl.UInt8)]
    for col in final_numeric:
        vals = df[col].cast(pl.Float64).drop_nulls()
        if len(vals) < 4:
            continue
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_mask = (df[col].cast(pl.Float64) < lower) | (df[col].cast(pl.Float64) > upper)
        iqr_indices = [i for i, v in enumerate(iqr_mask.to_list()) if v][:200]

        arr = vals.to_numpy()
        mean_v = float(arr.mean())
        std_v = float(arr.std())
        z3_count = int((np.abs((arr - mean_v) / (std_v + 1e-12)) > 3).sum()) if std_v > 0 else 0

        outliers[col] = {
            "iqr_lower": float(lower),
            "iqr_upper": float(upper),
            "iqr_outlier_count": len(iqr_indices),
            "iqr_outlier_indices_sample": iqr_indices[:200],
            "z3_outlier_count": z3_count,
        }

    # ── 13. Quality report ────────────────────────────────────────────────────
    null_rate = {c: float(df[c].is_null().mean()) for c in df.columns}

    quality_report = {
        "row_count_before": row_count_before,
        "row_count_after": len(df),
        "column_count": df.width,
        "null_rate": null_rate,
        "fixes": fixes,
        "outliers": outliers,
    }

    # ── 14. Write parquet ─────────────────────────────────────────────────────
    parquet_path = str(Path(output_dir) / "cleaned.parquet")
    df.write_parquet(parquet_path)
    print(f"Written cleaned.parquet: {parquet_path} ({len(df)} rows, {df.width} cols)")
    print(f"Schema: {df.schema}")

    return quality_report, time_column, detected_frequency, target_col_normalized


def main():
    parser = argparse.ArgumentParser(description="Step 10: CSV Read & Cleansing")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    global OUTPUT_DIR, RUN_ID
    OUTPUT_DIR = args.output_dir
    RUN_ID = args.run_id

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Initialize progress.json
    progress_path = Path(OUTPUT_DIR) / "progress.json"
    progress = {
        "run_id": RUN_ID,
        "csv_path": args.csv_path,
        "target_column": args.target_column,
        "status": "running",
        "current_step": "10-csv-read-cleansing",
        "completed_steps": [],
        "errors": [],
    }
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    try:
        quality_report, time_column, detected_frequency, target_col_normalized = load_and_clean_csv(
            args.csv_path, args.target_column, OUTPUT_DIR, RUN_ID
        )

        # Build step output JSON
        step_output = {
            "step": "10-csv-read-cleansing",
            "run_id": RUN_ID,
            "csv_path": args.csv_path,
            "target_column_original": args.target_column,
            "target_column_normalized": target_col_normalized,
            "time_column": time_column,
            "detected_frequency": detected_frequency,
            "row_count_before": quality_report["row_count_before"],
            "row_count_after": quality_report["row_count_after"],
            "column_count": quality_report["column_count"],
            "null_rate": quality_report["null_rate"],
            "fixes": quality_report["fixes"],
            "outliers": quality_report["outliers"],
            "artifacts": {
                "cleaned_parquet": str(Path(OUTPUT_DIR) / "cleaned.parquet"),
            },
        }

        out_path = Path(OUTPUT_DIR) / "step-10-cleanse.json"
        with open(out_path, "w") as f:
            json.dump(step_output, f, indent=2)
        print(f"Written: {out_path}")

        # Update progress
        with open(progress_path) as f:
            progress = json.load(f)
        progress["completed_steps"].append("10-csv-read-cleansing")
        progress["current_step"] = "10-csv-read-cleansing"
        progress["status"] = "running"
        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)

        print("Step 10 complete.")
        sys.exit(0)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR in step 10: {e}\n{tb}", file=sys.stderr)

        with open(progress_path) as f:
            progress = json.load(f)
        progress["status"] = "error"
        progress["errors"].append({"step": "10-csv-read-cleansing", "error": str(e), "traceback": tb})
        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)

        sys.exit(1)


if __name__ == "__main__":
    main()
