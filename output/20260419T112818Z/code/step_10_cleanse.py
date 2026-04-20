#!/usr/bin/env python3
"""
Step 10: CSV Read & Cleansing
Loads customer CSV robustly, normalizes column names, detects time column,
and produces cleaned parquet + metadata JSON.
"""

import json
import argparse
from pathlib import Path
import polars as pl
import sys


def run_step_10(csv_path: str, target_column: str, output_dir: str) -> dict:
    """
    Load CSV, cleanse, normalize column names, detect time column.
    
    Returns step output dict.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load CSV with automatic date parsing
    try:
        df = pl.read_csv(csv_path, try_parse_dates=True)
    except FileNotFoundError:
        error_msg = f"CSV file not found: {csv_path}"
        sys.exit(error_msg)
    except Exception as e:
        error_msg = f"Failed to read CSV: {e}"
        sys.exit(error_msg)
    
    if df.height == 0:
        sys.exit("CSV is empty (0 rows)")
    
    row_count_before = df.height
    column_count = df.width
    
    # Record original column names
    original_columns = df.columns.copy()
    
    # Normalize column names: strip, lowercase, replace spaces with underscores
    normalized_columns = [
        col.strip().lower().replace(" ", "_") for col in df.columns
    ]
    
    if normalized_columns != df.columns:
        df = df.rename(dict(zip(original_columns, normalized_columns, strict=False)))
    
    # Normalize target column name
    target_column_normalized = target_column.strip().lower().replace(" ", "_")
    
    # Check if target column exists
    if target_column_normalized not in df.columns:
        available = df.columns
        error_msg = (
            f"Target column '{target_column_normalized}' not found in CSV. "
            f"Available columns: {available}"
        )
        sys.exit(error_msg)
    
    # Detect time column (look for datetime dtype or column name with "date"/"time")
    time_column_detected = None
    for col in df.columns:
        if df[col].dtype == pl.Datetime:
            time_column_detected = col
            break
    
    if time_column_detected is None:
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                time_column_detected = col
                break
    
    # Compute null rates
    null_rate = {
        col: float(df.select(pl.col(col).is_null().mean()).item())
        for col in df.columns
    }
    
    # Get inferred dtypes as strings
    inferred_dtypes = {col: str(df[col].dtype) for col in df.columns}
    
    # Save cleaned parquet
    cleaned_parquet_path = output_path / "cleaned.parquet"
    df.write_parquet(cleaned_parquet_path)
    
    # Build step output
    step_output = {
        "step": "10-csv-read-cleansing",
        "row_count_before": row_count_before,
        "row_count_after": df.height,
        "column_count": column_count,
        "target_column_original": target_column,
        "target_column_normalized": target_column_normalized,
        "time_column_detected": time_column_detected,
        "null_rate": null_rate,
        "inferred_dtypes": inferred_dtypes,
        "artifacts": {
            "cleaned_parquet": str(cleaned_parquet_path)
        }
    }
    
    # Save step JSON
    step_json_path = output_path / "step-10-cleanse.json"
    with open(step_json_path, "w") as f:
        json.dump(step_output, f, indent=2)
    
    # Update progress
    progress_path = output_path / "progress.json"
    if progress_path.exists():
        with open(progress_path, "r") as f:
            progress = json.load(f)
    else:
        progress = {
            "run_id": "",
            "csv_path": csv_path,
            "target_column": target_column,
            "status": "running",
            "current_step": "10-csv-read-cleansing",
            "completed_steps": [],
            "errors": []
        }
    
    progress["status"] = "running"
    progress["current_step"] = "11-data-exploration"
    progress["completed_steps"].append("10-csv-read-cleansing")
    
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)
    
    return step_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 10: CSV Read & Cleansing")
    parser.add_argument("--csv-path", required=True, help="Path to input CSV")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument("--output-dir", required=True, help="Output directory for artifacts")
    
    args = parser.parse_args()
    
    try:
        result = run_step_10(args.csv_path, args.target_column, args.output_dir)
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except Exception as e:
        print(f"Error in Step 10: {e}", file=sys.stderr)
        sys.exit(1)
