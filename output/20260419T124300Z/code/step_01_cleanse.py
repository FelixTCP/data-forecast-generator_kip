#!/usr/bin/env python3
"""
Step 01: CSV Read & Cleansing
Load CSV robustly, normalize columns, and output cleaned parquet.
"""

import json
import os
import sys
import argparse
import polars as pl
from typing import Any

def normalize_column_names(columns: list[str]) -> dict[str, str]:
    """
    Normalize column names: strip, lowercase, replace spaces with underscores.
    Returns mapping of original -> normalized.
    """
    mapping = {}
    for col in columns:
        # Strip, lowercase, replace spaces/hyphens with underscores
        normalized = col.strip().lower().replace(" ", "_").replace("-", "_")
        # Handle special characters
        normalized = "".join(c if c.isalnum() or c == "_" else "_" for c in normalized)
        # Remove consecutive underscores
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        normalized = normalized.strip("_")
        mapping[col] = normalized
    return mapping

def load_and_clean_csv(csv_path: str, output_dir: str, run_id: str, target_column: str) -> None:
    """Main cleansing pipeline."""
    try:
        # Validate file exists
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        print(f"[Step 01] Loading CSV from {csv_path}...")
        
        # Lazy load with try_parse_dates
        lf = pl.scan_csv(csv_path, try_parse_dates=True)
        
        quality_report = {
            "step": "01-csv-read-cleansing",
            "run_id": run_id,
            "fixes_applied": [],
            "csv_path": csv_path,
        }
        
        # Record initial state
        initial_columns = lf.columns
        quality_report["initial_columns"] = len(initial_columns)
        quality_report["initial_column_names"] = initial_columns
        
        # Normalize column names
        col_mapping = normalize_column_names(initial_columns)
        if col_mapping != {c: c for c in initial_columns}:
            lf = lf.rename(col_mapping)
            quality_report["fixes_applied"].append("normalized_column_names")
            print(f"  ✓ Normalized {len(col_mapping)} column names")
        
        # Evaluate to collect
        df = lf.collect()
        quality_report["initial_row_count"] = df.height
        print(f"  ✓ Loaded {df.height} rows x {df.width} columns")
        
        # Note: String columns with numeric data will be handled in step 11
        # where they are converted as needed for MI computation
        quality_report["fixes_applied"].append("kept_string_types_for_step11")
        
        # Log initial schema
        quality_report["initial_schema"] = {str(col): str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
        
        # Detect and handle null values
        null_rates = {}
        for col in df.columns:
            null_count = df.select(pl.col(col).is_null().sum()).item()
            null_rate = null_count / df.height if df.height > 0 else 0.0
            null_rates[col] = {
                "null_count": int(null_count),
                "null_rate": float(null_rate)
            }
        
        quality_report["null_rates"] = null_rates
        
        # Normalize the target column name
        target_normalized = normalize_column_names([target_column])[target_column]
        if target_normalized not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found after normalization. Available: {df.columns}")
        quality_report["target_column_normalized"] = target_normalized
        
        # Drop rows where target is null (critical)
        rows_before = df.height
        df = df.filter(pl.col(target_normalized).is_not_null())
        rows_after = df.height
        if rows_after < rows_before:
            quality_report["fixes_applied"].append(f"dropped_null_target ({rows_before - rows_after} rows)")
            print(f"  ✓ Dropped {rows_before - rows_after} rows with null target")
        
        # Detect time column (datetime or contains 'date' in name)
        time_column = None
        for col in df.columns:
            if df[col].dtype in [pl.Date, pl.Datetime]:
                time_column = col
                break
            elif 'date' in col.lower() or 'time' in col.lower():
                time_column = col
                break
        
        if time_column:
            quality_report["time_column"] = time_column
            print(f"  ✓ Detected time column: {time_column}")
        else:
            quality_report["time_column"] = None
        
        # Final state
        quality_report["final_row_count"] = df.height
        quality_report["final_column_count"] = df.width
        quality_report["final_schema"] = {str(col): str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
        
        # Artifacts
        cleaned_parquet_path = os.path.join(output_dir, "cleaned.parquet")
        os.makedirs(output_dir, exist_ok=True)
        df.write_parquet(cleaned_parquet_path)
        quality_report["artifacts"] = {
            "cleaned_parquet": cleaned_parquet_path
        }
        print(f"  ✓ Wrote cleaned parquet to {cleaned_parquet_path}")
        
        # Write cleansing report
        report_path = os.path.join(output_dir, "step-01-cleanse.json")
        with open(report_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        print(f"  ✓ Wrote cleansing report to {report_path}")
        
        # Update progress
        progress_path = os.path.join(output_dir, "progress.json")
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        
        progress["current_step"] = "01-csv-read-cleansing"
        progress["completed_steps"].append("01-csv-read-cleansing")
        
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("✓ Step 01 completed successfully")
        sys.exit(0)
    
    except Exception as e:
        print(f"✗ Step 01 failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 01: CSV Read & Cleansing")
    parser.add_argument("--csv-path", required=True, help="Path to input CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory for artifacts")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--target-column", required=True, help="Target column name")
    
    args = parser.parse_args()
    
    load_and_clean_csv(args.csv_path, args.output_dir, args.run_id, args.target_column)
