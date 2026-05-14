#!/usr/bin/env python3
"""
Step 10: CSV Read & Cleansing

Load customer CSV robustly and produce a typed, clean polars.DataFrame 
exported to Parquet, alongside a tracked issues report.
"""
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
import polars as pl
import traceback


def normalize_column_name(col: str) -> str:
    """Normalize column name: strip, lowercase, replace spaces with underscores."""
    return col.strip().lower().replace(" ", "_")


def load_and_clean_csv(
    csv_path: str,
    target_column: str,
    output_dir: str,
    run_id: str,
) -> dict:
    """
    Load and clean CSV using polars.
    
    Returns:
        dict: Output JSON with cleansing report
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize progress
    progress_path = output_dir_path / "progress.json"
    progress = {
        "run_id": run_id,
        "csv_path": csv_path,
        "target_column": target_column,
        "status": "running",
        "current_step": "10-csv-read-cleansing",
        "completed_steps": [],
        "errors": []
    }
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)
    
    # Load CSV with polars
    try:
        lf = pl.scan_csv(csv_path, try_parse_dates=True)
        df = lf.collect()
    except Exception as e:
        error_msg = f"Failed to load CSV: {str(e)}\n{traceback.format_exc()}"
        progress["status"] = "error"
        progress["current_step"] = "10-csv-read-cleansing"
        progress["errors"].append(error_msg)
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        raise RuntimeError(error_msg)
    
    # Record initial state
    initial_row_count = df.height
    initial_columns = df.columns
    initial_dtypes = df.schema
    
    # Normalize column names
    normalized_columns = [normalize_column_name(c) for c in initial_columns]
    column_mapping = dict(zip(initial_columns, normalized_columns))
    
    # Check if there are changes
    column_name_fixes = []
    if normalized_columns != initial_columns:
        df = df.rename(column_mapping)
        column_name_fixes.append({
            "reason": "normalized_column_names",
            "changes": {old: new for old, new in column_mapping.items() if old != new}
        })
    
    # Normalize target column name
    target_normalized = normalize_column_name(target_column)
    
    # Verify target column exists
    if target_normalized not in df.columns:
        error_msg = f"Target column '{target_column}' (normalized: '{target_normalized}') not found in CSV. Available columns: {df.columns}"
        progress["status"] = "error"
        progress["errors"].append(error_msg)
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        raise ValueError(error_msg)
    
    # Compute null rates
    null_rates = {}
    for col in df.columns:
        null_rate = float(df.select(pl.col(col).is_null().mean()).item())
        null_rates[col] = null_rate
    
    # Check for high-missingness columns (log but don't drop)
    high_miss_cols = [col for col, rate in null_rates.items() if rate > 0.5]
    
    # Record final state
    final_row_count = df.height
    final_column_count = df.width
    
    # Build output JSON
    output_json = {
        "step": "10-csv-read-cleansing",
        "run_id": run_id,
        "csv_path": csv_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        
        # Counts
        "row_count_initial": initial_row_count,
        "row_count_after": final_row_count,
        "column_count_initial": len(initial_columns),
        "column_count_after": final_column_count,
        "rows_dropped": initial_row_count - final_row_count,
        
        # Target
        "target_column": target_column,
        "target_column_normalized": target_normalized,
        "target_column_dtype": str(df.schema[target_normalized]),
        
        # Column normalization
        "column_mapping": column_mapping,
        "column_name_fixes": column_name_fixes,
        
        # Quality checks
        "null_rate": null_rates,
        "null_rate_summary": {
            "max_null_rate": max(null_rates.values()) if null_rates else 0.0,
            "columns_with_50_percent_plus_nulls": high_miss_cols,
        },
        
        # Schema
        "schema": {col: str(dtype) for col, dtype in initial_dtypes.items()},
        "schema_after": {col: str(dtype) for col, dtype in df.schema.items()},
        
        # Artifacts
        "artifacts": {
            "cleaned_parquet": str(output_dir_path / "cleaned.parquet"),
        },
        
        # Summary
        "quality_report": {
            "total_fixes_applied": len(column_name_fixes),
            "fixes": column_name_fixes,
            "warnings": [f"High missingness detected in {col}: {null_rates[col]:.1%}" for col in high_miss_cols],
        },
        
        "context": {
            "dataset_id": run_id,
            "target_column": target_normalized,
            "time_column": None,  # Will be detected in step 11
            "features": [],
            "split_strategy": {},
            "model_candidates": [],
            "metrics": {},
            "artifacts": {},
            "notes": [
                f"Loaded {initial_row_count} rows from {csv_path}",
                f"Normalized {len([x for x in column_name_fixes])} column names",
                f"Final shape: {final_row_count} rows × {final_column_count} columns",
            ]
        }
    }
    
    # Write cleaned parquet
    parquet_path = output_dir_path / "cleaned.parquet"
    try:
        df.write_parquet(parquet_path)
    except Exception as e:
        error_msg = f"Failed to write parquet: {str(e)}\n{traceback.format_exc()}"
        progress["status"] = "error"
        progress["errors"].append(error_msg)
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        raise RuntimeError(error_msg)
    
    # Write step JSON
    step_json_path = output_dir_path / "step-10-cleanse.json"
    with open(step_json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    # Update progress
    progress["status"] = "running"
    progress["completed_steps"].append("10-csv-read-cleansing")
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"✓ Step 10 complete: {final_row_count} rows × {final_column_count} columns")
    print(f"  Parquet written to: {parquet_path}")
    print(f"  Report written to: {step_json_path}")
    
    return output_json


def main():
    parser = argparse.ArgumentParser(description="Step 10: CSV Read & Cleansing")
    parser.add_argument("--csv-path", required=True, help="Path to input CSV")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    try:
        output_json = load_and_clean_csv(
            csv_path=args.csv_path,
            target_column=args.target_column,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
        sys.exit(0)
    except Exception as e:
        print(f"✗ Step 10 failed: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
