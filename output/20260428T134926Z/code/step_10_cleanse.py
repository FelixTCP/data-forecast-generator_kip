#!/usr/bin/env python3
"""
Step 10: CSV Read & Cleansing
Loads CSV with polars, normalizes columns, validates target column presence,
and exports cleaned data to parquet.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import polars as pl

def normalize_column_name(col: str) -> str:
    """Normalize a column name: strip, lowercase, replace spaces with underscores."""
    return col.strip().lower().replace(" ", "_")

def load_and_clean_csv(
    csv_path: str,
    target_column: str,
    output_dir: str,
    run_id: str
) -> dict:
    """
    Load CSV, normalize columns, validate target, export to parquet.
    
    Args:
        csv_path: Path to input CSV
        target_column: Expected target column name (will be normalized)
        output_dir: Directory for outputs
        run_id: Run identifier
    
    Returns:
        Dictionary with step output JSON content
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Normalize target column name for comparison
    target_normalized = normalize_column_name(target_column)
    
    try:
        # Try to load CSV with polars
        print(f"[Step 10] Loading CSV from: {csv_path}")
        lf = pl.scan_csv(csv_path, try_parse_dates=True)
        
        # Get initial column info
        initial_columns = lf.collect_schema().names()
        print(f"[Step 10] Initial columns ({len(initial_columns)}): {initial_columns[:5]}...")
        
        # Normalize column names
        column_mapping = {}
        for col in initial_columns:
            normalized = normalize_column_name(col)
            column_mapping[col] = normalized
        
        # Check if any normalization was needed
        normalized_columns = list(column_mapping.values())
        normalization_applied = normalized_columns != initial_columns
        
        if normalization_applied:
            print(f"[Step 10] Column name normalization applied")
            lf = lf.rename(column_mapping)
        
        # Collect to DataFrame for analysis
        df = lf.collect()
        
        # Validate target column exists
        if target_normalized not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' (normalized: '{target_normalized}') "
                f"not found in CSV. Available columns: {df.columns}"
            )
        
        print(f"[Step 10] Target column detected: {target_normalized}")
        
        # Attempt numeric coercion for string columns that look numeric
        print("[Step 10] Attempting numeric coercion on string columns...")
        for col in df.columns:
            if df.schema[col] == pl.String or df.schema[col] == pl.Utf8:
                # Try to cast to float
                try:
                    df = df.with_columns(
                        pl.col(col).str.strip_chars().cast(pl.Float64, strict=False).alias(col)
                    )
                    print(f"[Step 10]   - Coerced {col} to Float64")
                except Exception:
                    # Keep as string if coercion fails
                    pass
        
        # Get initial row count
        initial_row_count = df.height
        print(f"[Step 10] Initial rows: {initial_row_count}")
        
        # Remove rows where target is null
        target_nulls_before = df.select(pl.col(target_normalized).is_null().sum()).item()
        if target_nulls_before > 0:
            print(f"[Step 10] Found {target_nulls_before} rows with null target, removing...")
            df = df.filter(pl.col(target_normalized).is_not_null())
        
        # Final row count
        final_row_count = df.height
        print(f"[Step 10] Final rows after cleaning: {final_row_count}")
        
        if final_row_count == 0:
            raise ValueError("No rows remaining after cleansing (all target values were null)")
        
        # Compute null rates per column
        null_rate = {}
        for col in df.columns:
            null_count = df.select(pl.col(col).is_null().sum()).item()
            null_rate[col] = float(null_count) / final_row_count
        
        # Try to coerce numeric columns (optional: attempt to parse numeric strings)
        # For appliances energy data, columns should already be numeric or dates
        
        # Export to parquet
        parquet_path = str(output_path / "cleaned.parquet")
        df.write_parquet(parquet_path)
        print(f"[Step 10] Exported cleaned parquet to: {parquet_path}")
        
        # Build step output
        step_output = {
            "step": "10-csv-read-cleansing",
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "row_count_initial": initial_row_count,
            "row_count_after": final_row_count,
            "rows_removed": initial_row_count - final_row_count,
            "column_count": df.width,
            "target_column": target_column,
            "target_column_normalized": target_normalized,
            "columns_normalized": normalization_applied,
            "null_rate": null_rate,
            "null_rows_removed": target_nulls_before,
            "schema": {col: str(dtype) for col, dtype in df.schema.items()},
            "artifacts": {
                "cleaned_parquet": parquet_path
            },
            "notes": [
                f"Loaded {initial_row_count} rows from CSV",
                f"Target column: {target_normalized}",
                f"Removed {target_nulls_before} rows with null target values",
                f"Final dataset: {final_row_count} rows × {df.width} columns"
            ]
        }
        
        return step_output
        
    except Exception as e:
        print(f"[Step 10 ERROR] {type(e).__name__}: {str(e)}", file=sys.stderr)
        raise

def update_progress_json(output_dir: str, step_output: dict, status: str, run_id: str, target_column: str) -> None:
    """Update progress.json with current step status."""
    progress_path = Path(output_dir) / "progress.json"
    
    try:
        with open(progress_path, 'r') as f:
            progress = json.load(f)
    except FileNotFoundError:
        progress = {
            "run_id": run_id,
            "target_column": target_column,
            "completed_steps": [],
            "errors": []
        }
    
    progress["current_step"] = "10-csv-read-cleansing"
    progress["status"] = status
    
    if status == "completed":
        if "10-csv-read-cleansing" not in progress["completed_steps"]:
            progress["completed_steps"].append("10-csv-read-cleansing")
    elif status == "error":
        progress["errors"].append({
            "step": "10-csv-read-cleansing",
            "message": step_output.get("error", "Unknown error")
        })
    
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="Step 10: CSV Read & Cleansing"
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--target-column",
        required=True,
        help="Name of target column for regression"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for artifacts"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier"
    )
    
    args = parser.parse_args()
    
    try:
        # Execute step
        step_output = load_and_clean_csv(
            csv_path=args.csv_path,
            target_column=args.target_column,
            output_dir=args.output_dir,
            run_id=args.run_id
        )
        
        # Write step output
        output_json_path = Path(args.output_dir) / "step-10-cleanse.json"
        with open(output_json_path, 'w') as f:
            json.dump(step_output, f, indent=2)
        print(f"[Step 10] Output written to: {output_json_path}")
        
        # Update progress
        update_progress_json(
            output_dir=args.output_dir,
            step_output=step_output,
            status="completed",
            run_id=args.run_id,
            target_column=args.target_column
        )
        
        print("[Step 10] ✓ Completed successfully")
        sys.exit(0)
        
    except Exception as e:
        print(f"[Step 10] ✗ Failed: {str(e)}", file=sys.stderr)
        step_output = {
            "step": "10-csv-read-cleansing",
            "error": str(e),
            "status": "error"
        }
        update_progress_json(
            output_dir=args.output_dir,
            step_output=step_output,
            status="error",
            run_id=args.run_id,
            target_column=args.target_column
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
