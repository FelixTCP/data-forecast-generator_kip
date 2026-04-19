#!/usr/bin/env python3
"""
Step 00: Data Profiling & Pre-Exploration
Safely inspect the CSV without loading entirely into memory.
"""

import json
import os
import sys
import argparse
import csv
from pathlib import Path
from typing import Any

def count_csv_rows(csv_path: str) -> int:
    """Count rows in CSV file without loading it entirely."""
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        return sum(1 for _ in f) - 1  # Subtract 1 for header

def sample_csv_rows(csv_path: str, sample_size: int = 10) -> tuple[list[str], list[list[str]]]:
    """Read header and a random sample of rows."""
    rows_list = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        all_rows = list(reader)
    
    # Sample: first 5, last 5, and random middle rows if available
    sample_rows = []
    if len(all_rows) >= sample_size:
        # First 5
        sample_rows.extend(all_rows[:5])
        # Last 5
        sample_rows.extend(all_rows[-5:])
        # Random middle rows (if more than 10 rows)
        if len(all_rows) > 10:
            import random
            random.seed(42)
            middle_rows = random.sample(all_rows[5:-5], min(sample_size - 10, len(all_rows) - 10))
            sample_rows.extend(middle_rows)
    else:
        sample_rows = all_rows
    
    return header, sample_rows

def infer_column_types(header: list[str], sample_rows: list[list[str]]) -> dict[str, str]:
    """Infer column types from sample rows."""
    inferred_types = {}
    
    for col_idx, col_name in enumerate(header):
        # Collect non-null values
        values = [row[col_idx] for row in sample_rows if col_idx < len(row) and row[col_idx].strip()]
        
        if not values:
            inferred_types[col_name] = "unknown (all null)"
            continue
        
        # Try integer
        try:
            for v in values:
                int(v)
            inferred_types[col_name] = "integer"
            continue
        except (ValueError, AttributeError):
            pass
        
        # Try float
        try:
            for v in values:
                float(v)
            inferred_types[col_name] = "float"
            continue
        except (ValueError, AttributeError):
            pass
        
        # Check for date-like
        if any(c in values[0] for c in ['-', '/', ':']):
            inferred_types[col_name] = "datetime-candidate"
            continue
        
        # Default to string
        inferred_types[col_name] = "string"
    
    return inferred_types

def detect_anomalies(csv_path: str, header: list[str], sample_rows: list[list[str]]) -> list[str]:
    """Detect structural anomalies."""
    anomalies = []
    expected_cols = len(header)
    
    # Check for inconsistent column counts
    for i, row in enumerate(sample_rows):
        if len(row) != expected_cols:
            anomalies.append(f"Row {i} has {len(row)} columns, expected {expected_cols}")
    
    # Check for null-heavy columns
    for col_idx, col_name in enumerate(header):
        null_count = sum(1 for row in sample_rows if col_idx >= len(row) or not row[col_idx].strip())
        if sample_rows and null_count / len(sample_rows) > 0.5:
            anomalies.append(f"Column '{col_name}' is >{50}% null in sample")
    
    return anomalies

def generate_profile(csv_path: str, output_dir: str, run_id: str) -> None:
    """Main profiling function."""
    try:
        # Validate CSV exists
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Read structure
        header, sample_rows = sample_csv_rows(csv_path)
        row_count = count_csv_rows(csv_path)
        inferred_types = infer_column_types(header, sample_rows)
        anomalies = detect_anomalies(csv_path, header, sample_rows)
        
        # Generate profiler JSON
        profiler_data = {
            "run_id": run_id,
            "step": "00-pre-exploration",
            "csv_path": csv_path,
            "file_size_bytes": os.path.getsize(csv_path),
            "total_rows": row_count,
            "total_columns": len(header),
            "header": header,
            "inferred_types": inferred_types,
            "sample_rows_count": len(sample_rows),
            "sample_rows": sample_rows[:10],  # First 10 for brevity
            "anomalies": anomalies,
            "recommendations": [
                "Use polars.scan_csv(try_parse_dates=True) for lazy evaluation",
                "Normalize column names (lowercase, underscores)",
                "Check for missing value representations (empty, 'NA', 'N/A', etc)",
            ]
        }
        
        profiler_path = os.path.join(output_dir, "step-00_profiler.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(profiler_path, 'w') as f:
            json.dump(profiler_data, f, indent=2)
        
        print(f"✓ Profiler JSON written to {profiler_path}")
        
        # Generate profiler report markdown
        report_md = f"""# Data Profile Report

## File Metadata
- **File Path**: {csv_path}
- **File Size**: {os.path.getsize(csv_path):,} bytes
- **Total Rows**: {row_count:,}
- **Total Columns**: {len(header)}

## Column Summary
| Column | Inferred Type | Notes |
|--------|---------------|-------|
"""
        for col in header:
            col_type = inferred_types.get(col, "unknown")
            report_md += f"| {col} | {col_type} | |\n"
        
        report_md += f"""

## Detected Anomalies
"""
        if anomalies:
            for anomaly in anomalies:
                report_md += f"- {anomaly}\n"
        else:
            report_md += "- None detected in sample\n"
        
        report_md += """

## Recommended Cleansing Steps (for Step 01)
1. Normalize column names: strip whitespace, convert to lowercase, replace spaces with underscores
2. Attempt date parsing: use `polars.scan_csv(try_parse_dates=True)` 
3. Validate numeric columns: attempt coercion, flag conversion failures
4. Check for duplicate rows: identify and log (do not silently drop)
5. Assess null patterns: log null rate per column
6. No silent drops: log row count before/after any filtering

## Sample Data (first 5 rows)
"""
        if sample_rows:
            report_md += f"\n```\n{header}\n"
            for row in sample_rows[:5]:
                report_md += f"{row}\n"
            report_md += "```\n"
        
        report_path = os.path.join(output_dir, "step-00_data_profile_report.md")
        with open(report_path, 'w') as f:
            f.write(report_md)
        
        print(f"✓ Profile report written to {report_path}")
        
        # Update progress.json
        progress_path = os.path.join(output_dir, "progress.json")
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        
        progress["current_step"] = "00-pre-exploration"
        progress["completed_steps"].append("00-pre-exploration")
        
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("✓ Step 00 completed successfully")
        sys.exit(0)
    
    except Exception as e:
        print(f"✗ Step 00 failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 00: Data Profiling")
    parser.add_argument("--csv-path", required=True, help="Path to input CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory for artifacts")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    
    args = parser.parse_args()
    
    generate_profile(args.csv_path, args.output_dir, args.run_id)
