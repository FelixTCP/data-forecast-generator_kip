#!/usr/bin/env python3
"""
Step 10: CSV Read & Cleansing

Load customer CSV robustly and produce a typed, clean polars.DataFrame exported to Parquet.
Includes extreme anomaly detection (|z-score| > 6) and outlier detection (IQR).
"""

import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Tuple, Dict, Any

import polars as pl
import numpy as np


def normalize_column_names(df: pl.DataFrame) -> Tuple[pl.DataFrame, bool]:
    """Normalize column names: strip, lowercase, replace spaces with underscores."""
    original_cols = df.columns
    normalized_cols = [c.strip().lower().replace(" ", "_") for c in original_cols]
    
    if normalized_cols != original_cols:
        df = df.rename(dict(zip(original_cols, normalized_cols)))
        return df, True
    return df, False


def detect_time_column(df: pl.DataFrame) -> str | None:
    """Detect time column by dtype or column name."""
    # First check by dtype
    for col in df.columns:
        if df[col].dtype in [pl.Date, pl.Datetime, pl.Time]:
            return col
    
    # Then check by name pattern
    for col in df.columns:
        col_lower = col.lower()
        if "date" in col_lower or "time" in col_lower:
            return col
    
    return None


def compute_zscore_anomalies(series: pl.Series) -> Tuple[list[int], float, float]:
    """Find extreme anomalies (|z-score| > 6). Returns (indices, mean, std)."""
    # Convert to numpy for computation
    arr = series.drop_nulls().to_numpy(use_pyarrow=False, allow_copy=True)
    
    if len(arr) == 0:
        return [], 0.0, 1.0
    
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    
    if std == 0:
        return [], mean, std
    
    # Compute z-scores
    z_scores = np.abs((arr - mean) / std)
    extreme_indices = np.where(z_scores > 6)[0]
    
    return extreme_indices.tolist(), mean, std


def smooth_extreme_anomalies(df: pl.DataFrame, numeric_cols: list[str]) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Null out extreme anomalies (|z-score| > 6) and interpolate."""
    fixes = []
    
    for col in numeric_cols:
        # Get the column
        series = df[col]
        
        # Find extreme anomalies
        extreme_indices, mean, std = compute_zscore_anomalies(series)
        
        if len(extreme_indices) == 0:
            continue
        
        # Get the null mask for the full column (including nulls)
        full_series = df[col]
        mask = pl.Series([False] * len(df))
        
        # Mark extreme anomalies as null
        for idx in extreme_indices:
            mask = mask.to_list()
            mask[idx] = True
            mask = pl.Series(mask)
        
        # Create a new column with extremes nulled
        new_col = pl.when(mask).then(None).otherwise(full_series)
        df = df.with_columns(new_col.alias(col))
        
        # Interpolate
        df = df.with_columns(pl.col(col).interpolate())
        df = df.with_columns(pl.col(col).forward_fill())
        df = df.with_columns(pl.col(col).backward_fill())
        
        fixes.append({
            "type": "extreme_anomaly_smoothed",
            "column": col,
            "zscore_threshold": 6,
            "count": len(extreme_indices)
        })
    
    return df, fixes


def compute_outliers_iqr(df: pl.DataFrame, numeric_cols: list[str]) -> Dict[str, Dict[str, Any]]:
    """Compute outlier statistics using IQR method."""
    outliers = {}
    
    for col in numeric_cols:
        series = df[col].drop_nulls()
        
        if len(series) == 0:
            continue
        
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Count outliers
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        iqr_outlier_count = int(outlier_mask.sum())
        
        # Z-score outliers
        arr = series.to_numpy(use_pyarrow=False, allow_copy=True)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std > 0:
            z_scores = np.abs((arr - mean) / std)
            zscore_outlier_count = int(np.sum(z_scores > 3))
        else:
            zscore_outlier_count = 0
        
        outlier_fraction = iqr_outlier_count / len(series) if len(series) > 0 else 0.0
        
        # Get outlier indices (sample first 200)
        outlier_indices = []
        for idx, val in enumerate(series.to_list()):
            if val is not None and (val < lower_bound or val > upper_bound):
                outlier_indices.append(idx)
                if len(outlier_indices) >= 200:
                    break
        
        outliers[col] = {
            "iqr_outlier_count": iqr_outlier_count,
            "zscore_outlier_count": zscore_outlier_count,
            "iqr_lower_bound": float(lower_bound),
            "iqr_upper_bound": float(upper_bound),
            "outlier_fraction": float(outlier_fraction),
            "outlier_indices_sample": outlier_indices
        }
    
    return outliers


def load_and_clean_csv(csv_path: str, output_dir: str, target_column: str) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Load and clean CSV file."""
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load with polars lazy evaluation
    lf = pl.scan_csv(str(csv_path), try_parse_dates=True)
    
    # Collect to DataFrame
    df = lf.collect()
    
    row_count_initial = len(df)
    column_count = len(df.columns)
    
    # Normalize column names
    df, names_normalized = normalize_column_names(df)
    
    # Detect target column normalized name
    target_column_normalized = target_column.strip().lower().replace(" ", "_")
    
    # Verify target exists
    if target_column_normalized not in df.columns:
        raise ValueError(f"Target column '{target_column_normalized}' not found in CSV. Available: {df.columns}")
    
    # Detect time column
    time_column = detect_time_column(df)
    
    # Compute null rates before
    initial_null_rate = {
        col: float(df[col].is_null().mean()) for col in df.columns
    }
    
    # Remove duplicate rows
    df_before_dedup = df
    df = df.unique()
    duplicate_rows_removed = len(df_before_dedup) - len(df)
    
    # Sort by time column (mandatory after dedup)
    if time_column is not None:
        df = df.sort(time_column)
    
    # Smooth extreme anomalies
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]]
    df, anomaly_fixes = smooth_extreme_anomalies(df, numeric_cols)
    
    # Compute outliers
    outliers = compute_outliers_iqr(df, numeric_cols)
    
    row_count_after = len(df)
    
    # Inferred dtypes
    inferred_dtypes = {col: str(df[col].dtype) for col in df.columns}
    
    # Compute null rates after
    final_null_rate = {
        col: float(df[col].is_null().mean()) for col in df.columns
    }
    
    # Build quality report
    quality_report = {
        "step": "10-csv-read-cleansing",
        "row_count_initial": row_count_initial,
        "row_count_after": row_count_after,
        "column_count": column_count,
        "target_column_normalized": target_column_normalized,
        "time_column_detected": time_column,
        "null_rate": final_null_rate,
        "duplicate_rows_removed": duplicate_rows_removed,
        "inferred_dtypes": inferred_dtypes,
        "outliers": outliers,
        "fixes": [
            {"type": "normalized_column_names"} if names_normalized else None,
            {"type": "removed_duplicates", "count": duplicate_rows_removed} if duplicate_rows_removed > 0 else None
        ] + anomaly_fixes,
        "artifacts": {
            "cleaned_parquet": str(Path(output_dir) / "cleaned.parquet")
        }
    }
    
    # Clean up None values in fixes list
    quality_report["fixes"] = [f for f in quality_report["fixes"] if f is not None]
    
    return df, quality_report


def main():
    parser = argparse.ArgumentParser(description="Step 10: CSV Read & Cleansing")
    parser.add_argument("--csv-path", required=True, help="Path to input CSV file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--target-column", required=True, help="Target column name")
    
    args = parser.parse_args()
    
    try:
        # Update progress
        progress_path = Path(args.output_dir) / "progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
        else:
            progress = {
                "run_id": args.run_id,
                "status": "running",
                "current_step": "10-csv-read-cleansing",
                "completed_steps": []
            }
        
        progress["status"] = "running"
        progress["current_step"] = "10-csv-read-cleansing"
        progress_path.write_text(json.dumps(progress, indent=2))
        
        # Load and clean
        df, quality_report = load_and_clean_csv(
            args.csv_path,
            args.output_dir,
            args.target_column
        )
        
        # Write parquet
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_path = output_dir / "cleaned.parquet"
        df.write_parquet(str(parquet_path))
        
        # Write step JSON
        step_json_path = output_dir / "step-10-cleanse.json"
        step_json_path.write_text(json.dumps(quality_report, indent=2))
        
        # Update progress
        progress["status"] = "running"
        progress["completed_steps"] = ["10-csv-read-cleansing"]
        progress_path.write_text(json.dumps(progress, indent=2))
        
        print(f"Step 10 completed: {quality_report['row_count_after']} rows, {quality_report['column_count']} columns")
        sys.exit(0)
        
    except Exception as e:
        print(f"Step 10 failed: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
