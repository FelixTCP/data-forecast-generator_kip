#!/usr/bin/env python3
"""
Step 10: CSV Read & Cleansing
Load CSV, cleanse, detect columns, handle anomalies, and export to Parquet.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import polars as pl
import numpy as np


def detect_time_column(df: pl.DataFrame) -> Optional[str]:
    """
    Detect the time column by dtype or name pattern.
    Returns column name if found, None otherwise.
    """
    # First check for datetime dtype
    for col in df.columns:
        if df[col].dtype in [pl.Date, pl.Datetime]:
            return col
    
    # Then check for name patterns
    time_keywords = ["date", "time", "datetime", "timestamp"]
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in time_keywords):
            return col
    
    return None


def synthesize_date_column(df: pl.DataFrame) -> Tuple[pl.DataFrame, Optional[str]]:
    """
    If date columns (year, month, day) exist, synthesize a date column.
    Returns (modified dataframe, new column name or None).
    """
    # Check for year/month/day columns
    year_col = next((c for c in df.columns if c.lower() == "year"), None)
    month_col = next((c for c in df.columns if c.lower() == "month"), None)
    day_col = next((c for c in df.columns if c.lower() == "day"), None)
    
    if year_col and month_col and day_col:
        # Synthesize date using polars.date()
        df = df.with_columns(
            pl.date(pl.col(year_col), pl.col(month_col), pl.col(day_col))
            .alias("_synthesized_date")
        )
        return df, "_synthesized_date"
    
    return df, None


def normalize_column_names(columns: list[str]) -> list[str]:
    """Normalize column names: strip, lowercase, replace spaces with underscores."""
    normalized = []
    for col in columns:
        col = col.strip().lower().replace(" ", "_").replace("-", "_")
        normalized.append(col)
    return normalized


def compute_extreme_anomalies(series: pl.Series, threshold: float = 6.0) -> Dict[str, Any]:
    """
    Detect extreme anomalies using z-score.
    Returns dict with count, indices, and z-scores.
    """
    # Convert to numpy for z-score computation
    values = series.drop_nulls().to_numpy()
    if len(values) == 0:
        return {"count": 0, "indices": []}
    
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return {"count": 0, "indices": []}
    
    z_scores = np.abs((values - mean) / std)
    anomaly_indices = np.where(z_scores > threshold)[0]
    
    return {
        "count": len(anomaly_indices),
        "indices": anomaly_indices.tolist()[:20],  # Sample first 20
        "threshold": threshold
    }


def handle_extreme_anomalies(df: pl.DataFrame, numeric_cols: list[str], threshold: float = 6.0) -> Tuple[pl.DataFrame, list[str]]:
    """
    Detect and interpolate extreme anomalies (|z-score| > threshold).
    Returns (modified dataframe, list of fixes applied).
    """
    fixes = []
    
    for col in numeric_cols:
        series = df[col]
        if series.dtype not in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            continue
        
        # Compute z-scores
        values = series.drop_nulls().to_numpy()
        if len(values) == 0:
            continue
        
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            continue
        
        # Find extreme anomalies
        z_scores = np.abs((values - mean) / std)
        extreme_mask = z_scores > threshold
        anomaly_count = np.sum(extreme_mask)
        
        if anomaly_count > 0:
            # Null out extremes and interpolate
            df = df.with_columns(
                pl.when(
                    pl.col(col).is_not_null() & 
                    ((pl.col(col) - pl.col(col).mean()).abs() / (pl.col(col).std() + 1e-8) > threshold)
                ).then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
            
            # Interpolate
            df = df.with_columns(
                pl.col(col).interpolate().forward_fill().backward_fill()
            )
            
            fixes.append(f"extreme_anomaly_smoothed: col='{col}', zscore_threshold={threshold}, count={anomaly_count}")
    
    return df, fixes


def compute_outlier_statistics(df: pl.DataFrame, numeric_cols: list[str]) -> Dict[str, Dict[str, Any]]:
    """
    Compute IQR-based outlier statistics for all numeric columns.
    """
    outliers_report = {}
    
    for col in numeric_cols:
        series = df[col]
        values = series.drop_nulls().to_numpy()
        
        if len(values) == 0:
            continue
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_count = np.sum(outlier_mask)
        outlier_fraction = outlier_count / len(values) if len(values) > 0 else 0
        
        # Get first 200 outlier indices
        outlier_indices = np.where(outlier_mask)[0].tolist()[:200]
        
        # Z-score outliers
        mean = np.mean(values)
        std = np.std(values)
        if std > 0:
            z_scores = np.abs((values - mean) / std)
            zscore_outlier_count = np.sum(z_scores > 3)
        else:
            zscore_outlier_count = 0
        
        outliers_report[col] = {
            "iqr_outlier_count": int(outlier_count),
            "zscore_outlier_count": int(zscore_outlier_count),
            "iqr_lower_bound": float(lower_bound),
            "iqr_upper_bound": float(upper_bound),
            "outlier_fraction": float(outlier_fraction),
            "outlier_indices_sample": outlier_indices
        }
    
    return outliers_report


def step_10_main(
    csv_path: str,
    output_dir: str,
    run_id: str,
    target_column: str
) -> int:
    """
    Main step 10 logic.
    Returns 0 on success, non-zero on failure.
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        progress_file = output_path / "progress.json"
        
        # Load CSV with polars
        print(f"[Step 10] Loading CSV from: {csv_path}")
        df = pl.scan_csv(csv_path, try_parse_dates=True).collect()
        
        row_count_initial = df.height
        print(f"[Step 10] Initial row count: {row_count_initial}")
        
        # Normalize column names
        original_columns = df.columns
        normalized_columns = normalize_column_names(original_columns)
        df = df.rename(dict(zip(original_columns, normalized_columns)))
        
        # Normalize target column name
        target_normalized = normalize_column_names([target_column])[0]
        
        # Detect time column (before or after synthesis)
        time_column = detect_time_column(df)
        
        # Try to synthesize date if not found
        if not time_column:
            df, synthesized = synthesize_date_column(df)
            if synthesized:
                time_column = synthesized
                print(f"[Step 10] Synthesized date column: {time_column}")
        
        if not time_column:
            raise RuntimeError(
                "No time column detected and no year/month/day columns found. "
                "Cannot guarantee chronological order. Aborting step 10."
            )
        
        print(f"[Step 10] Detected time column: {time_column}")
        
        # Identify numeric columns
        numeric_cols = [c for c in df.columns if df[c].dtype in 
                       [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        
        # Handle extreme anomalies
        print(f"[Step 10] Checking for extreme anomalies (z-score > 6)...")
        df, anomaly_fixes = handle_extreme_anomalies(df, numeric_cols, threshold=6.0)
        
        # Remove duplicates
        df_before = df.height
        df = df.unique()
        df_after = df.height
        duplicate_rows_removed = df_before - df_after
        
        if duplicate_rows_removed > 0:
            print(f"[Step 10] Removed {duplicate_rows_removed} duplicate rows")
        
        # Compute null rates
        null_rates = {}
        for col in df.columns:
            null_rate = df[col].is_null().mean()
            null_rates[col] = float(null_rate)
        
        print(f"[Step 10] Null rates computed")
        
        # Compute outlier statistics
        print(f"[Step 10] Computing outlier statistics...")
        outliers = compute_outlier_statistics(df, numeric_cols)
        
        # Get inferred dtypes
        inferred_dtypes = {col: str(df[col].dtype) for col in df.columns}
        
        # MANDATORY: Sort by time column LAST
        print(f"[Step 10] Sorting by time column: {time_column}")
        df = df.sort(time_column)
        
        # Write cleaned parquet
        cleaned_parquet = output_path / "cleaned.parquet"
        df.write_parquet(cleaned_parquet)
        print(f"[Step 10] Wrote cleaned.parquet: {cleaned_parquet}")
        
        # Prepare output JSON
        fixes = anomaly_fixes.copy()
        if duplicate_rows_removed > 0:
            fixes.append(f"removed_duplicates: count={duplicate_rows_removed}")
        fixes.append(f"final_chronological_sort_by={time_column}")
        if synthesized:
            fixes.append("synthesized_date_column")
        
        output_json = {
            "step": "10-csv-read-cleansing",
            "run_id": run_id,
            "row_count_initial": row_count_initial,
            "row_count_after": df.height,
            "column_count": df.width,
            "target_column_normalized": target_normalized,
            "time_column_detected": time_column,
            "null_rate": null_rates,
            "duplicate_rows_removed": duplicate_rows_removed,
            "inferred_dtypes": inferred_dtypes,
            "outliers": outliers,
            "sorted_by": time_column,
            "fixes": fixes,
            "artifacts": {
                "cleaned_parquet": str(cleaned_parquet)
            }
        }
        
        # Write output JSON
        step_json = output_path / "step-10-cleanse.json"
        with open(step_json, "w") as f:
            json.dump(output_json, f, indent=2)
        print(f"[Step 10] Wrote step-10-cleanse.json")
        
        # Update progress
        progress = {
            "run_id": run_id,
            "csv_path": csv_path,
            "target_column": target_column,
            "status": "running",
            "current_step": "11-data-exploration",
            "completed_steps": ["10-csv-read-cleansing"],
            "errors": []
        }
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)
        
        print(f"[Step 10] ✓ Completed successfully")
        return 0
        
    except Exception as e:
        print(f"[Step 10] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 10: CSV Read & Cleansing")
    parser.add_argument("--csv-path", required=True, help="Path to input CSV file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--target-column", required=True, help="Target column name")
    
    args = parser.parse_args()
    
    sys.exit(step_10_main(args.csv_path, args.output_dir, args.run_id, args.target_column))
