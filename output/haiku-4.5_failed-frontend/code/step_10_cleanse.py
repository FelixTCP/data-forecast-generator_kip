#!/usr/bin/env python
"""
STEP 10 — CSV Read & Cleansing

Loads customer CSV robustly, produces a typed, clean polars.DataFrame,
detects time column, applies extreme anomaly smoothing, and exports to Parquet.

Exit code: 0 on success, non-zero on failure.
"""

import sys
import json
import argparse
import polars as pl
import polars.selectors as cs
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def detect_time_column(df: pl.DataFrame) -> Optional[str]:
    """
    Detect the time column by:
    1. Looking for a datetime/date typed column
    2. Looking for column names containing 'date', 'time', 'datetime'
    3. Looking for year/month/day integer columns to synthesize a date
    """
    # Check existing datetime columns
    for col in df.columns:
        if df[col].dtype in [pl.Date, pl.Datetime]:
            logger.info(f"Detected datetime column: {col}")
            return col
    
    # Check column names for date/time keywords
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['date', 'time', 'datetime']):
            logger.info(f"Detected time column by name pattern: {col}")
            return col
    
    # Look for year/month/day columns to synthesize
    year_col = next((c for c in df.columns if c.lower() == 'year'), None)
    month_col = next((c for c in df.columns if c.lower() == 'month'), None)
    day_col = next((c for c in df.columns if c.lower() == 'day'), None)
    
    if year_col and month_col and day_col:
        logger.info(f"Detected year/month/day columns: {year_col}, {month_col}, {day_col}")
        return None  # Will synthesize below
    
    logger.warning("No time column detected")
    return None

def normalize_column_names(columns: list) -> list:
    """Normalize column names: strip, lowercase, space→underscore"""
    return [c.strip().lower().replace(" ", "_") for c in columns]

def synthesize_date_column(df: pl.DataFrame) -> Tuple[pl.DataFrame, Optional[str]]:
    """
    If year/month/day columns exist, synthesize a date column.
    Return (updated_df, date_column_name or None)
    """
    year_col = next((c for c in df.columns if c.lower() == 'year'), None)
    month_col = next((c for c in df.columns if c.lower() == 'month'), None)
    day_col = next((c for c in df.columns if c.lower() == 'day'), None)
    
    if year_col and month_col and day_col:
        logger.info(f"Synthesizing date column from {year_col}, {month_col}, {day_col}")
        df = df.with_columns(
            pl.date(pl.col(year_col), pl.col(month_col), pl.col(day_col)).alias("synthesized_date")
        )
        return df, "synthesized_date"
    
    return df, None

def smooth_extreme_anomalies(df: pl.DataFrame, time_col: str) -> Tuple[pl.DataFrame, list]:
    """
    Detect values with |z-score| > 6 and replace with linear interpolation.
    Return (cleaned_df, fixes_list)
    """
    fixes = []
    # Get all numeric columns
    numeric_cols = df.select(cs.numeric()).columns
    
    for col in numeric_cols:
        series = df[col]
        mean = series.mean()
        std = series.std()
        
        if std is None or std == 0:
            continue
        
        # Compute z-scores
        z_scores = (series - mean) / std
        
        # Detect anomalies
        anomaly_mask = z_scores.abs() > 6
        anomaly_count = anomaly_mask.sum()
        
        if anomaly_count > 0:
            logger.info(f"Found {anomaly_count} extreme anomalies in {col}")
            # Null out anomalies and interpolate
            df = df.with_columns(
                pl.when(anomaly_mask)
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
            # Interpolate, forward fill, backward fill
            df = df.with_columns(
                pl.col(col)
                .interpolate()
                .forward_fill()
                .backward_fill()
                .alias(col)
            )
            fixes.append(f"extreme_anomaly_smoothed: col='{col}', zscore_threshold=6, count={anomaly_count}")
    
    return df, fixes

def compute_outlier_stats(df: pl.DataFrame) -> dict:
    """
    Compute IQR-based and z-score based outlier statistics.
    """
    outliers = {}
    numeric_cols = df.select(cs.numeric()).columns
    
    for col in numeric_cols:
        series = df[col]
        
        # IQR method
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_indices = [i for i, is_outlier in enumerate(outlier_mask) if is_outlier][:200]
        
        # Z-score method
        mean = series.mean()
        std = series.std()
        if std is not None and std > 0:
            z_scores = (series - mean) / std
            zscore_outlier_count = (z_scores.abs() > 3).sum()
        else:
            zscore_outlier_count = 0
        
        outliers[col] = {
            "iqr_outlier_count": len(outlier_indices),
            "zscore_outlier_count": int(zscore_outlier_count),
            "iqr_lower_bound": float(lower_bound),
            "iqr_upper_bound": float(upper_bound),
            "outlier_fraction": len(outlier_indices) / df.height if df.height > 0 else 0.0,
            "outlier_indices_sample": outlier_indices
        }
    
    return outliers

def load_and_clean_csv(
    csv_path: str,
    target_column: str,
    output_dir: str,
    run_id: str
) -> Tuple[pl.DataFrame, dict]:
    """
    Load CSV with polars, apply cleansing, detect time column, 
    smooth anomalies, and export to Parquet.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading CSV from {csv_path}")
    
    # Load with try_parse_dates to auto-detect datetime
    lf = pl.scan_csv(csv_path, try_parse_dates=True)
    initial_columns = lf.columns
    logger.info(f"Initial columns: {initial_columns}")
    logger.info(f"Initial row count: {len(initial_columns)} columns")
    
    # Normalize column names
    normalized = normalize_column_names(initial_columns)
    fixes = []
    
    if normalized != initial_columns:
        logger.info(f"Normalizing column names")
        lf = lf.rename(dict(zip(initial_columns, normalized)))
        fixes.append("normalized_column_names")
    
    # Collect to DataFrame
    df = lf.collect()
    row_count_initial = df.height
    logger.info(f"Initial row count: {row_count_initial}")
    
    # Remove duplicates
    df_dedup = df.unique()
    duplicates_removed = row_count_initial - df_dedup.height
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows")
        df = df_dedup
        fixes.append(f"removed_duplicates: count={duplicates_removed}")
    
    # Detect time column (before synthesizing)
    time_col = detect_time_column(df)
    
    # Try to synthesize date if not found
    if time_col is None:
        df, synth_time = synthesize_date_column(df)
        time_col = synth_time
    
    if time_col is None:
        raise RuntimeError(
            "No time column detected and could not synthesize from year/month/day. "
            "Cannot guarantee chronological order. Aborting step 10."
        )
    
    logger.info(f"Using time column: {time_col}")
    
    # Sort by time BEFORE anomaly smoothing
    df = df.sort(time_col)
    logger.info(f"Sorted by {time_col}")
    
    # Smooth extreme anomalies
    df, anomaly_fixes = smooth_extreme_anomalies(df, time_col)
    fixes.extend(anomaly_fixes)
    
    # Normalize target column name (for contract compatibility)
    target_normalized = target_column.strip().lower().replace(" ", "_")
    logger.info(f"Target column normalized: {target_column} → {target_normalized}")
    
    # Verify target column exists
    if target_normalized not in df.columns:
        raise ValueError(f"Target column '{target_normalized}' not found in cleaned dataframe. Available: {df.columns}")
    
    # Compute null rates
    null_rates = {
        c: float(df.select(pl.col(c).is_null().mean()).item())
        for c in df.columns
    }
    
    # Compute inferred dtypes
    inferred_dtypes = {c: str(df[c].dtype) for c in df.columns}
    
    # Compute outlier statistics
    outlier_stats = compute_outlier_stats(df)
    
    # Final chronological sort (MANDATORY LAST OPERATION)
    df = df.sort(time_col)
    fixes.append(f"final_chronological_sort_by={time_col}")
    
    # Write to Parquet
    cleaned_parquet = output_dir / "cleaned.parquet"
    df.write_parquet(str(cleaned_parquet))
    logger.info(f"Wrote cleaned data to {cleaned_parquet}")
    
    # Build output JSON
    output_json = {
        "step": "10-csv-read-cleansing",
        "run_id": run_id,
        "row_count_initial": row_count_initial,
        "row_count_after": df.height,
        "column_count": df.width,
        "target_column_normalized": target_normalized,
        "time_column_detected": time_col,
        "null_rate": null_rates,
        "duplicate_rows_removed": duplicates_removed,
        "inferred_dtypes": inferred_dtypes,
        "outliers": outlier_stats,
        "sorted_by": time_col,
        "fixes": fixes,
        "artifacts": {
            "cleaned_parquet": str(cleaned_parquet)
        }
    }
    
    return df, output_json

def main():
    parser = argparse.ArgumentParser(description="STEP 10 — CSV Read & Cleansing")
    parser.add_argument("--csv-path", required=True, help="Path to input CSV")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    try:
        # Initialize progress
        output_dir = Path(args.output_dir)
        progress_file = output_dir / "progress.json"
        progress = {
            "run_id": args.run_id,
            "csv_path": args.csv_path,
            "target_column": args.target_column,
            "status": "running",
            "current_step": "10-csv-read-cleansing",
            "completed_steps": [],
            "errors": []
        }
        progress_file.write_text(json.dumps(progress, indent=2))
        
        # Run cleansing
        df, output_json = load_and_clean_csv(
            args.csv_path,
            args.target_column,
            args.output_dir,
            args.run_id
        )
        
        # Write output JSON
        step_json_path = output_dir / "step-10-cleanse.json"
        step_json_path.write_text(json.dumps(output_json, indent=2))
        logger.info(f"Wrote step JSON to {step_json_path}")
        
        # Update progress
        progress["status"] = "completed"
        progress["completed_steps"] = ["10-csv-read-cleansing"]
        progress["current_step"] = "11-data-exploration"
        progress_file.write_text(json.dumps(progress, indent=2))
        
        logger.info("STEP 10 completed successfully")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"STEP 10 failed: {e}", exc_info=True)
        progress_file = Path(args.output_dir) / "progress.json"
        progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
        progress["status"] = "error"
        progress["errors"] = progress.get("errors", []) + [str(e)]
        progress_file.write_text(json.dumps(progress, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
