#!/usr/bin/env python
"""
Step 10: CSV Read & Cleansing
Load raw CSV, detect types, normalize columns, handle anomalies, export cleaned parquet.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl


def detect_time_column(df: pl.DataFrame) -> str | None:
    """Detect time column by name pattern or datetime dtype."""
    # Priority 1: Look for "date" or "time" in column name
    for col in df.columns:
        col_lower = col.lower()
        if "date" in col_lower or "time" in col_lower:
            return col
    
    # Priority 2: Look for datetime dtype
    for col in df.columns:
        if df[col].dtype in [pl.Datetime, pl.Date]:
            return col
    
    # Priority 3: Check for year/month/day pattern
    cols_lower = {c: c.lower() for c in df.columns}
    if "year" in cols_lower.values() and "month" in cols_lower.values() and "day" in cols_lower.values():
        return "year/month/day"
    
    return None


def synthesize_datetime_from_ymd(df: pl.DataFrame) -> pl.DataFrame:
    """If year/month/day cols exist, synthesize a datetime column."""
    year_col = month_col = day_col = None
    for c in df.columns:
        c_lower = c.lower()
        if c_lower == "year":
            year_col = c
        elif c_lower == "month":
            month_col = c
        elif c_lower == "day":
            day_col = c
    
    if year_col and month_col and day_col:
        df = df.with_columns(
            pl.date(pl.col(year_col), pl.col(month_col), pl.col(day_col))
            .alias("_synthesized_date")
        )
        return df
    return df


def normalize_column_names(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Normalize column names: strip, lowercase, replace spaces with underscores."""
    original_cols = df.columns
    normalized_map = {}
    for col in original_cols:
        norm = col.strip().lower().replace(" ", "_")
        normalized_map[col] = norm
    
    df = df.rename(normalized_map)
    changed = [col for orig, col in zip(original_cols, df.columns) if orig != col]
    return df, changed


def detect_extreme_anomalies(df: pl.DataFrame, numeric_cols: list[str], zscore_threshold: float = 6.0) -> dict:
    """Detect and replace extreme anomalies (|z-score| > threshold) with linear interpolation."""
    fixes = []
    
    for col in numeric_cols:
        try:
            col_data = df[col]
            # Compute mean and std
            mean = col_data.mean()
            std = col_data.std()
            
            if std is None or std == 0:
                continue
            
            # Compute z-scores
            z_scores = ((col_data - mean) / std).abs()
            
            # Identify extreme anomalies
            mask = z_scores > zscore_threshold
            count_anomalies = mask.sum()
            
            if count_anomalies > 0:
                # Replace with null, then interpolate
                df = df.with_columns(
                    pl.when(mask)
                    .then(None)
                    .otherwise(pl.col(col))
                    .alias(col)
                )
                
                # Interpolate
                df = df.with_columns(
                    pl.col(col).interpolate().forward_fill().backward_fill()
                )
                
                fixes.append(
                    f"extreme_anomaly_smoothed: col='{col}', zscore_threshold={zscore_threshold}, count={count_anomalies}"
                )
        except Exception as e:
            # Silently skip columns that can't be processed
            pass
    
    return {"anomaly_fixes": fixes, "df": df}


def compute_outlier_stats(df: pl.DataFrame, numeric_cols: list[str]) -> dict:
    """Compute IQR-based outlier statistics for each numeric column."""
    outliers = {}
    
    for col in numeric_cols:
        try:
            col_data = df[col]
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Count IQR outliers
            iqr_mask = (col_data < lower_bound) | (col_data > upper_bound)
            iqr_count = iqr_mask.sum()
            
            # Count z-score outliers (|z| > 3)
            mean = col_data.mean()
            std = col_data.std()
            if std and std > 0:
                z_scores = ((col_data - mean) / std).abs()
                zscore_count = (z_scores > 3).sum()
            else:
                zscore_count = 0
            
            # Get sample outlier indices
            outlier_indices = [i for i, v in enumerate(iqr_mask) if v][:200]
            
            outliers[col] = {
                "iqr_outlier_count": int(iqr_count),
                "zscore_outlier_count": int(zscore_count),
                "iqr_lower_bound": float(lower_bound) if lower_bound is not None else None,
                "iqr_upper_bound": float(upper_bound) if upper_bound is not None else None,
                "outlier_fraction": float(iqr_count / len(col_data)) if len(col_data) > 0 else 0.0,
                "outlier_indices_sample": outlier_indices
            }
        except Exception as e:
            pass
    
    return outliers


def main():
    parser = argparse.ArgumentParser(description="Step 10: CSV Read & Cleansing")
    parser.add_argument("--csv-path", required=True, help="Path to input CSV")
    parser.add_argument("--target-column", required=True, help="Name of target column")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID for progress tracking")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    progress_file = output_dir / "progress.json"
    
    try:
        # Update progress: step started
        progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
        progress.update({
            "current_step": "10-csv-read-cleansing",
            "status": "running"
        })
        progress_file.write_text(json.dumps(progress, indent=2))
        
        # ===== LOAD CSV =====
        print("[Step 10] Loading CSV...")
        try:
            df = pl.read_csv(args.csv_path, try_parse_dates=True)
        except Exception as e:
            print(f"ERROR: Failed to read CSV: {e}")
            raise
        
        row_count_initial = df.height
        print(f"  Initial rows: {row_count_initial}")
        
        # ===== NORMALIZE COLUMN NAMES =====
        df, col_changes = normalize_column_names(df)
        fixes = []
        if col_changes:
            fixes.append("normalized_column_names")
            print(f"  Normalized columns: {col_changes}")
        
        # ===== SYNTHESIZE DATETIME IF NEEDED =====
        df = synthesize_datetime_from_ymd(df)
        
        # ===== DETECT TIME COLUMN =====
        time_column = detect_time_column(df)
        print(f"  Time column detected: {time_column}")
        
        # ===== SORT BY TIME (CHRONOLOGICAL MANDATORY) =====
        if time_column and time_column != "year/month/day":
            try:
                df = df.sort(time_column)
                print(f"  Sorted by {time_column}")
            except Exception as e:
                print(f"WARNING: Could not sort by {time_column}: {e}")
        elif "_synthesized_date" in df.columns:
            df = df.sort("_synthesized_date")
            time_column = "_synthesized_date"
            print(f"  Sorted by synthesized date")
        
        # ===== REMOVE DUPLICATES =====
        row_count_before_dedup = df.height
        df = df.unique()
        duplicates_removed = row_count_before_dedup - df.height
        if duplicates_removed > 0:
            fixes.append("removed_duplicates")
            print(f"  Removed {duplicates_removed} duplicate rows")
        
        # ===== VERIFY CHRONOLOGICAL ORDER PRESERVED =====
        if time_column and time_column in df.columns:
            # Re-sort after dedup to ensure chronological order
            df = df.sort(time_column)
            print(f"  Re-sorted after dedup to ensure chronological order")
        
        # ===== NORMALIZE TARGET COLUMN NAME =====
        target_normalized = args.target_column.strip().lower().replace(" ", "_")
        
        # Check if target exists
        if target_normalized not in df.columns:
            print(f"ERROR: Target column '{target_normalized}' not found in dataset")
            print(f"Available columns: {df.columns}")
            raise ValueError(f"Target column '{target_normalized}' not found")
        
        # ===== DETECT NUMERIC COLUMNS =====
        numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]]
        print(f"  Numeric columns: {numeric_cols}")
        
        # ===== DETECT & REPLACE EXTREME ANOMALIES =====
        anomaly_result = detect_extreme_anomalies(df, numeric_cols, zscore_threshold=6.0)
        df = anomaly_result["df"]
        fixes.extend(anomaly_result["anomaly_fixes"])
        if anomaly_result["anomaly_fixes"]:
            print(f"  Applied anomaly fixes: {len(anomaly_result['anomaly_fixes'])}")
        
        # ===== COMPUTE METRICS =====
        row_count_after = df.height
        column_count = df.width
        
        # Null rates
        null_rate = {}
        for col in df.columns:
            null_count = df[col].is_null().sum()
            null_pct = float(null_count / row_count_after) if row_count_after > 0 else 0.0
            null_rate[col] = null_pct
        
        # Inferred dtypes
        inferred_dtypes = {col: str(df[col].dtype) for col in df.columns}
        
        # Outlier statistics
        outliers = compute_outlier_stats(df, numeric_cols)
        
        # ===== WRITE PARQUET =====
        parquet_path = output_dir / "cleaned.parquet"
        df.write_parquet(str(parquet_path))
        print(f"  Wrote cleaned parquet to {parquet_path}")
        
        # ===== WRITE STEP JSON =====
        step_output = {
            "step": "10-csv-read-cleansing",
            "row_count_initial": row_count_initial,
            "row_count_after": row_count_after,
            "column_count": column_count,
            "target_column_normalized": target_normalized,
            "time_column_detected": time_column,
            "null_rate": null_rate,
            "duplicate_rows_removed": duplicates_removed,
            "inferred_dtypes": inferred_dtypes,
            "outliers": outliers,
            "fixes": fixes,
            "artifacts": {
                "cleaned_parquet": str(parquet_path)
            }
        }
        
        step_json_path = output_dir / "step-10-cleanse.json"
        step_json_path.write_text(json.dumps(step_output, indent=2))
        print(f"  Wrote step JSON to {step_json_path}")
        
        # ===== UPDATE PROGRESS =====
        progress = json.loads(progress_file.read_text())
        progress.update({
            "completed_steps": ["10-csv-read-cleansing"],
            "status": "running"
        })
        progress_file.write_text(json.dumps(progress, indent=2))
        
        print("[Step 10] SUCCESS ✓")
        return 0
        
    except Exception as e:
        print(f"[Step 10] FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        # Update progress with error
        try:
            progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
            if "errors" not in progress:
                progress["errors"] = []
            progress["errors"].append(f"Step 10 failed: {str(e)}")
            progress["status"] = "error"
            progress_file.write_text(json.dumps(progress, indent=2))
        except:
            pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
