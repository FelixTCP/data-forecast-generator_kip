#!/usr/bin/env python3
"""
Step 11: Data Exploration
Analyze dataset to identify valuable features for engineering.
"""

import json
import os
import sys
import argparse
import polars as pl
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr
from typing import Any

def load_exploration_data(parquet_path: str, target_column: str) -> tuple[pl.DataFrame, str]:
    """Load parquet and get numeric columns."""
    df = pl.read_parquet(parquet_path)
    
    # Ensure target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Columns: {df.columns}")
    
    return df, target_column

def get_numeric_columns(df: pl.DataFrame, target_column: str) -> list[str]:
    """Get all numeric columns except target."""
    numeric_cols = []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64]:
            if col != target_column:
                numeric_cols.append(col)
    return numeric_cols

def compute_variance_scaled(X: np.ndarray) -> np.ndarray:
    """Compute min-max scaled variance for each column."""
    variances = np.var(X, axis=0)
    X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
    # Avoid division by zero
    ranges = X_max - X_min
    ranges[ranges == 0] = 1
    scaled_var = variances / (ranges ** 2)
    return scaled_var

def detect_low_variance(df: pl.DataFrame, numeric_cols: list[str], threshold: float = 1e-4) -> dict:
    """Detect columns with near-zero variance."""
    low_variance = {}
    for col in numeric_cols:
        X = df[col].to_numpy()
        X = X[~np.isnan(X)]  # Remove NaN
        if len(X) > 0:
            scaled_var = compute_variance_scaled(X.reshape(-1, 1))[0]
            if scaled_var < threshold:
                low_variance[col] = float(scaled_var)
    return low_variance

def compute_mutual_information(df: pl.DataFrame, numeric_cols: list[str], target_column: str) -> dict:
    """Compute mutual information ranking for numeric features vs. target."""
    # Prepare target (ensure it's numeric)
    target_data = df[target_column]
    
    # If target is String, try to convert
    if target_data.dtype == pl.String:
        try:
            y = target_data.str.strip_chars().cast(pl.Float64).to_numpy()
        except:
            print(f"Warning: Cannot convert target '{target_column}' to numeric")
            return {}, 0.0, []
    elif target_data.dtype not in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                                   pl.Float32, pl.Float64]:
        # Try to cast to float
        try:
            y = target_data.cast(pl.Float64).to_numpy()
        except:
            print(f"Warning: Cannot convert target to numeric")
            return {}, 0.0, []
    else:
        y = target_data.to_numpy().astype(np.float64)
    
    # Remove NaN
    y = y[~np.isnan(y)]
    
    # Prepare feature matrix (clean NaN)
    X_list = []
    valid_cols = []
    for col in numeric_cols:
        x = df[col]
        
        # If column is String, try to cast to float
        if x.dtype == pl.String:
            try:
                x = x.cast(pl.Float64)
            except:
                print(f"  Warning: Could not convert '{col}' (String) to float, skipping")
                continue
        
        x_np = x.to_numpy().astype(np.float64).copy()
        x_nan_mask = np.isnan(x_np)
        
        if np.any(x_nan_mask):
            # Impute NaN with median
            x_clean = x_np.copy()
            x_clean[x_nan_mask] = np.nanmedian(x_np[~x_nan_mask])
            X_list.append(x_clean)
        else:
            X_list.append(x_np)
        valid_cols.append(col)
    
    if not X_list:
        return {}, 0.0, []
    
    X = np.column_stack(X_list)
    
    # Ensure y matches X shape
    if len(y) != X.shape[0]:
        y = y[:X.shape[0]]
        # Re-collect X to match y length
        X_list_trimmed = []
        for col in valid_cols:
            x = df[col].to_numpy()[:len(y)]
            x[np.isnan(x)] = np.nanmedian(x)
            X_list_trimmed.append(x)
        X = np.column_stack(X_list_trimmed)
    
    # Compute MI for real features
    try:
        mi_scores = mutual_info_regression(X, y, random_state=42)
    except Exception as e:
        print(f"Warning: MI computation failed: {e}")
        mi_scores = np.zeros(len(valid_cols))
    
    # Compute MI for noise baseline (5 random columns)
    noise_mi_scores = []
    for _ in range(5):
        noise = np.random.RandomState(42 + _).standard_normal(X.shape[0])
        try:
            mi_noise = mutual_info_regression(noise.reshape(-1, 1), y, random_state=42)
            noise_mi_scores.append(mi_noise[0])
        except:
            pass
    
    noise_mi_baseline = float(np.mean(noise_mi_scores)) if noise_mi_scores else 0.0
    
    # Ranking
    ranking = []
    below_baseline_cols = []
    for col, mi in zip(valid_cols, mi_scores):
        ranking.append({
            "feature": col,
            "mi_score": float(mi),
            "below_noise_baseline": float(mi) <= noise_mi_baseline
        })
        if float(mi) <= noise_mi_baseline:
            below_baseline_cols.append(col)
    
    ranking.sort(key=lambda x: x["mi_score"], reverse=True)
    
    return ranking, noise_mi_baseline, below_baseline_cols

def compute_pairwise_correlations(df: pl.DataFrame, numeric_cols: list[str], mi_ranking: dict) -> dict:
    """Detect redundant features (high correlation, low MI)."""
    # Create MI lookup
    mi_lookup = {item["feature"]: item["mi_score"] for item in mi_ranking}
    
    # Compute correlation matrix (skip columns that can't be converted)
    X_list = []
    cols_to_use = []
    for col in numeric_cols:
        x = df[col]
        
        # If column is String, try to cast to float
        if x.dtype == pl.String:
            try:
                x = x.cast(pl.Float64)
            except:
                continue
        
        x_np = x.to_numpy().astype(np.float64).copy()
        x_nan_mask = np.isnan(x_np)
        if np.any(x_nan_mask):
            x_np[x_nan_mask] = np.nanmedian(x_np[~x_nan_mask])
        X_list.append(x_np)
        cols_to_use.append(col)
    
    if not X_list:
        return {"max_pair": [], "max_corr": 0.0}, []
    
    X = np.column_stack(X_list)
    
    redundant = []
    max_corr_pair = []
    max_corr_val = 0.0
    
    for i in range(len(cols_to_use)):
        for j in range(i + 1, len(cols_to_use)):
            corr, _ = pearsonr(X[:, i], X[:, j])
            corr_abs = abs(corr)
            
            if corr_abs >= 0.90:
                col_i, col_j = cols_to_use[i], cols_to_use[j]
                mi_i = mi_lookup.get(col_i, 0.0)
                mi_j = mi_lookup.get(col_j, 0.0)
                
                # Flag the one with lower MI as redundant
                if mi_i < mi_j:
                    redundant.append(col_i)
                else:
                    redundant.append(col_j)
            
            if corr_abs > max_corr_val:
                max_corr_val = corr_abs
                max_corr_pair = [cols_to_use[i], cols_to_use[j]]
    
    return {
        "max_pair": max_corr_pair,
        "max_corr": float(max_corr_val)
    }, redundant

def detect_time_series_properties(df: pl.DataFrame, target_column: str, time_column: str = None) -> dict:
    """Detect time-series properties: significant lags and useful lag features."""
    result = {
        "time_series_detected": False,
        "time_column": None,
        "significant_lags": [],
        "useful_lag_features": []
    }
    
    if time_column is None or time_column not in df.columns:
        return result
    
    # Check if time column is datetime-like
    time_dtype = df[time_column].dtype
    if time_dtype not in [pl.Date, pl.Datetime]:
        return result
    
    result["time_series_detected"] = True
    result["time_column"] = time_column
    
    # Compute target autocorrelation at lags 1-24
    # Convert target to numeric first
    target_data = df[target_column]
    if target_data.dtype == pl.String:
        try:
            y = target_data.str.strip_chars().cast(pl.Float64).to_numpy()
        except:
            return result
    else:
        y = target_data.to_numpy().astype(np.float64)
    
    y = y[~np.isnan(y)]
    
    significant_lags = []
    max_lag = min(24, len(y) // 4)
    
    for lag in range(1, max_lag + 1):
        if lag < len(y):
            y1 = y[:-lag]
            y2 = y[lag:]
            if len(y1) > 1 and np.std(y1) > 0 and np.std(y2) > 0:
                try:
                    acf_val, _ = pearsonr(y1, y2)
                    if abs(acf_val) > 0.1:
                        significant_lags.append(lag)
                except:
                    pass
    
    result["significant_lags"] = significant_lags
    
    # Compute cross-correlation for features at lags 0-3
    useful_lags = []
    numeric_cols = get_numeric_columns(df, target_column)
    
    for col in numeric_cols:
        x = df[col].to_numpy().astype(np.float64).copy()  # Make writable copy
        if np.any(np.isnan(x)):
            x[np.isnan(x)] = np.nanmedian(x[~np.isnan(x)])
        
        for lag in range(4):
            if lag < len(y):
                x_lagged = x[lag:] if lag > 0 else x
                y_aligned = y[:len(x_lagged)]
                
                if len(x_lagged) > 1 and np.std(x_lagged) > 0 and np.std(y_aligned) > 0:
                    try:
                        xcorr, _ = pearsonr(x_lagged, y_aligned)
                        if abs(xcorr) > 0.15:
                            useful_lags.append({
                                "feature": col,
                                "lag": lag,
                                "xcorr": float(xcorr)
                            })
                    except:
                        pass
    
    result["useful_lag_features"] = useful_lags
    return result

def explore_data(parquet_path: str, output_dir: str, run_id: str, target_column: str) -> None:
    """Main exploration pipeline."""
    try:
        print(f"[Step 11] Loading parquet from {parquet_path}...")
        df, target_col = load_exploration_data(parquet_path, target_column)
        
        exploration_report = {
            "step": "11-data-exploration",
            "run_id": run_id,
            "shape": {"rows": df.height, "columns": df.width}
        }
        
        # Get numeric columns
        numeric_cols = get_numeric_columns(df, target_col)
        exploration_report["numeric_columns"] = numeric_cols
        print(f"  ✓ Found {len(numeric_cols)} numeric features")
        
        # Low variance filter
        low_var = detect_low_variance(df, numeric_cols)
        exploration_report["low_variance_columns"] = low_var
        if low_var:
            print(f"  ✓ Detected {len(low_var)} low-variance columns")
        
        # MI ranking
        mi_ranking, noise_baseline, below_baseline = compute_mutual_information(df, numeric_cols, target_col)
        exploration_report["mi_ranking"] = mi_ranking
        exploration_report["noise_mi_baseline"] = noise_baseline
        print(f"  ✓ Computed MI ranking (noise baseline: {noise_baseline:.6f})")
        
        # Redundancy filter
        corr_summary, redundant_cols = compute_pairwise_correlations(df, numeric_cols, mi_ranking)
        exploration_report["correlation_matrix_summary"] = corr_summary
        exploration_report["redundant_columns"] = list(set(redundant_cols))
        if redundant_cols:
            print(f"  ✓ Detected {len(set(redundant_cols))} redundant columns")
        
        # Time-series analysis (detect time column if present)
        time_column = None
        for col in df.columns:
            if df[col].dtype in [pl.Date, pl.Datetime]:
                time_column = col
                break
        
        ts_props = detect_time_series_properties(df, target_col, time_column)
        exploration_report.update(ts_props)
        if ts_props["time_series_detected"]:
            print(f"  ✓ Time-series detected: {len(ts_props['significant_lags'])} significant lags")
        
        # Recommended features: exclude low variance, below baseline MI, and redundant
        excluded = {}
        recommended = []
        
        for col in numeric_cols:
            if col in low_var:
                excluded[col] = "low_variance"
            elif col in below_baseline:
                excluded[col] = "below_noise_baseline"
            elif col in exploration_report.get("redundant_columns", []):
                excluded[col] = "redundant"
            else:
                recommended.append(col)
        
        # If all excluded, loosen threshold by 50%
        if not recommended and numeric_cols:
            loose_baseline = noise_baseline * 0.5
            recommended = [col for col in numeric_cols 
                          if col not in low_var and col not in exploration_report.get("redundant_columns", [])]
            # Filter by loosened baseline
            recommended = [col for col in recommended 
                          if next((m["mi_score"] for m in mi_ranking if m["feature"] == col), 0) > loose_baseline]
            
            if not recommended:
                # Last resort: take top 5 by MI
                recommended = [m["feature"] for m in mi_ranking[:5]]
            print(f"  ⚠ All features filtered; loosened threshold → {len(recommended)} features")
        
        exploration_report["recommended_features"] = recommended
        exploration_report["excluded_features"] = excluded
        print(f"  ✓ Recommended {len(recommended)} features for engineering")
        
        # Write report
        report_path = os.path.join(output_dir, "step-11-exploration.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(exploration_report, f, indent=2)
        print(f"  ✓ Wrote exploration report to {report_path}")
        
        # Update progress
        progress_path = os.path.join(output_dir, "progress.json")
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        
        progress["current_step"] = "11-data-exploration"
        progress["completed_steps"].append("11-data-exploration")
        
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("✓ Step 11 completed successfully")
        sys.exit(0)
    
    except Exception as e:
        print(f"✗ Step 11 failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 11: Data Exploration")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--target-column", required=True, help="Target column (normalized)")
    
    args = parser.parse_args()
    
    parquet_path = os.path.join(args.output_dir, "cleaned.parquet")
    explore_data(parquet_path, args.output_dir, args.run_id, args.target_column)
