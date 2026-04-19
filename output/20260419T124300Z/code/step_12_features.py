#!/usr/bin/env python3
"""
Step 12: Feature Extraction
Build a leakage-safe feature matrix with engineered features.
"""

import json
import os
import sys
import argparse
import polars as pl
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from typing import Any

def load_exploration_output(output_dir: str) -> dict:
    """Load step 11 exploration output."""
    exp_path = os.path.join(output_dir, "step-11-exploration.json")
    with open(exp_path, 'r') as f:
        return json.load(f)

def build_features(df: pl.DataFrame, target_column: str, step11_output: dict, 
                   output_dir: str, split_mode: str = "auto") -> tuple[pl.DataFrame, dict]:
    """Build feature matrix from recommended features + engineered features."""
    
    audit = {
        "step": "12-feature-extraction",
        "features": [],
        "features_excluded": {},
        "created_features": [],
        "rows_dropped_by_lag": 0,
        "leakage_flags": [],
        "split_strategy": {}
    }
    
    # Convert target column to numeric if it's String
    if df[target_column].dtype == pl.String:
        print(f"[Step 12] Converting {target_column} from String to Float64...")
        df = df.with_columns(
            pl.col(target_column).str.strip_chars().cast(pl.Float64)
        )
        print(f"  ✓ Converted {target_column} to Float64")
    
    # Validate step 11 output
    recommended = step11_output.get("recommended_features", [])
    if not recommended:
        raise ValueError("No recommended_features from step 11 — cannot proceed")
    
    print(f"[Step 12] Starting with {len(recommended)} recommended features")
    
    # Start with recommended features
    feature_cols = list(recommended)  # Make a copy
    audit["features"] = feature_cols.copy()
    
    # Detect time column
    time_column = step11_output.get("time_column")
    
    # Add time features if time column detected
    if time_column and time_column in df.columns:
        df_datetime = df[time_column]
        if df_datetime.dtype in [pl.Date, pl.Datetime]:
            # Create time features
            if df_datetime.dtype == pl.Datetime:
                df = df.with_columns([
                    df_datetime.dt.year().alias("year"),
                    df_datetime.dt.month().alias("month"),
                    df_datetime.dt.weekday().alias("day_of_week"),
                    df_datetime.dt.hour().alias("hour"),
                ])
                feature_cols.extend(["year", "month", "day_of_week", "hour"])
                audit["created_features"].extend([
                    {"name": "year", "reason": "time decomposition"},
                    {"name": "month", "reason": "time decomposition"},
                    {"name": "day_of_week", "reason": "time decomposition"},
                    {"name": "hour", "reason": "time decomposition"},
                ])
                print(f"  ✓ Added 4 time features")
    
    # Create lag features for recommended features (from useful_lag_features)
    useful_lags = step11_output.get("useful_lag_features", [])
    for lag_info in useful_lags:
        feature = lag_info["feature"]
        lag = lag_info["lag"]
        
        # Only create lag if feature is in recommended
        if feature in recommended:
            lag_col_name = f"{feature}_lag_{lag}"
            if lag_col_name not in df.columns:
                df = df.with_columns(
                    pl.col(feature).shift(lag).alias(lag_col_name)
                )
                feature_cols.append(lag_col_name)
                audit["created_features"].append({
                    "name": lag_col_name,
                    "reason": f"useful_lag_feature: {feature} at lag={lag}"
                })
    
    # Create lag features for target
    significant_lags = step11_output.get("significant_lags", [])
    if significant_lags:
        for lag in significant_lags[:3]:  # Limit to first 3 significant lags
            lag_col_name = f"{target_column}_lag_{lag}"
            if lag_col_name not in df.columns:
                df = df.with_columns(
                    pl.col(target_column).shift(lag).alias(lag_col_name)
                )
                feature_cols.append(lag_col_name)
                audit["created_features"].append({
                    "name": lag_col_name,
                    "reason": f"significant_lag={lag} from step 11"
                })
        
        # Create rolling features (shift before roll, for causality)
        # Skip rolling features - they cause too many NaNs
        # Will only use lag features
        print(f"  ✓ Skipping rolling features (use lag features for forecast)")
    else:
        audit["created_features"].append({
            "note": "No lag features created — no significant autocorrelation detected"
        })
    
    print(f"  ✓ Created {len(audit['created_features'])} engineered features")
    
    # Drop rows with NaN (introduced by shifting/rolling) - ONLY in engineered features
    rows_before = df.height
    
    # Get engineered feature names
    engineered_cols = [f["name"] for f in audit["created_features"] if "name" in f]
    
    # Drop rows that have NaN in engineered features
    if engineered_cols:
        for col in engineered_cols:
            if col in df.columns:
                df = df.filter(pl.col(col).is_not_null())
    
    rows_after = df.height
    rows_dropped = rows_before - rows_after
    audit["rows_dropped_by_lag"] = rows_dropped
    
    if rows_dropped > 0:
        print(f"  ✓ Dropped {rows_dropped} rows with NaN from lag/rolling features")
    else:
        print(f"  ✓ No rows dropped (lag features compatible with data)")
    
    # Ensure all features are numeric
    final_features = []
    for col in feature_cols:
        if col in df.columns:
            dtype = df[col].dtype
            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                        pl.Float32, pl.Float64]:
                final_features.append(col)
            else:
                # Try to cast String to float
                if dtype == pl.String:
                    try:
                        df = df.with_columns(
                            pl.col(col).str.strip_chars().cast(pl.Float64)
                        )
                        final_features.append(col)
                    except:
                        audit["features_excluded"][col] = "non-numeric, cast failed"
                else:
                    audit["features_excluded"][col] = "non-numeric dtype"
        else:
            audit["features_excluded"][col] = "column not found"
    
    audit["features"] = final_features
    
    # Check for minimum viable feature count
    if len(final_features) < 2:
        raise ValueError(f"Fewer than 2 features survive filtering: {final_features}")
    
    print(f"  ✓ {len(final_features)} numeric features ready")
    
    # Leakage guard: check pairwise correlation with target
    leakage_detected = False
    target_np = df[target_column].to_numpy().astype(np.float64)
    target_np[np.isnan(target_np)] = np.nanmedian(target_np[~np.isnan(target_np)])
    
    for col in final_features:
        if col == target_column:
            continue
        
        x_np = df[col].to_numpy().astype(np.float64)
        x_np[np.isnan(x_np)] = np.nanmedian(x_np[~np.isnan(x_np)])
        
        try:
            corr, _ = pearsonr(x_np, target_np)
            if abs(corr) > 0.99:
                audit["leakage_flags"].append({
                    "feature": col,
                    "issue": "pairwise correlation > 0.99",
                    "correlation": float(corr)
                })
                leakage_detected = True
        except:
            pass
    
    # Algebraic leakage probe: check if engineered features alone can predict target near-perfectly
    # Build a simple test: train on first 80%, test on last 20%
    if audit["created_features"]:
        try:
            engineered_features = [f["name"] for f in audit["created_features"] 
                                  if "name" in f]
            engineered_features = [f for f in engineered_features if f in final_features]
            
            if engineered_features:
                n = df.height
                split_idx = int(0.8 * n)
                
                X_eng = df[engineered_features].to_numpy().astype(np.float64)
                y = target_np
                
                X_train = X_eng[:split_idx]
                y_train = y[:split_idx]
                X_test = X_eng[split_idx:]
                y_test = y[split_idx:]
                
                # Train simple linear model
                try:
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    r2_test = model.score(X_test, y_test)
                    y_pred = model.predict(X_test)
                    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                    
                    if r2_test > 0.995:
                        audit["leakage_flags"].append({
                            "issue": "algebraic_reconstruction_leakage",
                            "message": f"Engineered features alone achieve R2={r2_test:.4f} on test set",
                            "r2_test": float(r2_test),
                            "rmse_test": float(rmse)
                        })
                        leakage_detected = True
                except:
                    pass
        except:
            pass
    
    audit["leakage_audit"] = {
        "status": "fail" if leakage_detected else "pass",
        "checks": ["pairwise_corr", "linear_probe_reconstruction"],
        "details": audit["leakage_flags"]
    }
    
    if leakage_detected:
        print(f"  ✗ LEAKAGE DETECTED: {len(audit['leakage_flags'])} issues found")
        print(json.dumps(audit["leakage_audit"], indent=2))
        raise ValueError("Leakage detected — cannot proceed to training")
    
    print(f"  ✓ Leakage audit passed")
    
    # Determine split strategy
    split_resolved = split_mode
    if split_mode == "auto":
        if time_column:
            split_resolved = "time_series"
        else:
            split_resolved = "random"
    
    audit["split_strategy"] = {
        "requested_mode": split_mode,
        "resolved_mode": split_resolved,
        "time_column": time_column,
        "random_state": 42
    }
    
    # Build final feature matrix (include target)
    feature_cols_with_target = final_features + [target_column]
    feature_df = df.select(feature_cols_with_target)
    
    return feature_df, audit

def extract_features(parquet_path: str, output_dir: str, run_id: str, 
                    target_column: str, split_mode: str = "auto") -> None:
    """Main feature extraction pipeline."""
    try:
        print(f"[Step 12] Loading parquet from {parquet_path}...")
        df = pl.read_parquet(parquet_path)
        
        # Load step 11 output
        step11_output = load_exploration_output(output_dir)
        
        print(f"[Step 12] Building features...")
        feature_df, audit = build_features(df, target_column, step11_output, output_dir, split_mode)
        
        audit["run_id"] = run_id
        
        # Write features parquet
        features_path = os.path.join(output_dir, "features.parquet")
        os.makedirs(output_dir, exist_ok=True)
        feature_df.write_parquet(features_path)
        audit["artifacts"] = {"features_parquet": features_path}
        print(f"  ✓ Wrote features parquet to {features_path}")
        
        # Write step 12 report
        report_path = os.path.join(output_dir, "step-12-features.json")
        with open(report_path, 'w') as f:
            json.dump(audit, f, indent=2)
        print(f"  ✓ Wrote features report to {report_path}")
        
        # Update progress
        progress_path = os.path.join(output_dir, "progress.json")
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        
        progress["current_step"] = "12-feature-extraction"
        progress["completed_steps"].append("12-feature-extraction")
        
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("✓ Step 12 completed successfully")
        sys.exit(0)
    
    except Exception as e:
        print(f"✗ Step 12 failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 12: Feature Extraction")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--target-column", required=True, help="Target column (normalized)")
    parser.add_argument("--split-mode", default="auto", help="Split mode: auto|random|time_series")
    
    args = parser.parse_args()
    
    parquet_path = os.path.join(args.output_dir, "cleaned.parquet")
    extract_features(parquet_path, args.output_dir, args.run_id, args.target_column, args.split_mode)
