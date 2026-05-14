#!/usr/bin/env python3
"""
Step 12: Feature Extraction

Build feature matrix starting strictly from recommended_features in step 11,
adding time features and lag features as appropriate.
"""
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
import traceback
import warnings
warnings.filterwarnings('ignore')

import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def extract_features(
    output_dir: str,
    run_id: str,
) -> dict:
    """
    Extract features from cleaned data using step-11 guidance.
    
    Returns:
        dict: Feature extraction output JSON
    """
    output_dir_path = Path(output_dir)
    
    # Load cleaned data and step 11 output
    cleaned_parquet = output_dir_path / "cleaned.parquet"
    df_pl = pl.read_parquet(cleaned_parquet)
    df_pd = df_pl.to_pandas()
    
    with open(output_dir_path / "step-10-cleanse.json") as f:
        step_10_data = json.load(f)
    
    with open(output_dir_path / "step-11-exploration.json") as f:
        step_11_data = json.load(f)
    
    target_normalized = step_11_data["target_candidates"][0]
    recommended_features = step_11_data["recommended_features"]
    time_column = step_11_data.get("time_column")
    useful_lag_features = step_11_data.get("useful_lag_features", [])
    significant_lags = step_11_data.get("significant_lags", [])
    
    # Start with recommended features
    features_list = list(recommended_features)
    feature_creation_log = []
    
    # === Add time features if time column detected ===
    if time_column and time_column in df_pd.columns:
        df_pd[time_column] = pd.to_datetime(df_pd[time_column], errors='coerce')
        
        if df_pd[time_column].dtype == 'datetime64[ns]':
            # Extract time features
            df_pd['year'] = df_pd[time_column].dt.year
            df_pd['month'] = df_pd[time_column].dt.month
            df_pd['day_of_week'] = df_pd[time_column].dt.dayofweek
            df_pd['hour'] = df_pd[time_column].dt.hour
            
            time_features = ['year', 'month', 'day_of_week', 'hour']
            for feat in time_features:
                if feat not in features_list:
                    features_list.append(feat)
                    feature_creation_log.append({
                        "feature": feat,
                        "reason": "time_feature",
                        "source": time_column
                    })
    
    # === Add lag features ===
    # Only from useful_lag_features
    for lag_entry in useful_lag_features:
        feat = lag_entry["feature"]
        lag = lag_entry["lag"]
        lag_feature_name = f"{feat}_lag_{lag}"
        
        if lag_feature_name not in features_list and lag > 0:
            df_pd[lag_feature_name] = df_pd[feat].shift(lag)
            features_list.append(lag_feature_name)
            feature_creation_log.append({
                "feature": lag_feature_name,
                "reason": "useful_lag_feature",
                "source": feat,
                "lag": lag
            })
    
    # Add target lags at significant_lags
    for lag in significant_lags:
        lag_feature_name = f"{target_normalized}_lag_{lag}"
        
        if lag_feature_name not in features_list and lag > 0:
            df_pd[lag_feature_name] = df_pd[target_normalized].shift(lag)
            features_list.append(lag_feature_name)
            feature_creation_log.append({
                "feature": lag_feature_name,
                "reason": "target_lag",
                "source": target_normalized,
                "lag": lag
            })
    
    # === Add rolling features for target at top lags ===
    top_lags = sorted(significant_lags)[:2] if len(significant_lags) >= 2 else significant_lags
    
    for lag in top_lags:
        if lag > 0:
            rolling_feature_name = f"{target_normalized}_rolling_{lag}"
            
            if rolling_feature_name not in features_list:
                # Use shift to avoid look-ahead bias
                df_pd[rolling_feature_name] = df_pd[target_normalized].shift(1).rolling(window=lag).mean()
                features_list.append(rolling_feature_name)
                feature_creation_log.append({
                    "feature": rolling_feature_name,
                    "reason": "target_rolling",
                    "source": target_normalized,
                    "window": lag
                })
    
    # Note: Don't dropna() yet - we'll do it when building final matrix
    rows_before = len(df_pd)
    
    # === Leakage detection ===
    leakage_status = "pass"
    leakage_candidates = []
    correlations = {}
    
    if target_normalized in df_pd.columns:
        for feat in features_list:
            if feat in df_pd.columns and feat != target_normalized:
                try:
                    # Compute Pearson correlation
                    valid_mask = (~df_pd[feat].isna()) & (~df_pd[target_normalized].isna())
                    if valid_mask.sum() > 1:
                        x = df_pd.loc[valid_mask, feat].values
                        y = df_pd.loc[valid_mask, target_normalized].values
                        corr, _ = pearsonr(x, y)
                        correlations[feat] = float(corr)
                        
                        if abs(corr) > 0.99:
                            leakage_candidates.append(feat)
                            leakage_status = "fail"
                except:
                    pass
    
    if leakage_status == "fail":
        error_msg = f"Leakage detected in features: {leakage_candidates}"
        raise RuntimeError(error_msg)
    
    # === Guard: exclude re-included dropped features ===
    excluded_in_step_11 = set(step_11_data.get("excluded_features", {}).keys())
    features_list_checked = [f for f in features_list if f not in excluded_in_step_11]
    
    # === Ensure minimum features ===
    if len(features_list_checked) < 2:
        error_msg = f"Fewer than 2 features after cleanup: {features_list_checked}. Cannot proceed."
        raise ValueError(error_msg)
    
    # === Build final feature matrix, excluding columns with all-NaN ===
    feature_cols = [
        col for col in features_list_checked 
        if col in df_pd.columns and df_pd[col].notna().any()  # Must have at least one non-null value
    ]
    
    # Only select rows where all selected features + target are non-null
    select_cols = [*feature_cols, target_normalized]
    df_final_pd = df_pd[select_cols].dropna().copy()
    
    rows_after = len(df_final_pd)
    rows_dropped = rows_before - rows_after
    
    # Convert back to polars
    df_final_pl = pl.from_pandas(df_final_pd)
    
    # === Write features parquet ===
    features_parquet = output_dir_path / "features.parquet"
    df_final_pl.write_parquet(features_parquet)
    
    # === Build output JSON ===
    output_json = {
        "step": "12-feature-extraction",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        
        "shape_before": {
            "rows": rows_before,
            "columns": len(df_pd.columns)
        },
        "shape_after": {
            "rows": rows_after,
            "columns": len(df_final_pl.columns)
        },
        "rows_dropped_by_lags": rows_dropped,
        
        "target": target_normalized,
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "feature_creation_log": feature_creation_log,
        
        "features_excluded": {
            col: reason for col, reason in step_11_data.get("excluded_features", {}).items()
            if col in features_list and col not in feature_cols
        },
        
        "leakage": {
            "status": leakage_status,
            "leakage_candidates": leakage_candidates,
            "correlations": correlations,
            "threshold": 0.99,
        },
        
        "split_strategy": {
            "resolved_mode": "time_series" if time_column else "random",
            "time_column": time_column,
        },
        
        "artifacts": {
            "features_parquet": str(features_parquet),
        },
        
        "context": {
            "dataset_id": run_id,
            "target_column": target_normalized,
            "time_column": time_column,
            "features": feature_cols,
            "split_strategy": {
                "resolved_mode": "time_series" if time_column else "random",
                "time_column": time_column,
            },
            "model_candidates": step_11_data.get("context", {}).get("model_candidates", []),
            "metrics": {},
            "artifacts": {"features_parquet": str(features_parquet)},
            "notes": [
                f"Started with {len(recommended_features)} recommended features",
                f"Added {len(feature_creation_log)} derived features (time, lag, rolling)",
                f"Final feature matrix: {rows_after} rows × {len(feature_cols)} features",
                f"Dropped {rows_dropped} rows due to NaN from lag operations",
            ]
        }
    }
    
    # Write JSON
    step_json_path = output_dir_path / "step-12-features.json"
    with open(step_json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    # Update progress
    progress_path = output_dir_path / "progress.json"
    with open(progress_path) as f:
        progress = json.load(f)
    
    progress["completed_steps"].append("12-feature-extraction")
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"✓ Step 12 complete")
    print(f"  Features: {len(feature_cols)} from {len(recommended_features)} recommended")
    print(f"  Leakage status: {leakage_status}")
    print(f"  Final matrix: {rows_after} rows × {len(feature_cols)} features")
    print(f"  Report written to: {step_json_path}")
    
    return output_json


def main():
    parser = argparse.ArgumentParser(description="Step 12: Feature Extraction")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    try:
        extract_features(
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
        sys.exit(0)
    except Exception as e:
        print(f"✗ Step 12 failed: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
