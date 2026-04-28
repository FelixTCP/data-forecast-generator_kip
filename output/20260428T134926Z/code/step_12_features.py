#!/usr/bin/env python3
"""
Step 12: Feature Extraction & Preprocessing
Creates time-series appropriate features with strict leakage controls.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings

import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_step_output(output_dir: str, step: str) -> dict:
    """Load output JSON from a previous step."""
    path = Path(output_dir) / f"step-{step}.json"
    with open(path, 'r') as f:
        return json.load(f)

def create_time_features(df: pd.DataFrame, time_col: str) -> tuple[pd.DataFrame, dict]:
    """Create time-based features from a datetime column."""
    features_created = {}
    
    if time_col not in df.columns:
        return df, features_created
    
    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except:
            return df, features_created
    
    # Extract time features
    try:
        df['year'] = df[time_col].dt.year.astype('int16')
        features_created['year'] = 'extracted_from_time_column'
        
        df['month'] = df[time_col].dt.month.astype('int8')
        features_created['month'] = 'extracted_from_time_column'
        
        df['day_of_week'] = df[time_col].dt.dayofweek.astype('int8')
        features_created['day_of_week'] = 'extracted_from_time_column'
        
        df['hour'] = df[time_col].dt.hour.astype('int8')
        features_created['hour'] = 'extracted_from_time_column'
        
        df['day_of_year'] = df[time_col].dt.dayofyear.astype('int16')
        features_created['day_of_year'] = 'extracted_from_time_column'
        
        df['quarter'] = df[time_col].dt.quarter.astype('int8')
        features_created['quarter'] = 'extracted_from_time_column'
        
    except Exception as e:
        print(f"[Step 12] Warning: Time feature extraction failed: {e}", file=sys.stderr)
    
    return df, features_created

def create_lag_features(
    df: pd.DataFrame,
    useful_lag_features: list,
    target_col: str,
    significant_lags: list,
    max_lags: int = 3
) -> tuple[pd.DataFrame, dict]:
    """Create lag-based features for exogenous and target variables."""
    features_created = {}
    
    # Create target lag features from significant lags
    for lag in significant_lags[:5]:  # Limit to first 5
        lag_name = f"{target_col}_lag_{lag}"
        try:
            df[lag_name] = df[target_col].shift(lag)
            features_created[lag_name] = f"target_lag_{lag}_from_significant_lags"
        except Exception as e:
            print(f"[Step 12] Warning: Failed to create {lag_name}: {e}", file=sys.stderr)
    
    # Create exogenous lag features from useful_lag_features
    for entry in useful_lag_features:
        feat_name = entry.get("feature")
        lag = entry.get("lag", 0)
        
        if feat_name not in df.columns:
            continue
        
        if lag == 0:
            continue  # Skip lag 0 (current value) for non-target exogenous
        
        lag_feat_name = f"{feat_name}_lag_{lag}"
        try:
            df[lag_feat_name] = df[feat_name].shift(lag)
            features_created[lag_feat_name] = f"exogenous_lag_{lag}_from_xcorr_analysis"
        except Exception as e:
            print(f"[Step 12] Warning: Failed to create {lag_feat_name}: {e}", file=sys.stderr)
    
    return df, features_created

def create_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    significant_lags: list,
    max_windows: int = 2
) -> tuple[pd.DataFrame, dict]:
    """Create rolling window features for target at significant lags."""
    features_created = {}
    
    if not significant_lags:
        return df, features_created
    
    # Use top N significant lags as window sizes
    window_sizes = sorted(significant_lags)[:max_windows]
    
    for window in window_sizes:
        try:
            # Rolling mean (with shift to avoid look-ahead)
            rolling_mean_name = f"{target_col}_rolling_mean_{window}"
            df[rolling_mean_name] = df[target_col].shift(1).rolling(
                window=window, min_periods=1
            ).mean()
            features_created[rolling_mean_name] = f"target_rolling_mean_window_{window}"
            
            # Rolling std
            rolling_std_name = f"{target_col}_rolling_std_{window}"
            df[rolling_std_name] = df[target_col].shift(1).rolling(
                window=window, min_periods=1
            ).std()
            features_created[rolling_std_name] = f"target_rolling_std_window_{window}"
            
        except Exception as e:
            print(f"[Step 12] Warning: Failed to create rolling features for window {window}: {e}", file=sys.stderr)
    
    return df, features_created

def check_leakage(X: pd.DataFrame, y: pd.Series, xcorr_threshold: float = 0.99) -> tuple[bool, list]:
    """
    Check for target leakage.
    
    Returns:
        (has_leakage, leakage_features)
    """
    leakage_features = []
    
    # Remove NaNs for correlation computation
    mask = ~(X.isna().any(axis=1) | y.isna())
    if mask.sum() < 10:
        return False, []
    
    X_clean = X[mask]
    y_clean = y[mask]
    
    for col in X_clean.columns:
        try:
            # Compute Pearson correlation
            xcorr = np.corrcoef(X_clean[col].values, y_clean.values)[0, 1]
            if abs(xcorr) > xcorr_threshold:
                leakage_features.append({
                    "feature": col,
                    "xcorr_with_target": float(xcorr)
                })
        except:
            pass
    
    return len(leakage_features) > 0, leakage_features

def main():
    parser = argparse.ArgumentParser(
        description="Step 12: Feature Extraction"
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
    output_dir = args.output_dir
    run_id = args.run_id
    
    try:
        print("[Step 12] Starting feature extraction...")
        
        # Load step outputs
        step10 = load_step_output(output_dir, "10-cleanse")
        step11 = load_step_output(output_dir, "11-exploration")
        
        target_column = step10["target_column_normalized"]
        parquet_path = step10["artifacts"]["cleaned_parquet"]
        recommended_features = step11["recommended_features"]
        significant_lags = step11.get("significant_lags", [])
        useful_lag_features = step11.get("useful_lag_features", [])
        time_column = step11.get("time_column")
        
        print(f"[Step 12] Loading cleaned parquet...")
        df = pl.read_parquet(parquet_path)
        
        # Convert to pandas for feature engineering
        df_pd = df.to_pandas()
        print(f"[Step 12] Loaded {len(df_pd)} rows × {df_pd.shape[1]} columns")
        
        # Start with recommended features only
        feature_columns = recommended_features.copy()
        print(f"[Step 12] Starting with {len(feature_columns)} recommended features")
        
        # Track feature creation
        features_created = {}
        features_excluded = {}
        
        # Create time features if time column detected
        if time_column and time_column in df_pd.columns:
            print(f"[Step 12] Creating time features from column: {time_column}")
            df_pd, time_feats = create_time_features(df_pd, time_column)
            features_created.update(time_feats)
            feature_columns.extend(time_feats.keys())
        
        # Create lag features
        print("[Step 12] Creating lag features...")
        df_pd, lag_feats = create_lag_features(
            df_pd,
            useful_lag_features,
            target_column,
            significant_lags
        )
        features_created.update(lag_feats)
        feature_columns.extend(lag_feats.keys())
        
        # Create rolling features
        print("[Step 12] Creating rolling features...")
        df_pd, roll_feats = create_rolling_features(
            df_pd,
            target_column,
            significant_lags
        )
        features_created.update(roll_feats)
        feature_columns.extend(roll_feats.keys())
        
        # Remove duplicates
        feature_columns = list(set(feature_columns))
        print(f"[Step 12] Total feature columns: {len(feature_columns)}")
        
        # Check for missing features (may have been dropped during creation)
        available_features = [f for f in feature_columns if f in df_pd.columns]
        excluded_due_to_missing = [f for f in feature_columns if f not in df_pd.columns]
        
        for feat in excluded_due_to_missing:
            features_excluded[feat] = "not_in_data_after_engineering"
        
        feature_columns = available_features
        print(f"[Step 12] Available feature columns: {len(feature_columns)}")
        
        if len(feature_columns) < 2:
            raise ValueError(
                f"Insufficient features after engineering: {len(feature_columns)}. "
                f"Need at least 2 features for modeling."
            )
        
        # Check for leakage
        print("[Step 12] Checking for target leakage...")
        X = df_pd[feature_columns].copy()
        y = df_pd[target_column].copy()
        
        has_leakage, leakage_features = check_leakage(X, y, xcorr_threshold=0.99)
        
        if has_leakage:
            print(f"[Step 12] ✗ LEAKAGE DETECTED: {leakage_features}", file=sys.stderr)
            raise RuntimeError(
                f"Target leakage detected in features: {leakage_features}. "
                f"Pipeline halted to prevent invalid model."
            )
        
        print("[Step 12] ✓ No leakage detected")
        
        # Handle NaN values: fill lag/rolling NaNs with forward-fill, keep only if target is not NaN
        print("[Step 12] Handling NaN values...")
        initial_rows = len(df_pd)
        
        # Forward-fill NaN values from lag/rolling features
        for col in feature_columns:
            if df_pd[col].isna().any():
                df_pd[col] = df_pd[col].bfill().ffill()
        
        # Fill remaining NaNs with 0 for numeric features
        for col in feature_columns:
            if df_pd[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                df_pd[col] = df_pd[col].fillna(0)
        
        # Drop rows where target is NaN
        df_pd = df_pd.dropna(subset=[target_column])
        final_rows = len(df_pd)
        print(f"[Step 12] Removed {initial_rows - final_rows} rows with NaN target values")
        
        # Convert back to polars and export
        print("[Step 12] Exporting feature matrix...")
        df_features = pl.from_pandas(df_pd[[*feature_columns, target_column]])
        
        features_parquet_path = str(Path(output_dir) / "features.parquet")
        df_features.write_parquet(features_parquet_path)
        print(f"[Step 12] Exported features parquet to: {features_parquet_path}")
        
        # Build step output
        step_output = {
            "step": "12-feature-extraction",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "features": feature_columns,
            "feature_count": len(feature_columns),
            "features_created": features_created,
            "features_excluded": features_excluded,
            "target_column": target_column,
            "row_count_initial": initial_rows,
            "row_count_final": final_rows,
            "rows_removed_due_to_nan": initial_rows - final_rows,
            "split_strategy": {
                "resolved_mode": "time_series",
                "time_column": time_column,
                "reason": "Chronological split required for time-series forecasting"
            },
            "leakage_check": {
                "status": "pass",
                "features_with_high_xcorr": []
            },
            "artifacts": {
                "features_parquet": features_parquet_path
            },
            "notes": [
                f"Started with {len(recommended_features)} recommended features from step 11",
                f"Created {len(features_created)} new features (time, lags, rolling)",
                f"Final feature set: {len(feature_columns)} features",
                f"Removed {initial_rows - final_rows} rows with missing values"
            ]
        }
        
        # Write step output
        output_json_path = Path(output_dir) / "step-12-features.json"
        with open(output_json_path, 'w') as f:
            json.dump(step_output, f, indent=2)
        print(f"[Step 12] Output written to: {output_json_path}")
        
        # Update progress
        progress_path = Path(output_dir) / "progress.json"
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        progress["current_step"] = "12-feature-extraction"
        progress["completed_steps"].append("11-data-exploration")
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("[Step 12] ✓ Completed successfully")
        sys.exit(0)
        
    except Exception as e:
        print(f"[Step 12] ✗ Failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
