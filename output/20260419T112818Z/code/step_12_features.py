#!/usr/bin/env python3
"""
Step 12: Feature Extraction
Builds time-based features (year, month, day_of_week, hour), lag features,
and rolling features based on lag analysis from step 11. Checks for leakage.
"""

import json
import argparse
from pathlib import Path
import polars as pl
import numpy as np
import sys
from sklearn.ensemble import RandomForestRegressor


def run_step_12(output_dir: str) -> dict:
    """
    Extract features: time features, lags, rolling features. Check leakage.
    
    Returns step output dict.
    """
    output_path = Path(output_dir)
    
    # Load prior step outputs
    step_10_path = output_path / "step-10-cleanse.json"
    step_11_path = output_path / "step-11-exploration.json"
    cleaned_parquet = output_path / "cleaned.parquet"
    
    if not step_10_path.exists() or not step_11_path.exists():
        sys.exit("Step 10 or 11 output not found")
    
    with open(step_10_path) as f:
        step_10 = json.load(f)
    with open(step_11_path) as f:
        step_11 = json.load(f)
    
    df = pl.read_parquet(cleaned_parquet)
    
    target_column = step_10["target_column_normalized"]
    time_column = step_10["time_column_detected"]
    recommended_features = step_11["recommended_features"]
    significant_lags = step_11["time_series"].get("significant_lags", [])
    useful_lag_features = step_11["time_series"].get("useful_lag_features", [])
    
    if len(recommended_features) == 0:
        sys.exit("No recommended features from step 11")
    
    # Start with recommended features only
    feature_df = df.select([target_column] + recommended_features).clone()
    
    # Add time-based features if time column exists
    if time_column is not None:
        if time_column in df.columns:
            time_data = df[time_column]
            feature_df = feature_df.with_columns([
                time_data.dt.year().alias("year"),
                time_data.dt.month().alias("month"),
                time_data.dt.weekday().alias("day_of_week"),
                time_data.dt.hour().alias("hour")
            ])
            recommended_features.extend(["year", "month", "day_of_week", "hour"])
    
    # Add target lags (from significant_lags)
    rows_dropped_by_lags = 0
    max_lag = 0
    if significant_lags:
        max_lag = max(significant_lags)
        for lag in significant_lags[:5]:  # Limit to top 5 lags
            lag_col = f"target_lag_{lag}"
            feature_df = feature_df.with_columns(
                pl.col(target_column).shift(lag).alias(lag_col)
            )
            recommended_features.append(lag_col)
        
        # Drop rows with NaN from lagging
        feature_df = feature_df.drop_nulls()
        rows_dropped_by_lags = df.height - feature_df.height
    
    # Add feature lags (from useful_lag_features)
    for lag_feat in useful_lag_features[:5]:  # Limit to top 5
        feat_name = lag_feat["feature"]
        lag_val = lag_feat["lag"]
        if feat_name in feature_df.columns:
            lag_col = f"{feat_name}_lag_{lag_val}"
            feature_df = feature_df.with_columns(
                pl.col(feat_name).shift(lag_val).alias(lag_col)
            )
            recommended_features.append(lag_col)
    
    # Drop remaining nulls
    feature_df = feature_df.drop_nulls()
    
    if feature_df.height == 0:
        sys.exit("All rows dropped during feature engineering")
    
    # Get all feature names (excluding target)
    feature_names_final = [c for c in feature_df.columns if c != target_column]
    
    # Leakage check: compute correlation of each feature with target
    leakage_candidates = []
    
    # Get only numeric columns for leakage check, and convert to float
    numeric_feature_cols = []
    for c in feature_names_final:
        if feature_df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
            numeric_feature_cols.append(c)
        else:
            # Try to convert
            try:
                feature_df = feature_df.with_columns(pl.col(c).cast(pl.Float64).alias(c))
                numeric_feature_cols.append(c)
            except:
                pass
    
    if len(numeric_feature_cols) > 0:
        # Create numpy array from only numeric columns
        X_numeric_only = feature_df.select(numeric_feature_cols).to_numpy(allow_copy=True).astype(np.float64)
        y = feature_df[target_column].to_numpy(allow_copy=True).astype(np.float64)
        
        for i, fname in enumerate(numeric_feature_cols):
            if feature_df.height > 1 and not np.isnan(X_numeric_only[:, i]).all():
                try:
                    corr = np.corrcoef(X_numeric_only[:, i], y)[0, 1]
                    if abs(corr) > 0.98:
                        leakage_candidates.append({
                            "feature": fname,
                            "correlation": float(corr)
                        })
                except:
                    pass
        
        leakage_status = "pass"
        if leakage_candidates:
            # Try to confirm with RF R²
            try:
                rf = RandomForestRegressor(n_estimators=10, random_state=42)
                X_leakage = X_numeric_only[:, [numeric_feature_cols.index(c["feature"]) for c in leakage_candidates]]
                rf.fit(X_leakage, y)
                r2 = rf.score(X_leakage, y)
                if r2 > 0.999:
                    leakage_status = "fail"
            except:
                leakage_status = "fail"
    else:
        leakage_status = "pass"
    
    # Save features.parquet
    features_parquet = output_path / "features.parquet"
    feature_df.write_parquet(features_parquet)
    
    # Build step output
    recommended_features_final = feature_names_final.copy()
    step_output = {
        "step": "12-feature-extraction",
        "features": recommended_features_final,
        "feature_count": len(recommended_features_final),
        "rows_after_lags": feature_df.height,
        "rows_dropped_by_lags": rows_dropped_by_lags,
        "time_column": time_column,
        "time_features_added": ["year", "month", "day_of_week", "hour"] if time_column else [],
        "target_lags_added": [f"target_lag_{lag}" for lag in significant_lags[:5]],
        "feature_lags_added": [f"{lag_feat['feature']}_lag_{lag_feat['lag']}" for lag_feat in useful_lag_features[:5]],
        "leakage": {
            "status": leakage_status,
            "leakage_candidates": leakage_candidates
        },
        "split_strategy": {
            "resolved_mode": "time_series" if time_column else "random"
        },
        "artifacts": {
            "features_parquet": str(features_parquet)
        }
    }
    
    # Save step JSON
    step_json_path = output_path / "step-12-features.json"
    with open(step_json_path, "w") as f:
        json.dump(step_output, f, indent=2)
    
    # Update progress
    progress_path = output_path / "progress.json"
    with open(progress_path, "r") as f:
        progress = json.load(f)
    progress["status"] = "running"
    progress["current_step"] = "13-model-training"
    progress["completed_steps"].append("12-feature-extraction")
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)
    
    return step_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 12: Feature Extraction")
    parser.add_argument("--output-dir", required=True, help="Output directory for artifacts")
    
    args = parser.parse_args()
    
    try:
        result = run_step_12(args.output_dir)
        print(json.dumps(result, indent=2), flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"Error in Step 12: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
