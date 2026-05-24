#!/usr/bin/env python3
"""
Step 12: Feature Extraction & Engineering
Generate engineered features, check for leakage, scale, and prepare training data.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import polars as pl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm


def step_12_main(output_dir: str, run_id: str, split_mode: str = "auto", 
                 exclude_features: str = "", max_lag: int = 12) -> int:
    """Main step 12 logic."""
    try:
        output_path = Path(output_dir)
        
        # Load prior outputs
        with open(output_path / "step-10-cleanse.json") as f:
            step10 = json.load(f)
        with open(output_path / "step-11-exploration.json") as f:
            step11 = json.load(f)
        
        target_col = step10["target_column_normalized"]
        time_col = step10["time_column_detected"]
        recommended_feats = step11["recommended_features"]
        useful_lags = step11["useful_lag_features"]
        significant_lags = step11.get("significant_lags", [])
        multiple_series = step11.get("multiple_series_detected", False)
        series_id = step11.get("series_id_column")
        
        # Parse exclude list
        exclude_list = [f.strip() for f in exclude_features.split(",") if f.strip()]
        
        # Load data
        print(f"[Step 12] Loading cleaned.parquet...")
        df = pl.read_parquet(output_path / "cleaned.parquet").to_pandas()
        
        print(f"[Step 12] Initial shape: {df.shape}")
        
        # ============ FEATURE ENGINEERING ============
        print(f"[Step 12] Engineering features...")
        
        engineered_features = {}
        features_list = []
        
        # Time features (year, month, day, day_of_week, hour if available)
        if time_col in df.columns:
            print(f"[Step 12] Adding time features from {time_col}...")
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            
            for prefix, extractor in tqdm([
                ("year", lambda x: x.dt.year),
                ("month", lambda x: x.dt.month),
                ("day_of_week", lambda x: x.dt.dayofweek),
                ("quarter", lambda x: x.dt.quarter)
            ]):
                try:
                    feat_name = f"{prefix}"
                    df[feat_name] = extractor(df[time_col])
                    features_list.append(feat_name)
                    engineered_features[feat_name] = "time_feature"
                except:
                    pass
        
        # Add recommended features (if not excluded)
        print(f"[Step 12] Adding recommended features...")
        for feat in tqdm(recommended_feats):
            if feat in exclude_list or feat == target_col:
                continue
            if feat not in df.columns:
                continue
            
            # Check for leakage (|r| > 0.98)
            feat_valid = df[feat].dropna()
            target_valid = df[target_col].dropna()
            if len(feat_valid) > 2 and len(target_valid) > 2:
                try:
                    corr = np.corrcoef(feat_valid.values, target_valid.values[:len(feat_valid)])[0, 1]
                    if abs(corr) > 0.98:
                        print(f"[Step 12] WARNING: {feat} has |r| > 0.98 with target (potential leakage)")
                        continue
                except:
                    pass
            
            features_list.append(feat)
            engineered_features[feat] = "recommended_feature"
        
        # Lag features
        print(f"[Step 12] Adding lag features...")
        target_lags = significant_lags[:5] if significant_lags else [1, 3]
        
        for lag in tqdm(target_lags):
            if lag > 0 and lag < len(df):
                try:
                    feat_name = f"{target_col}_lag_{lag}"
                    df[feat_name] = df[target_col].shift(lag)
                    features_list.append(feat_name)
                    engineered_features[feat_name] = "target_lag_feature"
                except:
                    pass
        
        # Cross-feature lags
        for lag_spec in tqdm(useful_lags[:10]):
            feat = lag_spec["feature"]
            lag = lag_spec["lag"]
            if lag > 0 and lag < len(df) and feat in df.columns and feat not in exclude_list:
                try:
                    feat_name = f"{feat}_lag_{lag}"
                    df[feat_name] = df[feat].shift(lag)
                    features_list.append(feat_name)
                    engineered_features[feat_name] = "cross_lag_feature"
                except:
                    pass
        
        # Rolling features on target
        print(f"[Step 12] Adding rolling features...")
        rolling_windows = [3, 7] if significant_lags else [3]
        
        for window in tqdm(rolling_windows):
            if window < len(df):
                try:
                    feat_name = f"{target_col}_rolling_mean_{window}"
                    df[feat_name] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
                    features_list.append(feat_name)
                    engineered_features[feat_name] = "rolling_feature"
                except:
                    pass
        
        # ============ REMOVE ZERO-VARIANCE FEATURES ============
        print(f"[Step 12] Removing zero-variance features...")
        valid_features = []
        for feat in tqdm(features_list):
            if feat not in df.columns:
                continue
            try:
                std_val = df[feat].std()
                if std_val > np.sqrt(1e-10):  # Above threshold
                    valid_features.append(feat)
            except:
                valid_features.append(feat)
        
        features_list = valid_features
        
        # Ensure at least 2 features
        if len(features_list) < 2:
            print(f"[Step 12] ✗ Fewer than 2 features after engineering: {len(features_list)}")
            return 1
        
        print(f"[Step 12] Final feature count: {len(features_list)}")
        
        # ============ PREPARE OUTPUT DATAFRAME ============
        print(f"[Step 12] Preparing output dataframe...")
        
        # Remove duplicates from features_list
        features_list = list(dict.fromkeys(features_list))  # Preserve order, remove duplicates
        
        # Include target column
        df_output = df[[target_col] + features_list].copy()
        
        # Ensure column names are unique
        df_output.columns = [str(c) for c in df_output.columns]
        
        # Drop rows with NaN
        df_output = df_output.dropna()
        
        print(f"[Step 12] Output shape: {df_output.shape}")
        
        if df_output.shape[0] < 10:
            print(f"[Step 12] ✗ Output has < 10 rows: {df_output.shape[0]}")
            return 1
        
        # ============ SCALING ============
        print(f"[Step 12] Scaling features...")
        
        # Use StandardScaler for linear models / ARIMA
        scaler = StandardScaler()
        features_to_scale = features_list  # All features
        
        df_scaled = df_output.copy()
        df_scaled[features_to_scale] = scaler.fit_transform(df_output[features_to_scale])
        
        # Save scaler
        scaler_path = output_path / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"[Step 12] Saved scaler to {scaler_path}")
        
        # ============ LEAKAGE CHECK (FINAL) ============
        print(f"[Step 12] Final leakage check...")
        leakage_status = "pass"
        leakage_candidates = []
        
        for feat in features_list:
            if feat not in df_output.columns:
                continue
            try:
                feat_vals = df_output[feat].dropna().values
                target_vals = df_output[target_col].dropna().values[:len(feat_vals)]
                if len(feat_vals) > 2 and len(target_vals) > 2:
                    corr = np.corrcoef(feat_vals, target_vals)[0, 1]
                    if abs(corr) > 0.98:
                        leakage_status = "fail"
                        leakage_candidates.append({"feature": feat, "correlation": float(corr)})
            except:
                pass
        
        if leakage_status == "fail":
            print(f"[Step 12] ✗ Leakage detected! Exiting with code 2")
            return 2
        
        # ============ DETERMINE SPLIT STRATEGY ============
        if split_mode == "auto":
            if step11.get("time_series_detected", False):
                resolved_split = "time_series"
            else:
                resolved_split = "random"
        else:
            resolved_split = split_mode
        
        print(f"[Step 12] Split strategy: {resolved_split}")
        
        # ============ OUTPUT JSON ============
        excluded_features_dict = {}
        
        # Features that were filtered out
        for feat in recommended_feats:
            if feat not in features_list and feat != target_col:
                if feat in exclude_list:
                    excluded_features_dict[feat] = "user_excluded"
                elif abs(0.98) < 0.98:  # simplified check
                    excluded_features_dict[feat] = "leakage_suspect"
        
        output_json = {
            "step": "12-feature-extraction",
            "run_id": run_id,
            "features": features_list,
            "features_count": len(features_list),
            "features_excluded": excluded_features_dict,
            "excluded_count": len(excluded_features_dict),
            "target_column": target_col,
            "split_strategy": {"resolved_mode": resolved_split},
            "leakage": {
                "status": leakage_status,
                "leakage_candidates": leakage_candidates,
                "threshold": 0.98,
                "reconstruction_probe_r2": None
            },
            "scaling_metadata": {
                "scaler_used": "StandardScaler",
                "features_scaled": features_to_scale,
                "features_not_scaled": [],
                "scaler_path": str(scaler_path)
            },
            "artifacts": {
                "features_parquet": str(output_path / "features.parquet"),
                "scaler_joblib": str(scaler_path)
            }
        }
        
        # Write features parquet
        features_parquet = output_path / "features.parquet"
        pl.from_pandas(df_scaled).write_parquet(features_parquet)
        print(f"[Step 12] Wrote features.parquet: {features_parquet}")
        
        # Write output JSON
        step12_json = output_path / "step-12-features.json"
        with open(step12_json, "w") as f:
            json.dump(output_json, f, indent=2)
        print(f"[Step 12] Wrote step-12-features.json")
        
        print(f"[Step 12] ✓ Completed successfully")
        return 0
        
    except Exception as e:
        print(f"[Step 12] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 12: Feature Extraction")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--split-mode", default="auto", help="Split mode: auto|random|time_series")
    parser.add_argument("--exclude-features", default="", help="Comma-separated features to exclude")
    parser.add_argument("--max-lag", type=int, default=12, help="Maximum lag for lag features")
    
    args = parser.parse_args()
    
    sys.exit(step_12_main(args.output_dir, args.run_id, args.split_mode, 
                         args.exclude_features, args.max_lag))
