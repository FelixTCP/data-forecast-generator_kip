#!/usr/bin/env python3
"""
Step 12: Feature Extraction & Engineering

Transform raw features into a clean, scaled feature matrix ready for model training.
Includes lag features, rolling statistics, temporal features, and leakage detection.
"""

import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Tuple, Dict, Any, List

import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from tqdm import tqdm
import joblib


def auto_detect_target_column(df: pd.DataFrame, numeric_cols: List[str], explicit_target: str = None) -> Tuple[str, Dict[str, Any]]:
    """Auto-detect or validate target column."""
    if explicit_target:
        if explicit_target not in numeric_cols:
            raise ValueError(f"Target '{explicit_target}' not in numeric columns")
        return explicit_target, {"method": "explicit", "score": 1.0}
    
    # Return highest variance column
    variances = {col: df[col].var() for col in numeric_cols}
    best_col = max(variances, key=variances.get)
    return best_col, {"method": "highest_variance", "score": variances[best_col]}


def compute_lag_features(df: pd.DataFrame, target_col: str, max_lag: int = 12) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """Compute lag features for target and top features."""
    lag_features = []
    excluded = {}
    
    # Lag features for target
    for lag in range(1, min(4, max_lag + 1)):  # Top 3 lags
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = df[target_col].shift(lag)
        lag_features.append(col_name)
    
    return df, lag_features, excluded


def compute_rolling_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """Compute rolling statistics."""
    rolling_features = []
    
    for window in [7, 30]:
        # Rolling mean
        col_name = f"{target_col}_rolling_mean_{window}"
        df[col_name] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        rolling_features.append(col_name)
        
        # Rolling std
        col_name = f"{target_col}_rolling_std_{window}"
        df[col_name] = df[target_col].shift(1).rolling(window=window, min_periods=1).std()
        rolling_features.append(col_name)
    
    return df, rolling_features


def compute_temporal_features(df: pd.DataFrame, time_col: str = None) -> Tuple[pd.DataFrame, List[str]]:
    """Add temporal features from date column or year/month/day."""
    temporal_features = []
    
    # Synthesize date if needed
    if time_col is None and all(col in df.columns for col in ['year', 'month', 'day']):
        try:
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']].rename(columns={'year': 'Y', 'month': 'm', 'day': 'd'}))
            time_col = 'date'
        except Exception:
            return df, temporal_features
    
    if time_col is not None and time_col in df.columns:
        # Hour of day
        if 'hour' in df.columns:
            df['hour'] = df['hour']
            temporal_features.append('hour')
        
        # Day of week
        if 'date' in df.columns or time_col == 'date':
            try:
                df['day_of_week'] = pd.to_datetime(df[time_col]).dt.dayofweek
                temporal_features.append('day_of_week')
            except Exception:
                pass
        
        # Month
        if 'month' not in df.columns or df['month'].dtype != 'int64':
            try:
                df['month_of_year'] = pd.to_datetime(df[time_col]).dt.month
                temporal_features.append('month_of_year')
            except Exception:
                pass
        
        # Year
        if 'year' not in df.columns or df['year'].dtype != 'int64':
            try:
                df['year_val'] = pd.to_datetime(df[time_col]).dt.year
                temporal_features.append('year_val')
            except Exception:
                pass
    
    return df, temporal_features


def remove_zero_variance_features(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """Remove zero or near-zero variance features."""
    zero_var_threshold = np.sqrt(1e-10)
    excluded = {}
    
    remaining_features = []
    for col in features:
        if col not in df.columns:
            continue
        
        std_val = df[col].std()
        if pd.isna(std_val) or std_val <= zero_var_threshold:
            excluded[col] = "zero_variance"
        else:
            remaining_features.append(col)
    
    if len(remaining_features) < 2:
        raise ValueError(f"Fewer than 2 features after zero-variance removal: {remaining_features}")
    
    return df, remaining_features, excluded


def detect_feature_leakage(X: np.ndarray, y: np.ndarray, feature_names: List[str], threshold: float = 0.98) -> Tuple[bool, List[str], float]:
    """Detect potential feature leakage."""
    
    leakage_candidates = []
    
    # Step 1: Pearson correlation check
    for i, col_name in enumerate(feature_names):
        if col_name.startswith("target_"):
            continue  # Skip target-derived features
        
        try:
            corr, _ = pearsonr(X[:, i], y)
            if abs(corr) >= threshold:
                leakage_candidates.append(col_name)
        except Exception:
            pass
    
    # Step 2: Random Forest reconstruction probe
    probe_r2 = None
    if leakage_candidates:
        try:
            # Get indices of leakage candidates
            leak_indices = [i for i, name in enumerate(feature_names) if name in leakage_candidates]
            X_leak = X[:, leak_indices]
            
            # Train RF on leakage features only
            rf = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=42)
            rf.fit(X_leak, y)
            probe_r2 = float(rf.score(X_leak, y))
            
            if probe_r2 > 0.999:
                return False, leakage_candidates, probe_r2
        except Exception:
            pass
    
    return True, leakage_candidates, probe_r2


def apply_scaling(X: pd.DataFrame, target_col: str, recommended_models: List[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply feature scaling based on recommended models."""
    
    scaling_metadata = {
        "scaler_used": "None",
        "features_scaled": [],
        "features_not_scaled": list(X.columns),
        "scaler_path": None
    }
    
    # Determine if scaling is needed
    if recommended_models is None:
        recommended_models = []
    
    # Extract model name from list of dicts if needed
    if recommended_models and isinstance(recommended_models[0], dict):
        model_type = recommended_models[0].get("model_class", "random_forest").lower()
    elif recommended_models:
        model_type = recommended_models[0].lower()
    else:
        model_type = "random_forest"
    
    # Tree-based models don't need scaling
    if any(tree_model in model_type for tree_model in ["tree", "forest", "xgboost", "lightgbm", "gradient"]):
        return X, scaling_metadata
    
    # Linear models and SARIMA need StandardScaler
    if any(linear_model in model_type for linear_model in ["ridge", "lasso", "linear", "elastic", "sarima"]):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        scaling_metadata = {
            "scaler_used": "StandardScaler",
            "features_scaled": list(X.columns),
            "features_not_scaled": [],
            "scaler_path": "scaler.joblib"
        }
        
        return pd.DataFrame(X_scaled, columns=X.columns), scaling_metadata
    
    # LSTM and neural nets need MinMaxScaler
    if any(nn_model in model_type for nn_model in ["lstm", "neural", "cnn"]):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        scaling_metadata = {
            "scaler_used": "MinMaxScaler",
            "features_scaled": list(X.columns),
            "features_not_scaled": [],
            "scaler_path": "scaler.joblib"
        }
        
        return pd.DataFrame(X_scaled, columns=X.columns), scaling_metadata
    
    return X, scaling_metadata


def main():
    parser = argparse.ArgumentParser(description="Step 12: Feature Extraction")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--split-mode", default="auto", choices=["auto", "random", "time_series"])
    parser.add_argument("--exclude-features", default="", help="Comma-separated features to exclude")
    
    args = parser.parse_args()
    
    try:
        output_dir = Path(args.output_dir)
        
        # Load progress and prior step outputs
        progress_path = output_dir / "progress.json"
        progress = json.loads(progress_path.read_text())
        target_col = progress.get("target_column", "").lower().replace(" ", "_")
        
        step11_path = output_dir / "step-11-exploration.json"
        step11 = json.loads(step11_path.read_text())
        recommended_features = step11.get("recommended_features", [])
        
        step10_path = output_dir / "step-10-cleanse.json"
        step10 = json.loads(step10_path.read_text())
        time_col = step10.get("time_column_detected") or None
        
        # Load cleaned data
        cleaned_path = output_dir / "cleaned.parquet"
        df_pl = pl.read_parquet(str(cleaned_path))
        df = df_pl.to_pandas()
        
        # Get numeric columns
        numeric_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
        
        # Ensure target is in df
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Parse exclude-features
        exclude_features_list = [f.strip() for f in args.exclude_features.split(",") if f.strip()]
        
        # Start with recommended features
        current_features = [f for f in recommended_features if f in numeric_cols and f != target_col]
        current_features = [f for f in current_features if f not in exclude_features_list]
        
        excluded_features = {}
        
        # Add temporal features
        df, temporal_feats = compute_temporal_features(df, time_col)
        current_features.extend(temporal_feats)
        
        # Add lag features
        df, lag_feats, lag_excluded = compute_lag_features(df, target_col)
        current_features.extend(lag_feats)
        excluded_features.update(lag_excluded)
        
        # Add rolling features
        df, rolling_feats = compute_rolling_features(df, target_col)
        current_features.extend(rolling_feats)
        
        # Remove zero-variance features
        df, current_features, zero_var_excluded = remove_zero_variance_features(df, current_features)
        excluded_features.update(zero_var_excluded)
        
        # Ensure target is not in features
        if target_col in current_features:
            excluded_features[target_col] = "data_leakage_target_column"
            current_features.remove(target_col)
            sys.exit(2)
        
        # Prepare feature matrix
        X = df[current_features].fillna(df[current_features].mean())
        y = df[target_col].fillna(df[target_col].mean())
        
        # Detect leakage
        leakage_pass, leakage_cands, probe_r2 = detect_feature_leakage(X.values, y.values, current_features)
        
        if not leakage_pass:
            print(f"Leakage detected in features: {leakage_cands}", file=sys.stderr)
            sys.exit(2)
        
        # Apply scaling
        X_scaled, scaling_metadata = apply_scaling(X, target_col, step11.get("model_class_recommendations", []))
        
        # Save scaler if used
        if scaling_metadata["scaler_path"]:
            scaler_path = output_dir / scaling_metadata["scaler_path"]
            scaler = StandardScaler() if scaling_metadata["scaler_used"] == "StandardScaler" else MinMaxScaler()
            scaler.fit(X.values)
            joblib.dump(scaler, str(scaler_path))
        
        # Prepare output features dataframe (X + y)
        df_features = X_scaled.copy()
        df_features[target_col] = y.values
        
        # Save features parquet
        features_parquet = output_dir / "features.parquet"
        df_pl_features = pl.from_pandas(df_features)
        df_pl_features.write_parquet(str(features_parquet))
        
        # Determine split strategy
        split_strategy = {
            "resolved_mode": "time_series" if time_col else "random",
            "test_size": 0.2,
            "random_state": 42
        }
        
        # Build output JSON
        output_json = {
            "step": "12-feature-extraction",
            "run_id": args.run_id,
            "features": current_features,
            "features_count": len(current_features),
            "features_excluded": excluded_features,
            "excluded_count": len(excluded_features),
            "target_column": target_col,
            "split_strategy": split_strategy,
            "leakage": {
                "status": "pass" if leakage_pass else "fail",
                "leakage_candidates": leakage_cands,
                "threshold": 0.98,
                "reconstruction_probe_r2": probe_r2
            },
            "scaling_metadata": scaling_metadata,
            "artifacts": {
                "features_parquet": str(features_parquet),
                "scaler_joblib": str(output_dir / scaling_metadata["scaler_path"]) if scaling_metadata["scaler_path"] else None
            }
        }
        
        # Write output JSON
        step_json_path = output_dir / "step-12-features.json"
        step_json_path.write_text(json.dumps(output_json, indent=2))
        
        # Update progress
        progress["status"] = "running"
        progress["current_step"] = "13-model-training"
        progress["completed_steps"] = ["10-csv-read-cleansing", "11-data-exploration", "12-feature-extraction"]
        progress_path.write_text(json.dumps(progress, indent=2))
        
        print(f"Step 12 completed: {len(current_features)} features extracted")
        sys.exit(0)
        
    except Exception as e:
        print(f"Step 12 failed: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
