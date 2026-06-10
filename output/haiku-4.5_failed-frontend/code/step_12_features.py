#!/usr/bin/env python
"""
STEP 12 — Feature Extraction & Model Preselection

Transforms raw features into a clean, scaled feature matrix.
Uses step-11 exploration output to guide lag/rolling feature creation.

Exit code: 0=success, 1=error, 2=leakage_detected
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
import polars as pl
import polars.selectors as cs
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_progress(output_dir: Path) -> dict:
    return json.loads((output_dir / "progress.json").read_text())

def load_step10(output_dir: Path) -> dict:
    return json.loads((output_dir / "step-10-cleanse.json").read_text())

def load_step11(output_dir: Path) -> dict:
    return json.loads((output_dir / "step-11-exploration.json").read_text())

def create_lag_features(
    df: pl.DataFrame,
    target_col: str,
    useful_lag_features: list,
    significant_lags: list,
    group_col: str = None
) -> pl.DataFrame:
    """
    Create lag features strictly from step-11 recommendations.
    useful_lag_features: [{"feature": "feat", "lag": 1, "xcorr": 0.23}, ...]
    significant_lags: [1, 3, 6, ...] for target
    """
    logger.info("Creating lag features...")
    
    # Target lags
    for lag in significant_lags[:5]:  # Cap at 5 lags for speed
        try:
            if group_col and group_col in df.columns:
                df = df.with_columns(
                    pl.col(target_col)
                    .shift(lag)
                    .over(group_col)
                    .alias(f"{target_col}_lag_{lag}")
                )
            else:
                df = df.with_columns(
                    pl.col(target_col)
                    .shift(lag)
                    .alias(f"{target_col}_lag_{lag}")
                )
        except Exception as e:
            logger.warning(f"Failed to create lag {lag} for target: {e}")
    
    # Feature lags from useful_lag_features
    for item in tqdm(useful_lag_features[:10], desc="Feature lags"):
        feature = item.get("feature")
        lag = item.get("lag")
        
        if feature not in df.columns:
            continue
        
        try:
            if group_col and group_col in df.columns:
                df = df.with_columns(
                    pl.col(feature)
                    .shift(lag)
                    .over(group_col)
                    .alias(f"{feature}_lag_{lag}")
                )
            else:
                df = df.with_columns(
                    pl.col(feature)
                    .shift(lag)
                    .alias(f"{feature}_lag_{lag}")
                )
        except Exception as e:
            logger.warning(f"Failed to create lag {lag} for {feature}: {e}")
    
    return df

def create_rolling_features(
    df: pl.DataFrame,
    target_col: str,
    group_col: str = None,
    windows: list = None
) -> pl.DataFrame:
    """Create rolling mean/std features for target."""
    if windows is None:
        windows = [7, 30]
    
    logger.info("Creating rolling features...")
    
    for window in tqdm(windows, desc="Rolling windows"):
        try:
            # Rolling mean - with shift for causality
            if group_col and group_col in df.columns:
                df = df.with_columns(
                    pl.col(target_col)
                    .shift(1)
                    .rolling_mean(window_size=window)
                    .over(group_col)
                    .alias(f"{target_col}_rolling_mean_{window}")
                )
            else:
                df = df.with_columns(
                    pl.col(target_col)
                    .shift(1)
                    .rolling_mean(window_size=window)
                    .alias(f"{target_col}_rolling_mean_{window}")
                )
            
            # Rolling std
            if group_col and group_col in df.columns:
                df = df.with_columns(
                    pl.col(target_col)
                    .shift(1)
                    .rolling_std(window_size=window)
                    .over(group_col)
                    .alias(f"{target_col}_rolling_std_{window}")
                )
            else:
                df = df.with_columns(
                    pl.col(target_col)
                    .shift(1)
                    .rolling_std(window_size=window)
                    .alias(f"{target_col}_rolling_std_{window}")
                )
        except Exception as e:
            logger.warning(f"Failed to create rolling {window}: {e}")
    
    return df

def add_calendar_features(df: pl.DataFrame, time_col: str) -> pl.DataFrame:
    """Add hour, day-of-week, month if time column is datetime."""
    if time_col not in df.columns:
        return df
    
    logger.info("Adding calendar features...")
    
    try:
        # Ensure datetime
        if df[time_col].dtype != pl.Datetime and df[time_col].dtype != pl.Date:
            logger.warning(f"Time column {time_col} is not datetime, skipping calendar features")
            return df
        
        df = df.with_columns([
            pl.col(time_col).dt.hour().alias("hour"),
            pl.col(time_col).dt.weekday().alias("day_of_week"),
            pl.col(time_col).dt.month().alias("month"),
        ])
    except Exception as e:
        logger.warning(f"Failed to add calendar features: {e}")
    
    return df

def add_trend_feature(df: pl.DataFrame, time_col: str) -> pl.DataFrame:
    """Add trend_elapsed_days feature."""
    if time_col not in df.columns:
        return df
    
    logger.info("Adding trend feature...")
    
    try:
        # Create elapsed days since start
        df = df.with_columns(
            (pl.col(time_col) - pl.col(time_col).min())
            .dt.days()
            .alias("trend_elapsed_days")
        )
    except Exception as e:
        logger.warning(f"Failed to add trend feature: {e}")
    
    return df

def remove_zero_variance_features(
    df: pl.DataFrame,
    target_col: str,
    variance_threshold: float = 1e-10
) -> tuple[pl.DataFrame, dict]:
    """
    Remove constant or near-constant features.
    Return (cleaned_df, {"feature": "zero_variance", ...})
    """
    logger.info("Removing zero-variance features...")
    
    numeric_cols = df.select(cs.numeric()).columns
    excluded = {}
    
    for col in tqdm(numeric_cols, desc="Variance check"):
        if col == target_col:
            continue
        
        series = df[col]
        variance = series.var()
        
        if variance is None or variance < variance_threshold:
            logger.warning(f"Removing {col}: zero/near-zero variance (var={variance})")
            excluded[col] = "zero_variance"
    
    # Drop excluded columns
    cols_to_keep = [c for c in df.columns if c not in excluded]
    df = df.select(cols_to_keep)
    
    return df, excluded

def detect_leakage(
    df: pl.DataFrame,
    target_col: str,
    pearson_threshold: float = 0.98
) -> tuple[bool, list]:
    """
    Detect leakage via Pearson correlation and RandomForest probe.
    Return (has_leakage, leakage_candidates_list)
    """
    logger.info("Detecting feature leakage...")
    
    # Convert to pandas for sklearn operations
    df_pd = df.to_pandas()
    
    leakage_candidates = []
    
    # Pearson correlation with target
    numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    
    if not numeric_cols:
        return False, []
    
    X = df_pd[numeric_cols].dropna()
    y = df_pd.loc[X.index, target_col].dropna()
    X = X.loc[y.index]
    
    if len(X) < 10:
        return False, []
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        for col in tqdm(numeric_cols, desc="Leakage detection"):
            # Pearson check
            if col in X.columns:
                corr = np.abs(X[col].corr(y))
                
                if corr >= pearson_threshold:
                    leakage_candidates.append({
                        "feature": col,
                        "reason": "high_pearson_correlation",
                        "correlation": float(corr)
                    })
                    logger.warning(f"Leakage suspect: {col} (Pearson |r|={corr:.3f})")
                    
                    # RF probe for confirmation
                    try:
                        rf = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=42)
                        rf.fit(X[[col]], y)
                        r2 = rf.score(X[[col]], y)
                        
                        if r2 > 0.999:
                            logger.error(f"LEAKAGE CONFIRMED: {col} (RF R²={r2:.4f})")
                            return True, leakage_candidates
                    except:
                        pass
    
    except Exception as e:
        logger.warning(f"Leakage detection failed: {e}")
    
    return False, leakage_candidates

def apply_scaling(
    df: pl.DataFrame,
    target_col: str,
    model_type: str = "tree"
) -> tuple[pl.DataFrame, dict]:
    """
    Apply feature scaling based on model type.
    Return (scaled_df, scaling_metadata_dict)
    """
    logger.info(f"Applying scaling for model type: {model_type}")
    
    metadata = {
        "scaler_used": "None",
        "features_scaled": [],
        "features_not_scaled": [],
        "scaler_path": None
    }
    
    # Trees don't need scaling
    if model_type in ["tree", "xgboost", "random_forest", "gradient_boosting"]:
        metadata["scaler_used"] = "None"
        metadata["features_not_scaled"] = [c for c in df.columns if c != target_col]
        return df, metadata
    
    # Linear/SARIMA → StandardScaler
    if model_type in ["linear", "ridge", "lasso", "sarima", "prophet"]:
        try:
            from sklearn.preprocessing import StandardScaler
            import joblib
            
            df_pd = df.to_pandas()
            numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != target_col]
            
            if numeric_cols:
                scaler = StandardScaler()
                df_pd[numeric_cols] = scaler.fit_transform(df_pd[numeric_cols])
                
                metadata["scaler_used"] = "StandardScaler"
                metadata["features_scaled"] = numeric_cols
                metadata["features_not_scaled"] = [target_col]
                metadata["scaler_path"] = "scaler.joblib"
                
                df = pl.from_pandas(df_pd)
        except Exception as e:
            logger.warning(f"Scaling failed: {e}")
    
    return df, metadata

def main():
    parser = argparse.ArgumentParser(description="STEP 12 — Feature Extraction")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--split-mode", default="auto", choices=["auto", "random", "time_series"])
    parser.add_argument("--exclude-features", default="", help="Comma-separated features to exclude")
    
    args = parser.parse_args()
    
    try:
        output_dir = Path(args.output_dir)
        
        # Load inputs
        progress = load_progress(output_dir)
        step10 = load_step10(output_dir)
        step11 = load_step11(output_dir)
        
        target_col = progress["target_column"]
        time_col = step10.get("time_column_detected")
        recommended_features = step11.get("recommended_features", [])
        useful_lag_features = step11.get("useful_lag_features", [])
        significant_lags = step11.get("significant_lags", [])
        excluded_from_step11 = step11.get("excluded_features", {})
        multiple_series = step11.get("multiple_series_detected", False)
        series_id_col = step11.get("series_id_column")
        
        # Load cleaned data
        df = pl.read_parquet(output_dir / "cleaned.parquet")
        
        logger.info(f"Loaded {df.height} rows, {df.width} columns")
        logger.info(f"Recommended features: {recommended_features}")
        
        # Filter by recommended features + target
        features_to_use = [f for f in recommended_features if f in df.columns]
        features_to_use = [target_col] + features_to_use + (
            [time_col] if time_col and time_col in df.columns else []
        )
        
        df = df.select(features_to_use)
        
        # Apply user exclusions
        user_exclude = [f.strip() for f in args.exclude_features.split(",") if f.strip()]
        if user_exclude:
            logger.info(f"Applying user exclusions: {user_exclude}")
            df = df.drop([c for c in user_exclude if c in df.columns])
        
        # Feature engineering
        df = create_lag_features(df, target_col, useful_lag_features, significant_lags, series_id_col)
        df = create_rolling_features(df, target_col, series_id_col)
        
        if time_col and time_col in df.columns:
            df = add_calendar_features(df, time_col)
            df = add_trend_feature(df, time_col)
        
        # Remove time column and series ID from features
        cols_to_drop = [c for c in [time_col, series_id_col] if c and c in df.columns]
        df = df.drop(cols_to_drop)
        
        logger.info(f"After engineering: {df.width} features")
        
        # Remove zero-variance features
        df_clean, zero_var_excluded = remove_zero_variance_features(df, target_col)
        
        # Check minimum features
        final_feature_count = df_clean.width - 1  # exclude target
        if final_feature_count < 2:
            logger.error(f"Too few features after cleanup: {final_feature_count}")
            sys.exit(1)
        
        logger.info(f"After cleanup: {df_clean.width} columns ({final_feature_count} features + target)")
        
        # Detect leakage
        has_leakage, leakage_cands = detect_leakage(df_clean, target_col)
        
        if has_leakage:
            logger.error("LEAKAGE DETECTED - exiting with code 2")
            sys.exit(2)
        
        # Check for target column in features
        numeric_cols = df_clean.select(cs.numeric()).columns
        if target_col in numeric_cols and target_col != target_col:
            logger.error(f"Target column {target_col} found in features - leakage!")
            sys.exit(2)
        
        # Apply scaling
        df_scaled, scaling_metadata = apply_scaling(df_clean, target_col, "tree")
        
        # Get final feature list
        all_cols = df_scaled.columns
        final_features = [c for c in all_cols if c != target_col]
        
        # Build excluded dict
        features_excluded = dict(zero_var_excluded)
        for item in leakage_cands:
            features_excluded[item["feature"]] = "leakage_suspect"
        for item in user_exclude:
            if item not in features_excluded:
                features_excluded[item] = "user_excluded"
        
        # Ensure no re-inclusion of step-11 excluded features
        for feat in features_excluded:
            if feat in excluded_from_step11:
                logger.warning(f"Feature {feat} was excluded in step 11, re-excluded")
        
        # Write features parquet
        df_scaled.write_parquet(output_dir / "features.parquet")
        logger.info(f"Wrote {df_scaled.height} rows x {df_scaled.width} cols to features.parquet")
        
        # Determine split strategy
        if args.split_mode == "auto":
            if time_col:
                split_mode = "time_series"
            else:
                split_mode = "random"
        else:
            split_mode = args.split_mode
        
        # Build output JSON
        output = {
            "step": "12-feature-extraction",
            "run_id": args.run_id,
            "features": final_features,
            "features_count": len(final_features),
            "features_excluded": features_excluded,
            "excluded_count": len(features_excluded),
            "target_column": target_col,
            "split_strategy": {
                "resolved_mode": split_mode,
                "time_column": time_col,
                "multiple_series": multiple_series
            },
            "leakage": {
                "status": "fail" if has_leakage else "pass",
                "leakage_candidates": leakage_cands,
                "threshold": 0.98
            },
            "scaling_metadata": scaling_metadata,
            "artifacts": {
                "features_parquet": str(output_dir / "features.parquet"),
                "scaler_joblib": str(output_dir / "scaler.joblib") if scaling_metadata.get("scaler_path") else None
            }
        }
        
        # Write output JSON
        (output_dir / "step-12-features.json").write_text(json.dumps(output, indent=2))
        logger.info(f"Wrote step-12-features.json")
        
        # Update progress
        progress["status"] = "completed"
        progress["completed_steps"].append("12-feature-extraction")
        progress["current_step"] = "13-model-training"
        (output_dir / "progress.json").write_text(json.dumps(progress, indent=2))
        
        logger.info("STEP 12 completed successfully")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"STEP 12 failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
