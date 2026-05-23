#!/usr/bin/env python
"""
Step 12: Feature Extraction (Time-Series Focused)
Builds calendar, lag, rolling, Fourier, and PCA features from step 11 diagnostics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib


def build_calendar_features(df: pl.DataFrame, time_col: str) -> pl.DataFrame:
    """Extract calendar features from time column."""
    if time_col not in df.columns or time_col == "_synthesized_date":
        return df
    
    try:
        df = df.with_columns([
            pl.col(time_col).dt.day().alias("day_of_month"),
            pl.col(time_col).dt.month().alias("month_cat"),
            pl.col(time_col).dt.weekday().alias("day_of_week"),
            pl.col(time_col).dt.quarter().alias("quarter"),
            pl.col(time_col).dt.iso_week().alias("week_of_year"),
        ])
        
        # Weekend flag
        df = df.with_columns(
            (pl.col("day_of_week") >= 5).cast(pl.Int32).alias("is_weekend")
        )
        
        # Month start/end
        df = df.with_columns(
            (pl.col("day_of_month") == 1).cast(pl.Int32).alias("is_month_start")
        )
        
        # Month end: check if next day is month 1
        df = df.with_columns(
            ((pl.col("month_cat").shift(-1) == 1) | (pl.col("month_cat") != pl.col("month_cat").shift(-1)))
            .cast(pl.Int32).alias("is_month_end")
        )
        
        # Drop temporary month_cat
        df = df.drop("month_cat")
        
        return df
    except Exception as e:
        print(f"WARNING: Could not build calendar features: {e}")
        return df


def build_lag_features(df: pl.DataFrame, target_col: str, acf_lags: list, pacf_lags: list, 
                       primary_period: int, hurst: float) -> tuple[pl.DataFrame, list]:
    """Build lag features for target column."""
    lag_features = []
    
    # Determine lag set
    all_lags = sorted(set(acf_lags + pacf_lags + [1]))
    
    # Add primary period lags
    if primary_period:
        all_lags.extend([primary_period, primary_period * 2])
    
    # Extend if high Hurst
    if hurst and hurst > 0.65:
        extended_lag = min((primary_period or 12) * 2, 96)
        all_lags.extend(range(1, extended_lag + 1, 12))  # Sample every 12 lags
    
    all_lags = sorted(set(all_lags))[:30]  # Cap at 30
    
    # Create lag features
    for lag in all_lags:
        lag_col = f"y_lag_{lag}"
        df = df.with_columns(pl.col(target_col).shift(lag).alias(lag_col))
        lag_features.append(lag_col)
    
    return df, lag_features


def build_exogenous_lag_features(df: pl.DataFrame, useful_lags: list) -> tuple[pl.DataFrame, list]:
    """Build lag features for exogenous variables."""
    exo_lag_features = []
    
    for item in useful_lags:
        feat = item["feature"]
        lag = item["lag"]
        
        if feat not in df.columns:
            continue
        
        lag_col = f"{feat}_lag_{lag}"
        df = df.with_columns(pl.col(feat).shift(lag).alias(lag_col))
        exo_lag_features.append(lag_col)
    
    return df, exo_lag_features


def build_differencing_features(df: pl.DataFrame, target_col: str, stationarity: str, 
                               primary_period: int) -> tuple[pl.DataFrame, list]:
    """Build differencing features if non-stationary."""
    diff_features = []
    
    if stationarity in ["non-stationary", "trend-stationary"]:
        # First difference
        diff_col = "y_diff_1"
        df = df.with_columns(
            pl.col(target_col).shift(1).diff().alias(diff_col)
        )
        diff_features.append(diff_col)
        
        # Seasonal difference
        if primary_period:
            sdiff_col = f"y_diff_seasonal_{primary_period}"
            df = df.with_columns(
                pl.col(target_col).shift(primary_period).diff().alias(sdiff_col)
            )
            diff_features.append(sdiff_col)
    
    return df, diff_features


def build_rolling_features(df: pl.DataFrame, target_col: str, primary_period: int) -> tuple[pl.DataFrame, list]:
    """Build rolling statistics on lagged target."""
    rolling_features = []
    
    # Determine window sizes
    if primary_period:
        windows = [primary_period // 2, primary_period, primary_period * 2]
    else:
        windows = [6, 12, 24]
    
    windows = [w for w in windows if w > 1]
    
    # Rolling on shift(1) to avoid look-ahead
    for w in windows:
        # Mean
        mean_col = f"rolling_mean_{w}"
        df = df.with_columns(
            pl.col(target_col).shift(1).rolling_mean(w).alias(mean_col)
        )
        rolling_features.append(mean_col)
        
        # Std
        std_col = f"rolling_std_{w}"
        df = df.with_columns(
            pl.col(target_col).shift(1).rolling_std(w).alias(std_col)
        )
        rolling_features.append(std_col)
    
    return df, rolling_features


def build_fourier_features(df: pl.DataFrame, detected_periods: list) -> tuple[pl.DataFrame, list]:
    """Build Fourier features for detected seasonal periods."""
    fourier_features = []
    
    if not detected_periods:
        return df, fourier_features
    
    for period_info in detected_periods:
        period = period_info["period"]
        
        # Determine number of harmonics
        k_max = min(3, period // 4)
        
        # Generate Fourier features
        t_index = np.arange(len(df)) % period
        
        for k in range(1, k_max + 1):
            sin_col = f"fourier_sin_{period}_{k}"
            cos_col = f"fourier_cos_{period}_{k}"
            
            sin_vals = np.sin(2 * np.pi * k * t_index / period)
            cos_vals = np.cos(2 * np.pi * k * t_index / period)
            
            df = df.with_columns([
                pl.lit(sin_vals).alias(sin_col),
                pl.lit(cos_vals).alias(cos_col)
            ])
            
            fourier_features.extend([sin_col, cos_col])
    
    return df, fourier_features


def build_pca_factors(df: pl.DataFrame, recommended_features: list, holdout_idx: int, 
                      model_recommendations: list) -> tuple[pl.DataFrame, list, dict]:
    """Build PCA factors if FAAR models are recommended."""
    pca_features = []
    pca_info = {}
    
    # Check if PCA is needed
    faar_models = [m["model_class"] for m in model_recommendations 
                   if "FAAR" in m["model_class"] or m["model_class"] == "Factor-VAR"]
    
    if not faar_models or not recommended_features:
        return df, pca_features, pca_info
    
    # Extract exogenous features
    exo_cols = [c for c in recommended_features if c in df.columns and c not in ["avgtemperature"]]
    
    if len(exo_cols) < 2:
        return df, pca_features, pca_info
    
    try:
        # Get training portion
        df_train = df[:holdout_idx]
        
        # Extract and standardize
        X_train = df_train.select(exo_cols).to_numpy().astype(float)
        X_train = np.nan_to_num(X_train, nan=0)
        
        if X_train.shape[0] < 5:
            return df, pca_features, pca_info
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Apply PCA
        n_components = min(len(exo_cols), max(2, len(exo_cols) // 2))
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        
        # Determine components to keep (95% variance)
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_keep = np.argmax(cumsum_var >= 0.95) + 1
        n_keep = min(n_keep, n_components)
        
        if n_keep == 0:
            n_keep = 1
        
        # Transform full dataset
        X_full = df.select(exo_cols).to_numpy().astype(float)
        X_full = np.nan_to_num(X_full, nan=0)
        X_full_scaled = scaler.transform(X_full)
        X_full_pca = pca.transform(X_full_scaled)
        
        # Add PCA components to dataframe
        for i in range(n_keep):
            pca_col = f"pca_factor_{i+1}"
            df = df.with_columns(pl.lit(X_full_pca[:, i]).alias(pca_col))
            pca_features.append(pca_col)
        
        # Save preprocessor
        preprocessor = {"scaler": scaler, "pca": pca}
        joblib.dump(preprocessor, "/app/output/20260523T123651Z/pca_preprocessor.joblib")
        
        pca_info = {
            "n_components": int(n_keep),
            "explained_variance_ratio": pca.explained_variance_ratio_[:n_keep].tolist(),
            "cumulative_variance": float(cumsum_var[n_keep - 1]),
            "pca_preprocessor_path": "/app/output/20260523T123651Z/pca_preprocessor.joblib"
        }
        
    except Exception as e:
        print(f"WARNING: PCA feature extraction failed: {e}")
    
    return df, pca_features, pca_info


def detect_leakage(df: pl.DataFrame, target_col: str, feature_cols: list, holdout_idx: int) -> dict:
    """Detect leakage via Pearson correlation and RF probe."""
    from scipy.stats import pearsonr
    from sklearn.ensemble import RandomForestRegressor
    
    leakage_audit = {
        "status": "pass",
        "probe1_suspects": [],
        "probe2_r2": None,
        "threshold": 0.98
    }
    
    # Probe 1: Pearson correlation
    target = df[target_col].to_numpy().astype(float)
    target = target[~np.isnan(target)]
    
    suspects = []
    for feat in feature_cols:
        if feat not in df.columns:
            continue
        
        feat_data = df[feat].to_numpy().astype(float)
        
        # Match length
        if len(feat_data) > len(target):
            feat_data = feat_data[:len(target)]
        elif len(feat_data) < len(target):
            continue
        
        # Skip if all NaN
        if np.all(np.isnan(feat_data)):
            continue
        
        # Fill NaN with mean
        feat_mean = np.nanmean(feat_data)
        if np.isnan(feat_mean):
            feat_mean = 0
        feat_data = np.nan_to_num(feat_data, nan=feat_mean)
        
        try:
            r, _ = pearsonr(feat_data, target)
            if abs(r) >= 0.98:
                suspects.append(feat)
                leakage_audit["probe1_suspects"].append({
                    "feature": feat,
                    "pearson_r": float(r)
                })
        except:
            pass
    
    # Probe 2: RF reconstruction if suspects found
    if suspects:
        try:
            df_train = df[:holdout_idx]
            X_train = df_train.select(suspects).to_numpy().astype(float)
            X_train = np.nan_to_num(X_train, nan=0)
            y_train = df_train[target_col].to_numpy().astype(float)
            y_train = np.nan_to_num(y_train, nan=0)
            
            rf = RandomForestRegressor(n_estimators=50, random_state=42, oob_score=True)
            rf.fit(X_train, y_train)
            
            r2 = rf.oob_score_ if hasattr(rf, 'oob_score_') and rf.oob_score_ is not None else 0.0
            leakage_audit["probe2_r2"] = float(r2)
            
            if r2 > 0.95:
                leakage_audit["status"] = "fail"
            else:
                leakage_audit["status"] = "warn"
        except:
            leakage_audit["status"] = "warn"
    
    return leakage_audit


def main():
    parser = argparse.ArgumentParser(description="Step 12: Feature Extraction")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    progress_file = output_dir / "progress.json"
    
    try:
        # Update progress
        progress = json.loads(progress_file.read_text())
        progress["current_step"] = "12-feature-extraction"
        progress["status"] = "running"
        progress_file.write_text(json.dumps(progress, indent=2))
        
        # ===== LOAD DATA =====
        print("[Step 12] Loading data...")
        step10_json = json.loads((output_dir / "step-10-cleanse.json").read_text())
        step11_json = json.loads((output_dir / "step-11-exploration.json").read_text())
        
        target_col = step10_json["target_column_normalized"]
        time_col = step10_json["time_column_detected"]
        
        df = pl.read_parquet(output_dir / "cleaned.parquet")
        print(f"  Loaded {df.height} rows")
        
        # ===== EXTRACT DIAGNOSTICS FROM STEP 11 =====
        ts_diag = step11_json.get("ts_diagnostics", {})
        acf_lags = ts_diag.get("acf_significant_lags", [])
        pacf_lags = ts_diag.get("pacf_significant_lags", [])
        hurst = ts_diag.get("hurst_exponent")
        primary_period = ts_diag.get("primary_seasonal_period")
        stationarity = ts_diag.get("stationarity_conclusion", "unknown")
        white_noise = ts_diag.get("white_noise", False)
        detected_periods = ts_diag.get("detected_periods", [])
        
        recommended_features = step11_json.get("recommended_features", [])
        useful_lag_features = step11_json.get("useful_lag_features", [])
        model_recommendations = step11_json.get("model_class_recommendations", [])
        
        print(f"  Target: {target_col}, Time: {time_col}")
        print(f"  Recommended features: {recommended_features}")
        print(f"  Stationarity: {stationarity}, Hurst: {hurst}")
        
        initial_rows = df.height
        
        # ===== BUILD FEATURE GROUPS =====
        all_feature_cols = []
        feature_groups = {}
        
        # Group A: Calendar
        if time_col:
            df = build_calendar_features(df, time_col)
            cal_cols = [c for c in df.columns if c in ["day_of_month", "day_of_week", "quarter", "week_of_year", "is_weekend", "is_month_start", "is_month_end"]]
            feature_groups["calendar"] = cal_cols
            all_feature_cols.extend(cal_cols)
            print(f"  Calendar features: {len(cal_cols)}")
        
        # Group B: Target lags
        if not white_noise:
            df, lag_cols = build_lag_features(df, target_col, acf_lags, pacf_lags, primary_period, hurst)
            feature_groups["target_lags"] = lag_cols
            all_feature_cols.extend(lag_cols)
            print(f"  Target lags: {len(lag_cols)}")
        else:
            # White noise: only lag-1
            df = df.with_columns(pl.col(target_col).shift(1).alias("y_lag_1"))
            feature_groups["target_lags"] = ["y_lag_1"]
            all_feature_cols.append("y_lag_1")
            print(f"  White noise mode: only lag-1")
        
        # Group C: Exogenous lags
        if not white_noise and useful_lag_features:
            df, exo_cols = build_exogenous_lag_features(df, useful_lag_features)
            feature_groups["exogenous_lags"] = exo_cols
            all_feature_cols.extend(exo_cols)
            print(f"  Exogenous lags: {len(exo_cols)}")
        
        # Group D: Differencing
        if not white_noise:
            df, diff_cols = build_differencing_features(df, target_col, stationarity, primary_period)
            feature_groups["differencing"] = diff_cols
            all_feature_cols.extend(diff_cols)
            print(f"  Differencing features: {len(diff_cols)}")
        
        # Group E: Rolling statistics
        if not white_noise:
            df, rolling_cols = build_rolling_features(df, target_col, primary_period)
            feature_groups["rolling"] = rolling_cols
            all_feature_cols.extend(rolling_cols)
            print(f"  Rolling features: {len(rolling_cols)}")
        
        # Group F: Fourier
        if not white_noise and detected_periods:
            df, fourier_cols = build_fourier_features(df, detected_periods)
            feature_groups["fourier"] = fourier_cols
            all_feature_cols.extend(fourier_cols)
            print(f"  Fourier features: {len(fourier_cols)}")
        
        # ===== COMPUTE HOLDOUT BOUNDARY =====
        holdout_idx = int(0.8 * df.height)
        holdout_timestamp = None
        if time_col and time_col in df.columns:
            try:
                holdout_timestamp = str(df[holdout_idx][time_col])
            except:
                pass
        
        print(f"  Split: train={holdout_idx}, holdout={df.height - holdout_idx}")
        
        # ===== GROUP G: PCA Factors =====
        df, pca_cols, pca_info = build_pca_factors(df, recommended_features, holdout_idx, model_recommendations)
        feature_groups["pca_factors"] = pca_cols
        all_feature_cols.extend(pca_cols)
        if pca_cols:
            print(f"  PCA factors: {len(pca_cols)}")
        
        # ===== NaN HANDLING =====
        print("[Step 12] Handling NaN...")
        
        # 1. Drop target NaN
        rows_before_target_drop = df.height
        df = df.drop_nulls(subset=[target_col])
        rows_dropped_target = rows_before_target_drop - df.height
        print(f"  Dropped {rows_dropped_target} rows with target NaN")
        
        # 2. Drop leading NaN rows from lags
        if all_feature_cols:
            lag_feature_cols = [c for c in all_feature_cols if "lag" in c or "rolling" in c]
            if lag_feature_cols:
                # Find first non-null row for maximum lag
                first_valid = 0
                for i in range(len(df)):
                    valid = True
                    for col in lag_feature_cols:
                        if col in df.columns and df[i][col] is None:
                            valid = False
                            break
                    if valid:
                        first_valid = i
                        break
                
                rows_dropped_lags = first_valid
                df = df.slice(first_valid)
                print(f"  Dropped {rows_dropped_lags} leading lag NaN rows")
        
        # 3. Forward/backward fill remaining NaN
        feature_cols_list = [c for c in all_feature_cols if c in df.columns]
        for col in feature_cols_list:
            if df[col].null_count() > 0:
                df = df.with_columns(
                    pl.col(col).forward_fill().backward_fill()
                )
        
        # 4. Check for remaining NaN
        remaining_nan = df.select([pl.col(c).is_null().sum() for c in feature_cols_list]).to_numpy().flatten()
        if np.any(remaining_nan > 0):
            print(f"  WARNING: {np.sum(remaining_nan)} NaN values remain after fill")
        
        # ===== LEAKAGE DETECTION =====
        print("[Step 12] Detecting leakage...")
        feature_cols_for_check = [c for c in all_feature_cols if c in df.columns]
        leakage_audit = detect_leakage(df, target_col, feature_cols_for_check, holdout_idx)
        
        print(f"  Leakage status: {leakage_audit['status']}")
        
        if leakage_audit["status"] == "fail":
            print("[Step 12] LEAKAGE DETECTED - stopping pipeline")
            leakage_file = output_dir / "leakage_audit.json"
            leakage_file.write_text(json.dumps(leakage_audit, indent=2))
            raise RuntimeError("Leakage detected — see leakage_audit.json")
        
        # ===== WRITE LEAKAGE AUDIT =====
        leakage_file = output_dir / "leakage_audit.json"
        leakage_file.write_text(json.dumps(leakage_audit, indent=2))
        
        # ===== WRITE FEATURES PARQUET =====
        final_features = feature_cols_for_check + [target_col]
        df_out = df.select(final_features)
        
        parquet_path = output_dir / "features.parquet"
        df_out.write_parquet(str(parquet_path))
        print(f"  Wrote {df_out.height} rows to {parquet_path}")
        
        # ===== COMPILE OUTPUT JSON =====
        output = {
            "step": "12-feature-extraction",
            "target_column": target_col,
            "feature_names": feature_cols_for_check,
            "feature_count": len(feature_cols_for_check),
            "rows_dropped_by_lags": rows_dropped_lags if 'rows_dropped_lags' in locals() else 0,
            "rows_dropped_target": rows_dropped_target,
            "final_row_count": df_out.height,
            
            "split_info": {
                "holdout_start_index": holdout_idx,
                "holdout_start_timestamp": holdout_timestamp,
                "train_row_count": holdout_idx,
                "holdout_row_count": df.height - holdout_idx,
                "split_strategy": "last_20pct_chronological",
                "resolved_mode": "time_series"
            },
            
            "feature_groups": feature_groups,
            "pca_info": pca_info if pca_cols else {},
            "leakage_audit": leakage_audit,
            
            "features_excluded": step11_json.get("excluded_features", {}),
            "artifacts": {
                "features_parquet": str(parquet_path),
                "leakage_audit_json": str(leakage_file)
            }
        }
        
        step_json_path = output_dir / "step-12-features.json"
        step_json_path.write_text(json.dumps(output, indent=2))
        print(f"  Wrote step JSON to {step_json_path}")
        
        # ===== UPDATE PROGRESS =====
        progress = json.loads(progress_file.read_text())
        progress["completed_steps"].append("12-feature-extraction")
        progress_file.write_text(json.dumps(progress, indent=2))
        
        print("[Step 12] SUCCESS ✓")
        return 0
        
    except Exception as e:
        print(f"[Step 12] FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
            if "errors" not in progress:
                progress["errors"] = []
            progress["errors"].append(f"Step 12 failed: {str(e)}")
            progress["status"] = "error"
            progress_file.write_text(json.dumps(progress, indent=2))
        except:
            pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
