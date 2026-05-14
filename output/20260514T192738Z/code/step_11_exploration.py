#!/usr/bin/env python3
"""
Step 11: Data Exploration

Generate a decision-ready profile that critically evaluates feature quality 
AND performs deep time-series profiling.
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
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from scipy.stats import pearsonr
try:
    from statsmodels.tsa.stattools import adfuller, acf, ccf, kpss
    from statsmodels.graphics.tsaplots import plot_acf
except ImportError:
    adfuller = None
    acf = None
    ccf = None
    kpss = None


def detect_time_column(df_pl: pl.DataFrame) -> str | None:
    """Detect time column by dtype or name heuristics."""
    for col in df_pl.columns:
        if df_pl.schema[col] == pl.Date or df_pl.schema[col] == pl.Datetime:
            return col
    
    # Fallback: look for common names
    for col in df_pl.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ["date", "time", "timestamp"]):
            return col
    
    return None


def explore_data(
    output_dir: str,
    run_id: str,
    target_column: str,
) -> dict:
    """
    Perform comprehensive data exploration.
    
    Returns:
        dict: Exploration output JSON
    """
    output_dir_path = Path(output_dir)
    
    # Load cleaned data
    cleaned_parquet = output_dir_path / "cleaned.parquet"
    df_pl = pl.read_parquet(cleaned_parquet)
    
    # Load step 10 JSON for context
    with open(output_dir_path / "step-10-cleanse.json") as f:
        step_10_data = json.load(f)
    
    target_normalized = step_10_data["target_column_normalized"]
    
    # Convert to pandas for sklearn/statsmodels
    df_pd = df_pl.to_pandas()
    
    # Detect time column
    time_column = detect_time_column(df_pl)
    
    # Identify numeric columns
    numeric_columns = [col for col in df_pl.columns if df_pl.schema[col] in [
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    ]]
    
    # Check if target is numeric
    is_target_numeric = target_normalized in numeric_columns
    
    if not is_target_numeric:
        # Try to find a numeric target
        print(f"Warning: Target column '{target_normalized}' is not numeric. Attempting to find suitable numeric target...")
        
        # Look for likely target columns (avgtemperature, target, value, etc.)
        likely_targets = [col for col in numeric_columns if any(
            x in col.lower() for x in ["temperature", "target", "value", "price", "sales", "consumption"]
        )]
        
        if likely_targets:
            print(f"Found potential numeric targets: {likely_targets}")
            target_normalized = likely_targets[0]
            print(f"Using '{target_normalized}' as target for regression")
        else:
            # Use first numeric column as target
            if numeric_columns:
                target_normalized = numeric_columns[0]
                print(f"Using first numeric column '{target_normalized}' as target")
            else:
                raise ValueError(f"No numeric columns found for regression target")
    
    # Remove target from feature list for MI computation
    feature_candidates = [col for col in numeric_columns if col != target_normalized]
    
    # === FILTER 1: Near-zero variance ===
    low_variance_columns = []
    if len(feature_candidates) > 0:
        for col in feature_candidates:
            data = df_pd[col].dropna()
            if len(data) > 1:
                # Min-max scaling then compute variance
                x_min, x_max = data.min(), data.max()
                if x_max != x_min:
                    scaled = (data - x_min) / (x_max - x_min)
                    var = scaled.var()
                else:
                    var = 0.0
                
                if var < 1e-4:
                    low_variance_columns.append(col)
    
    # === FILTER 2: High cardinality (for string columns) ===
    high_cardinality_columns = []
    for col in df_pl.columns:
        if df_pl.schema[col] == pl.Utf8:  # String type
            n_unique = df_pd[col].nunique()
            if n_unique > 50:
                high_cardinality_columns.append(col)
    
    # === Recommended features before MI filtering ===
    excluded_features = {}
    
    for col in low_variance_columns:
        excluded_features[col] = "low_variance"
    
    for col in high_cardinality_columns:
        excluded_features[col] = "high_cardinality"
    
    features_for_mi = [col for col in feature_candidates if col not in excluded_features]
    
    # === FILTER 3: Mutual Information Ranking ===
    mi_ranking = []
    noise_mi_baseline = 0.0
    
    if len(features_for_mi) > 0 and target_normalized in df_pd.columns:
        X = df_pd[features_for_mi].fillna(df_pd[features_for_mi].mean()).values
        y = df_pd[target_normalized].fillna(df_pd[target_normalized].mean()).values
        
        # Compute MI for real features
        try:
            mi_scores = mutual_info_regression(X, y, random_state=42)
            for feat, mi_score in zip(features_for_mi, mi_scores):
                mi_ranking.append({
                    "feature": feat,
                    "mi_score": float(mi_score),
                    "below_noise_baseline": False  # Will update after noise baseline
                })
        except Exception as e:
            print(f"Warning: MI computation failed: {e}")
            mi_ranking = []
        
        # Compute noise baseline
        n_noise_cols = 5
        noise_mi_scores = []
        for i in range(n_noise_cols):
            noise_col = np.random.RandomState(42 + i).randn(len(y))
            try:
                noise_mi = mutual_info_regression(noise_col.reshape(-1, 1), y, random_state=42)
                noise_mi_scores.append(float(noise_mi[0]))
            except:
                pass
        
        noise_mi_baseline = float(np.mean(noise_mi_scores)) if noise_mi_scores else 0.0
        
        # Flag features below noise baseline
        for entry in mi_ranking:
            if entry["mi_score"] <= noise_mi_baseline:
                entry["below_noise_baseline"] = True
                excluded_features[entry["feature"]] = "below_noise_baseline"
        
        # Sort by MI descending
        mi_ranking.sort(key=lambda x: x["mi_score"], reverse=True)
    
    # === FILTER 4: Pairwise Correlation & Redundancy ===
    redundant_columns = []
    if len(features_for_mi) > 1:
        X_corr = df_pd[features_for_mi].fillna(df_pd[features_for_mi].mean())
        corr_matrix = X_corr.corr().abs()
        
        for i in range(len(features_for_mi)):
            for j in range(i+1, len(features_for_mi)):
                feat_i = features_for_mi[i]
                feat_j = features_for_mi[j]
                corr = corr_matrix.loc[feat_i, feat_j]
                
                if corr >= 0.90:
                    # Find MI scores for both
                    mi_i = next((x["mi_score"] for x in mi_ranking if x["feature"] == feat_i), 0.0)
                    mi_j = next((x["mi_score"] for x in mi_ranking if x["feature"] == feat_j), 0.0)
                    
                    # Mark lower MI one as redundant
                    lower_mi_feat = feat_i if mi_i <= mi_j else feat_j
                    if lower_mi_feat not in excluded_features:
                        excluded_features[lower_mi_feat] = "redundant"
                        redundant_columns.append(lower_mi_feat)
    
    # === FILTER 5: Leakage detection (lag-0 xcorr > 0.98) ===
    leakage_suspects = []
    for col in features_for_mi:
        if col not in excluded_features and target_normalized in df_pd.columns:
            x_valid = df_pd[col].dropna()
            y_valid = df_pd[target_normalized].dropna()
            
            # Align indices
            common_idx = x_valid.index.intersection(y_valid.index)
            if len(common_idx) > 1:
                x_aligned = df_pd.loc[common_idx, col].values
                y_aligned = df_pd.loc[common_idx, target_normalized].values
                
                try:
                    xcorr, _ = pearsonr(x_aligned, y_aligned)
                    if abs(xcorr) > 0.98:
                        excluded_features[col] = "leakage_suspect"
                        leakage_suspects.append(col)
                except:
                    pass
    
    # === TIME-SERIES PROFILING ===
    time_series_characteristics = {
        "trend_detected": False,
        "seasonality_detected": False,
        "stationarity": "unknown",
        "white_noise": False,
    }
    significant_lags = []
    useful_lag_features = []
    model_recommendations = []
    
    if time_column is not None and target_normalized in df_pd.columns:
        # Sort by time if available
        df_ts = df_pd.sort_values(time_column)
        target_series = df_ts[target_normalized].dropna().values
        
        if len(target_series) > 10 and adfuller is not None:
            # ADF stationarity test
            try:
                adf_result = adfuller(target_series, autolag='AIC')
                p_value = adf_result[1]
                time_series_characteristics["stationarity"] = "stationary" if p_value < 0.05 else "non-stationary"
            except:
                pass
            
            # Autocorrelation analysis
            try:
                acf_values = acf(target_series, nlags=min(24, len(target_series)//4), fft=False)
                for lag in range(1, len(acf_values)):
                    if abs(acf_values[lag]) > 0.1:
                        significant_lags.append(lag)
                
                # Detect seasonality from ACF
                if len(significant_lags) > 2:
                    lags_diff = [significant_lags[i+1] - significant_lags[i] for i in range(len(significant_lags)-1)]
                    if len(set(lags_diff)) <= 2:  # Regular pattern
                        time_series_characteristics["seasonality_detected"] = True
            except:
                pass
            
            # Detect trend (simple: check if first half mean != second half mean)
            if len(target_series) > 20:
                mid = len(target_series) // 2
                mean_first = np.mean(target_series[:mid])
                mean_second = np.mean(target_series[mid:])
                if abs(mean_first - mean_second) > 0.1 * np.std(target_series):
                    time_series_characteristics["trend_detected"] = True
        
        # Cross-correlation for lag features
        for feat in feature_candidates:
            if feat != target_normalized and feat in df_ts.columns:
                feat_series = df_ts[feat].dropna().values
                if len(feat_series) > 3 and len(target_series) > 3:
                    min_len = min(len(feat_series), len(target_series))
                    try:
                        for lag in range(0, 4):
                            if lag < len(feat_series) and lag < len(target_series):
                                xcorr = np.corrcoef(
                                    feat_series[:-lag] if lag > 0 else feat_series,
                                    target_series[lag:] if lag > 0 else target_series
                                )[0, 1]
                                if abs(xcorr) > 0.15:
                                    useful_lag_features.append({
                                        "feature": feat,
                                        "lag": lag,
                                        "xcorr": float(xcorr)
                                    })
                    except:
                        pass
    
    # === MODEL RECOMMENDATIONS ===
    model_recommendations = ["ridge", "gradient_boosting"]  # Always include these
    
    if time_column is not None:
        model_recommendations.extend(["arima", "holt_winters"])
        if time_series_characteristics["seasonality_detected"]:
            model_recommendations.insert(0, "sarima")
    
    # === RECOMMENDED FEATURES ===
    recommended_features = [
        col for col in features_for_mi
        if col not in excluded_features and col != target_normalized
    ]
    
    # Ensure non-empty
    if not recommended_features and features_for_mi:
        # Loosen threshold by 50%
        print("Warning: All features filtered. Loosening thresholds...")
        excluded_features_orig = excluded_features.copy()
        excluded_features = {
            k: v for k, v in excluded_features.items()
            if v not in ["below_noise_baseline"]  # Remove the most liberal filter
        }
        recommended_features = [
            col for col in features_for_mi
            if col not in excluded_features and col != target_normalized
        ]
    
    # === CLIENT FACING SUMMARY ===
    summary_parts = []
    if time_column:
        summary_parts.append(f"Time-series data detected (time column: {time_column}).")
    
    if time_series_characteristics["trend_detected"]:
        summary_parts.append("Strong trend detected in target variable.")
    
    if time_series_characteristics["seasonality_detected"]:
        summary_parts.append("Seasonal patterns detected.")
    
    if recommended_features:
        summary_parts.append(f"Key predictive features: {', '.join(recommended_features[:3])}.")
    
    if excluded_features:
        summary_parts.append(f"Features excluded due to low information: {len(excluded_features)} columns removed.")
    
    client_facing_summary = " ".join(summary_parts) if summary_parts else "Standard regression dataset with no strong temporal patterns."
    
    # === BUILD OUTPUT JSON ===
    output_json = {
        "step": "11-data-exploration",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        
        "shape": {
            "rows": df_pl.height,
            "columns": df_pl.width,
        },
        
        "numeric_columns": numeric_columns,
        "high_cardinality": high_cardinality_columns,
        "low_variance_columns": low_variance_columns,
        
        "time_series_detected": time_column is not None,
        "time_column": time_column,
        "multiple_series_detected": False,  # Not implemented yet
        
        "time_series_characteristics": time_series_characteristics,
        "model_recommendations": model_recommendations,
        
        "mi_ranking": mi_ranking,
        "noise_mi_baseline": float(noise_mi_baseline),
        
        "redundant_columns": redundant_columns,
        "correlation_matrix_summary": {
            "max_corr": 0.0,  # Simplified
        },
        
        "significant_lags": significant_lags,
        "useful_lag_features": useful_lag_features,
        
        "recommended_features": recommended_features,
        "excluded_features": excluded_features,
        
        "target_candidates": [target_normalized],
        "client_facing_summary": client_facing_summary,
        
        "context": {
            "dataset_id": run_id,
            "target_column": target_normalized,
            "time_column": time_column,
            "features": recommended_features,
            "split_strategy": {
                "resolved_mode": "time_series" if time_column else "random"
            },
            "model_candidates": [{"name": m} for m in model_recommendations],
            "metrics": {},
            "artifacts": {},
            "notes": [
                f"Explored {len(numeric_columns)} numeric features",
                f"Recommended {len(recommended_features)} features for modeling",
                f"Excluded {len(excluded_features)} features (reasons: {set(excluded_features.values())})",
            ]
        }
    }
    
    # Write JSON
    step_json_path = output_dir_path / "step-11-exploration.json"
    with open(step_json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    # Update progress
    progress_path = output_dir_path / "progress.json"
    with open(progress_path) as f:
        progress = json.load(f)
    
    progress["completed_steps"].append("11-data-exploration")
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"✓ Step 11 complete")
    print(f"  Time column: {time_column}")
    print(f"  Recommended features: {len(recommended_features)} / {len(feature_candidates)}")
    print(f"  Model recommendations: {model_recommendations}")
    print(f"  Report written to: {step_json_path}")
    
    return output_json


def main():
    parser = argparse.ArgumentParser(description="Step 11: Data Exploration")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--target-column", required=True, help="Target column")
    
    args = parser.parse_args()
    
    try:
        explore_data(
            output_dir=args.output_dir,
            run_id=args.run_id,
            target_column=args.target_column,
        )
        sys.exit(0)
    except Exception as e:
        print(f"✗ Step 11 failed: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
