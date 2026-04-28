#!/usr/bin/env python3
"""
Step 11: Data Exploration
Performs comprehensive data profiling including MI ranking, correlation analysis,
time-series characterization, and feature filtering.
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
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

def load_step_output(output_dir: str, step: str) -> dict:
    """Load output JSON from a previous step."""
    path = Path(output_dir) / f"step-{step}.json"
    with open(path, 'r') as f:
        return json.load(f)

def detect_time_column(df: pl.DataFrame, columns: list) -> str | None:
    """Detect if there's a datetime column by type or name heuristic."""
    for col in columns:
        if df.schema[col] == pl.Date or df.schema[col] == pl.Datetime:
            return col
        if col.lower() in ['date', 'time', 'datetime', 'timestamp']:
            return col
    return None

def compute_mi_ranking(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42
) -> tuple[list, float]:
    """
    Compute mutual information ranking.
    
    Returns:
        (mi_ranking_list, noise_baseline)
    """
    # Remove any rows with NaN in X or y
    mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[mask].values
    y_clean = y[mask].values
    
    if len(X_clean) < 10:
        raise ValueError("Not enough clean samples for MI computation")
    
    # Compute MI for real features
    mi_scores = mutual_info_regression(X_clean, y_clean, random_state=random_state)
    
    mi_ranking = []
    for feat_idx, feat_name in enumerate(X.columns):
        mi_ranking.append({
            "feature": feat_name,
            "mi_score": float(mi_scores[feat_idx]),
            "below_noise_baseline": False  # Will be updated after noise baseline
        })
    
    # Generate 5 random noise columns and compute their MI
    n_samples = len(X_clean)
    noise_mis = []
    for _ in range(5):
        noise = np.random.RandomState(random_state).normal(size=n_samples)
        noise_mi = mutual_info_regression(
            noise.reshape(-1, 1),
            y_clean,
            random_state=random_state
        )[0]
        noise_mis.append(noise_mi)
    
    noise_baseline = float(np.mean(noise_mis))
    
    # Mark features below baseline
    for entry in mi_ranking:
        if entry["mi_score"] <= noise_baseline:
            entry["below_noise_baseline"] = True
    
    # Sort by MI descending
    mi_ranking.sort(key=lambda x: x["mi_score"], reverse=True)
    
    return mi_ranking, noise_baseline

def filter_low_variance_columns(df: pd.DataFrame, threshold: float = 1e-4) -> list:
    """Identify low-variance numeric columns."""
    low_variance = []
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Min-max scale variance
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max - col_min > 0:
                col_range = col_max - col_min
                scaled_var = df[col].var() / (col_range ** 2)
                if scaled_var < threshold:
                    low_variance.append(col)
    return low_variance

def detect_redundant_columns(
    df: pd.DataFrame,
    mi_ranking: list,
    corr_threshold: float = 0.90
) -> list:
    """Detect redundant columns via correlation and MI comparison."""
    redundant = []
    
    # Build MI score dict for quick lookup
    mi_dict = {entry["feature"]: entry["mi_score"] for entry in mi_ranking}
    
    # Compute correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Find high-correlation pairs
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= corr_threshold:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                
                # Mark the one with lower MI as redundant
                mi_i = mi_dict.get(col_i, 0)
                mi_j = mi_dict.get(col_j, 0)
                
                if mi_i < mi_j:
                    redundant.append(col_i)
                else:
                    redundant.append(col_j)
    
    return list(set(redundant))

def detect_leakage_suspect(
    X: pd.DataFrame,
    y: pd.Series,
    xcorr_threshold: float = 0.98
) -> list:
    """Detect features that are essentially copies of the target."""
    leakage_suspects = []
    
    # Remove NaNs for correlation computation
    mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[mask]
    y_clean = y[mask]
    
    for col in X.columns:
        # Compute lag-0 cross-correlation
        xcorr = np.corrcoef(X_clean[col].values, y_clean.values)[0, 1]
        if abs(xcorr) > xcorr_threshold:
            leakage_suspects.append(col)
    
    return leakage_suspects

def perform_time_series_analysis(
    df: pl.DataFrame,
    target_col: str,
    time_col: str | None
) -> dict:
    """Perform time-series specific analysis if time column detected."""
    
    ts_analysis = {
        "time_series_detected": False,
        "time_column": time_col,
        "multiple_series_detected": False,
        "time_series_characteristics": {
            "trend_detected": False,
            "seasonality_detected": False,
            "stationarity": "unknown",
            "white_noise": False
        },
        "model_recommendations": [],
        "significant_lags": [],
        "useful_lag_features": []
    }
    
    if time_col is None:
        return ts_analysis
    
    ts_analysis["time_series_detected"] = True
    
    try:
        # Convert to pandas for statsmodels
        df_pd = df.to_pandas()
        
        # Sort by time column
        if time_col in df_pd.columns:
            df_pd = df_pd.sort_values(by=time_col).reset_index(drop=True)
        
        target_series = df_pd[target_col].values
        
        # Remove NaNs
        target_series = target_series[~np.isnan(target_series)]
        
        if len(target_series) < 20:
            return ts_analysis
        
        # Stationarity test (ADF)
        try:
            adf_result = adfuller(target_series, autolag='AIC')
            p_value = adf_result[1]
            ts_analysis["time_series_characteristics"]["stationarity"] = (
                "stationary" if p_value < 0.05 else "non-stationary"
            )
        except Exception as e:
            print(f"[Step 11] ADF test failed: {e}", file=sys.stderr)
        
        # White noise test (Ljung-Box)
        try:
            lb_result = acorr_ljungbox(target_series, lags=min(10, len(target_series) // 4))
            if len(lb_result) > 0:
                # If all p-values > 0.05, likely white noise
                if (lb_result[1] > 0.05).all():
                    ts_analysis["time_series_characteristics"]["white_noise"] = True
        except Exception as e:
            print(f"[Step 11] Ljung-Box test failed: {e}", file=sys.stderr)
        
        # Autocorrelation analysis for significant lags
        try:
            acf_values = stats.pearsonr(target_series[:-1], target_series[1:])[0]
            
            # Check lags 1-24 for autocorrelation
            significant_lags = []
            max_lag = min(24, len(target_series) // 4)
            for lag in range(1, max_lag + 1):
                if lag < len(target_series):
                    try:
                        lag_corr = abs(np.corrcoef(target_series[:-lag], target_series[lag:])[0, 1])
                        if lag_corr > 0.1:
                            significant_lags.append(lag)
                    except:
                        pass
            
            ts_analysis["significant_lags"] = significant_lags[:10]  # Limit to top 10
        except Exception as e:
            print(f"[Step 11] Autocorrelation analysis failed: {e}", file=sys.stderr)
        
        # Detect trend (simple heuristic: regression slope significance)
        try:
            x = np.arange(len(target_series))
            z = np.polyfit(x, target_series, 1)
            slope = z[0]
            if abs(slope) > np.std(target_series) / len(target_series):
                ts_analysis["time_series_characteristics"]["trend_detected"] = True
        except:
            pass
        
        # Detect seasonality (heuristic: check if autocorrelation peaks at seasonal lags)
        try:
            if len(ts_analysis["significant_lags"]) > 2:
                ts_analysis["time_series_characteristics"]["seasonality_detected"] = True
        except:
            pass
        
        # Model recommendations based on characteristics
        recommendations = []
        if ts_analysis["time_series_characteristics"]["stationarity"] == "non-stationary":
            if ts_analysis["time_series_characteristics"]["seasonality_detected"]:
                recommendations.append("SARIMA")
            else:
                recommendations.append("ARIMA")
        
        if not ts_analysis["time_series_characteristics"]["white_noise"]:
            recommendations.append("XGBoost")
        
        if not recommendations:
            recommendations.append("Naive")
        
        ts_analysis["model_recommendations"] = recommendations[:3]
        
        # Cross-correlation analysis for useful lag features
        numeric_cols = df.select_dtypes([pl.Float64, pl.Float32, pl.Int64, pl.Int32]).columns
        useful_lags = []
        
        for feat_col in numeric_cols[:10]:  # Limit to first 10 numeric features
            if feat_col != target_col:
                feat_series = df_pd[feat_col].values
                if not pd.isna(feat_series).all():
                    feat_series = feat_series[~np.isnan(feat_series)]
                    
                    for lag in range(0, 4):
                        if lag < len(feat_series):
                            try:
                                xcorr = abs(np.corrcoef(feat_series[lag:], target_series[:len(feat_series)-lag])[0, 1])
                                if xcorr > 0.15:
                                    useful_lags.append({
                                        "feature": feat_col,
                                        "lag": lag,
                                        "xcorr": float(xcorr)
                                    })
                            except:
                                pass
        
        ts_analysis["useful_lag_features"] = useful_lags[:20]
        
    except Exception as e:
        print(f"[Step 11] Time-series analysis failed: {e}", file=sys.stderr)
    
    return ts_analysis

def main():
    parser = argparse.ArgumentParser(
        description="Step 11: Data Exploration"
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
        print("[Step 11] Starting data exploration...")
        
        # Load step 10 output and cleaned data
        step10 = load_step_output(output_dir, "10-cleanse")
        target_column = step10["target_column_normalized"]
        parquet_path = step10["artifacts"]["cleaned_parquet"]
        
        # Load cleaned parquet
        df = pl.read_parquet(parquet_path)
        print(f"[Step 11] Loaded {df.height} rows × {df.width} columns")
        
        # Convert to pandas for sklearn/statsmodels compatibility
        df_pd = df.to_pandas()
        
        # Identify numeric columns
        numeric_columns = df_pd.select_dtypes(include=[np.number]).columns.tolist()
        print(f"[Step 11] Numeric columns: {len(numeric_columns)}")
        
        if target_column not in numeric_columns:
            raise ValueError(f"Target column {target_column} is not numeric")
        
        # Prepare feature matrix (exclude target)
        X = df_pd[numeric_columns].drop(columns=[target_column])
        y = df_pd[target_column]
        
        # Filter low-variance columns
        low_variance_cols = filter_low_variance_columns(X)
        print(f"[Step 11] Low-variance columns: {low_variance_cols}")
        
        # Compute MI ranking
        print("[Step 11] Computing mutual information ranking...")
        mi_ranking, noise_baseline = compute_mi_ranking(X, y)
        print(f"[Step 11] Noise MI baseline: {noise_baseline:.6f}")
        
        # Detect redundant columns
        print("[Step 11] Detecting redundant columns...")
        redundant_cols = detect_redundant_columns(df_pd[numeric_columns], mi_ranking)
        print(f"[Step 11] Redundant columns: {redundant_cols}")
        
        # Detect leakage suspects
        print("[Step 11] Detecting leakage suspects...")
        leakage_suspects = detect_leakage_suspect(X, y)
        print(f"[Step 11] Leakage suspects: {leakage_suspects}")
        
        # Build excluded features dict
        excluded_features = {}
        for col in low_variance_cols:
            excluded_features[col] = "low_variance"
        for col in redundant_cols:
            excluded_features[col] = "redundant"
        for col in leakage_suspects:
            excluded_features[col] = "leakage_suspect"
        
        # Build recommended features (all numeric except those excluded)
        all_features = set(numeric_columns) - {target_column}
        recommended_features = [
            f for f in all_features
            if f not in excluded_features
        ]
        
        # If no features remain, loosen threshold
        if not recommended_features:
            print("[Step 11] WARNING: No features passed filters, loosening threshold...")
            # Use features that are not obvious leakage suspects
            recommended_features = [
                f for f in all_features
                if f not in leakage_suspects
            ]
        
        if not recommended_features:
            recommended_features = list(all_features)[:10]  # Fallback: use first 10
        
        print(f"[Step 11] Recommended features: {len(recommended_features)}")
        
        # Time-series analysis
        time_col = detect_time_column(df, df.columns)
        ts_analysis = perform_time_series_analysis(df, target_column, time_col)
        
        # Build step output
        step_output = {
            "step": "11-data-exploration",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "shape": {
                "rows": df.height,
                "columns": df.width
            },
            "numeric_columns": numeric_columns,
            "high_cardinality": [],
            "low_variance_columns": low_variance_cols,
            "time_series_detected": ts_analysis["time_series_detected"],
            "time_column": ts_analysis["time_column"],
            "multiple_series_detected": ts_analysis["multiple_series_detected"],
            "time_series_characteristics": ts_analysis["time_series_characteristics"],
            "model_recommendations": ts_analysis["model_recommendations"],
            "mi_ranking": mi_ranking,
            "noise_mi_baseline": noise_baseline,
            "redundant_columns": redundant_cols,
            "correlation_matrix_summary": {
                "max_pair": [],
                "max_corr": 0.0
            },
            "significant_lags": ts_analysis["significant_lags"],
            "useful_lag_features": ts_analysis["useful_lag_features"],
            "recommended_features": recommended_features,
            "excluded_features": excluded_features,
            "target_candidates": [target_column],
            "client_facing_summary": (
                f"Dataset contains {len(recommended_features)} predictive features. "
                f"Target variable shows {'trend' if ts_analysis['time_series_characteristics']['trend_detected'] else 'no trend'} "
                f"and {'seasonality' if ts_analysis['time_series_characteristics']['seasonality_detected'] else 'no seasonality'}. "
                f"Recommended models: {', '.join(ts_analysis['model_recommendations'])}"
            ),
            "context": {}
        }
        
        # Write step output
        output_json_path = Path(output_dir) / "step-11-exploration.json"
        with open(output_json_path, 'w') as f:
            json.dump(step_output, f, indent=2)
        print(f"[Step 11] Output written to: {output_json_path}")
        
        # Update progress
        progress_path = Path(output_dir) / "progress.json"
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        progress["current_step"] = "11-data-exploration"
        progress["status"] = "running"
        progress["completed_steps"].append("10-csv-read-cleansing")
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("[Step 11] ✓ Completed successfully")
        sys.exit(0)
        
    except Exception as e:
        print(f"[Step 11] ✗ Failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
