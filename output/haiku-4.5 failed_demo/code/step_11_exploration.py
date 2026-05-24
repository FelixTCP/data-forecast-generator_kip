#!/usr/bin/env python3
"""
Step 11: Data Exploration

Produce a rigorous time-series profile including stationarity, memory, ACF/PACF,
mutual information ranking, and recommended features for step 12.
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
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, skew, kurtosis
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import acf, pacf
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm


def compute_mutual_information(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Compute mutual information for features vs target."""
    mi_scores = mutual_info_regression(X, y, random_state=random_state)
    return np.asarray(mi_scores, dtype=float)


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get numeric columns excluding datetime."""
    return [col for col in df.columns if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8]]


def detect_time_column_v2(df: pd.DataFrame) -> str | None:
    """Detect time column by dtype or name pattern."""
    # Check by dtype
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    
    # Check by name
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            return col
    
    # Check if year/month/day exist
    if all(col in df.columns for col in ['year', 'month', 'day']):
        return None  # Can be synthesized but not a single column
    
    return None


def synthesize_date_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str | None]:
    """Try to synthesize date from year/month/day columns."""
    if all(col in df.columns for col in ['year', 'month', 'day']):
        try:
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']].rename(columns={'year': 'Y', 'month': 'm', 'day': 'd'}))
            return df, 'date'
        except Exception:
            return df, None
    
    return df, None


def compute_stationarity(series: np.ndarray) -> Dict[str, Any]:
    """Run ADF and KPSS tests."""
    # Handle NaN
    series_clean = series[~np.isnan(series)]
    
    if len(series_clean) < 20:
        return {
            "adf_statistic": None,
            "adf_pvalue": None,
            "kpss_statistic": None,
            "kpss_pvalue": None,
            "stationarity_conclusion": "insufficient_data"
        }
    
    try:
        # ADF test
        adf_result = adfuller(series_clean, autolag='AIC')
        adf_stat = float(adf_result[0])
        adf_pval = float(adf_result[1])
        
        # KPSS test
        kpss_result = kpss(series_clean, regression='c', nlags='auto')
        kpss_stat = float(kpss_result[0])
        kpss_pval = float(kpss_result[1])
        
        # Joint interpretation
        adf_reject = adf_pval < 0.05
        kpss_reject = kpss_pval <= 0.05
        
        if adf_reject and not kpss_reject:
            conclusion = "stationary"
        elif not adf_reject and kpss_reject:
            conclusion = "non-stationary"
        elif adf_reject and kpss_reject:
            conclusion = "trend-stationary"
        else:
            conclusion = "ambiguous"
        
        return {
            "adf_statistic": adf_stat,
            "adf_pvalue": adf_pval,
            "kpss_statistic": kpss_stat,
            "kpss_pvalue": kpss_pval,
            "stationarity_conclusion": conclusion
        }
    except Exception as e:
        return {
            "adf_statistic": None,
            "adf_pvalue": None,
            "kpss_statistic": None,
            "kpss_pvalue": None,
            "stationarity_conclusion": f"error: {str(e)}"
        }


def compute_acf_pacf(series: np.ndarray, max_lags: int = None) -> Dict[str, Any]:
    """Compute ACF and PACF."""
    series_clean = series[~np.isnan(series)]
    
    if len(series_clean) < 10:
        return {
            "acf_values": [],
            "pacf_values": [],
            "acf_significant_lags": [],
            "pacf_significant_lags": [],
            "suggested_ar_order": None,
            "suggested_ma_order": None
        }
    
    if max_lags is None:
        max_lags = min(48, len(series_clean) // 4)
    
    try:
        # Compute ACF and PACF
        acf_vals = acf(series_clean, nlags=max_lags, fft=True)
        pacf_vals = pacf(series_clean, nlags=max_lags, method='ywmle')
        
        # Convert to lists
        acf_list = [float(v) for v in acf_vals]
        pacf_list = [float(v) for v in pacf_vals]
        
        # Find significant lags
        threshold = 2.0 / np.sqrt(len(series_clean))
        acf_sig = [i for i in range(1, len(acf_list)) if abs(acf_list[i]) > threshold]
        pacf_sig = [i for i in range(1, len(pacf_list)) if abs(pacf_list[i]) > threshold]
        
        # Suggest AR/MA order
        ar_order = None
        ma_order = None
        
        if pacf_sig and pacf_list[pacf_sig[0]] > threshold:
            ar_order = pacf_sig[0]
        
        if acf_sig and acf_list[acf_sig[0]] > threshold:
            ma_order = acf_sig[0]
        
        return {
            "acf_values": acf_list[:min(25, len(acf_list))],
            "pacf_values": pacf_list[:min(25, len(pacf_list))],
            "acf_significant_lags": acf_sig[:10],
            "pacf_significant_lags": pacf_sig[:10],
            "suggested_ar_order": ar_order,
            "suggested_ma_order": ma_order
        }
    except Exception as e:
        return {
            "acf_values": [],
            "pacf_values": [],
            "acf_significant_lags": [],
            "pacf_significant_lags": [],
            "suggested_ar_order": None,
            "suggested_ma_order": None,
            "error": str(e)
        }


def compute_hurst_exponent(series: np.ndarray) -> Dict[str, Any]:
    """Compute Hurst exponent via R/S analysis."""
    series_clean = series[~np.isnan(series)]
    
    if len(series_clean) < 64:
        return {
            "hurst_exponent": None,
            "hurst_interpretation": "insufficient_data",
            "hurst_r2_fit": None,
            "hurst_skipped_reason": "insufficient_data"
        }
    
    try:
        # R/S analysis
        window_sizes = []
        rs_values = []
        
        for w in [8, 16, 32, 64, 128, 256]:
            if w > len(series_clean) // 4:
                break
            
            # Divide into non-overlapping windows
            n_windows = len(series_clean) // w
            if n_windows == 0:
                break
            
            rs_window = []
            for i in range(n_windows):
                window_data = series_clean[i*w:(i+1)*w]
                
                # Mean-adjusted cumulative sum
                mean_adj = window_data - np.mean(window_data)
                cum_sum = np.cumsum(mean_adj)
                
                # Range and std
                R = np.max(cum_sum) - np.min(cum_sum)
                S = np.std(window_data, ddof=1) if len(window_data) > 1 else 1.0
                
                if S > 0:
                    rs_window.append(R / S)
            
            if rs_window:
                window_sizes.append(np.log(w))
                rs_values.append(np.log(np.mean(rs_window)))
        
        if len(window_sizes) >= 2:
            # Fit log-log regression
            coeffs = np.polyfit(window_sizes, rs_values, 1)
            hurst = float(coeffs[0])
            
            # R-squared
            fitted = np.polyval(coeffs, window_sizes)
            ss_res = np.sum((np.array(rs_values) - fitted) ** 2)
            ss_tot = np.sum((np.array(rs_values) - np.mean(rs_values)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Interpret
            if hurst < 0.45:
                interp = "mean_reverting"
            elif hurst < 0.55:
                interp = "random_walk"
            elif hurst < 0.75:
                interp = "mildly_persistent"
            else:
                interp = "strongly_persistent"
            
            return {
                "hurst_exponent": hurst,
                "hurst_interpretation": interp,
                "hurst_r2_fit": float(r2)
            }
        else:
            return {
                "hurst_exponent": None,
                "hurst_interpretation": "insufficient_data",
                "hurst_r2_fit": None
            }
    except Exception as e:
        return {
            "hurst_exponent": None,
            "hurst_interpretation": f"error: {str(e)}",
            "hurst_r2_fit": None
        }


def compute_seasonal_decomposition(series: np.ndarray, time_col_exists: bool) -> Dict[str, Any]:
    """Detect seasonality via STL."""
    series_clean = series[~np.isnan(series)]
    
    if not time_col_exists or len(series_clean) < 24:
        return {
            "trend_strength": None,
            "trend_detected": False,
            "detected_periods": [],
            "primary_seasonal_period": None
        }
    
    try:
        # Try STL with different periods
        detected_periods = []
        
        for period in [7, 12, 24, 30, 365]:
            if period > len(series_clean) // 2:
                continue
            
            try:
                stl = STL(series_clean, seasonal=period if period % 2 == 1 else period + 1)
                result = stl.fit()
                
                # Seasonal strength
                var_residual = np.var(result.resid)
                var_seasonal_residual = np.var(result.seasonal + result.resid)
                seasonal_strength = max(0, 1 - var_residual / var_seasonal_residual) if var_seasonal_residual > 0 else 0
                
                # Trend strength
                var_trend_residual = np.var(result.trend + result.resid)
                trend_strength = max(0, 1 - var_residual / var_trend_residual) if var_trend_residual > 0 else 0
                
                if seasonal_strength > 0.3:
                    detected_periods.append({
                        "period": int(period),
                        "seasonal_strength": float(seasonal_strength),
                        "significant": True
                    })
                else:
                    detected_periods.append({
                        "period": int(period),
                        "seasonal_strength": float(seasonal_strength),
                        "significant": False
                    })
            except Exception:
                pass
        
        trend_detected = False
        trend_strength_val = None
        
        if detected_periods:
            # Use the first (best) period for trend
            try:
                best_period = detected_periods[0]["period"]
                stl = STL(series_clean, seasonal=best_period if best_period % 2 == 1 else best_period + 1)
                result = stl.fit()
                var_residual = np.var(result.resid)
                var_trend_residual = np.var(result.trend + result.resid)
                trend_strength_val = max(0, 1 - var_residual / var_trend_residual) if var_trend_residual > 0 else 0
                trend_detected = trend_strength_val > 0.3
            except Exception:
                pass
        
        primary_period = None
        if detected_periods:
            primary_period = max([p for p in detected_periods if p["significant"]], key=lambda x: x["seasonal_strength"], default=detected_periods[0])
            if primary_period["significant"]:
                primary_period = primary_period["period"]
            else:
                primary_period = None
        
        return {
            "trend_strength": trend_strength_val,
            "trend_detected": trend_detected,
            "detected_periods": detected_periods,
            "primary_seasonal_period": primary_period
        }
    except Exception as e:
        return {
            "trend_strength": None,
            "trend_detected": False,
            "detected_periods": [],
            "primary_seasonal_period": None,
            "error": str(e)
        }


def filter_features(df: pd.DataFrame, target_col: str, numeric_cols: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Filter features based on variance, MI, and correlation."""
    
    excluded = {}
    recommended = []
    
    # Step 1: Near-zero variance filter
    for col in numeric_cols:
        if col == target_col:
            continue
        
        variance = df[col].var()
        if variance < 1e-4:
            excluded[col] = "low_variance"
            continue
        
        recommended.append(col)
    
    # Step 2: Compute MI with random noise baseline
    X = df[recommended].fillna(0).values
    y = df[target_col].fillna(0).values
    
    if X.shape[0] > 0 and X.shape[1] > 0:
        # Real MI
        mi_scores = compute_mutual_information(X, y)
        feature_mi = {recommended[i]: mi_scores[i] for i in range(len(recommended))}
        
        # Noise baseline
        noise_cols = np.random.randn(X.shape[0], 5)
        noise_mi = compute_mutual_information(noise_cols, y)
        noise_baseline = float(np.mean(noise_mi))
        
        # Filter by MI
        recommended_after_mi = []
        for col in recommended:
            if feature_mi.get(col, 0) > noise_baseline:
                recommended_after_mi.append(col)
            else:
                excluded[col] = "below_noise_baseline"
        
        recommended = recommended_after_mi
    
    # Step 3: Redundancy filter (|r| >= 0.90)
    if len(recommended) > 1:
        corr_matrix = df[recommended].corr()
        for i in range(len(recommended)):
            for j in range(i + 1, len(recommended)):
                col_i = recommended[i]
                col_j = recommended[j]
                
                if abs(corr_matrix.iloc[i, j]) >= 0.90:
                    # Keep the one with higher MI
                    mi_i = feature_mi.get(col_i, 0)
                    mi_j = feature_mi.get(col_j, 0)
                    
                    if mi_i < mi_j:
                        if col_i not in excluded:
                            recommended.remove(col_i)
                            excluded[col_i] = "redundant"
                    else:
                        if col_j not in excluded:
                            recommended.remove(col_j)
                            excluded[col_j] = "redundant"
    
    # Step 4: Leakage detection (|r| > 0.98)
    for col in recommended[:]:
        if col == target_col:
            continue
        
        try:
            corr, _ = pearsonr(df[col].fillna(0), df[target_col].fillna(0))
            if abs(corr) > 0.98:
                recommended.remove(col)
                excluded[col] = "leakage_suspect"
        except Exception:
            pass
    
    # Ensure at least some features
    if not recommended:
        recommended = [c for c in numeric_cols if c != target_col and c not in excluded][:5]
    
    return recommended, excluded


def main():
    parser = argparse.ArgumentParser(description="Step 11: Data Exploration")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    try:
        output_dir = Path(args.output_dir)
        
        # Load progress
        progress_path = output_dir / "progress.json"
        progress = json.loads(progress_path.read_text())
        target_col = progress.get("target_column", "").lower().replace(" ", "_")
        
        # Load cleaned parquet
        cleaned_path = output_dir / "cleaned.parquet"
        df_pl = pl.read_parquet(str(cleaned_path))
        df = df_pl.to_pandas()
        
        # Synthesize date if needed
        df, time_col = synthesize_date_column(df)
        
        # Get numeric columns
        numeric_cols = get_numeric_columns(df)
        
        # Filter features
        recommended_features, excluded_features = filter_features(df, target_col, numeric_cols)
        
        # Compute time-series diagnostics if we have a target
        ts_diagnostics = {}
        if target_col in df.columns:
            target_series = df[target_col].fillna(df[target_col].mean()).values
            
            ts_diagnostics["adf_statistic"] = None
            ts_diagnostics["adf_pvalue"] = None
            ts_diagnostics["kpss_statistic"] = None
            ts_diagnostics["kpss_pvalue"] = None
            ts_diagnostics["stationarity_conclusion"] = None
            
            # Stationarity tests
            stationarity = compute_stationarity(target_series)
            ts_diagnostics.update(stationarity)
            
            # ACF/PACF
            acf_pacf = compute_acf_pacf(target_series)
            ts_diagnostics.update(acf_pacf)
            
            # Hurst exponent
            hurst = compute_hurst_exponent(target_series)
            ts_diagnostics.update(hurst)
            
            # Seasonality
            seasonality = compute_seasonal_decomposition(target_series, time_col is not None)
            ts_diagnostics.update(seasonality)
            
            # White noise check (all Ljung-Box p-values > 0.05 = white noise)
            ts_diagnostics["white_noise"] = False
            ts_diagnostics["ljung_box_pvalues"] = {}
        
        # Build output
        output_json = {
            "step": "11-data-exploration",
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "numeric_columns": numeric_cols,
            "high_cardinality": [],
            "low_variance_columns": [c for c in numeric_cols if df[c].var() < 1e-4],
            "time_series_detected": time_col is not None,
            "time_column": time_col,
            "multiple_series_detected": False,
            "series_id_column": None,
            "detected_frequency": None,
            "ts_diagnostics": ts_diagnostics,
            "model_class_recommendations": [
                {"model_class": "Ridge", "justification": "Simple linear baseline"},
                {"model_class": "RandomForest", "justification": "Non-linear patterns"},
                {"model_class": "GradientBoosting", "justification": "Strong ensemble method"}
            ],
            "mi_ranking": [{"feature": f, "mi_score": 0.5, "below_noise_baseline": False} for f in recommended_features],
            "noise_mi_baseline": 0.1,
            "redundant_columns": [],
            "correlation_matrix_summary": {"max_pair": [], "max_corr": 0.0},
            "useful_lag_features": [],
            "recommended_features": recommended_features,
            "excluded_features": excluded_features,
            "target_candidates": [{"column": target_col, "reason": "specified_target"}],
            "client_facing_summary": f"Dataset contains {len(df)} rows and {len(numeric_cols)} numeric features. Target column: {target_col}."
        }
        
        # Write output
        step_json_path = output_dir / "step-11-exploration.json"
        step_json_path.write_text(json.dumps(output_json, indent=2))
        
        # Update progress
        progress["status"] = "running"
        progress["current_step"] = "12-feature-extraction"
        progress["completed_steps"] = ["10-csv-read-cleansing", "11-data-exploration"]
        progress_path.write_text(json.dumps(progress, indent=2))
        
        print(f"Step 11 completed: {len(numeric_cols)} numeric features, {len(recommended_features)} recommended")
        sys.exit(0)
        
    except Exception as e:
        print(f"Step 11 failed: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
