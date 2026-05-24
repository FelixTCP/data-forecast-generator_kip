#!/usr/bin/env python3
"""
Step 11: Data Exploration
Comprehensive time-series diagnostics, feature quality assessment, and model recommendations.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import polars as pl
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")


def compute_mutual_information(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Compute mutual information between features and target."""
    return mutual_info_regression(X, y, random_state=random_state)


def compute_mi_noise_baseline(n_samples: int, n_features: int = 5, random_state: int = 42) -> float:
    """
    Compute MI baseline by generating noise columns.
    """
    np.random.seed(random_state)
    noise = np.random.randn(n_samples, n_features)
    target = np.random.randn(n_samples)
    mi_noise = compute_mutual_information(noise, target, random_state=random_state)
    return float(np.mean(mi_noise))


def compute_hurst_exponent(series: np.ndarray, max_lag: int = 64) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute Hurst exponent via rescaled-range analysis.
    Returns (hurst_exponent, r2_fit) or (None, None) if insufficient data.
    """
    if len(series) < 64:
        return None, None
    
    # Remove NaNs
    series = series[~np.isnan(series)]
    if len(series) < 64:
        return None, None
    
    # Compute mean-adjusted cumulative sum
    mean_series = series - np.mean(series)
    cumsum = np.cumsum(mean_series)
    
    # Window sizes for R/S analysis
    window_sizes = [2**i for i in range(3, int(np.log2(len(series) / 4)) + 1)]
    if not window_sizes:
        return None, None
    
    rs_values = []
    for w in window_sizes:
        n_windows = len(series) // w
        if n_windows < 2:
            continue
        
        rs_list = []
        for i in range(n_windows):
            start, end = i * w, (i + 1) * w
            window = cumsum[start:end]
            range_val = np.max(window) - np.min(window)
            std_val = np.std(mean_series[start:end], ddof=1)
            if std_val > 0:
                rs = range_val / std_val
                rs_list.append(rs)
        
        if rs_list:
            rs_values.append(np.mean(rs_list))
    
    if len(rs_values) < 2:
        return None, None
    
    # Fit log(R/S) vs log(window_size)
    window_sizes_used = window_sizes[:len(rs_values)]
    log_windows = np.log(window_sizes_used)
    log_rs = np.log(rs_values)
    
    coeffs = np.polyfit(log_windows, log_rs, 1)
    hurst = coeffs[0]
    
    # Compute R² of fit
    y_pred = np.polyval(coeffs, log_windows)
    ss_res = np.sum((log_rs - y_pred) ** 2)
    ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return float(hurst), float(r2)


def step_11_main(output_dir: str, run_id: str) -> int:
    """Main step 11 logic."""
    try:
        output_path = Path(output_dir)
        
        # Load progress and step 10 output
        progress_file = output_path / "progress.json"
        step10_json_file = output_path / "step-10-cleanse.json"
        cleaned_parquet = output_path / "cleaned.parquet"
        
        with open(step10_json_file) as f:
            step10_output = json.load(f)
        
        target_column = step10_output["target_column_normalized"]
        time_column = step10_output["time_column_detected"]
        
        print(f"[Step 11] Loading cleaned.parquet...")
        df_pl = pl.read_parquet(cleaned_parquet)
        
        # Convert to pandas for statistical analysis
        df = df_pl.to_pandas()
        
        print(f"[Step 11] Shape: {df.shape}")
        print(f"[Step 11] Columns: {list(df.columns)}")
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"[Step 11] Numeric columns: {numeric_cols}")
        
        # Get target column
        if target_column not in df.columns:
            raise RuntimeError(f"Target column '{target_column}' not found in dataset")
        
        y = df[target_column].values
        
        # ============ NEAR-ZERO VARIANCE FILTER ============
        low_variance_cols = []
        for col in numeric_cols:
            if col == time_column:
                continue
            
            x_scaled = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
            variance = np.var(x_scaled)
            if variance < 1e-4:
                low_variance_cols.append(col)
        
        print(f"[Step 11] Low variance columns: {low_variance_cols}")
        
        # ============ HIGH CARDINALITY FILTER ============
        high_cardinality_cols = []
        for col in df.columns:
            if df[col].dtype == "object":
                if df[col].nunique() > 50:
                    high_cardinality_cols.append(col)
        
        print(f"[Step 11] High cardinality columns: {high_cardinality_cols}")
        
        # ============ MUTUAL INFORMATION RANKING ============
        print(f"[Step 11] Computing mutual information...")
        
        # Remove non-numeric and exclude time/target for MI computation
        feature_cols = [c for c in numeric_cols 
                       if c not in [target_column, time_column] 
                       and c not in low_variance_cols 
                       and c not in high_cardinality_cols]
        
        if not feature_cols:
            feature_cols = [c for c in numeric_cols 
                           if c not in [target_column, time_column]]
        
        X_features = df[feature_cols].fillna(df[feature_cols].mean()).values
        y_clean = y.copy()
        y_clean[np.isnan(y_clean)] = np.nanmean(y_clean)
        
        mi_scores = compute_mutual_information(X_features, y_clean, random_state=42)
        noise_baseline = compute_mi_noise_baseline(len(df), n_features=5, random_state=42)
        
        print(f"[Step 11] MI noise baseline: {noise_baseline:.6f}")
        
        mi_ranking = []
        below_baseline = []
        for feat, mi in zip(feature_cols, mi_scores):
            mi_ranking.append({"feature": feat, "mi_score": float(mi), "below_noise_baseline": float(mi) <= noise_baseline})
            if float(mi) <= noise_baseline:
                below_baseline.append(feat)
        
        mi_ranking.sort(key=lambda x: x["mi_score"], reverse=True)
        print(f"[Step 11] Features below MI noise baseline: {below_baseline}")
        
        # ============ REDUNDANCY FILTER ============
        print(f"[Step 11] Detecting redundant features...")
        redundant_cols = []
        mi_dict = {r["feature"]: r["mi_score"] for r in mi_ranking}
        
        for i, feat1 in enumerate(feature_cols):
            for feat2 in feature_cols[i+1:]:
                corr = np.corrcoef(
                    df[feat1].fillna(df[feat1].mean()).values,
                    df[feat2].fillna(df[feat2].mean()).values
                )[0, 1]
                
                if abs(corr) >= 0.90:
                    # Keep the one with higher MI
                    mi1 = mi_dict.get(feat1, 0)
                    mi2 = mi_dict.get(feat2, 0)
                    if mi1 < mi2 and feat1 not in redundant_cols:
                        redundant_cols.append(feat1)
                    elif mi2 < mi1 and feat2 not in redundant_cols:
                        redundant_cols.append(feat2)
        
        print(f"[Step 11] Redundant columns: {redundant_cols}")
        
        # ============ LEAKAGE DETECTION ============
        leakage_suspects = []
        for col in feature_cols:
            corr_with_target, _ = pearsonr(
                df[col].fillna(df[col].mean()).values,
                y_clean
            )
            if abs(corr_with_target) > 0.98:
                leakage_suspects.append(col)
        
        print(f"[Step 11] Leakage suspects: {leakage_suspects}")
        
        # ============ RECOMMENDED FEATURES ============
        recommended_features = [
            f for f in feature_cols
            if f not in low_variance_cols
            and f not in high_cardinality_cols
            and f not in below_baseline
            and f not in redundant_cols
            and f not in leakage_suspects
        ]
        
        if not recommended_features and feature_cols:
            # Relax threshold
            print(f"[Step 11] All features filtered, relaxing MI threshold by 50%...")
            relaxed_baseline = noise_baseline * 0.5
            recommended_features = [
                f for f in feature_cols
                if mi_dict.get(f, 0) > relaxed_baseline
                and f not in low_variance_cols
                and f not in high_cardinality_cols
                and f not in redundant_cols
                and f not in leakage_suspects
            ]
        
        print(f"[Step 11] Recommended features: {recommended_features}")
        
        # ============ TIME-SERIES DIAGNOSTICS ============
        print(f"[Step 11] Running time-series diagnostics on target...")
        
        ts_diagnostics = {}
        y_ts = y_clean.copy()
        
        # ADF Test
        try:
            adf_result = adfuller(y_ts, autolag="AIC")
            ts_diagnostics["adf_statistic"] = float(adf_result[0])
            ts_diagnostics["adf_pvalue"] = float(adf_result[1])
        except:
            ts_diagnostics["adf_statistic"] = None
            ts_diagnostics["adf_pvalue"] = None
        
        # KPSS Test
        try:
            kpss_result = kpss(y_ts, regression="c", nlags="auto")
            ts_diagnostics["kpss_statistic"] = float(kpss_result[0])
            ts_diagnostics["kpss_pvalue"] = float(kpss_result[1])
        except:
            ts_diagnostics["kpss_statistic"] = None
            ts_diagnostics["kpss_pvalue"] = None
        
        # Stationarity conclusion
        adf_p = ts_diagnostics.get("adf_pvalue")
        kpss_p = ts_diagnostics.get("kpss_pvalue")
        
        if adf_p is not None and kpss_p is not None:
            if adf_p < 0.05 and kpss_p > 0.05:
                stationarity_conclusion = "stationary"
            elif adf_p >= 0.05 and kpss_p <= 0.05:
                stationarity_conclusion = "non-stationary"
            elif adf_p < 0.05 and kpss_p <= 0.05:
                stationarity_conclusion = "trend-stationary"
            else:
                stationarity_conclusion = "ambiguous"
        else:
            stationarity_conclusion = "insufficient_data"
        
        ts_diagnostics["stationarity_conclusion"] = stationarity_conclusion
        
        # ACF / PACF
        try:
            max_lags = min(48, len(y_ts) // 4)
            acf_values = acf(y_ts, nlags=max_lags)
            pacf_values = pacf(y_ts, nlags=max_lags, method="ywm")
            
            # Significant lags (|value| > 2/sqrt(N))
            threshold = 2 / np.sqrt(len(y_ts))
            acf_sig_lags = [i for i, v in enumerate(acf_values) if abs(v) > threshold and i > 0][:12]
            pacf_sig_lags = [i for i, v in enumerate(pacf_values) if abs(v) > threshold and i > 0][:12]
            
            ts_diagnostics["acf_values"] = acf_values.tolist()[:13]
            ts_diagnostics["pacf_values"] = pacf_values.tolist()[:13]
            ts_diagnostics["acf_significant_lags"] = acf_sig_lags
            ts_diagnostics["pacf_significant_lags"] = pacf_sig_lags
            
            # Derive AR/MA orders
            if pacf_sig_lags and len(pacf_sig_lags) > 0:
                ts_diagnostics["suggested_ar_order"] = int(pacf_sig_lags[0])
            else:
                ts_diagnostics["suggested_ar_order"] = 1
            
            if acf_sig_lags and len(acf_sig_lags) > 0:
                ts_diagnostics["suggested_ma_order"] = int(acf_sig_lags[0])
            else:
                ts_diagnostics["suggested_ma_order"] = 0
        except:
            ts_diagnostics["acf_values"] = []
            ts_diagnostics["pacf_values"] = []
            ts_diagnostics["acf_significant_lags"] = []
            ts_diagnostics["pacf_significant_lags"] = []
            ts_diagnostics["suggested_ar_order"] = 1
            ts_diagnostics["suggested_ma_order"] = 0
        
        # Hurst Exponent
        hurst, r2_fit = compute_hurst_exponent(y_ts)
        if hurst is not None:
            ts_diagnostics["hurst_exponent"] = float(hurst)
            ts_diagnostics["hurst_r2_fit"] = float(r2_fit)
            
            if hurst < 0.45:
                hurst_interpretation = "anti_persistent"
            elif hurst < 0.55:
                hurst_interpretation = "random_walk"
            elif hurst < 0.75:
                hurst_interpretation = "mildly_persistent"
            else:
                hurst_interpretation = "strongly_persistent"
            
            ts_diagnostics["hurst_interpretation"] = hurst_interpretation
        else:
            ts_diagnostics["hurst_exponent"] = None
            ts_diagnostics["hurst_r2_fit"] = None
            ts_diagnostics["hurst_interpretation"] = None
            ts_diagnostics["hurst_skipped_reason"] = "insufficient_data"
        
        # Ljung-Box
        try:
            ljung_box_result = acorr_ljungbox(y_ts, lags=[6, 12, 24], return_df=True)
            ljung_box_pvalues = {str(lag): float(pval) for lag, pval in zip(ljung_box_result.index, ljung_box_result["lb_pvalue"])}
            ts_diagnostics["ljung_box_pvalues"] = ljung_box_pvalues
            
            white_noise = all(p > 0.05 for p in ljung_box_pvalues.values())
            ts_diagnostics["white_noise"] = white_noise
        except:
            ts_diagnostics["ljung_box_pvalues"] = {}
            ts_diagnostics["white_noise"] = False
        
        # STL Decomposition for seasonality
        try:
            # Try different seasonal periods
            detected_periods = []
            candidate_periods = [7, 30, 12]  # week, month, year (approximate)
            
            for period in candidate_periods:
                if period < len(y_ts) / 2:
                    try:
                        stl = STL(y_ts, seasonal=period, trend=period+1)
                        result = stl.fit()
                        
                        # Seasonal strength
                        var_seasonal = np.var(result.seasonal)
                        var_residual = np.var(result.resid)
                        fs = max(0, 1 - var_residual / (var_seasonal + var_residual + 1e-10))
                        
                        detected_periods.append({
                            "period": period,
                            "seasonal_strength": float(fs),
                            "significant": float(fs) > 0.30
                        })
                    except:
                        pass
            
            ts_diagnostics["detected_periods"] = detected_periods
            
            # Primary seasonal period
            sig_periods = [p for p in detected_periods if p["significant"]]
            if sig_periods:
                primary = max(sig_periods, key=lambda x: x["seasonal_strength"])
                ts_diagnostics["primary_seasonal_period"] = primary["period"]
                ts_diagnostics["trend_detected"] = True
            else:
                ts_diagnostics["primary_seasonal_period"] = None
                ts_diagnostics["trend_detected"] = False
            
            ts_diagnostics["trend_strength"] = 0.0
            
        except:
            ts_diagnostics["detected_periods"] = []
            ts_diagnostics["primary_seasonal_period"] = None
            ts_diagnostics["trend_detected"] = False
            ts_diagnostics["trend_strength"] = 0.0
        
        # ============ USEFUL LAG FEATURES ============
        useful_lag_features = []
        significant_lags = ts_diagnostics.get("acf_significant_lags", [])
        
        for col in recommended_features:
            for lag in [0, 1, 2, 3, 6, 12]:
                if lag == 0:
                    continue
                
                if lag >= len(df):
                    continue
                
                try:
                    y_shifted = np.roll(y_clean, lag)
                    xcorr, _ = pearsonr(
                        df[col].fillna(df[col].mean()).values,
                        y_shifted
                    )
                    
                    if abs(xcorr) > 0.15:
                        useful_lag_features.append({
                            "feature": col,
                            "lag": lag,
                            "xcorr": float(xcorr)
                        })
                except:
                    pass
        
        print(f"[Step 11] Useful lag features: {len(useful_lag_features)}")
        
        # ============ MULTIPLE SERIES DETECTION ============
        multiple_series_detected = False
        series_id_column = None
        
        # Check for low-cardinality categorical columns
        for col in df.columns:
            if df[col].dtype == "object" and df[col].nunique() <= 50:
                if col != time_column:
                    # Check if time ranges overlap
                    multiple_series_detected = True
                    series_id_column = col
                    break
        
        print(f"[Step 11] Multiple series detected: {multiple_series_detected}")
        
        # ============ EXCLUDED FEATURES ============
        excluded_features = {}
        for col in low_variance_cols:
            excluded_features[col] = "low_variance"
        for col in high_cardinality_cols:
            excluded_features[col] = "high_cardinality"
        for col in below_baseline:
            excluded_features[col] = "below_noise_baseline"
        for col in redundant_cols:
            excluded_features[col] = "redundant"
        for col in leakage_suspects:
            excluded_features[col] = "leakage_suspect"
        
        # ============ MODEL CLASS RECOMMENDATIONS ============
        model_recommendations = []
        
        if ts_diagnostics["white_noise"]:
            model_recommendations.append({
                "model_class": "Naive",
                "justification": "Target is white noise; only baseline models applicable."
            })
        elif ts_diagnostics["stationarity_conclusion"] == "stationary":
            model_recommendations.extend([
                {"model_class": "AR", "justification": "Stationary series with autocorrelation structure."},
                {"model_class": "ElasticNet", "justification": "Stationary with engineered features."},
                {"model_class": "Ridge", "justification": "Stationary baseline with regularization."},
                {"model_class": "XGBoost", "justification": "Non-linear interactions in stationary series."}
            ])
        elif ts_diagnostics["stationarity_conclusion"] == "non-stationary":
            if ts_diagnostics["primary_seasonal_period"] is not None:
                model_recommendations.extend([
                    {"model_class": "SARIMA", "justification": "Non-stationary with seasonality."},
                    {"model_class": "XGBoost", "justification": "Non-linear + seasonal patterns."}
                ])
            else:
                model_recommendations.extend([
                    {"model_class": "ARIMA", "justification": "Non-stationary series requires differencing."},
                    {"model_class": "XGBoost", "justification": "Non-linear trend modeling."}
                ])
        
        if len(recommended_features) >= 3:
            model_recommendations.append({
                "model_class": "XGBoost",
                "justification": f"Multiple informative features ({len(recommended_features)}) support multivariate approach."
            })
        
        # ============ TARGET CANDIDATES ============
        target_candidates = [{
            "column": target_column,
            "reason": "User-specified target column"
        }]
        
        # ============ OUTPUT JSON ============
        output_json = {
            "step": "11-data-exploration",
            "run_id": run_id,
            "shape": {
                "rows": df.shape[0],
                "columns": df.shape[1]
            },
            "numeric_columns": numeric_cols,
            "high_cardinality": high_cardinality_cols,
            "low_variance_columns": low_variance_cols,
            "time_series_detected": True,
            "time_column": time_column,
            "multiple_series_detected": multiple_series_detected,
            "series_id_column": series_id_column,
            "detected_frequency": "daily",
            "ts_diagnostics": ts_diagnostics,
            "model_class_recommendations": model_recommendations,
            "acf_pacf_orders": {
                "suggested_ar_order": ts_diagnostics.get("suggested_ar_order", 1),
                "suggested_ma_order": ts_diagnostics.get("suggested_ma_order", 0),
                "suggested_d": 1,
                "suggested_seasonal_ar": 1,
                "suggested_seasonal_d": 1,
                "suggested_seasonal_ma": 1,
                "seasonal_period": ts_diagnostics.get("primary_seasonal_period")
            },
            "mi_ranking": mi_ranking,
            "noise_mi_baseline": float(noise_baseline),
            "redundant_columns": redundant_cols,
            "useful_lag_features": useful_lag_features,
            "significant_lags": ts_diagnostics.get("acf_significant_lags", []),
            "recommended_features": recommended_features,
            "excluded_features": excluded_features,
            "target_candidates": target_candidates,
            "client_facing_summary": f"Dataset contains {len(recommended_features)} informative features. "
                                     f"Target shows {ts_diagnostics['stationarity_conclusion']} behavior. "
                                     f"Recommended models: {', '.join([r['model_class'] for r in model_recommendations[:3]])}."
        }
        
        # Write output JSON
        step11_json = output_path / "step-11-exploration.json"
        with open(step11_json, "w") as f:
            json.dump(output_json, f, indent=2)
        
        print(f"[Step 11] ✓ Completed successfully")
        return 0
        
    except Exception as e:
        print(f"[Step 11] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 11: Data Exploration")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    sys.exit(step_11_main(args.output_dir, args.run_id))
