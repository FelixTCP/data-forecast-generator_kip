#!/usr/bin/env python
"""
STEP 11 — Data Exploration

Produces a rigorous time-series profile including stationarity tests,
ACF/PACF analysis, Hurst exponent, seasonality detection, MI ranking,
and model class recommendations.

Exit code: 0 on success, non-zero on failure.
"""

import sys
import json
import argparse
import warnings
import polars as pl
import polars.selectors as cs
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def json_serialize_safe(obj):
    """Helper to convert numpy/pandas types to native Python types for JSON."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, (np.ndarray, list)):
        return [json_serialize_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: json_serialize_safe(v) for k, v in obj.items()}
    return obj

def load_progress(output_dir: Path) -> dict:
    """Load progress.json to get target column and run_id."""
    progress_file = output_dir / "progress.json"
    return json.loads(progress_file.read_text())

def load_step10(output_dir: Path) -> dict:
    """Load step-10-cleanse.json for time column info."""
    step10_file = output_dir / "step-10-cleanse.json"
    return json.loads(step10_file.read_text())

def compute_mi_ranking(X: pd.DataFrame, y: pd.Series, n_noise: int = 5) -> Tuple[List[dict], float]:
    """
    Compute mutual information ranking vs. target.
    Create random noise baseline.
    Return (mi_ranking_list, noise_baseline_mean)
    """
    from sklearn.feature_selection import mutual_info_regression
    
    # Compute MI for all features
    mi_scores = mutual_info_regression(X, y, random_state=42)
    
    # Create noise baseline
    noise_mis = []
    for _ in range(n_noise):
        noise_col = np.random.randn(len(y))
        noise_mi = mutual_info_regression(noise_col.reshape(-1, 1), y, random_state=42)[0]
        noise_mis.append(noise_mi)
    
    noise_baseline = np.mean(noise_mis)
    
    # Build ranking
    ranking = []
    for feature, mi in zip(X.columns, mi_scores):
        ranking.append({
            "feature": feature,
            "mi_score": float(mi),
            "below_noise_baseline": mi <= noise_baseline
        })
    
    # Sort by MI descending
    ranking.sort(key=lambda x: x["mi_score"], reverse=True)
    
    return ranking, noise_baseline

def compute_stationarity(target_series: np.ndarray) -> dict:
    """
    Run ADF and KPSS tests. Return joint stationarity conclusion.
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    
    result = {
        "adf_statistic": None,
        "adf_pvalue": None,
        "kpss_statistic": None,
        "kpss_pvalue": None,
        "stationarity_conclusion": None
    }
    
    if len(target_series) < 20:
        result["stationarity_conclusion"] = "insufficient_data"
        return result
    
    try:
        # ADF test
        adf_res = adfuller(target_series, autolag='AIC')
        adf_stat, adf_pval = adf_res[0], adf_res[1]
        result["adf_statistic"] = float(adf_stat)
        result["adf_pvalue"] = float(adf_pval)
        
        # KPSS test
        try:
            kpss_res = kpss(target_series, regression='c', nlags='auto')
            kpss_stat, kpss_pval = kpss_res[0], kpss_res[1]
        except:
            kpss_stat, kpss_pval = np.nan, np.nan
        
        result["kpss_statistic"] = float(kpss_stat) if not np.isnan(kpss_stat) else None
        result["kpss_pvalue"] = float(kpss_pval) if not np.isnan(kpss_pval) else None
        
        # Joint interpretation
        adf_reject = adf_pval < 0.05
        kpss_reject = (kpss_pval <= 0.05) if kpss_pval is not None else False
        
        if adf_reject and not kpss_reject:
            result["stationarity_conclusion"] = "stationary"
        elif not adf_reject and kpss_reject:
            result["stationarity_conclusion"] = "non-stationary"
        elif adf_reject and kpss_reject:
            result["stationarity_conclusion"] = "trend-stationary"
        else:
            result["stationarity_conclusion"] = "ambiguous"
    except Exception as e:
        logger.warning(f"Stationarity test failed: {e}")
        result["stationarity_conclusion"] = "test_failed"
    
    return result

def compute_acf_pacf(target_series: np.ndarray) -> dict:
    """Compute ACF/PACF and extract significant lags."""
    from statsmodels.tsa.stattools import acf, pacf
    
    result = {
        "acf_values": [],
        "pacf_values": [],
        "acf_significant_lags": [],
        "pacf_significant_lags": [],
        "suggested_ar_order": 0,
        "suggested_ma_order": 0
    }
    
    if len(target_series) < 10:
        return result
    
    try:
        n = len(target_series)
        max_lag = min(48, n // 4)
        threshold = 2 / np.sqrt(n)
        
        # ACF
        acf_vals = acf(target_series, nlags=max_lag, fft=False)
        result["acf_values"] = [float(v) for v in acf_vals[1:]]
        
        # PACF
        pacf_vals = pacf(target_series, nlags=max_lag, method='ywm')
        result["pacf_values"] = [float(v) for v in pacf_vals[1:]]
        
        # Significant lags
        acf_sig = [i+1 for i, v in enumerate(acf_vals[1:]) if abs(v) > threshold]
        pacf_sig = [i+1 for i, v in enumerate(pacf_vals[1:]) if abs(v) > threshold]
        
        result["acf_significant_lags"] = acf_sig[:10]  # Limit to first 10
        result["pacf_significant_lags"] = pacf_sig[:10]
        
        # Suggested orders (simple heuristic)
        if pacf_sig and all(pacf_sig[i] <= pacf_sig[i+1] + 2 for i in range(len(pacf_sig)-1)):
            result["suggested_ar_order"] = min(pacf_sig[0] if pacf_sig else 1, 3)
        
        if acf_sig:
            result["suggested_ma_order"] = 0  # MA detection is more complex
    
    except Exception as e:
        logger.warning(f"ACF/PACF computation failed: {e}")
    
    return result

def compute_hurst_exponent(target_series: np.ndarray) -> dict:
    """Compute Hurst exponent via R/S rescaled-range analysis."""
    result = {
        "hurst_exponent": None,
        "hurst_interpretation": None,
        "hurst_r2_fit": None,
        "hurst_skipped_reason": None
    }
    
    if len(target_series) < 64:
        result["hurst_skipped_reason"] = "insufficient_data"
        return result
    
    try:
        # Handle NaN
        target_series = target_series[~np.isnan(target_series)]
        if len(target_series) < 64:
            result["hurst_skipped_reason"] = "insufficient_data_after_nan_removal"
            return result
        
        # R/S analysis
        window_sizes = [8, 16, 32, 64, 128, 256]
        window_sizes = [w for w in window_sizes if w < len(target_series) // 2]
        
        if len(window_sizes) < 2:
            result["hurst_skipped_reason"] = "insufficient_data_for_rs_analysis"
            return result
        
        rs_ratios = []
        for window_size in window_sizes:
            n_windows = len(target_series) // window_size
            ratio_sum = 0
            
            for i in range(n_windows):
                window = target_series[i*window_size:(i+1)*window_size]
                mean_adj = window - np.mean(window)
                cumsum = np.cumsum(mean_adj)
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(window, ddof=1)
                if S > 0:
                    ratio_sum += R / S
            
            rs_ratios.append(ratio_sum / n_windows if n_windows > 0 else 1)
        
        # Fit log-log
        log_windows = np.log(window_sizes)
        log_ratios = np.log(rs_ratios)
        
        coeffs = np.polyfit(log_windows, log_ratios, 1)
        hurst = coeffs[0]
        
        # R² fit quality
        y_pred = np.polyval(coeffs, log_windows)
        ss_res = np.sum((log_ratios - y_pred) ** 2)
        ss_tot = np.sum((log_ratios - np.mean(log_ratios)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        result["hurst_exponent"] = float(hurst)
        result["hurst_r2_fit"] = float(r2)
        
        # Interpretation
        if hurst < 0.45:
            result["hurst_interpretation"] = "anti-persistent"
        elif hurst < 0.55:
            result["hurst_interpretation"] = "random_walk"
        elif hurst < 0.75:
            result["hurst_interpretation"] = "mildly_persistent"
        else:
            result["hurst_interpretation"] = "strongly_persistent"
    
    except Exception as e:
        logger.warning(f"Hurst computation failed: {e}")
        result["hurst_skipped_reason"] = "computation_error"
    
    return result

def compute_ljung_box(target_series: np.ndarray) -> dict:
    """Run Ljung-Box test for white noise."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    result = {
        "ljung_box_pvalues": {},
        "white_noise": False
    }
    
    if len(target_series) < 30:
        return result
    
    try:
        lags = [6, 12, 24]
        lags = [l for l in lags if l < len(target_series) // 5]
        
        if not lags:
            lags = [6]
        
        lb_result = acorr_ljungbox(target_series, lags=lags, return_df=True)
        
        for lag, pval in zip(lb_result.index, lb_result['lb_pvalue']):
            result["ljung_box_pvalues"][str(lag)] = float(pval)
        
        # White noise if all p-values > 0.05
        result["white_noise"] = all(p > 0.05 for p in result["ljung_box_pvalues"].values())
    
    except Exception as e:
        logger.warning(f"Ljung-Box test failed: {e}")
    
    return result

def detect_seasonality(target_series: np.ndarray, frequency: str) -> dict:
    """Detect seasonal periods and strength using STL."""
    from statsmodels.tsa.seasonal import STL
    
    result = {
        "detected_periods": [],
        "primary_seasonal_period": None,
        "trend_strength": 0.0,
        "trend_detected": False
    }
    
    if len(target_series) < 100:
        return result
    
    # Map frequency to candidate periods
    freq_map = {
        "10min": [6, 12, 36, 144, 1008],
        "hourly": [24, 168],
        "daily": [7, 30, 365],
        "monthly": [12]
    }
    
    candidate_periods = freq_map.get(frequency, [12, 24, 30])
    
    # Filter out periods larger than N/2
    candidate_periods = [p for p in candidate_periods if p < len(target_series) // 2]
    
    if not candidate_periods:
        candidate_periods = [min(12, len(target_series) // 4)]
    
    try:
        # Try STL with the first suitable period
        for period in candidate_periods[:3]:  # Limit to first 3 to save time
            try:
                stl = STL(target_series, seasonal=period+1 if period % 2 == 0 else period)
                res = stl.fit()
                
                # Seasonal strength
                seasonal_var = np.var(res.seasonal)
                residual_var = np.var(res.resid)
                fs = max(0, 1 - residual_var / (seasonal_var + residual_var)) if (seasonal_var + residual_var) > 0 else 0
                
                # Trend strength
                trend_var = np.var(res.trend)
                ft = max(0, 1 - residual_var / (trend_var + residual_var)) if (trend_var + residual_var) > 0 else 0
                
                result["detected_periods"].append({
                    "period": period,
                    "seasonal_strength": float(fs),
                    "significant": fs > 0.30
                })
                
                if result["trend_strength"] == 0:
                    result["trend_strength"] = float(ft)
                    result["trend_detected"] = ft > 0.30
            except:
                continue
        
        # Primary period is the one with highest seasonal strength among significant ones
        sig_periods = [p for p in result["detected_periods"] if p["significant"]]
        if sig_periods:
            result["primary_seasonal_period"] = max(sig_periods, key=lambda x: x["seasonal_strength"])["period"]
    
    except Exception as e:
        logger.warning(f"Seasonality detection failed: {e}")
    
    return result

def compute_cross_correlation_lags(feature_df: pd.DataFrame, target: pd.Series) -> List[dict]:
    """Compute cross-correlation lag analysis."""
    from scipy.stats import pearsonr
    
    useful_lags = []
    lags_to_test = [1, 2, 3, 6, 12]
    
    for feature_col in feature_df.columns:
        feature = feature_df[feature_col].values
        target_vals = target.values
        
        for lag in lags_to_test:
            if lag >= len(feature):
                continue
            
            # Shift feature backwards by lag (feature at t-lag aligned with target at t)
            xcorr, pval = pearsonr(feature[:-lag], target_vals[lag:])
            
            if abs(xcorr) > 0.15:
                useful_lags.append({
                    "feature": feature_col,
                    "lag": lag,
                    "xcorr": float(xcorr)
                })
    
    return useful_lags

def detect_multiple_series(df: pl.DataFrame, time_col: str) -> Tuple[bool, Optional[str]]:
    """Detect if dataset contains multiple time series."""
    # Look for categorical columns with low cardinality
    for col in df.columns:
        if col == time_col or df[col].dtype == pl.Categorical or df[col].dtype == pl.String:
            n_unique = df[col].n_unique()
            if 2 <= n_unique <= 50:
                # Check if time ranges overlap per group
                return True, col
    
    return False, None

def main():
    parser = argparse.ArgumentParser(description="STEP 11 — Data Exploration")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    try:
        output_dir = Path(args.output_dir)
        
        # Load progress and step 10 output
        progress = load_progress(output_dir)
        step10 = load_step10(output_dir)
        
        target_col = progress["target_column"]
        time_col = step10["time_column_detected"]
        
        logger.info(f"Target column: {target_col}, Time column: {time_col}")
        
        # Load cleaned data
        df = pl.read_parquet(output_dir / "cleaned.parquet")
        df_pd = df.to_pandas()
        
        # Get numeric and categorical columns
        numeric_cols = df.select(cs.numeric()).columns
        numeric_cols = [c for c in numeric_cols if c != target_col and c != time_col]
        
        logger.info(f"Numeric columns: {numeric_cols}")
        
        # Prepare target series
        target_series = df_pd[target_col].dropna().values
        
        # Initialize output
        output = {
            "step": "11-data-exploration",
            "run_id": args.run_id,
            "shape": {"rows": df.height, "columns": df.width},
            "numeric_columns": numeric_cols,
            "high_cardinality": [],
            "low_variance_columns": [],
            "time_series_detected": time_col is not None,
            "time_column": time_col,
            "multiple_series_detected": False,
            "series_id_column": None,
            "detected_frequency": "unknown",
            "ts_diagnostics": {},
            "model_class_recommendations": [],
            "acf_pacf_orders": {},
            "mi_ranking": [],
            "noise_mi_baseline": 0.0,
            "redundant_columns": [],
            "useful_lag_features": [],
            "significant_lags": [],
            "recommended_features": [],
            "excluded_features": {},
            "target_candidates": [{"column": target_col, "reason": "primary_target"}],
            "client_facing_summary": "",
            "context": {}
        }
        
        # Step 1: Near-zero variance filter
        for col in numeric_cols:
            series = df_pd[col].dropna()
            if len(series) > 0:
                min_val = series.min()
                max_val = series.max()
                if max_val == min_val:
                    range_val = 1.0
                else:
                    range_val = max_val - min_val
                
                scaled_var = np.var(series) / (range_val ** 2) if range_val > 0 else 0
                
                if scaled_var < 1e-4:
                    output["low_variance_columns"].append(col)
        
        # Step 2-4: MI ranking and leakage/redundancy detection
        features_for_mi = [c for c in numeric_cols if c not in output["low_variance_columns"]]
        
        if features_for_mi:
            X = df_pd[features_for_mi].dropna()
            y = df_pd[target_col].dropna()
            
            # Align indices
            valid_idx = X.index.intersection(y.index)
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
            
            if len(X) > 10:
                mi_ranking, noise_baseline = compute_mi_ranking(X, y)
                output["mi_ranking"] = mi_ranking
                output["noise_mi_baseline"] = float(noise_baseline)
                
                # Compute Pearson correlation for redundancy and leakage detection
                corr_matrix = X.corr()
                
                # Leakage suspects (|r| > 0.98 with target)
                target_corr = X.corrwith(y)
                
                for col in features_for_mi:
                    if col in target_corr.index:
                        r = abs(target_corr[col])
                        if r > 0.98:
                            output["excluded_features"][col] = "leakage_suspect"
                
                # Redundancy (|r| >= 0.90)
                for i, col1 in enumerate(features_for_mi):
                    for col2 in features_for_mi[i+1:]:
                        if col1 in corr_matrix.index and col2 in corr_matrix.columns:
                            r = abs(corr_matrix.loc[col1, col2])
                            if r >= 0.90:
                                # Keep higher MI
                                mi1 = next((m["mi_score"] for m in output["mi_ranking"] if m["feature"] == col1), 0)
                                mi2 = next((m["mi_score"] for m in output["mi_ranking"] if m["feature"] == col2), 0)
                                
                                if mi1 < mi2:
                                    output["excluded_features"][col1] = "redundant"
                                else:
                                    output["excluded_features"][col2] = "redundant"
                
                # Features below noise baseline
                for item in output["mi_ranking"]:
                    if item["below_noise_baseline"]:
                        output["excluded_features"][item["feature"]] = "below_noise_baseline"
        
        # Recommended features = numeric cols - excluded
        recommended = [c for c in numeric_cols if c not in output["excluded_features"] and c not in output["low_variance_columns"]]
        
        if not recommended and numeric_cols:
            # Loosen threshold
            logger.warning("All features excluded; loosening noise baseline threshold by 50%")
            recommended = numeric_cols[:5]  # Use top 5
            output["client_facing_summary"] += "\n[Warning: Feature filtering threshold was relaxed due to insufficient passing features.]"
        
        output["recommended_features"] = recommended
        
        # Time-series diagnostics (if time column exists)
        if time_col and len(target_series) >= 20:
            logger.info("Running time-series diagnostics...")
            
            # Stationarity
            stationarity = compute_stationarity(target_series)
            output["ts_diagnostics"].update(stationarity)
            
            # ACF/PACF
            acf_pacf = compute_acf_pacf(target_series)
            output["ts_diagnostics"].update(acf_pacf)
            output["acf_pacf_orders"]["suggested_ar_order"] = acf_pacf["suggested_ar_order"]
            output["acf_pacf_orders"]["suggested_ma_order"] = acf_pacf["suggested_ma_order"]
            
            # Extract significant lags from ACF
            output["significant_lags"] = acf_pacf["acf_significant_lags"]
            
            # Hurst exponent
            hurst = compute_hurst_exponent(target_series)
            output["ts_diagnostics"].update(hurst)
            
            # Ljung-Box
            ljung_box = compute_ljung_box(target_series)
            output["ts_diagnostics"].update(ljung_box)
            
            # Seasonality
            seasonality = detect_seasonality(target_series, "daily")
            output["ts_diagnostics"].update(seasonality)
            
            # Cross-correlation lags
            if output["recommended_features"]:
                features_df = df_pd[output["recommended_features"]].dropna()
                target_aligned = df_pd[target_col].loc[features_df.index]
                useful_lags = compute_cross_correlation_lags(features_df, target_aligned)
                output["useful_lag_features"] = useful_lags
        
        # Multiple series detection
        multi_series, series_col = detect_multiple_series(df, time_col)
        output["multiple_series_detected"] = multi_series
        output["series_id_column"] = series_col
        
        # Model class recommendations (simplified version)
        recommendations = []
        
        if output["ts_diagnostics"].get("white_noise"):
            recommendations.append({"model_class": "Naive", "justification": "Target is white noise; complex models not beneficial."})
        else:
            if output["ts_diagnostics"].get("stationarity_conclusion") == "stationary":
                recommendations.append({"model_class": "ARMA", "justification": "Stationary series suitable for ARMA models."})
                recommendations.append({"model_class": "Ridge", "justification": "Interpretable linear baseline."})
            elif output["ts_diagnostics"].get("stationarity_conclusion") == "non-stationary":
                recommendations.append({"model_class": "ARIMA", "justification": "Non-stationary series requires differencing."})
            
            if output["ts_diagnostics"].get("detected_periods"):
                recommendations.append({"model_class": "SARIMA", "justification": "Seasonal patterns detected."})
            
            if len(output["recommended_features"]) >= 3:
                recommendations.append({"model_class": "XGBoost", "justification": "Multiple features available for ensemble learning."})
                recommendations.append({"model_class": "ElasticNet", "justification": "Regularized linear regression with collinearity handling."})
        
        output["model_class_recommendations"] = recommendations
        
        # Client-facing summary
        summary_parts = [
            f"Dataset contains {df.height} records across {df.width} columns.",
            f"Target: {target_col}. Time column: {time_col}.",
            f"Numeric features: {len(numeric_cols)}. Recommended for modeling: {len(output['recommended_features'])}."
        ]
        
        if output["ts_diagnostics"].get("stationarity_conclusion") == "stationary":
            summary_parts.append("The target series is stationary, suitable for linear forecasting methods.")
        elif output["ts_diagnostics"].get("stationarity_conclusion") == "non-stationary":
            summary_parts.append("The target series is non-stationary; differencing is recommended.")
        
        if output["ts_diagnostics"].get("detected_periods"):
            summary_parts.append("Seasonal patterns detected in the data.")
        
        output["client_facing_summary"] = " ".join(summary_parts)
        
        # Write output JSON
        step_json_path = output_dir / "step-11-exploration.json"
        output = json_serialize_safe(output)
        step_json_path.write_text(json.dumps(output, indent=2))
        logger.info(f"Wrote step JSON to {step_json_path}")
        
        # Update progress
        progress["status"] = "completed"
        progress["completed_steps"].append("11-data-exploration")
        progress["current_step"] = "12-feature-extraction"
        (output_dir / "progress.json").write_text(json.dumps(progress, indent=2))
        
        logger.info("STEP 11 completed successfully")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"STEP 11 failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
