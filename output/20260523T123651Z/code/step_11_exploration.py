#!/usr/bin/env python
"""
Step 11: Data Exploration
Rigorous time-series profiling, feature quality gates, model class recommendations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import kpss as kpss_test, adfuller as adf_test
from statsmodels.stats.diagnostic import acorr_ljungbox


def compute_hurst_exponent(series: np.ndarray, min_window: int = 8, max_window: int = None) -> tuple[float, float, str]:
    """Compute Hurst exponent via R/S rescaled-range analysis."""
    if len(series) < 64:
        return None, None, "insufficient_data"
    
    series = series[~np.isnan(series)]
    if len(series) < 64:
        return None, None, "insufficient_data"
    
    # Use shorter window for smaller datasets
    if max_window is None:
        max_window = len(series) // 4
    
    # Generate window sizes
    window_sizes = []
    w = min_window
    while w <= max_window:
        window_sizes.append(w)
        w = int(w * 1.5) + 1  # Geometric progression
    
    if len(window_sizes) < 3:
        return None, None, "insufficient_windows"
    
    rs_values = []
    
    for window in window_sizes:
        # Split series into non-overlapping chunks
        n_chunks = len(series) // window
        if n_chunks < 1:
            continue
        
        chunk_rs = []
        for i in range(n_chunks):
            chunk = series[i*window:(i+1)*window]
            
            # Mean-adjusted cumsum
            mean_adj = chunk - np.mean(chunk)
            cumsum = np.cumsum(mean_adj)
            
            # Range
            r = np.max(cumsum) - np.min(cumsum)
            
            # Std dev
            s = np.std(chunk, ddof=1)
            
            if s > 0:
                chunk_rs.append(r / s)
        
        if chunk_rs:
            rs_values.append(np.mean(chunk_rs))
        else:
            continue
    
    if len(rs_values) < 3:
        return None, None, "insufficient_rs_values"
    
    # Fit log(R/S) vs log(n)
    log_window = np.log([window_sizes[i] for i in range(len(rs_values))])
    log_rs = np.log(rs_values)
    
    # OLS fit
    coeffs = np.polyfit(log_window, log_rs, 1)
    h = coeffs[0]
    
    # Compute R²
    y_pred = np.polyval(coeffs, log_window)
    ss_res = np.sum((log_rs - y_pred) ** 2)
    ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Interpret
    if h < 0.45:
        interpretation = "anti_persistent_mean_reverting"
    elif h < 0.55:
        interpretation = "random_walk"
    elif h < 0.75:
        interpretation = "mildly_persistent"
    else:
        interpretation = "strongly_persistent_trending"
    
    return h, r2, interpretation


def compute_mi_ranking(X: np.ndarray, y: np.ndarray, feature_names: list) -> tuple[dict, float]:
    """Compute MI scores and establish noise baseline."""
    # Generate noise baseline
    np.random.seed(42)
    noise_scores = []
    for _ in range(5):
        noise_col = np.random.randn(len(y))
        mi_noise = mutual_info_regression(noise_col.reshape(-1, 1), y, random_state=42)
        noise_scores.extend(mi_noise)
    
    noise_baseline = np.mean(noise_scores)
    
    # Compute MI for real features
    mi_scores = mutual_info_regression(X, y, random_state=42)
    
    mi_ranking = []
    for i, (feat, score) in enumerate(zip(feature_names, mi_scores)):
        mi_ranking.append({
            "feature": feat,
            "mi_score": float(score),
            "below_noise_baseline": bool(float(score) <= noise_baseline)
        })
    
    mi_ranking.sort(key=lambda x: x["mi_score"], reverse=True)
    
    return mi_ranking, float(noise_baseline)


def main():
    parser = argparse.ArgumentParser(description="Step 11: Data Exploration")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    progress_file = output_dir / "progress.json"
    
    try:
        # Update progress
        progress = json.loads(progress_file.read_text())
        progress["current_step"] = "11-data-exploration"
        progress["status"] = "running"
        progress_file.write_text(json.dumps(progress, indent=2))
        
        # ===== LOAD DATA =====
        print("[Step 11] Loading cleaned data...")
        step10_json = json.loads((output_dir / "step-10-cleanse.json").read_text())
        target_col = step10_json["target_column_normalized"]
        time_col = step10_json["time_column_detected"]
        
        df = pl.read_parquet(output_dir / "cleaned.parquet")
        df_pd = df.to_pandas()
        
        print(f"  Loaded {df.height} rows, {df.width} cols")
        print(f"  Target: {target_col}, Time: {time_col}")
        
        # ===== EXTRACT NUMERIC COLUMNS =====
        numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]]
        numeric_cols = [c for c in numeric_cols if c != time_col and c != "_synthesized_date"]
        print(f"  Numeric features: {numeric_cols}")
        
        target_series = df_pd[target_col].values.astype(float)
        target_series = target_series[~np.isnan(target_series)]
        
        # ===== NEAR-ZERO VARIANCE FILTER =====
        low_variance = []
        feature_cols = [c for c in numeric_cols if c != target_col]
        X = df_pd[feature_cols].fillna(df_pd[feature_cols].mean()).values.astype(float)
        
        for i, col in enumerate(feature_cols):
            col_data = X[:, i]
            col_min, col_max = np.min(col_data), np.max(col_data)
            if col_max == col_min:
                scaled_var = 0.0
            else:
                scaled = (col_data - col_min) / (col_max - col_min)
                scaled_var = np.var(scaled)
            
            if scaled_var < 1e-4:
                low_variance.append(col)
                print(f"  Low variance: {col} (scaled_var={scaled_var:.2e})")
        
        # ===== MI RANKING =====
        print("[Step 11] Computing MI ranking...")
        X_filtered = X[:, [i for i, c in enumerate(feature_cols) if c not in low_variance]]
        features_filtered = [c for c in feature_cols if c not in low_variance]
        
        if len(features_filtered) == 0:
            print("  WARNING: All features dropped by variance filter. Using all features.")
            X_filtered = X
            features_filtered = feature_cols
        
        mi_ranking, noise_baseline = compute_mi_ranking(X_filtered, target_series, features_filtered)
        
        print(f"  Noise MI baseline: {noise_baseline:.4f}")
        print(f"  Top 5 features by MI:")
        for item in mi_ranking[:5]:
            print(f"    {item['feature']}: {item['mi_score']:.4f}")
        
        # ===== LEAKAGE DETECTION & REDUNDANCY =====
        excluded = {}
        
        # Pearson correlation for leakage
        for i, col in enumerate(features_filtered):
            col_data = X_filtered[:, i]
            r, _ = stats.pearsonr(col_data, target_series)
            if abs(r) > 0.98:
                excluded[col] = "leakage_suspect"
                print(f"  HARD EXCLUDE: {col} (r={r:.4f} > 0.98)")
        
        # Redundancy (|r| >= 0.90, keep higher MI)
        remaining_features = [c for c in features_filtered if c not in excluded]
        remaining_idx = [i for i, c in enumerate(features_filtered) if c not in excluded]
        
        if len(remaining_features) > 1:
            X_remain = X_filtered[:, remaining_idx]
            corr_matrix = np.corrcoef(X_remain.T)
            
            mi_dict = {item["feature"]: item["mi_score"] for item in mi_ranking}
            
            for i in range(len(remaining_features)):
                for j in range(i + 1, len(remaining_features)):
                    if abs(corr_matrix[i, j]) >= 0.90:
                        feat_i, feat_j = remaining_features[i], remaining_features[j]
                        mi_i, mi_j = mi_dict.get(feat_i, 0), mi_dict.get(feat_j, 0)
                        
                        if feat_i not in excluded and feat_j not in excluded:
                            # Remove lower MI
                            if mi_i < mi_j:
                                excluded[feat_i] = "redundant"
                                print(f"  EXCLUDE (redundant): {feat_i} (r={corr_matrix[i,j]:.4f}, lower MI)")
                            else:
                                excluded[feat_j] = "redundant"
                                print(f"  EXCLUDE (redundant): {feat_j} (r={corr_matrix[i,j]:.4f}, lower MI)")
        
        # Features below noise baseline
        for item in mi_ranking:
            if item["below_noise_baseline"] and item["feature"] not in excluded:
                excluded[item["feature"]] = "below_noise_baseline"
                print(f"  EXCLUDE (noise): {item['feature']} (MI={item['mi_score']:.4f})")
        
        recommended = [f for f in features_filtered if f not in excluded]
        print(f"  Recommended features: {recommended} ({len(recommended)} features)")
        
        # ===== TIME-SERIES DIAGNOSTICS =====
        ts_diagnostics = {}
        
        if time_col and time_col in df.columns:
            print("[Step 11] Running TS diagnostics...")
            
            # Stationarity tests
            if len(target_series) >= 20:
                try:
                    adf_result = adfuller(target_series, autolag="AIC")
                    adf_stat, adf_p, adf_lags = adf_result[0], adf_result[1], adf_result[2]
                    
                    kpss_result = kpss(target_series, regression="c", nlags="auto")
                    kpss_stat, kpss_p = kpss_result[0], kpss_result[1]
                    
                    # Joint interpretation
                    if adf_p < 0.05 and kpss_p > 0.05:
                        stationarity = "stationary"
                    elif adf_p >= 0.05 and kpss_p <= 0.05:
                        stationarity = "non-stationary"
                    elif adf_p < 0.05 and kpss_p <= 0.05:
                        stationarity = "trend-stationary"
                    else:
                        stationarity = "ambiguous"
                    
                    ts_diagnostics["adf_statistic"] = float(adf_stat)
                    ts_diagnostics["adf_pvalue"] = float(adf_p)
                    ts_diagnostics["kpss_statistic"] = float(kpss_stat)
                    ts_diagnostics["kpss_pvalue"] = float(kpss_p)
                    ts_diagnostics["stationarity_conclusion"] = stationarity
                    print(f"  Stationarity: {stationarity} (ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f})")
                    
                except Exception as e:
                    print(f"  WARNING: Stationarity tests failed: {e}")
                    ts_diagnostics["stationarity_conclusion"] = "error"
                
                # ACF/PACF
                try:
                    max_lags = min(48, len(target_series) // 4)
                    
                    acf_vals = acf(target_series, nlags=max_lags)
                    pacf_vals = pacf(target_series, nlags=max_lags, method="ywm")
                    
                    conf_band = 2.0 / np.sqrt(len(target_series))
                    acf_sig_lags = [i for i, v in enumerate(acf_vals[1:], 1) if abs(v) > conf_band][:12]
                    pacf_sig_lags = [i for i, v in enumerate(pacf_vals[1:], 1) if abs(v) > conf_band][:12]
                    
                    ts_diagnostics["acf_values"] = acf_vals[:13].tolist()
                    ts_diagnostics["pacf_values"] = pacf_vals[:13].tolist()
                    ts_diagnostics["acf_significant_lags"] = acf_sig_lags
                    ts_diagnostics["pacf_significant_lags"] = pacf_sig_lags
                    
                    # Derive AR/MA order
                    suggested_ar = pacf_sig_lags[0] if pacf_sig_lags else 0
                    suggested_ma = acf_sig_lags[0] if acf_sig_lags else 0
                    
                    ts_diagnostics["suggested_ar_order"] = int(suggested_ar)
                    ts_diagnostics["suggested_ma_order"] = int(suggested_ma)
                    
                    print(f"  ACF lags: {acf_sig_lags[:5]}, PACF lags: {pacf_sig_lags[:5]}")
                    
                except Exception as e:
                    print(f"  WARNING: ACF/PACF failed: {e}")
                
                # Hurst exponent
                try:
                    h, h_r2, h_interp = compute_hurst_exponent(target_series)
                    if h is not None:
                        ts_diagnostics["hurst_exponent"] = float(h)
                        ts_diagnostics["hurst_r2_fit"] = float(h_r2)
                        ts_diagnostics["hurst_interpretation"] = h_interp
                        print(f"  Hurst exponent: {h:.3f} ({h_interp})")
                    else:
                        ts_diagnostics["hurst_exponent"] = None
                        ts_diagnostics["hurst_skipped_reason"] = h_interp
                except Exception as e:
                    print(f"  WARNING: Hurst computation failed: {e}")
                
                # Ljung-Box
                try:
                    lb_lags = [6, 12, 24]
                    lb_pvals = {}
                    for lag in lb_lags:
                        if lag < len(target_series) // 5:
                            result = acorr_ljungbox(target_series, lags=[lag], return_df=False)
                            lb_pvals[str(lag)] = float(result[1][0])
                    
                    white_noise = all(p > 0.05 for p in lb_pvals.values()) if lb_pvals else False
                    ts_diagnostics["ljung_box_pvalues"] = lb_pvals
                    ts_diagnostics["white_noise"] = white_noise
                    print(f"  Ljung-Box: white_noise={white_noise}")
                    
                except Exception as e:
                    print(f"  WARNING: Ljung-Box failed: {e}")
                    ts_diagnostics["white_noise"] = False
                
                # STL decomposition for seasonality
                try:
                    if len(target_series) > 24:
                        # Try with daily period
                        seasonal_period = 365 if len(target_series) > 365 else 24
                        if len(target_series) < seasonal_period * 2:
                            seasonal_period = max(7, len(target_series) // 10)
                        
                        stl = STL(target_series, seasonal=seasonal_period)
                        result = stl.fit()
                        
                        # Compute strengths
                        residual_var = np.var(result.resid)
                        trend_var = np.var(result.trend)
                        seasonal_var = np.var(result.seasonal)
                        
                        trend_strength = max(0, 1 - residual_var / (trend_var + residual_var)) if (trend_var + residual_var) > 0 else 0
                        seasonal_strength = max(0, 1 - residual_var / (seasonal_var + residual_var)) if (seasonal_var + residual_var) > 0 else 0
                        
                        ts_diagnostics["trend_strength"] = float(trend_strength)
                        ts_diagnostics["trend_detected"] = trend_strength > 0.30
                        
                        detected_periods = []
                        if seasonal_strength > 0.30:
                            detected_periods.append({
                                "period": int(seasonal_period),
                                "seasonal_strength": float(seasonal_strength),
                                "significant": True
                            })
                        
                        ts_diagnostics["detected_periods"] = detected_periods
                        ts_diagnostics["primary_seasonal_period"] = int(seasonal_period) if seasonal_strength > 0.30 else None
                        
                        print(f"  STL: trend={trend_strength:.3f}, seasonal={seasonal_strength:.3f}")
                        
                except Exception as e:
                    print(f"  WARNING: STL failed: {e}")
            else:
                ts_diagnostics["stationarity_conclusion"] = "insufficient_data"
        
        # ===== USEFUL LAG FEATURES =====
        useful_lag_features = []
        
        if recommended and len(target_series) > 12:
            print("[Step 11] Computing lag features...")
            
            for feat in recommended:
                if feat not in df_pd.columns:
                    continue
                
                feat_data = df_pd[feat].fillna(df_pd[feat].mean()).values.astype(float)
                
                for lag in [1, 2, 3, 6, 12]:
                    if lag < len(feat_data):
                        feat_lagged = feat_data[lag:]
                        target_trimmed = target_series[:len(feat_lagged)]
                        
                        if len(target_trimmed) > 0 and np.std(feat_lagged) > 0 and np.std(target_trimmed) > 0:
                            xcorr, _ = stats.pearsonr(feat_lagged, target_trimmed)
                            
                            if abs(xcorr) > 0.15:
                                useful_lag_features.append({
                                    "feature": feat,
                                    "lag": int(lag),
                                    "xcorr": float(xcorr)
                                })
                                print(f"  Lag feature: {feat} lag={lag}, xcorr={xcorr:.4f}")
        
        # ===== MODEL RECOMMENDATIONS =====
        model_recommendations = []
        
        if ts_diagnostics.get("white_noise", False):
            model_recommendations.append({
                "model_class": "Naive",
                "justification": "Target appears to be white noise; naive forecasting is baseline."
            })
        else:
            if ts_diagnostics.get("stationarity_conclusion") == "stationary":
                model_recommendations.extend([
                    {"model_class": "AR", "justification": "Stationary target suitable for AR models."},
                    {"model_class": "ARMA", "justification": "Stationary target suitable for ARMA models."},
                    {"model_class": "ElasticNet", "justification": "Stationary target; linear model with regularization."},
                    {"model_class": "XGBoost", "justification": "Stationary target; gradient boosting for non-linear patterns."}
                ])
            
            elif ts_diagnostics.get("stationarity_conclusion") == "non-stationary":
                if ts_diagnostics.get("detected_periods"):
                    model_recommendations.extend([
                        {"model_class": "SARIMA", "justification": "Non-stationary target with seasonality detected."},
                        {"model_class": "HoltWinters", "justification": "Trend and seasonality; exponential smoothing family."},
                        {"model_class": "XGBoost", "justification": "Non-stationary target; XGBoost can capture non-linear trends."}
                    ])
                else:
                    model_recommendations.extend([
                        {"model_class": "ARIMA", "justification": "Non-stationary target; ARIMA differencing indicated."},
                        {"model_class": "ElasticNet", "justification": "Non-stationary; can use differenced target."},
                        {"model_class": "XGBoost", "justification": "Non-stationary target; gradient boosting for flexibility."}
                    ])
            
            else:
                model_recommendations.extend([
                    {"model_class": "Ridge", "justification": "Baseline robust regressor for uncertain stationarity."},
                    {"model_class": "ElasticNet", "justification": "Robust linear model with collinearity handling."},
                    {"model_class": "XGBoost", "justification": "Flexible gradient boosting regardless of stationarity."}
                ])
        
        if len(recommended) >= 3:
            model_recommendations.append({
                "model_class": "RandomForestRegressor",
                "justification": f"Multivariate features available ({len(recommended)} features)."
            })
        
        # ===== COMPILE OUTPUT =====
        # Convert numpy bool to Python bool to ensure JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return bool(obj) if isinstance(obj, np.bool_) else int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        output = {
            "step": "11-data-exploration",
            "shape": {"rows": int(df.height), "columns": int(df.width)},
            "numeric_columns": numeric_cols,
            "high_cardinality": [],
            "low_variance_columns": low_variance,
            "time_series_detected": bool(time_col is not None),
            "time_column": time_col,
            "multiple_series_detected": False,
            "series_id_column": None,
            "ts_diagnostics": convert_to_serializable(ts_diagnostics),
            "model_class_recommendations": model_recommendations,
            "acf_pacf_orders": {
                "suggested_ar_order": int(ts_diagnostics.get("suggested_ar_order", 1)),
                "suggested_ma_order": int(ts_diagnostics.get("suggested_ma_order", 0)),
                "suggested_d": 1 if ts_diagnostics.get("stationarity_conclusion") == "non-stationary" else 0,
                "suggested_seasonal_ar": 1,
                "suggested_seasonal_d": 1 if ts_diagnostics.get("detected_periods") else 0,
                "suggested_seasonal_ma": 1,
                "seasonal_period": int(ts_diagnostics.get("primary_seasonal_period", 365)) if ts_diagnostics.get("primary_seasonal_period") else 365
            },
            "mi_ranking": mi_ranking,
            "noise_mi_baseline": float(noise_baseline),
            "redundant_columns": [k for k, v in excluded.items() if v == "redundant"],
            "correlation_matrix_summary": {},
            "useful_lag_features": useful_lag_features,
            "recommended_features": recommended,
            "excluded_features": excluded,
            "target_candidates": [{"column": target_col, "reason": "provided_target"}],
            "client_facing_summary": f"Dataset has {df.height} rows with target '{target_col}'. Identified {len(recommended)} key features for modeling. " + 
                                      (f"Time series with {ts_diagnostics.get('stationarity_conclusion', 'unknown')} structure. " if ts_diagnostics else "") +
                                      ("White noise pattern detected; naive forecasting recommended. " if ts_diagnostics.get("white_noise") else "")
        }
        
        # Write step JSON
        step_json_path = output_dir / "step-11-exploration.json"
        step_json_path.write_text(json.dumps(output, indent=2))
        print(f"\n[Step 11] Wrote exploration to {step_json_path}")
        
        # Update progress
        progress = json.loads(progress_file.read_text())
        progress["completed_steps"].append("11-data-exploration")
        progress_file.write_text(json.dumps(progress, indent=2))
        
        print("[Step 11] SUCCESS ✓")
        return 0
        
    except Exception as e:
        print(f"[Step 11] FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
            if "errors" not in progress:
                progress["errors"] = []
            progress["errors"].append(f"Step 11 failed: {str(e)}")
            progress["status"] = "error"
            progress_file.write_text(json.dumps(progress, indent=2))
        except:
            pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
