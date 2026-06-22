"""Step 11 — Data Exploration."""
import argparse
import json
import os
import sys
import warnings
import numpy as np
import polars as pl
from datetime import datetime, timezone

warnings.filterwarnings("ignore")


def update_progress(output_dir, step, status, run_id=None, extra=None):
    progress_path = os.path.join(output_dir, "progress.json")
    with open(progress_path) as f:
        progress = json.load(f)
    progress["status"] = status
    progress["current_step"] = step
    if extra:
        progress.update(extra)
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def hurst_exponent(series: np.ndarray):
    n = len(series)
    if n < 64:
        return None, None, "insufficient_data"
    window_sizes = []
    w = 8
    while w <= n // 4:
        window_sizes.append(w)
        w *= 2
    if not window_sizes:
        return None, None, "insufficient_data"
    rs_values = []
    for size in window_sizes:
        rs_per_window = []
        for start in range(0, n - size + 1, size):
            sub = series[start:start + size].astype(float)
            sub = sub - np.mean(sub)
            cumsum = np.cumsum(sub)
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(series[start:start + size], ddof=1)
            if S > 0:
                rs_per_window.append(R / S)
        if rs_per_window:
            rs_values.append(np.mean(rs_per_window))
        else:
            rs_values.append(np.nan)
    valid = [(np.log(w), np.log(rs)) for w, rs in zip(window_sizes, rs_values) if not np.isnan(rs)]
    if len(valid) < 2:
        return None, None, "insufficient_valid_windows"
    log_w, log_rs = zip(*valid)
    coeffs = np.polyfit(log_w, log_rs, 1)
    H = coeffs[0]
    y_pred = np.polyval(coeffs, log_w)
    ss_res = np.sum((np.array(log_rs) - y_pred) ** 2)
    ss_tot = np.sum((np.array(log_rs) - np.mean(log_rs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(H), float(r2), None


def interpret_hurst(H):
    if H < 0.45:
        return "anti_persistent_mean_reverting"
    elif H < 0.55:
        return "random_walk"
    elif H < 0.75:
        return "mildly_persistent"
    else:
        return "strongly_persistent"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    run_id = args.run_id

    update_progress(output_dir, "11-data-exploration", "running")

    # Load inputs
    step10 = json.load(open(os.path.join(output_dir, "step-10-cleanse.json")))
    target_col = step10["target_column_normalized"]
    time_col = step10["time_column_detected"]

    df_pl = pl.read_parquet(os.path.join(output_dir, "cleaned.parquet"))
    df = df_pl.to_pandas()

    # Shape
    n_rows, n_cols = df.shape

    # Identify numeric columns (excluding time col)
    numeric_cols = [c for c in df.columns
                    if df[c].dtype in (float, int) or str(df[c].dtype).startswith('float') or str(df[c].dtype).startswith('int')
                    and c != time_col]
    # More robust
    import pandas as pd
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if time_col in numeric_cols:
        numeric_cols.remove(time_col)
    # Also exclude synthesized date columns
    for c in list(numeric_cols):
        if c == time_col:
            numeric_cols.remove(c)

    feature_candidates = [c for c in numeric_cols if c != target_col]

    # Near-zero variance filter
    from sklearn.preprocessing import MinMaxScaler
    low_variance_cols = []
    if feature_candidates:
        sub = df[feature_candidates].fillna(df[feature_candidates].median())
        if sub.shape[0] > 1:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(sub)
            variances = np.var(scaled, axis=0)
            for col, var in zip(feature_candidates, variances):
                if var < 1e-4:
                    low_variance_cols.append(col)

    # High cardinality
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if time_col in cat_cols:
        cat_cols.remove(time_col)
    high_cardinality = [c for c in cat_cols if df[c].nunique() > 50]

    # Initial candidate features (not low-variance, not high-cardinality, not target, not time)
    candidate_features = [c for c in feature_candidates
                          if c not in low_variance_cols and c not in high_cardinality]

    # Target series
    y = df[target_col].fillna(df[target_col].median()).values.astype(float)

    # Lag-0 leakage detection
    from scipy.stats import pearsonr
    excluded_features = {}
    leakage_suspects = []
    for col in list(candidate_features):
        x = df[col].fillna(df[col].median()).values.astype(float)
        if len(x) < 3:
            continue
        try:
            r, _ = pearsonr(x, y)
            if abs(r) > 0.98:
                leakage_suspects.append(col)
                excluded_features[col] = "leakage_suspect_lag0_pearson"
                candidate_features.remove(col)
        except Exception:
            pass

    # Mutual information ranking
    from sklearn.feature_selection import mutual_info_regression
    rng = np.random.default_rng(42)
    mi_ranking = []
    noise_mi_baseline = 0.0

    if candidate_features:
        X_feat = df[candidate_features].fillna(df[candidate_features].median()).values

        # 5 noise columns for baseline
        noise_cols = rng.standard_normal((n_rows, 5))
        mi_noise = mutual_info_regression(noise_cols, y, random_state=42)
        noise_mi_baseline = float(np.mean(mi_noise))

        mi_scores = mutual_info_regression(X_feat, y, random_state=42)
        for col, mi in zip(candidate_features, mi_scores):
            mi_ranking.append({
                "feature": col,
                "mi_score": float(mi),
                "below_noise_baseline": bool(mi <= noise_mi_baseline)
            })
        mi_ranking.sort(key=lambda x: x["mi_score"], reverse=True)

        # Exclude below-noise baseline
        below_noise = [r["feature"] for r in mi_ranking if r["below_noise_baseline"]]
        for col in below_noise:
            if col in candidate_features:
                excluded_features[col] = "below_noise_baseline"
                candidate_features.remove(col)

    # If all features filtered, relax threshold by 50%
    if not candidate_features and mi_ranking:
        relaxed_threshold = noise_mi_baseline * 0.5
        print("WARNING: All features below noise baseline — relaxing threshold by 50%")
        for r in mi_ranking:
            if r["mi_score"] > relaxed_threshold:
                candidate_features.append(r["feature"])
                # Remove from excluded if it was added
                excluded_features.pop(r["feature"], None)

    # Pairwise correlation redundancy
    import pandas as pd
    if len(candidate_features) > 1:
        mi_map = {r["feature"]: r["mi_score"] for r in mi_ranking}
        feat_df = df[candidate_features].fillna(df[candidate_features].median())
        corr_matrix = feat_df.corr(method="pearson")
        redundant = set()
        for i, col_i in enumerate(candidate_features):
            for j, col_j in enumerate(candidate_features):
                if j <= i:
                    continue
                if abs(corr_matrix.loc[col_i, col_j]) >= 0.90:
                    mi_i = mi_map.get(col_i, 0)
                    mi_j = mi_map.get(col_j, 0)
                    drop = col_i if mi_i < mi_j else col_j
                    redundant.add(drop)
        for col in redundant:
            if col in candidate_features:
                excluded_features[col] = "redundant_high_correlation"
                candidate_features.remove(col)
        max_corr_pair = None
        max_corr_val = 0.0
        for i, col_i in enumerate(candidate_features + list(redundant)):
            for j, col_j in enumerate(candidate_features + list(redundant)):
                if j <= i:
                    continue
                try:
                    v = abs(corr_matrix.loc[col_i, col_j])
                    if v > max_corr_val:
                        max_corr_val = v
                        max_corr_pair = [col_i, col_j]
                except Exception:
                    pass
    else:
        max_corr_pair = []
        max_corr_val = 0.0

    # Time series diagnostics
    ts_diagnostics = {}
    significant_lags = []
    useful_lag_features = []

    if time_col:
        from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from statsmodels.tsa.seasonal import STL

        y_clean = y.copy()
        # Forward-fill short NaN gaps
        mask = np.isnan(y_clean)
        if mask.any():
            idx = np.where(~mask)[0]
            y_clean[mask] = np.interp(np.where(mask)[0], idx, y_clean[idx])

        N = len(y_clean)
        max_lag_acf = min(48, N // 4)
        conf_band = 2 / np.sqrt(N)

        # ADF
        try:
            adf_result = adfuller(y_clean, autolag='AIC')
            adf_stat = float(adf_result[0])
            adf_pvalue = float(adf_result[1])
        except Exception as e:
            adf_stat, adf_pvalue = None, None

        # KPSS
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_result = kpss(y_clean, regression='c', nlags='auto')
            kpss_stat = float(kpss_result[0])
            kpss_pvalue = float(kpss_result[1])
        except Exception:
            kpss_stat, kpss_pvalue = None, None

        # Joint stationarity interpretation
        stationarity_conclusion = "unknown"
        if adf_pvalue is not None and kpss_pvalue is not None:
            adf_reject = adf_pvalue < 0.05
            kpss_reject = kpss_pvalue <= 0.05
            if adf_reject and not kpss_reject:
                stationarity_conclusion = "stationary"
            elif not adf_reject and kpss_reject:
                stationarity_conclusion = "non-stationary"
            elif adf_reject and kpss_reject:
                stationarity_conclusion = "trend-stationary"
            else:
                stationarity_conclusion = "ambiguous"

        # ACF / PACF
        try:
            acf_vals = acf(y_clean, nlags=max_lag_acf, fft=True)
            pacf_vals = pacf(y_clean, nlags=min(max_lag_acf, N//2 - 1))
            acf_significant = [int(i) for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > conf_band]
            pacf_significant = [int(i) for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > conf_band]
            # Suggest AR/MA orders
            suggested_ar_order = pacf_significant[-1] if pacf_significant else 1
            suggested_ma_order = 0
            significant_lags = acf_significant[:10]  # top 10 significant lags
        except Exception:
            acf_vals, pacf_vals = [], []
            acf_significant, pacf_significant = [], []
            suggested_ar_order, suggested_ma_order = 1, 0

        # Hurst exponent
        H, h_r2, h_skip = hurst_exponent(y_clean)
        hurst_interp = interpret_hurst(H) if H is not None else None

        # Ljung-Box
        max_lb_lag = min(24, N // 5)
        lb_lags = [l for l in [6, 12, 24] if l <= max_lb_lag]
        if not lb_lags:
            lb_lags = [min(6, max_lb_lag)]
        try:
            lb_result = acorr_ljungbox(y_clean, lags=lb_lags, return_df=True)
            lb_pvalues = {str(int(row["lags"])): float(row["lb_pvalue"]) for _, row in lb_result.iterrows()}
            white_noise = all(p > 0.05 for p in lb_pvalues.values())
        except Exception:
            lb_pvalues = {}
            white_noise = False

        # STL / seasonal decomposition
        detected_periods = []
        trend_strength = 0.0
        trend_detected = False
        primary_seasonal_period = None

        # Determine sampling frequency from time deltas
        time_series_pd = pd.to_datetime(df[time_col]) if time_col in df.columns else None
        detected_frequency = "unknown"
        if time_series_pd is not None:
            deltas = time_series_pd.diff().dropna()
            median_delta = deltas.median()
            if median_delta is not None:
                days = median_delta.total_seconds() / 86400
                if days < 0.02:
                    detected_frequency = "10min"
                    candidate_periods = [6, 12, 36, 144, 1008]
                elif days < 0.1:
                    detected_frequency = "hourly"
                    candidate_periods = [24, 168]
                elif days < 2:
                    detected_frequency = "daily"
                    candidate_periods = [7, 30, 365]
                elif days < 10:
                    detected_frequency = "weekly"
                    candidate_periods = [4, 52]
                else:
                    detected_frequency = "monthly"
                    candidate_periods = [12]
            else:
                candidate_periods = [7, 30, 365]
        else:
            candidate_periods = [7, 30, 365]

        for period in candidate_periods:
            if period < 2 or N < period * 2:
                continue
            try:
                stl = STL(y_clean, period=period, robust=True)
                res = stl.fit()
                var_resid = np.var(res.resid)
                var_seasonal_resid = np.var(res.seasonal + res.resid)
                var_trend_resid = np.var(res.trend + res.resid)
                fs = max(0.0, 1 - var_resid / var_seasonal_resid) if var_seasonal_resid > 0 else 0.0
                ft = max(0.0, 1 - var_resid / var_trend_resid) if var_trend_resid > 0 else 0.0
                significant = fs > 0.30
                detected_periods.append({"period": period, "seasonal_strength": round(fs, 4), "significant": significant})
                if ft > trend_strength:
                    trend_strength = ft
                    trend_detected = ft > 0.30
            except Exception:
                pass

        significant_periods = [p for p in detected_periods if p["significant"]]
        if significant_periods:
            primary_seasonal_period = max(significant_periods, key=lambda x: x["seasonal_strength"])["period"]

        ts_diagnostics = {
            "adf_statistic": adf_stat,
            "adf_pvalue": adf_pvalue,
            "kpss_statistic": kpss_stat,
            "kpss_pvalue": kpss_pvalue,
            "stationarity_conclusion": stationarity_conclusion,
            "acf_values": [float(v) for v in acf_vals[:20]] if hasattr(acf_vals, '__len__') else [],
            "pacf_values": [float(v) for v in pacf_vals[:20]] if hasattr(pacf_vals, '__len__') else [],
            "acf_significant_lags": acf_significant,
            "pacf_significant_lags": pacf_significant,
            "suggested_ar_order": int(suggested_ar_order),
            "suggested_ma_order": int(suggested_ma_order),
            "hurst_exponent": H,
            "hurst_interpretation": hurst_interp,
            "hurst_r2_fit": h_r2,
            "hurst_skipped_reason": h_skip,
            "ljung_box_pvalues": lb_pvalues,
            "white_noise": white_noise,
            "trend_strength": float(trend_strength),
            "trend_detected": trend_detected,
            "detected_periods": detected_periods,
            "primary_seasonal_period": primary_seasonal_period,
            "detected_frequency": detected_frequency,
        }

        # Cross-correlation lag analysis for recommended features
        for col in candidate_features:
            x = df[col].fillna(df[col].median()).values.astype(float)
            for lag in [1, 2, 3, 6, 12]:
                if lag >= len(y_clean):
                    continue
                try:
                    x_shifted = x[:-lag]
                    y_target = y_clean[lag:]
                    if len(x_shifted) < 10:
                        continue
                    r, _ = pearsonr(x_shifted, y_target)
                    if abs(r) > 0.15:
                        useful_lag_features.append({"feature": col, "lag": lag, "xcorr": round(float(r), 4)})
                except Exception:
                    pass

    # Multiple series detection
    multiple_series_detected = False
    series_id_column = None
    # Check for duplicate timestamps
    if time_col and time_col in df.columns:
        n_unique_ts = df[time_col].nunique()
        n_rows_total = len(df)
        if n_unique_ts < n_rows_total:
            multiple_series_detected = True
            # Find grouping column
            for col in df.columns:
                if col == time_col or col == target_col:
                    continue
                nuniq = df[col].nunique()
                if 3 <= nuniq <= 100:
                    series_id_column = col
                    break

    # Model class recommendations
    model_recs = []
    seasonality_detected = bool(primary_seasonal_period) if 'primary_seasonal_period' in dir() else False

    if white_noise:
        model_recs.append({"model_class": "Naive", "justification": "White noise target — naive/seasonal-naive only."})
        model_recs.append({"model_class": "SeasonalNaive", "justification": "White noise target — seasonal-naive baseline."})
    else:
        stat = ts_diagnostics.get("stationarity_conclusion", "unknown")
        if stat == "stationary":
            model_recs += [
                {"model_class": "Ridge", "justification": "Stationary series; linear model with L2 regularization."},
                {"model_class": "ElasticNet", "justification": "Stationary series with collinearity-robust regularization."},
                {"model_class": "XGBoost", "justification": "Non-linear interactions in stationary series."},
            ]
        elif stat in ("non-stationary", "trend-stationary"):
            if seasonality_detected:
                model_recs += [
                    {"model_class": "XGBoost", "justification": "Non-stationary + seasonal target; tree model with lag features."},
                    {"model_class": "GradientBoosting", "justification": "Robust to trend and seasonality."},
                    {"model_class": "RandomForest", "justification": "Ensemble for non-stationary seasonal data."},
                ]
            else:
                model_recs += [
                    {"model_class": "XGBoost", "justification": "Non-stationary series; gradient boosting handles trends."},
                    {"model_class": "ElasticNet", "justification": "Regularized linear baseline for non-stationary series."},
                ]
        else:
            model_recs += [
                {"model_class": "RandomForest", "justification": "General-purpose ensemble."},
                {"model_class": "GradientBoosting", "justification": "Gradient boosting for complex patterns."},
                {"model_class": "Ridge", "justification": "Linear baseline with regularization."},
            ]

        H = ts_diagnostics.get("hurst_exponent")
        if H and H > 0.65:
            model_recs.append({"model_class": "XGBoost", "justification": f"Hurst={H:.2f} persistent memory; extended lag window beneficial."})
        if n_rows > 10000 and len(candidate_features) > 5:
            model_recs.append({"model_class": "HistGradientBoosting", "justification": f"Large dataset (N={n_rows}), many features."})

    # target candidates
    target_candidates = [{"column": target_col, "reason": "user_specified_target"}]

    # acf_pacf_orders
    acf_pacf_orders = {
        "suggested_ar_order": ts_diagnostics.get("suggested_ar_order", 1),
        "suggested_ma_order": ts_diagnostics.get("suggested_ma_order", 0),
        "suggested_d": 1 if ts_diagnostics.get("stationarity_conclusion") in ("non-stationary", "trend-stationary") else 0,
        "suggested_seasonal_ar": 1 if primary_seasonal_period else 0,
        "suggested_seasonal_d": 1 if primary_seasonal_period else 0,
        "suggested_seasonal_ma": 1 if primary_seasonal_period else 0,
        "seasonal_period": primary_seasonal_period,
    }

    # Filter counts
    filter_counts = {
        "initial_features": len(feature_candidates),
        "low_variance_removed": len(low_variance_cols),
        "high_cardinality_removed": len(high_cardinality),
        "leakage_suspects_removed": len(leakage_suspects),
        "below_noise_removed": len([r for r in mi_ranking if r["below_noise_baseline"]]),
        "redundant_removed": len([c for c, r in excluded_features.items() if r == "redundant_high_correlation"]),
        "final_recommended": len(candidate_features),
    }

    client_facing_summary = (
        f"The {target_col} data spans {n_rows} rows with {detected_frequency} frequency. "
        f"The series is {ts_diagnostics.get('stationarity_conclusion', 'unknown')} and "
        f"{'has' if not white_noise else 'has no'} exploitable autocorrelation structure. "
        f"{'Seasonal patterns detected at periods ' + str([p['period'] for p in detected_periods if p['significant']]) + '.' if significant_periods else 'No strong seasonal pattern detected.'} "
        f"{len(candidate_features)} features recommended for modeling."
    )

    result = {
        "step": "11-data-exploration",
        "shape": {"rows": n_rows, "columns": n_cols},
        "numeric_columns": numeric_cols,
        "high_cardinality": high_cardinality,
        "low_variance_columns": low_variance_cols,
        "time_series_detected": bool(time_col),
        "time_column": time_col,
        "multiple_series_detected": multiple_series_detected,
        "series_id_column": series_id_column,
        "detected_frequency": detected_frequency,
        "ts_diagnostics": ts_diagnostics,
        "model_class_recommendations": model_recs,
        "acf_pacf_orders": acf_pacf_orders,
        "mi_ranking": mi_ranking,
        "noise_mi_baseline": noise_mi_baseline,
        "redundant_columns": [c for c, r in excluded_features.items() if r == "redundant_high_correlation"],
        "correlation_matrix_summary": {"max_pair": max_corr_pair, "max_corr": float(max_corr_val)},
        "useful_lag_features": useful_lag_features,
        "significant_lags": significant_lags,
        "recommended_features": candidate_features,
        "excluded_features": excluded_features,
        "target_candidates": target_candidates,
        "filter_counts": filter_counts,
        "client_facing_summary": client_facing_summary,
        "context": {
            "target_column": target_col,
            "time_column": time_col,
        }
    }

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            return str(obj)

    out_json = os.path.join(output_dir, "step-11-exploration.json")
    with open(out_json, "w") as f:
        json.dump(make_serializable(result), f, indent=2)

    # Update progress
    with open(os.path.join(output_dir, "progress.json")) as f:
        progress = json.load(f)
    progress["status"] = "running"
    progress["current_step"] = "12-feature-extraction"
    if "11-data-exploration" not in progress.get("completed_steps", []):
        progress["completed_steps"].append("11-data-exploration")
    with open(os.path.join(output_dir, "progress.json"), "w") as f:
        json.dump(progress, f, indent=2)

    print(f"Step 11 complete. Recommended features: {candidate_features}")
    print(f"Significant lags: {significant_lags}")
    print(f"Multiple series: {multiple_series_detected}")
    sys.exit(0)


if __name__ == "__main__":
    main()
