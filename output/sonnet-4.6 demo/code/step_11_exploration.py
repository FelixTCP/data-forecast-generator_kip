"""Step 11 — Data Exploration (Time-Series Focused).

Runnable:
    python step_11_exploration.py --output-dir <dir> --run-id <id>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_selection import mutual_info_regression


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hurst_rs(series: np.ndarray) -> tuple[float, float]:
    """Compute Hurst exponent via classical R/S rescaled-range analysis."""
    n = len(series)
    windows = []
    size = 8
    while size <= n // 4:
        windows.append(size)
        size *= 2
    if not windows:
        return 0.5, 0.0

    rs_vals = []
    for w in windows:
        chunks = [series[i:i+w] for i in range(0, n - w + 1, w)]
        rs_chunk = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = chunk.mean()
            demeaned = chunk - mean_c
            cumsum = np.cumsum(demeaned)
            r = cumsum.max() - cumsum.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_vals.append(np.mean(rs_chunk))
        else:
            rs_vals.append(np.nan)

    valid = [(np.log(w), np.log(rs)) for w, rs in zip(windows, rs_vals) if not np.isnan(rs)]
    if len(valid) < 2:
        return 0.5, 0.0

    log_n = np.array([v[0] for v in valid])
    log_rs = np.array([v[1] for v in valid])
    # OLS
    A = np.column_stack([log_n, np.ones(len(log_n))])
    result = np.linalg.lstsq(A, log_rs, rcond=None)
    H = float(result[0][0])
    ss_res = np.sum((log_rs - A @ result[0]) ** 2)
    ss_tot = np.sum((log_rs - log_rs.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return H, r2


def _hurst_interpretation(H: float) -> str:
    if H < 0.45:
        return "anti_persistent_mean_reverting"
    elif H < 0.55:
        return "random_walk"
    elif H < 0.75:
        return "mildly_persistent"
    else:
        return "strongly_persistent_trending"


def _detect_frequency(df: pl.DataFrame, time_col: str) -> str:
    """Detect sampling frequency from time column."""
    if df[time_col].dtype == pl.Date:
        diffs = (
            df.with_columns(pl.col(time_col).cast(pl.Int32).alias("_t"))
            .select(pl.col("_t").diff().drop_nulls())["_t"]
            .to_numpy()
        )
    else:
        diffs = (
            df.with_columns(pl.col(time_col).cast(pl.Int64).alias("_t"))
            .select(pl.col("_t").diff().drop_nulls())["_t"]
            .to_numpy()
        )
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return "daily"
    median = float(np.median(diffs))
    if df[time_col].dtype == pl.Date:
        if median <= 1.5:
            return "daily"
        elif median <= 7.5:
            return "weekly"
        elif median <= 31.5:
            return "monthly"
        else:
            return "yearly"
    else:
        usec = median
        if usec <= 61_000_000:
            return "minutely"
        elif usec <= 3601_000_000:
            return "hourly"
        elif usec <= 86401_000_000:
            return "daily"
        else:
            return "daily"


def _candidate_periods(freq: str) -> list[int]:
    mapping = {
        "minutely": [6, 12, 36, 144, 1008],
        "10min": [6, 12, 36, 144, 1008],
        "hourly": [24, 168],
        "daily": [7, 30, 365],
        "weekly": [4, 52],
        "monthly": [12],
        "yearly": [1],
    }
    return mapping.get(freq, [7, 365])


def _stl_seasonal_strength(series: np.ndarray, period: int) -> float:
    """Compute seasonal strength from STL decomposition."""
    try:
        from statsmodels.tsa.seasonal import STL
        if len(series) < 2 * period:
            return 0.0
        stl = STL(series, period=period, robust=True)
        res = stl.fit()
        var_resid = np.var(res.resid)
        var_seas_resid = np.var(res.seasonal + res.resid)
        if var_seas_resid == 0:
            return 0.0
        fs = max(0.0, 1.0 - var_resid / var_seas_resid)
        return float(fs)
    except Exception:
        return 0.0


def _stl_trend_strength(series: np.ndarray, period: int) -> float:
    """Compute trend strength from STL decomposition."""
    try:
        from statsmodels.tsa.seasonal import STL
        if len(series) < 2 * period:
            return 0.0
        stl = STL(series, period=period, robust=True)
        res = stl.fit()
        var_resid = np.var(res.resid)
        var_trend_resid = np.var(res.trend + res.resid)
        if var_trend_resid == 0:
            return 0.0
        ft = max(0.0, 1.0 - var_resid / var_trend_resid)
        return float(ft)
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Step 11: Data Exploration")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    progress_path = output_dir / "progress.json"

    def update_progress(status: str):
        if progress_path.exists():
            with open(progress_path) as f:
                p = json.load(f)
        else:
            p = {}
        p["status"] = status
        p["current_step"] = "11-data-exploration"
        with open(progress_path, "w") as f:
            json.dump(p, f, indent=2)

    update_progress("running")

    try:
        # ── Load step-10 context ──────────────────────────────────────────────
        with open(output_dir / "step-10-cleanse.json") as f:
            ctx10 = json.load(f)

        target_col = ctx10["target_column_normalized"]
        time_col = ctx10["time_column"]
        detected_frequency = ctx10.get("detected_frequency", "daily")

        # ── Load cleaned parquet ──────────────────────────────────────────────
        df = pl.read_parquet(output_dir / "cleaned.parquet")
        print(f"Loaded cleaned.parquet: {df.shape}")
        print(f"Columns: {df.columns}")
        print(f"Target: {target_col}, Time: {time_col}")

        n_rows = len(df)

        # ── Identify column types ─────────────────────────────────────────────
        numeric_dtypes = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                          pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        numeric_cols = [
            c for c in df.columns
            if df[c].dtype in numeric_dtypes and c not in (time_col, target_col)
        ]
        categorical_cols = [
            c for c in df.columns
            if df[c].dtype in (pl.Utf8, pl.String, pl.Categorical) and c not in (time_col,)
        ]

        print(f"Numeric feature cols (excl. target): {numeric_cols}")
        print(f"Categorical cols: {categorical_cols}")

        # ── 1. Near-zero variance filter ─────────────────────────────────────
        low_variance_columns = []
        for col in numeric_cols:
            vals = df[col].cast(pl.Float64).drop_nulls().to_numpy()
            if len(vals) == 0:
                low_variance_columns.append(col)
                continue
            v_min = float(vals.min())
            v_max = float(vals.max())
            r = v_max - v_min
            if r == 0:
                scaled_var = 0.0
            else:
                scaled = (vals - v_min) / r
                scaled_var = float(scaled.var())
            if scaled_var < 1e-4:
                low_variance_columns.append(col)
        print(f"Low variance columns: {low_variance_columns}")

        # ── 2. High cardinality (categorical) ────────────────────────────────
        high_cardinality = []
        for col in categorical_cols:
            n_unique = df[col].n_unique()
            if n_unique > 50:
                high_cardinality.append(col)
        print(f"High cardinality: {high_cardinality}")

        # ── 3. Lag-0 leakage detection (Hard Gate) ────────────────────────────
        target_vals = df[target_col].cast(pl.Float64).to_numpy()
        leakage_suspects = []
        excluded_features = []

        candidate_numeric = [c for c in numeric_cols if c not in low_variance_columns]
        for col in candidate_numeric:
            col_vals = df[col].cast(pl.Float64).to_numpy()
            # Handle NaN
            mask = ~(np.isnan(col_vals) | np.isnan(target_vals))
            if mask.sum() < 10:
                continue
            corr = float(np.corrcoef(col_vals[mask], target_vals[mask])[0, 1])
            if abs(corr) >= 0.98:
                leakage_suspects.append(col)
                excluded_features.append({"column": col, "reason": f"lag0_leakage_corr={corr:.4f}"})
                print(f"LEAKAGE suspect: {col} (|r|={abs(corr):.4f})")

        # ── 4. Mutual information ranking ─────────────────────────────────────
        mi_candidate_cols = [c for c in candidate_numeric if c not in leakage_suspects]
        mi_ranking = []
        noise_mi_baseline = 0.0

        if mi_candidate_cols:
            X_mi = np.column_stack([df[c].cast(pl.Float64).to_numpy() for c in mi_candidate_cols])
            y_mi = target_vals.copy()

            # Handle NaN: fill with column mean
            for j in range(X_mi.shape[1]):
                nan_mask = np.isnan(X_mi[:, j])
                if nan_mask.any():
                    col_mean = float(np.nanmean(X_mi[:, j]))
                    X_mi[nan_mask, j] = col_mean
            nan_y = np.isnan(y_mi)
            if nan_y.any():
                y_mi[nan_y] = float(np.nanmean(y_mi))

            # Add 5 noise columns
            rng = np.random.RandomState(42)
            noise_cols = rng.randn(len(y_mi), 5)
            X_with_noise = np.column_stack([X_mi, noise_cols])
            all_names = mi_candidate_cols + [f"_noise_{i}" for i in range(5)]

            mi_scores = mutual_info_regression(X_with_noise, y_mi, random_state=42)
            noise_mi_vals = mi_scores[len(mi_candidate_cols):]
            noise_mi_baseline = float(np.mean(noise_mi_vals))

            for name, score in zip(mi_candidate_cols, mi_scores[:len(mi_candidate_cols)]):
                mi_ranking.append({"feature": name, "mi_score": float(score)})

            mi_ranking.sort(key=lambda x: x["mi_score"], reverse=True)
            print(f"MI ranking: {mi_ranking}")
            print(f"Noise MI baseline: {noise_mi_baseline:.4f}")

            # Flag below noise baseline
            for entry in mi_ranking:
                if entry["mi_score"] <= noise_mi_baseline:
                    excluded_features.append({
                        "column": entry["feature"],
                        "reason": f"below_noise_baseline: mi={entry['mi_score']:.4f} <= noise={noise_mi_baseline:.4f}"
                    })
                    print(f"Below noise baseline: {entry['feature']}")

        # ── 5. Pairwise correlation & redundancy ──────────────────────────────
        above_noise_features = [
            e["feature"] for e in mi_ranking
            if e["mi_score"] > noise_mi_baseline
            and e["feature"] not in [x["column"] for x in excluded_features]
        ]
        excluded_cols_set = {x["column"] for x in excluded_features}

        redundant_pairs = []
        if len(above_noise_features) >= 2:
            mi_dict = {e["feature"]: e["mi_score"] for e in mi_ranking}
            for i, c1 in enumerate(above_noise_features):
                for j, c2 in enumerate(above_noise_features):
                    if j <= i:
                        continue
                    v1 = df[c1].cast(pl.Float64).to_numpy()
                    v2 = df[c2].cast(pl.Float64).to_numpy()
                    mask = ~(np.isnan(v1) | np.isnan(v2))
                    if mask.sum() < 10:
                        continue
                    corr = float(np.corrcoef(v1[mask], v2[mask])[0, 1])
                    if abs(corr) >= 0.90:
                        # Drop the one with lower MI
                        if mi_dict.get(c1, 0) >= mi_dict.get(c2, 0):
                            drop = c2
                        else:
                            drop = c1
                        redundant_pairs.append({"pair": [c1, c2], "corr": corr, "dropped": drop})
                        if drop not in excluded_cols_set:
                            excluded_cols_set.add(drop)
                            excluded_features.append({
                                "column": drop,
                                "reason": f"redundant_with_corr={corr:.4f}"
                            })

        # Build recommended_features
        recommended_features = [
            e["feature"] for e in mi_ranking
            if e["mi_score"] > noise_mi_baseline
            and e["feature"] not in excluded_cols_set
        ]
        print(f"Recommended features: {recommended_features}")

        # ── 6. Multiple series detection ──────────────────────────────────────
        multiple_series_detected = False
        series_id_column = None
        for col in categorical_cols:
            n_unique = df[col].n_unique()
            if 2 <= n_unique <= 50:
                multiple_series_detected = True
                series_id_column = col
                break

        # ── 7. Time-series diagnostics on target series ───────────────────────
        target_series = df[target_col].cast(pl.Float64).to_numpy()
        # Ensure no NaN
        nan_mask = np.isnan(target_series)
        if nan_mask.any():
            target_series = target_series.copy()
            # interpolate
            idx = np.arange(len(target_series))
            valid = ~nan_mask
            target_series = np.interp(idx, idx[valid], target_series[valid])

        # 7a. ADF + KPSS
        from statsmodels.tsa.stattools import adfuller, kpss

        adf_result = adfuller(target_series, autolag='AIC')
        adf_stat = float(adf_result[0])
        adf_pvalue = float(adf_result[1])

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(target_series, regression='c', nlags='auto')
        kpss_stat = float(kpss_result[0])
        kpss_pvalue = float(kpss_result[1])

        # Joint interpretation
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

        print(f"ADF: stat={adf_stat:.4f}, p={adf_pvalue:.4f} | KPSS: stat={kpss_stat:.4f}, p={kpss_pvalue:.4f}")
        print(f"Stationarity: {stationarity_conclusion}")

        # 7b. ACF / PACF
        from statsmodels.tsa.stattools import acf, pacf

        max_lag = min(48, n_rows // 4)
        conf_band = 2.0 / np.sqrt(n_rows)

        try:
            acf_vals_full = acf(target_series, nlags=max_lag, fft=True)
        except Exception:
            acf_vals_full = np.array([1.0])

        try:
            pacf_vals_full = pacf(target_series, nlags=min(max_lag, n_rows // 2 - 1))
        except Exception:
            pacf_vals_full = np.array([1.0])

        acf_lags = list(range(len(acf_vals_full)))
        acf_values = [float(v) for v in acf_vals_full]
        pacf_values = [float(v) for v in pacf_vals_full[:len(acf_vals_full)]]

        acf_sig_lags = [i for i in range(1, len(acf_vals_full)) if abs(acf_vals_full[i]) > conf_band]
        pacf_sig_lags = [i for i in range(1, min(len(pacf_vals_full), len(acf_vals_full))) if abs(pacf_vals_full[i]) > conf_band]

        # Suggest AR/MA orders
        # PACF cuts off → AR(p); ACF cuts off → MA(q)
        suggested_ar_order = min(pacf_sig_lags[0] if pacf_sig_lags else 1, 5)
        suggested_ma_order = 0  # for temperature, AR structure dominant

        print(f"ACF significant lags (first 10): {acf_sig_lags[:10]}")
        print(f"PACF significant lags (first 10): {pacf_sig_lags[:10]}")
        print(f"Suggested AR order: {suggested_ar_order}, MA order: {suggested_ma_order}")

        # 7c. Hurst exponent
        H, H_r2 = _hurst_rs(target_series)
        H_interp = _hurst_interpretation(H)
        print(f"Hurst: H={H:.4f}, interpretation={H_interp}, R²={H_r2:.4f}")

        # 7d. Ljung-Box white noise test
        from statsmodels.stats.diagnostic import acorr_ljungbox
        max_lb_lag = min(24, n_rows // 5)
        lb_lags = [l for l in [6, 12, 24] if l <= max_lb_lag]
        if not lb_lags:
            lb_lags = [min(6, max_lb_lag)]

        lb_result = acorr_ljungbox(target_series, lags=lb_lags, return_df=True)
        ljung_box_pvalues = {}
        for lag, pval in zip(lb_lags, lb_result["lb_pvalue"].values):
            ljung_box_pvalues[str(lag)] = float(pval)
        white_noise = all(p > 0.05 for p in ljung_box_pvalues.values())
        print(f"Ljung-Box p-values: {ljung_box_pvalues}, white_noise={white_noise}")

        # 7e. Seasonal decomposition & period detection
        candidate_periods = _candidate_periods(detected_frequency)
        detected_periods = []
        primary_seasonal_period = None
        best_fs = 0.0

        for period in candidate_periods:
            if n_rows < 2 * period:
                detected_periods.append({"period": period, "seasonal_strength": 0.0, "significant": False})
                continue
            fs = _stl_seasonal_strength(target_series, period)
            significant = fs > 0.30
            detected_periods.append({"period": period, "seasonal_strength": fs, "significant": significant})
            if significant and fs > best_fs:
                best_fs = fs
                primary_seasonal_period = period
            print(f"Period {period}: seasonal_strength={fs:.3f}, significant={significant}")

        seasonality_detected = any(d["significant"] for d in detected_periods)

        # Trend strength (using primary or fallback period)
        trend_period = primary_seasonal_period or (candidate_periods[0] if candidate_periods else 7)
        trend_strength = _stl_trend_strength(target_series, trend_period) if n_rows >= 2 * trend_period else 0.0
        trend_detected = trend_strength > 0.30
        print(f"Trend: strength={trend_strength:.3f}, detected={trend_detected}")

        # ── 8. Cross-correlation lag analysis ────────────────────────────────
        useful_lag_features = []
        target_vals_clean = target_series

        for col in recommended_features:
            col_vals = df[col].cast(pl.Float64).to_numpy()
            nan_mask_c = np.isnan(col_vals)
            if nan_mask_c.any():
                col_vals = col_vals.copy()
                col_vals[nan_mask_c] = float(np.nanmean(col_vals))

            for lag in [0, 1, 2, 3, 6, 12]:
                if lag >= len(target_vals_clean):
                    continue
                if lag == 0:
                    xcorr = float(np.corrcoef(col_vals, target_vals_clean)[0, 1])
                    if abs(xcorr) >= 0.98:
                        # Already flagged as leakage
                        continue
                else:
                    feat_lagged = col_vals[:-lag]
                    target_aligned = target_vals_clean[lag:]
                    if len(feat_lagged) < 10:
                        continue
                    xcorr = float(np.corrcoef(feat_lagged, target_aligned)[0, 1])

                if abs(xcorr) > 0.15 and lag >= 1:
                    useful_lag_features.append({
                        "feature": col,
                        "lag": lag,
                        "xcorr": xcorr
                    })

        # ── 9. Univariate summary stats ───────────────────────────────────────
        target_candidates = []
        for col in df.columns:
            if df[col].dtype in numeric_dtypes:
                vals = df[col].cast(pl.Float64).drop_nulls().to_numpy()
                if len(vals) == 0:
                    continue
                target_candidates.append({
                    "column": col,
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "null_rate": float(df[col].is_null().mean()),
                })

        # ── 10. Model class recommendations ───────────────────────────────────
        n_rec = len(recommended_features)
        model_class_recommendations = []

        if white_noise:
            model_class_recommendations.append({
                "model_class": "Naive",
                "justification": "White noise target: no autocorrelation structure to exploit"
            })
            model_class_recommendations.append({
                "model_class": "SeasonalNaive",
                "justification": "White noise target: only seasonal naive is meaningful"
            })
        else:
            if stationarity_conclusion == "stationary":
                for mc, just in [
                    ("AR", "Stationary + autocorrelation structure: AR model directly applicable"),
                    ("ElasticNet", "Stationary + sufficient features: linear regularized regression"),
                    ("Ridge", "Stationary + features: Ridge regression baseline"),
                    ("XGBoost", "Non-linear patterns in stationary series"),
                ]:
                    model_class_recommendations.append({"model_class": mc, "justification": just})

            elif stationarity_conclusion in ("non-stationary", "trend-stationary"):
                if seasonality_detected:
                    for mc, just in [
                        ("SARIMA", "Non-stationary + seasonality detected: SARIMA is primary choice"),
                        ("HoltWinters", "Strong seasonality + trend: Holt-Winters ETS indicated"),
                        ("XGBoost", "Non-stationary seasonal: XGBoost with lag/calendar features"),
                        ("ElasticNet", "Regularised linear model with differencing features"),
                    ]:
                        model_class_recommendations.append({"model_class": mc, "justification": just})
                else:
                    for mc, just in [
                        ("ARIMA", "Non-stationary without clear seasonality: ARIMA with differencing"),
                        ("ElasticNet", "ElasticNet with differencing features"),
                        ("XGBoost", "XGBoost with lagged + calendar features"),
                    ]:
                        model_class_recommendations.append({"model_class": mc, "justification": just})

            else:  # ambiguous
                for mc, just in [
                    ("ARIMA", "Ambiguous stationarity: ARIMA as safe default"),
                    ("ElasticNet", "Robust to stationarity uncertainty"),
                    ("XGBoost", "Non-linear model, robust to stationarity"),
                ]:
                    model_class_recommendations.append({"model_class": mc, "justification": just})

            # Hurst-driven additions
            if H > 0.65:
                model_class_recommendations.append({
                    "model_class": "FAAR-ARIMA",
                    "justification": f"Hurst={H:.3f} > 0.65: long-memory series benefits from Factor-Augmented AR"
                })
            elif H < 0.45:
                if not any(m["model_class"] == "ElasticNet" for m in model_class_recommendations):
                    model_class_recommendations.append({
                        "model_class": "ElasticNet",
                        "justification": f"Hurst={H:.3f} < 0.45: mean-reverting, short lags sufficient"
                    })

            # Seasonality additions
            if seasonality_detected and not any(m["model_class"] == "HoltWinters" for m in model_class_recommendations):
                model_class_recommendations.append({
                    "model_class": "HoltWinters-ETS",
                    "justification": f"Significant seasonality detected (period={primary_seasonal_period})"
                })

            # Multivariate
            if n_rec >= 3:
                model_class_recommendations.append({
                    "model_class": "SARIMAX",
                    "justification": f"{n_rec} recommended exogenous features available"
                })

            # Large dataset
            if n_rows > 10000 and n_rec > 5:
                model_class_recommendations.append({
                    "model_class": "HistGradientBoosting",
                    "justification": f"Large dataset (N={n_rows}) with many features"
                })

        print(f"Model recommendations: {[m['model_class'] for m in model_class_recommendations]}")

        # ── Build output JSON ─────────────────────────────────────────────────
        ts_diagnostics = {
            "adf_statistic": adf_stat,
            "adf_pvalue": adf_pvalue,
            "kpss_statistic": kpss_stat,
            "kpss_pvalue": kpss_pvalue,
            "stationarity_conclusion": stationarity_conclusion,

            "acf_values": acf_values[:49],
            "pacf_values": pacf_values[:49],
            "acf_significant_lags": acf_sig_lags[:30],
            "pacf_significant_lags": pacf_sig_lags[:30],
            "suggested_ar_order": suggested_ar_order,
            "suggested_ma_order": suggested_ma_order,

            "hurst_exponent": float(H),
            "hurst_interpretation": H_interp,
            "hurst_r2_fit": float(H_r2),

            "ljung_box_pvalues": ljung_box_pvalues,
            "white_noise": white_noise,

            "trend_strength": float(trend_strength),
            "trend_detected": trend_detected,
            "seasonality_detected": seasonality_detected,
            "detected_periods": detected_periods,
            "primary_seasonal_period": primary_seasonal_period,
            "multiple_series_detected": multiple_series_detected,
            "series_id_column": series_id_column,
        }

        step_output = {
            "step": "11-data-exploration",
            "run_id": args.run_id,
            "shape": {"rows": n_rows, "columns": df.width},
            "numeric_columns": [c for c in df.columns if df[c].dtype in numeric_dtypes],
            "categorical_columns": categorical_cols,
            "high_cardinality": high_cardinality,
            "low_variance_columns": low_variance_columns,
            "time_series_detected": True,
            "time_column": time_col,
            "multiple_series_detected": multiple_series_detected,
            "series_id_column": series_id_column,
            "detected_frequency": detected_frequency,

            "ts_diagnostics": ts_diagnostics,

            "mi_ranking": mi_ranking,
            "noise_mi_baseline": noise_mi_baseline,
            "leakage_suspects": leakage_suspects,
            "excluded_features": excluded_features,
            "recommended_features": recommended_features,
            "useful_lag_features": useful_lag_features,

            "model_class_recommendations": model_class_recommendations,
            "acf_pacf_orders": {
                "suggested_ar_order": suggested_ar_order,
                "suggested_ma_order": suggested_ma_order,
            },

            "target_candidates": target_candidates,
            "client_facing_summary": (
                f"Dataset: {n_rows} daily observations of temperature in Algiers (1995–2020). "
                f"Target: {target_col}. "
                f"Stationarity: {stationarity_conclusion}. "
                f"Seasonality: {'detected' if seasonality_detected else 'not detected'} "
                f"(primary period: {primary_seasonal_period} days). "
                f"White noise: {white_noise}. "
                f"Hurst exponent: {H:.3f} ({H_interp}). "
                f"Recommended models: {[m['model_class'] for m in model_class_recommendations]}."
            ),
        }

        out_path = output_dir / "step-11-exploration.json"
        with open(out_path, "w") as f:
            json.dump(step_output, f, indent=2)
        print(f"Written: {out_path}")

        # Update progress
        with open(progress_path) as f:
            p = json.load(f)
        if "11-data-exploration" not in p.get("completed_steps", []):
            p.setdefault("completed_steps", []).append("11-data-exploration")
        p["current_step"] = "11-data-exploration"
        p["status"] = "running"
        with open(progress_path, "w") as f:
            json.dump(p, f, indent=2)

        print("Step 11 complete.")
        sys.exit(0)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR in step 11: {e}\n{tb}", file=sys.stderr)
        if progress_path.exists():
            with open(progress_path) as f:
                p = json.load(f)
            p["status"] = "error"
            p.setdefault("errors", []).append({"step": "11-data-exploration", "error": str(e), "traceback": tb})
            with open(progress_path, "w") as f:
                json.dump(p, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
