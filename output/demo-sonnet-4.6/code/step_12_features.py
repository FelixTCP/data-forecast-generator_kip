"""Step 12 — Feature Extraction & Model Preselection."""
import argparse
import json
import os
import sys
import warnings
import numpy as np
import polars as pl
from tqdm import tqdm

warnings.filterwarnings("ignore")


def update_progress(output_dir, step, status, extra=None):
    progress_path = os.path.join(output_dir, "progress.json")
    with open(progress_path) as f:
        progress = json.load(f)
    progress["status"] = status
    progress["current_step"] = step
    if extra:
        progress.update(extra)
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


# Z
def auto_detect_target_column(df, numeric_cols, explicit_target=None):
    if explicit_target:
        if explicit_target in numeric_cols:
            return explicit_target, {"method": "explicit", "score": None}
        raise ValueError(f"Explicit target '{explicit_target}' not in numeric cols: {numeric_cols}")
    best = max(numeric_cols, key=lambda c: df[c].var())
    return best, {"method": "highest_variance", "score": float(df[best].var())}


# A
def compute_lag_mutual_information(df_pd, target_col, max_lag=12):
    from sklearn.feature_selection import mutual_info_regression
    results = []
    y = df_pd[target_col].fillna(df_pd[target_col].median()).values
    for lag in range(1, max_lag + 1):
        y_shifted = df_pd[target_col].shift(lag).fillna(method='bfill').fillna(method='ffill').values
        mi = mutual_info_regression(y_shifted.reshape(-1, 1), y, random_state=42)[0]
        results.append({"lag": lag, "mutual_information": float(mi)})
    results.sort(key=lambda x: x["mutual_information"], reverse=True)
    import pandas as pd
    return pd.DataFrame(results)


# B
def find_best_lags(df_pd, target_col, max_lag=12, top_n=3):
    from statsmodels.tsa.stattools import acf, pacf
    y = df_pd[target_col].fillna(df_pd[target_col].median()).values
    N = len(y)
    actual_max_lag = min(max_lag, N // 4)
    conf_band = 2 / np.sqrt(N)
    try:
        acf_vals = acf(y, nlags=actual_max_lag, fft=True)
        best_lags_acf = sorted(
            range(1, len(acf_vals)),
            key=lambda i: abs(acf_vals[i]),
            reverse=True
        )[:top_n]
    except Exception:
        best_lags_acf = [1, 2, 3]
    try:
        pacf_vals = pacf(y, nlags=min(actual_max_lag, N // 2 - 1))
        best_lags_pacf = sorted(
            range(1, len(pacf_vals)),
            key=lambda i: abs(pacf_vals[i]),
            reverse=True
        )[:top_n]
    except Exception:
        best_lags_pacf = [1, 2, 3]
    mi_df = compute_lag_mutual_information(df_pd, target_col, max_lag=actual_max_lag)
    best_lags_mi = mi_df.head(top_n)["lag"].tolist()
    # Combine (union of top lags by all methods), max 3
    combined = list(dict.fromkeys(best_lags_acf + best_lags_mi))[:top_n]
    return {"best_lags_acf": best_lags_acf, "best_lags_mi": best_lags_mi, "recommended_lags": combined}


# C
def detect_seasonality(df_pd, target_col, time_col):
    from statsmodels.tsa.seasonal import STL
    y = df_pd[target_col].fillna(df_pd[target_col].median()).values
    N = len(y)
    # Try period=365 (yearly), 7 (weekly), 30 (monthly)
    candidate_periods = [365, 30, 7]
    best_period = None
    best_strength = 0.0
    for period in candidate_periods:
        if period >= N // 2:
            continue
        try:
            stl = STL(y, period=period, robust=True)
            res = stl.fit()
            var_resid = np.var(res.resid)
            var_seasonal_resid = np.var(res.seasonal + res.resid)
            fs = max(0.0, 1 - var_resid / var_seasonal_resid) if var_seasonal_resid > 0 else 0.0
            if fs > best_strength:
                best_strength = fs
                best_period = period
        except Exception:
            pass
    has_seasonality = best_strength > 0.30
    return {"has_seasonality": has_seasonality, "dominant_period": best_period, "strength": float(best_strength)}


# D
def analyze_target_distribution(df_pd, target_col):
    y = df_pd[target_col].dropna()
    mean = float(y.mean())
    std = float(y.std())
    skewness = float(y.skew())
    kurtosis = float(y.kurtosis())
    cv = std / abs(mean) if mean != 0 else float('inf')
    outlier_frac = float(((y - mean).abs() > 3 * std).mean())
    tree_suitable = "yes" if outlier_frac < 0.05 and cv < 1.0 else "no"
    return {"mean": mean, "std": std, "skewness": skewness, "kurtosis": kurtosis,
            "cv": cv, "outlier_frac": outlier_frac, "tree_model_suitable": tree_suitable}


# E
def compute_state_space_embedding(series, embedding_dim=3):
    # Simplified: use lag 1 as delay
    N = len(series)
    if N < embedding_dim + 2:
        return {"embedding_matrix": None, "chosen_delay": 1}
    tau = 1
    rows = N - (embedding_dim - 1) * tau
    matrix = np.column_stack([series[i * tau:i * tau + rows] for i in range(embedding_dim)])
    return {"embedding_matrix": matrix.tolist()[:5], "chosen_delay": tau}


# F
def create_strata_features(df_pd, time_col, target_col):
    import pandas as pd
    from scipy.stats import f_oneway
    strata = {}
    active = []
    time_series = pd.to_datetime(df_pd[time_col]) if time_col in df_pd.columns else None
    if time_series is None:
        return {"strata_features": {}, "active_strata": []}
    for feat_name, feat_vals in [
        ("month_of_year", time_series.dt.month),
        ("day_of_week", time_series.dt.dayofweek),
        ("day_of_month", time_series.dt.day),
    ]:
        groups = [df_pd[target_col].dropna().values[df_pd[feat_name if feat_name in df_pd.columns else time_col].values == v]
                  if feat_name in df_pd.columns else [] for v in feat_vals.unique()]
        strata[feat_name] = feat_vals.values.tolist()[:5]
        active.append(feat_name)
    return {"strata_features": strata, "active_strata": active}


# G
def engineer_timeseries_features(df_pl, target_col, time_col, lags, rolling_windows, group_col=None, top_features=None, exclude_features=None):
    """Build time-series features. Uses polars for efficiency."""
    if exclude_features is None:
        exclude_features = []
    if top_features is None:
        top_features = []
    
    feature_df = df_pl.clone()
    metadata = {"lag_features": [], "rolling_features": [], "calendar_features": [], "trend_features": []}
    
    # Calendar features from time_col
    if time_col and time_col in feature_df.columns:
        try:
            # Try to cast to date/datetime if needed
            time_series = feature_df[time_col]
            if time_series.dtype not in (pl.Date, pl.Datetime):
                time_series = time_series.cast(pl.Date)
            feature_df = feature_df.with_columns([
                time_series.dt.month().alias("cal_month"),
                time_series.dt.day().alias("cal_day"),
                time_series.dt.weekday().alias("cal_day_of_week"),
                time_series.dt.year().alias("cal_year"),
            ])
            metadata["calendar_features"] = ["cal_month", "cal_day", "cal_day_of_week", "cal_year"]
        except Exception as e:
            print(f"Warning: calendar feature creation failed: {e}")

    # Trend feature (elapsed days instead of monotone index)
    if time_col and time_col in feature_df.columns:
        try:
            time_series = feature_df[time_col]
            if time_series.dtype not in (pl.Date, pl.Datetime):
                time_series = time_series.cast(pl.Date)
            first_date = time_series.min()
            feature_df = feature_df.with_columns(
                ((time_series - first_date).dt.total_days()).alias("trend_elapsed_days")
            )
            metadata["trend_features"] = ["trend_elapsed_days"]
        except Exception as e:
            print(f"Warning: trend feature creation failed: {e}")

    # Lag features for target (only top lags from ACF/MI analysis)
    valid_lags = [l for l in lags if isinstance(l, int) and l > 0][:3]  # max 3
    for lag in tqdm(valid_lags, desc="Creating lag features"):
        col_name = f"{target_col}_lag_{lag}"
        if group_col and group_col in feature_df.columns:
            feature_df = feature_df.with_columns(
                pl.col(target_col).shift(lag).over(group_col).alias(col_name)
            )
        else:
            feature_df = feature_df.with_columns(
                pl.col(target_col).shift(lag).alias(col_name)
            )
        metadata["lag_features"].append(col_name)

    # Rolling features for target only (2 windows: 7 and 30)
    for window in tqdm(rolling_windows[:2], desc="Creating rolling features"):
        mean_col = f"{target_col}_roll_mean_{window}"
        std_col = f"{target_col}_roll_std_{window}"
        if group_col and group_col in feature_df.columns:
            feature_df = feature_df.with_columns([
                pl.col(target_col).shift(1).over(group_col).rolling_mean(window_size=window).alias(mean_col),
                pl.col(target_col).shift(1).over(group_col).rolling_std(window_size=window).alias(std_col),
            ])
        else:
            feature_df = feature_df.with_columns([
                pl.col(target_col).shift(1).rolling_mean(window_size=window).alias(mean_col),
                pl.col(target_col).shift(1).rolling_std(window_size=window).alias(std_col),
            ])
        metadata["rolling_features"].extend([mean_col, std_col])

    # Lag features for top exogenous features (from useful_lag_features in step 11)
    # These are passed via the useful_lag_features parameter in the caller

    return feature_df, metadata


# H
def preselect_models(feature_matrix, analysis_data, best_lags):
    stationarity = analysis_data.get("ts_diagnostics", {}).get("stationarity_conclusion", "unknown")
    season = analysis_data.get("ts_diagnostics", {}).get("primary_seasonal_period")
    n_features = feature_matrix.shape[1] if hasattr(feature_matrix, 'shape') else 5
    n_rows = feature_matrix.shape[0] if hasattr(feature_matrix, 'shape') else 1000
    
    candidates = []
    if stationarity == "stationary":
        candidates = ["ridge", "random_forest", "gradient_boosting"]
    elif stationarity in ("non-stationary", "trend-stationary"):
        if season:
            candidates = ["gradient_boosting", "random_forest", "ridge"]
        else:
            candidates = ["gradient_boosting", "random_forest", "ridge"]
    else:
        candidates = ["random_forest", "gradient_boosting", "ridge"]
    
    return {"top_recommendation": candidates[0] if candidates else "random_forest",
            "top_3": candidates[:3],
            "reasoning": {"stationarity": stationarity, "seasonal": bool(season)}}


# I
def add_features_for_models(feature_matrix, target_col, recommended_models, analysis_data):
    """Add model-specific features (diffs for ARIMA-style models)."""
    new_features = []
    stationarity = analysis_data.get("ts_diagnostics", {}).get("stationarity_conclusion", "")
    if stationarity in ("non-stationary", "trend-stationary") and target_col in feature_matrix.columns:
        feat = feature_matrix.with_columns(
            pl.col(target_col).diff(1).alias(f"{target_col}_diff1")
        )
        new_features.append(f"{target_col}_diff1")
        return feat, new_features
    return feature_matrix, new_features


# J
def detect_feature_leakage(feature_matrix_pd, target_col, threshold=0.98):
    from scipy.stats import pearsonr
    from sklearn.ensemble import RandomForestRegressor
    
    y = feature_matrix_pd[target_col].fillna(feature_matrix_pd[target_col].median()).values
    candidates = []
    
    for col in feature_matrix_pd.columns:
        if col == target_col:
            continue
        # Skip lag features of target — they are NOT leakage
        if f"{target_col}_lag" in col or f"{target_col}_roll" in col or f"{target_col}_diff" in col:
            continue
        x = feature_matrix_pd[col].fillna(0).values.astype(float)
        try:
            r, _ = pearsonr(x, y)
            if abs(r) >= threshold:
                candidates.append(col)
        except Exception:
            pass
    
    if not candidates:
        return {"status": "pass", "leakage_candidates": [], "threshold": threshold, "reconstruction_probe_r2": None}
    
    # RF probe
    X_cand = feature_matrix_pd[candidates].fillna(0).values
    try:
        rf = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=42)
        rf.fit(X_cand, y)
        probe_r2 = float(rf.score(X_cand, y))
        if probe_r2 > 0.999:
            return {"status": "fail", "leakage_candidates": candidates, "threshold": threshold, "reconstruction_probe_r2": probe_r2}
    except Exception:
        probe_r2 = None
    
    return {"status": "pass", "leakage_candidates": candidates, "threshold": threshold, "reconstruction_probe_r2": probe_r2 if 'probe_r2' in dir() else None}


# K
def remove_zero_variance_features(feature_matrix_pd, target_col, variance_threshold=1e-10):
    removed = {}
    for col in feature_matrix_pd.columns:
        if col == target_col:
            continue
        std = feature_matrix_pd[col].std()
        if std is not None and std <= np.sqrt(variance_threshold):
            removed[col] = "zero_variance"
    cleaned = feature_matrix_pd.drop(columns=list(removed.keys()))
    remaining_features = [c for c in cleaned.columns if c != target_col]
    if len(remaining_features) < 2:
        print(f"ERROR: Only {len(remaining_features)} feature(s) remain after zero-variance removal. Minimum is 2.", file=sys.stderr)
        sys.exit(1)
    return cleaned, removed


# L
def compute_scaling_metadata(feature_matrix_pd, target_col, recommended_models, output_dir):
    import joblib
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    model_name = recommended_models[0] if recommended_models else "random_forest"
    linear_models = ["linear_regression", "ridge", "lasso", "sarima", "prophet", "elasticnet"]
    nn_models = ["lstm", "temporal_cnn", "neural_net"]
    tree_models = ["random_forest", "gradient_boosting", "xgboost", "lightgbm", "histgradientboosting", "gradient_boosting", "ridge"]
    
    # Default: no scaling for trees (most common)
    scaler_type = "None"
    
    if model_name.lower() in linear_models:
        scaler_type = "StandardScaler"
    elif model_name.lower() in nn_models:
        scaler_type = "MinMaxScaler"
    else:
        scaler_type = "None"
    
    feature_cols = [c for c in feature_matrix_pd.columns if c != target_col]
    binary_cols = [c for c in feature_cols if set(feature_matrix_pd[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
    scalable_cols = [c for c in feature_cols if c not in binary_cols]
    
    scaler_path = None
    if scaler_type != "None":
        scaler = StandardScaler() if scaler_type == "StandardScaler" else MinMaxScaler()
        feature_matrix_pd[scalable_cols] = scaler.fit_transform(feature_matrix_pd[scalable_cols].fillna(0))
        scaler_path = os.path.join(output_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
    
    metadata = {
        "scaler_used": scaler_type,
        "features_scaled": scalable_cols if scaler_type != "None" else [],
        "features_not_scaled": ([target_col] + binary_cols) if scaler_type != "None" else feature_cols + [target_col],
        "scaler_path": scaler_path,
    }
    return feature_matrix_pd, metadata


# M
def generate_future_inference_rows(feature_matrix_pd, feat_info, k_future=10):
    """Generate future rows for inference."""
    import pandas as pd
    try:
        last_row = feature_matrix_pd.iloc[-1].copy()
        target_col = feat_info.get("target_col")
        time_step = feat_info.get("time_step", "daily")
        feature_cols = [c for c in feature_matrix_pd.columns if c != target_col]
        
        future_rows = []
        for s in range(1, k_future + 1):
            row = last_row[feature_cols].copy()
            # Update calendar features if present
            for col in feature_cols:
                if "lag" in col:
                    row[col] = last_row.get(target_col, np.nan)
                elif "roll_mean" in col or "roll_std" in col:
                    row[col] = last_row[col]  # forward fill
            future_rows.append(row)
        
        future_df = pd.DataFrame(future_rows).reset_index(drop=True)
        future_df["is_future"] = True
        
        last_known_date = str(feat_info.get("last_known_date", ""))
        future_meta = {
            "k_future": k_future,
            "last_known_date": last_known_date,
            "time_step": time_step,
            "placeholder_lags": list(range(1, k_future)),
            "features_future_parquet": None,
        }
        return future_df, future_meta
    except Exception as e:
        print(f"Warning: future row generation failed: {e}. Skipping.")
        return None, {"k_future": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split-mode", default="auto")
    parser.add_argument("--exclude-features", default="")
    parser.add_argument("--max-lag", type=int, default=12)
    parser.add_argument("--seasonal-features", default="false")
    args = parser.parse_args()

    output_dir = args.output_dir
    run_id = args.run_id
    split_mode = args.split_mode
    exclude_from_cli = [f.strip() for f in args.exclude_features.split(",") if f.strip()]

    update_progress(output_dir, "12-feature-extraction", "running")

    # Load inputs
    step10 = json.load(open(os.path.join(output_dir, "step-10-cleanse.json")))
    step11 = json.load(open(os.path.join(output_dir, "step-11-exploration.json")))

    target_col = step10["target_column_normalized"]
    time_col = step10["time_column_detected"]
    recommended_features = step11.get("recommended_features", [])
    excluded_from_step11 = step11.get("excluded_features", {})
    useful_lag_features = step11.get("useful_lag_features", [])
    significant_lags = step11.get("significant_lags", [])
    multiple_series = step11.get("multiple_series_detected", False)
    group_col = step11.get("series_id_column")
    primary_period = step11.get("ts_diagnostics", {}).get("primary_seasonal_period")

    # Start with recommended_features from step 11 (never include step 11 exclusions)
    candidate_features = [f for f in recommended_features if f not in exclude_from_cli]

    # Remove features excluded by CLI (remediation)
    features_excluded = dict(excluded_from_step11)
    for f in exclude_from_cli:
        if f not in features_excluded:
            features_excluded[f] = "excluded_by_orchestrator"

    # Load parquet
    df_pl = pl.read_parquet(os.path.join(output_dir, "cleaned.parquet"))
    import pandas as pd
    df_pd = df_pl.to_pandas()

    # Verify target is in data
    if target_col not in df_pd.columns:
        print(f"ERROR: target '{target_col}' not found in parquet columns: {df_pd.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    # Verify candidate features exist
    available_cols = df_pd.columns.tolist()
    candidate_features = [f for f in candidate_features if f in available_cols]

    # Z: auto-detect target
    numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
    if time_col in numeric_cols:
        numeric_cols.remove(time_col)
    target_info = auto_detect_target_column(df_pd, numeric_cols, explicit_target=target_col)

    # A: lag MI
    print("A: Computing lag mutual information...")
    lag_mi_df = compute_lag_mutual_information(df_pd, target_col, max_lag=min(args.max_lag, 12))

    # B: best lags
    print("B: Finding best lags...")
    best_lags_info = find_best_lags(df_pd, target_col, max_lag=min(args.max_lag, 12), top_n=3)
    recommended_lags = best_lags_info["recommended_lags"]

    # Merge with significant_lags from step 11
    all_lags = list(dict.fromkeys(recommended_lags + significant_lags[:5]))[:3]

    # C: seasonality
    print("C: Detecting seasonality...")
    season_info = detect_seasonality(df_pd, target_col, time_col)
    has_seasonality = season_info["has_seasonality"]
    dominant_period = season_info["dominant_period"] or primary_period

    # D: target distribution
    print("D: Analyzing target distribution...")
    target_dist = analyze_target_distribution(df_pd, target_col)

    # E: state space embedding (fast)
    print("E: State space embedding...")
    y_vals = df_pd[target_col].fillna(df_pd[target_col].median()).values
    embedding_info = compute_state_space_embedding(y_vals, embedding_dim=3)

    # F: strata features
    print("F: Strata features...")
    strata_info = create_strata_features(df_pd, time_col, target_col)

    # G: engineer time series features
    print("G: Engineering time-series features...")
    rolling_windows = [7, 30]
    feature_df_pl, feat_metadata = engineer_timeseries_features(
        df_pl, target_col, time_col,
        lags=all_lags,
        rolling_windows=rolling_windows,
        group_col=group_col if multiple_series else None,
        top_features=candidate_features,
        exclude_features=list(features_excluded.keys()),
    )

    # Add exogenous lag features from step 11 useful_lag_features
    for ulf in tqdm(useful_lag_features, desc="Creating exo lag features"):
        feat = ulf["feature"]
        lag = ulf["lag"]
        if feat in feature_df_pl.columns and feat != target_col and feat not in list(features_excluded.keys()):
            col_name = f"{feat}_lag_{lag}"
            if col_name not in feature_df_pl.columns:
                if multiple_series and group_col and group_col in feature_df_pl.columns:
                    feature_df_pl = feature_df_pl.with_columns(
                        pl.col(feat).shift(lag).over(group_col).alias(col_name)
                    )
                else:
                    feature_df_pl = feature_df_pl.with_columns(
                        pl.col(feat).shift(lag).alias(col_name)
                    )
                feat_metadata["lag_features"].append(col_name)

    # Add Fourier features if seasonality is strong enough
    if has_seasonality and dominant_period and season_info["strength"] > 0.30:
        print(f"Adding Fourier features for period={dominant_period}")
        try:
            t = np.arange(len(feature_df_pl))
            for k in [1, 2]:
                sin_col = f"fourier_sin_{dominant_period}_{k}"
                cos_col = f"fourier_cos_{dominant_period}_{k}"
                feature_df_pl = feature_df_pl.with_columns([
                    pl.Series(sin_col, np.sin(2 * np.pi * k * t / dominant_period).tolist()),
                    pl.Series(cos_col, np.cos(2 * np.pi * k * t / dominant_period).tolist()),
                ])
        except Exception as e:
            print(f"Warning: Fourier features failed: {e}")

    # Convert to pandas for further processing
    feature_df_pd = feature_df_pl.to_pandas()

    # Drop rows with NaN in target
    feature_df_pd = feature_df_pd.dropna(subset=[target_col])

    # H: model preselection
    print("H: Model preselection...")
    model_recs = preselect_models(
        feature_df_pd.drop(columns=[target_col] + ([time_col] if time_col in feature_df_pd.columns else [])),
        step11,
        best_lags_info
    )

    # I: add model-specific features
    print("I: Adding model-specific features...")
    feature_df_pd_ext, new_feats = add_features_for_models(
        feature_df_pl, target_col, model_recs["top_3"], step11
    )
    if hasattr(feature_df_pd_ext, 'to_pandas'):
        feature_df_pd = feature_df_pd_ext.to_pandas()
        feature_df_pd = feature_df_pd.dropna(subset=[target_col])

    # Determine feature columns (all except target and time)
    drop_cols = [target_col]
    if time_col and time_col in feature_df_pd.columns:
        drop_cols.append(time_col)
    # Drop categorical/non-numeric columns
    categorical_cols = feature_df_pd.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in categorical_cols:
        if c not in drop_cols:
            drop_cols.append(c)
            features_excluded[c] = "non_numeric"
    feature_cols = [c for c in feature_df_pd.columns if c not in drop_cols]

    # Ensure no excluded step-11 features crept in
    for exc_feat in list(excluded_from_step11.keys()):
        if exc_feat in feature_cols:
            feature_cols.remove(exc_feat)
            features_excluded[exc_feat] = excluded_from_step11[exc_feat]

    # Build feature matrix
    feat_matrix = feature_df_pd[feature_cols + [target_col]].copy()

    # Fill NaN with forward/backward fill, then median
    feat_matrix = feat_matrix.ffill().bfill()
    feat_matrix = feat_matrix.fillna(feat_matrix.median(numeric_only=True))

    # J: leakage detection
    print("J: Detecting leakage...")
    leakage_result = detect_feature_leakage(feat_matrix, target_col, threshold=0.98)
    if leakage_result["status"] == "fail":
        print(f"LEAKAGE DETECTED: {leakage_result['leakage_candidates']}", file=sys.stderr)
        sys.exit(2)

    # K: zero variance removal
    print("K: Removing zero-variance features...")
    feat_matrix, zv_removed = remove_zero_variance_features(feat_matrix, target_col)
    for col, reason in zv_removed.items():
        features_excluded[col] = reason
    feature_cols = [c for c in feat_matrix.columns if c != target_col]

    # Leakage guard: Pearson |r| > 0.99 on full dataset
    from scipy.stats import pearsonr
    y_full = feat_matrix[target_col].values
    for col in list(feature_cols):
        try:
            r, _ = pearsonr(feat_matrix[col].values, y_full)
            if abs(r) > 0.99:
                print(f"WARNING: Leakage guard triggered for {col} (|r|={abs(r):.4f}). Removing.")
                features_excluded[col] = "leakage_pearson_r_above_0.99"
                feature_cols.remove(col)
                feat_matrix = feat_matrix.drop(columns=[col])
        except Exception:
            pass

    # Final check: at least 2 features
    if len(feature_cols) < 2:
        print(f"ERROR: Only {len(feature_cols)} feature(s) after processing. Minimum is 2.", file=sys.stderr)
        sys.exit(1)

    # L: scaling metadata
    print("L: Computing scaling metadata...")
    feat_matrix, scaling_meta = compute_scaling_metadata(
        feat_matrix.copy(), target_col, model_recs["top_3"], output_dir
    )

    # Determine split strategy
    if split_mode == "auto":
        if time_col:
            resolved_mode = "time_series"
        else:
            resolved_mode = "random"
    else:
        resolved_mode = split_mode

    # Write features.parquet
    features_parquet_path = os.path.join(output_dir, "features.parquet")
    feat_pl = pl.from_pandas(feat_matrix)
    feat_pl.write_parquet(features_parquet_path)

    # M: future inference rows
    print("M: Generating future inference rows...")
    last_date = str(feature_df_pd[time_col].max()) if time_col in feature_df_pd.columns else None
    future_df, future_meta = generate_future_inference_rows(
        feat_matrix,
        feat_info={"target_col": target_col, "time_step": "daily", "last_known_date": last_date},
        k_future=10
    )
    if future_df is not None:
        try:
            future_parquet = os.path.join(output_dir, "features_future.parquet")
            pl.from_pandas(future_df).write_parquet(future_parquet)
            future_meta["features_future_parquet"] = future_parquet
        except Exception as e:
            print(f"Warning: Failed to write future parquet: {e}")

    # Build output JSON
    result = {
        "step": "12-feature-extraction",
        "run_id": run_id,
        "features": feature_cols,
        "features_count": len(feature_cols),
        "features_excluded": features_excluded,
        "excluded_count": len(features_excluded),
        "target_column": target_col,
        "split_strategy": {"resolved_mode": resolved_mode},
        "leakage": leakage_result,
        "scaling_metadata": scaling_meta,
        "future_inference": future_meta,
        "artifacts": {
            "features_parquet": features_parquet_path,
            "scaler_joblib": scaling_meta.get("scaler_path"),
        },
        "engineered_metadata": feat_metadata,
        "model_preselection": model_recs,
        "seasonality": season_info,
        "target_distribution": target_dist,
        "context": {
            "target_column": target_col,
            "time_column": time_col,
            "features": feature_cols,
            "split_mode": resolved_mode,
        }
    }

    # JSON serialization helper
    def make_ser(obj):
        if isinstance(obj, dict):
            return {k: make_ser(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_ser(v) for v in obj]
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

    out_json = os.path.join(output_dir, "step-12-features.json")
    with open(out_json, "w") as f:
        json.dump(make_ser(result), f, indent=2)

    # Update progress
    with open(os.path.join(output_dir, "progress.json")) as f:
        progress = json.load(f)
    progress["status"] = "running"
    progress["current_step"] = "13-model-training"
    if "12-feature-extraction" not in progress.get("completed_steps", []):
        progress["completed_steps"].append("12-feature-extraction")
    with open(os.path.join(output_dir, "progress.json"), "w") as f:
        json.dump(progress, f, indent=2)

    print(f"Step 12 complete. Features: {feature_cols}")
    print(f"Split mode: {resolved_mode}")
    sys.exit(0)


if __name__ == "__main__":
    main()
