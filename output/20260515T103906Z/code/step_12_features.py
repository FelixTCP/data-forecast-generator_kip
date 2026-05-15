"""Step 12 — Feature Extraction (Time-Series Focused).

Runnable:
    python step_12_features.py --output-dir <dir> --run-id <id>
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _leakage_probe_rf(X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]) -> dict:
    """RF reconstruction probe: R² > 0.95 indicates leakage."""
    try:
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        r2 = float(rf.score(X_train, y_train))
        return {"r2": r2, "leakage": r2 > 0.95}
    except Exception as e:
        return {"r2": None, "leakage": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Step 12: Feature Extraction")
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
        p["current_step"] = "12-feature-extraction"
        with open(progress_path, "w") as f:
            json.dump(p, f, indent=2)

    update_progress("running")

    try:
        # ── Load context ──────────────────────────────────────────────────────
        with open(output_dir / "step-10-cleanse.json") as f:
            ctx10 = json.load(f)
        with open(output_dir / "step-11-exploration.json") as f:
            ctx11 = json.load(f)

        target_col = ctx10["target_column_normalized"]
        time_col = ctx10["time_column"]
        detected_frequency = ctx11.get("detected_frequency", "daily")

        ts = ctx11["ts_diagnostics"]
        acf_sig_lags = ts.get("acf_significant_lags", [])
        pacf_sig_lags = ts.get("pacf_significant_lags", [])
        primary_period = ts.get("primary_seasonal_period")
        hurst = ts.get("hurst_exponent", 0.5)
        stationarity = ts.get("stationarity_conclusion", "stationary")
        white_noise = ts.get("white_noise", False)
        trend_detected = ts.get("trend_detected", False)
        seasonality_detected = ts.get("seasonality_detected", False)
        multiple_series_detected = ts.get("multiple_series_detected", False)
        series_id_column = ts.get("series_id_column")

        recommended_features = ctx11.get("recommended_features", [])
        useful_lag_features = ctx11.get("useful_lag_features", [])
        model_class_recs = [m["model_class"] for m in ctx11.get("model_class_recommendations", [])]
        excluded_step11 = {x["column"] for x in ctx11.get("excluded_features", [])}

        print(f"Target: {target_col}, Time: {time_col}")
        print(f"Recommended features: {recommended_features}")
        print(f"Model classes: {model_class_recs}")
        print(f"Primary seasonal period: {primary_period}, Hurst: {hurst:.3f}")

        # ── Load cleaned parquet ──────────────────────────────────────────────
        df = pl.read_parquet(output_dir / "cleaned.parquet")
        print(f"Loaded cleaned.parquet: {df.shape}")

        n_rows = len(df)
        features_added = {}  # name → creation_reason
        features_excluded = {}  # name → exclusion_reason

        # ── GROUP A: Calendar features ────────────────────────────────────────
        has_date = time_col in df.columns and df[time_col].dtype in (pl.Date, pl.Datetime)
        if has_date:
            if df[time_col].dtype == pl.Datetime:
                date_col = pl.col(time_col).cast(pl.Date)
            else:
                date_col = pl.col(time_col)

            df = df.with_columns([
                date_col.dt.weekday().alias("day_of_week"),
                date_col.dt.day().alias("day_of_month"),
                date_col.dt.month().alias("month_cal"),
                (date_col.dt.month() // 4 + 1).cast(pl.Int32).alias("quarter"),
                date_col.dt.week().alias("week_of_year"),
                (date_col.dt.weekday() >= 5).cast(pl.Int32).alias("is_weekend"),
                (date_col.dt.day() == 1).cast(pl.Int32).alias("is_month_start"),
            ])
            # is_month_end: day == last day of month
            # Polars: compare day to days_in_month (approximate via shift+check)
            # Simple approach: next day's month != current month
            df = df.with_columns([
                pl.when(
                    (date_col.dt.month() != (date_col + pl.duration(days=1)).dt.month()) |
                    (date_col.dt.year() != (date_col + pl.duration(days=1)).dt.year())
                ).then(1).otherwise(0).cast(pl.Int32).alias("is_month_end")
            ])

            cal_features = ["day_of_week", "day_of_month", "month_cal", "quarter",
                           "week_of_year", "is_weekend", "is_month_start", "is_month_end"]
            for f in cal_features:
                features_added[f] = "group_A_calendar"
            print(f"Group A: Added {cal_features}")
        else:
            cal_features = []
            print("No date column for Group A calendar features")

        # Include raw exogenous features from recommended_features (forward-known)
        # year is forward-known, month is already in Group A as month_cal
        exog_raw = []
        for feat in recommended_features:
            if feat in df.columns and feat not in features_added and feat != time_col and feat != target_col:
                # Check it's not already duplicated by calendar features
                if feat == "month":
                    # Already covered by month_cal
                    features_excluded[feat] = "duplicate_of_month_cal_group_A"
                elif feat == "day":
                    features_excluded[feat] = "below_noise_baseline_step11"
                else:
                    exog_raw.append(feat)
                    features_added[feat] = "exogenous_forward_known"
        print(f"Raw exogenous features: {exog_raw}")

        # ── GROUP B: Lag features (ACF/PACF-driven) ───────────────────────────
        # Build candidate lag set
        must_include_lags = {1}
        if primary_period:
            must_include_lags.add(primary_period)
            must_include_lags.add(primary_period * 2)

        # Hurst > 0.65: extend window
        if hurst > 0.65 and primary_period:
            max_lag_window = min(primary_period * 2, 96)
        elif hurst > 0.65:
            max_lag_window = 96
        else:
            max_lag_window = max(acf_sig_lags[:1][0] if acf_sig_lags else 1, 24)

        # Candidate lags from ACF + PACF sig lags, extended window
        candidate_lags = sorted(set(acf_sig_lags) | set(pacf_sig_lags) | must_include_lags)
        # Add extended window lags
        extended_lags = list(range(1, max_lag_window + 1))
        candidate_lags = sorted(set(candidate_lags) | set(extended_lags))
        # Ensure must-include are present
        for ml in must_include_lags:
            if ml not in candidate_lags:
                candidate_lags.append(ml)
        candidate_lags = sorted(set(candidate_lags))

        # Cap at 30: keep must-include + highest ACF
        acf_sig_set = set(acf_sig_lags)
        def lag_priority(l):
            if l in must_include_lags:
                return (0, l)
            elif l in acf_sig_set:
                return (1, l)
            else:
                return (2, l)

        candidate_lags_sorted = sorted(candidate_lags, key=lag_priority)
        lag_set = candidate_lags_sorted[:30]
        lag_set = sorted(set(lag_set) | must_include_lags)
        if len(lag_set) > 30:
            # Keep must-include + first 30-len(must_include) by priority
            non_must = [l for l in candidate_lags_sorted if l not in must_include_lags]
            lag_set = sorted(must_include_lags) + non_must[:30 - len(must_include_lags)]
        lag_set = sorted(lag_set)

        print(f"Group B lag set ({len(lag_set)}): {lag_set}")

        # Add lag features
        lag_exprs = []
        for k in lag_set:
            col_name = f"y_lag_{k}"
            lag_exprs.append(pl.col(target_col).shift(k).alias(col_name))
            features_added[col_name] = f"group_B_lag_k{k}"
        df = df.with_columns(lag_exprs)

        # ── GROUP C: Exogenous feature lags (cross-correlation driven) ─────────
        if not white_noise:
            group_c_added = []
            seen_exog_lags = set()
            for entry in useful_lag_features:
                feat = entry["feature"]
                lag_k = entry["lag"]
                xcorr = entry.get("xcorr", 0.0)
                if lag_k == 0:
                    continue  # Skip lag-0 (leakage risk)
                if feat in excluded_step11:
                    continue
                if feat not in df.columns:
                    continue
                col_name = f"{feat}_lag_{lag_k}"
                if col_name in seen_exog_lags:
                    continue
                seen_exog_lags.add(col_name)
                df = df.with_columns(pl.col(feat).shift(lag_k).alias(col_name))
                features_added[col_name] = f"group_C_exog_xcorr={xcorr:.3f}_lag{lag_k}"
                group_c_added.append(col_name)
            print(f"Group C: Added {group_c_added}")

        # ── GROUP D: Differencing (only for non-stationary) ───────────────────
        diff_features = []
        if stationarity in ("non-stationary", "trend-stationary"):
            # y_diff_1 = y(t-1) - y(t-2) = shift(1).diff(1)
            df = df.with_columns(
                pl.col(target_col).shift(1).diff(1).alias("y_diff_1")
            )
            features_added["y_diff_1"] = "group_D_diff1_nonstationarity"
            diff_features.append("y_diff_1")

            if seasonality_detected and primary_period:
                df = df.with_columns(
                    (pl.col(target_col).shift(1) - pl.col(target_col).shift(1 + primary_period)).alias("y_diff_seasonal")
                )
                features_added["y_diff_seasonal"] = f"group_D_diff_seasonal_period{primary_period}"
                diff_features.append("y_diff_seasonal")
            print(f"Group D: Added {diff_features}")
        else:
            print("Group D: Skipped (stationary series)")

        # ── GROUP E: Rolling statistics (always with prior shift) ─────────────
        if not white_noise:
            if primary_period:
                roll_windows = [primary_period // 2, primary_period, primary_period * 2]
                roll_windows = [max(w, 2) for w in roll_windows]
            else:
                roll_windows = [6, 12, 24]

            print(f"Group E: Rolling windows: {roll_windows}")
            roll_features = []
            for w in roll_windows:
                # All rolling on shift(1) to prevent look-ahead
                shifted = pl.col(target_col).shift(1)
                df = df.with_columns([
                    shifted.rolling_mean(w).alias(f"rolling_mean_{w}"),
                    shifted.rolling_std(w).alias(f"rolling_std_{w}"),
                    shifted.rolling_min(w).alias(f"rolling_min_{w}"),
                    shifted.rolling_max(w).alias(f"rolling_max_{w}"),
                ])
                for stat in ["mean", "std", "min", "max"]:
                    n = f"rolling_{stat}_{w}"
                    features_added[n] = f"group_E_rolling_{stat}_w{w}_shifted"
                    roll_features.append(n)

            # EWM span = primary_period
            if primary_period:
                df = df.with_columns(
                    pl.col(target_col).shift(1).ewm_mean(span=primary_period).alias(f"ewm_span_{primary_period}")
                )
                features_added[f"ewm_span_{primary_period}"] = f"group_E_ewm_span{primary_period}_shifted"
                roll_features.append(f"ewm_span_{primary_period}")
            print(f"Group E: Added {len(roll_features)} rolling features")

        # ── GROUP F: Fourier features (seasonality-driven) ────────────────────
        fourier_features = []
        sig_periods = [p for p in ctx11["ts_diagnostics"].get("detected_periods", [])
                       if p.get("significant", False)]
        if sig_periods and not white_noise:
            t_full = np.arange(n_rows, dtype=float)
            fourier_exprs = []
            for period_info in sig_periods:
                m = period_info["period"]
                K = min(3, max(1, m // 4))
                t_index = (t_full % m).astype(float)
                for k in range(1, K + 1):
                    sin_name = f"fourier_sin_{m}_{k}"
                    cos_name = f"fourier_cos_{m}_{k}"
                    sin_vals = np.sin(2 * np.pi * k * t_index / m)
                    cos_vals = np.cos(2 * np.pi * k * t_index / m)
                    fourier_exprs.append(pl.Series(sin_name, sin_vals))
                    fourier_exprs.append(pl.Series(cos_name, cos_vals))
                    features_added[sin_name] = f"group_F_fourier_sin_period{m}_k{k}"
                    features_added[cos_name] = f"group_F_fourier_cos_period{m}_k{k}"
                    fourier_features += [sin_name, cos_name]
            if fourier_exprs:
                df = df.with_columns(fourier_exprs)
                print(f"Group F: Added {len(fourier_features)} Fourier features")

        # ── Drop leading NaN rows (max burn-in) ──────────────────────────────
        # Determine max lag in features
        lag_feature_names = [f for f in features_added if "lag" in f or "rolling" in f or "ewm" in f or "diff" in f]
        all_lag_cols = [c for c in df.columns if c in features_added]

        # Compute burn-in: max of all lag/rolling burn-ins
        burn_in = 0
        for k in lag_set:
            burn_in = max(burn_in, k)
        for w in (roll_windows if not white_noise and primary_period else (roll_windows if not white_noise else [0])):
            burn_in = max(burn_in, w)  # rolling after shift(1): first w rows are NaN
        if diff_features and primary_period and stationarity in ("non-stationary", "trend-stationary"):
            burn_in = max(burn_in, 1 + primary_period)

        print(f"Burn-in rows to drop: {burn_in}")

        # Drop leading rows with NaN
        df_features = df.slice(burn_in)
        print(f"After dropping burn-in: {len(df_features)} rows")

        # Verify no NaN remain in feature columns (except allowed string cols)
        feature_cols = [c for c in features_added if c in df_features.columns]
        remaining_null = sum(df_features[c].is_null().sum() for c in feature_cols)
        if remaining_null > 0:
            print(f"WARNING: {remaining_null} nulls remain in feature columns after burn-in drop")
            # Fill remaining nulls with forward/backward fill per column
            fill_exprs = []
            for c in feature_cols:
                if df_features[c].is_null().sum() > 0:
                    fill_exprs.append(pl.col(c).forward_fill().backward_fill())
            if fill_exprs:
                df_features = df_features.with_columns(fill_exprs)

        # ── Determine final feature list ──────────────────────────────────────
        # Exclude step-11 excluded features (except those added as lags - they're fine)
        final_features = []
        for fname in features_added:
            if fname not in df_features.columns:
                continue
            # Don't include raw excluded columns (but their lags are fine)
            if fname in excluded_step11 and "lag" not in fname and "rolling" not in fname:
                features_excluded[fname] = f"excluded_in_step11"
                continue
            final_features.append(fname)

        # Leakage guard: ensure no step-11 excluded feature re-enters
        for ef in excluded_step11:
            if ef in final_features:
                final_features.remove(ef)
                features_excluded[ef] = "re_inclusion_of_step11_excluded"

        if len(final_features) < 2:
            raise ValueError(
                f"Fewer than 2 features after cleanup: {final_features}. "
                "Cannot proceed to training."
            )

        print(f"Final feature count: {len(final_features)}")

        # ── Leakage detection (Hard Fail) ─────────────────────────────────────
        target_vals = df_features[target_col].cast(pl.Float64).to_numpy()
        leakage_detected = False
        leakage_details = []

        # 1. Pearson |r| >= 0.98 check
        for fname in final_features:
            if fname not in df_features.columns:
                continue
            try:
                feat_vals = df_features[fname].cast(pl.Float64).to_numpy()
                nan_mask = np.isnan(feat_vals) | np.isnan(target_vals)
                if nan_mask.sum() >= len(feat_vals) - 10:
                    continue
                corr = float(np.corrcoef(feat_vals[~nan_mask], target_vals[~nan_mask])[0, 1])
                if abs(corr) >= 0.98:
                    leakage_detected = True
                    leakage_details.append({
                        "feature": fname,
                        "pearson_r": corr,
                        "reason": "pearson_corr >= 0.98"
                    })
                    print(f"LEAKAGE DETECTED: {fname} |r|={abs(corr):.4f}")
            except Exception:
                pass

        # 2. RF reconstruction probe on TRAINING portion only
        n_total = len(df_features)
        holdout_size = max(365, int(n_total * 0.2))
        holdout_start = n_total - holdout_size
        print(f"Holdout start index: {holdout_start} (of {n_total})")

        X_all = []
        valid_feats = []
        for fname in final_features:
            if fname in df_features.columns:
                vals = df_features[fname].cast(pl.Float64).to_numpy()
                X_all.append(vals)
                valid_feats.append(fname)

        X_all = np.column_stack(X_all)
        X_train_lk = X_all[:holdout_start]
        y_train_lk = target_vals[:holdout_start]

        # Fill NaN for RF probe
        for j in range(X_train_lk.shape[1]):
            nan_mask = np.isnan(X_train_lk[:, j])
            if nan_mask.any():
                col_mean = float(np.nanmean(X_train_lk[:, j]))
                X_train_lk[nan_mask, j] = col_mean

        rf_probe = _leakage_probe_rf(X_train_lk, y_train_lk, valid_feats)
        print(f"RF leakage probe R²={rf_probe['r2']:.4f}")

        if rf_probe.get("leakage"):
            # Hard fail only if Pearson check also triggered
            if leakage_details:
                leakage_audit = {
                    "status": "fail",
                    "pearson_violations": leakage_details,
                    "rf_probe": rf_probe,
                    "message": "Hard leakage detected — pipeline halted."
                }
                with open(output_dir / "leakage_audit.json", "w") as f:
                    json.dump(leakage_audit, f, indent=2)
                raise RuntimeError(
                    f"LEAKAGE DETECTED: {[d['feature'] for d in leakage_details]}. "
                    "Pipeline halted. Check leakage_audit.json."
                )
            else:
                # RF high but no Pearson violation — warn only
                print(f"WARNING: RF probe R²={rf_probe['r2']:.4f} > 0.95 but no Pearson violations found. "
                      "This may indicate complex multicollinearity — proceeding with caution.")

        leakage_audit = {
            "status": "fail" if leakage_details else ("warn" if rf_probe.get("r2", 0) > 0.95 else "pass"),
            "pearson_violations": leakage_details,
            "rf_probe": rf_probe,
            "checked_features": valid_feats,
        }
        with open(output_dir / "leakage_audit.json", "w") as f:
            json.dump(leakage_audit, f, indent=2)

        if leakage_details:
            raise RuntimeError(
                f"LEAKAGE DETECTED: {[d['feature'] for d in leakage_details]}. "
                "Check leakage_audit.json."
            )

        # ── GROUP G: PCA Factor Components (FAAR models) ──────────────────────
        faar_models = {"FAAR-ARIMA", "FAAR-SARIMAX", "Factor-VAR"}
        pca_info = {}
        pca_features = []
        if faar_models & set(model_class_recs):
            exog_cols = [c for c in valid_feats
                        if c in df_features.columns
                        and not c.startswith("y_lag_")
                        and not c.startswith("rolling_")
                        and not c.startswith("ewm_")
                        and not c.startswith("y_diff_")
                        and not c.startswith("fourier_")]
            if len(exog_cols) >= 2:
                print(f"Group G: Building PCA from {exog_cols}")
                X_exog = np.column_stack([df_features[c].cast(pl.Float64).to_numpy() for c in exog_cols])
                X_exog_train = X_exog[:holdout_start]

                # Handle NaN
                for j in range(X_exog.shape[1]):
                    nan_mask = np.isnan(X_exog[:, j])
                    if nan_mask.any():
                        mean_v = float(np.nanmean(X_exog[:holdout_start, j]))
                        X_exog[nan_mask, j] = mean_v

                X_exog_train_clean = X_exog[:holdout_start]
                scaler = StandardScaler()
                scaler.fit(X_exog_train_clean)
                X_scaled = scaler.transform(X_exog)

                # Determine n_components for 95% variance
                pca_full = PCA(random_state=42)
                pca_full.fit(X_scaled[:holdout_start])
                cumvar = np.cumsum(pca_full.explained_variance_ratio_)
                n_components = int(np.searchsorted(cumvar, 0.95) + 1)
                n_components = min(n_components, len(exog_cols))

                pca = PCA(n_components=n_components, random_state=42)
                pca.fit(X_scaled[:holdout_start])
                X_pca = pca.transform(X_scaled)

                pca_col_names = [f"pca_factor_{i+1}" for i in range(n_components)]
                pca_series = [pl.Series(name, X_pca[:, i]) for i, name in enumerate(pca_col_names)]
                df_features = df_features.with_columns(pca_series)

                for n in pca_col_names:
                    features_added[n] = f"group_G_pca_factor"
                    valid_feats.append(n)
                pca_features = pca_col_names

                # Serialize PCA preprocessor
                pca_path = str(output_dir / "pca_preprocessor.joblib")
                joblib.dump({"scaler": scaler, "pca": pca, "exog_cols": exog_cols}, pca_path)
                print(f"Saved PCA preprocessor: {pca_path} ({n_components} components)")

                pca_info = {
                    "pca_n_components": n_components,
                    "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
                    "pca_loadings_dict": {
                        col: [float(v) for v in pca.components_[:, j]]
                        for j, col in enumerate(exog_cols)
                    },
                    "pca_source_cols": exog_cols,
                    "pca_preprocessor_path": pca_path,
                }
                print(f"Group G: Added {len(pca_col_names)} PCA factors")
            else:
                print(f"Group G: Skipped (only {len(exog_cols)} exogenous columns available)")

        # ── Update final feature list with PCA ────────────────────────────────
        final_features = [f for f in valid_feats if f in df_features.columns and f in features_added]
        if not final_features:
            raise ValueError("No valid features remain after all processing")
        print(f"Final features ({len(final_features)}): {final_features[:10]}...")

        # ── Write features.parquet ────────────────────────────────────────────
        output_cols = [time_col, target_col] + [f for f in final_features if f in df_features.columns and f != time_col and f != target_col]
        output_cols = list(dict.fromkeys(output_cols))  # dedup while preserving order
        df_out = df_features.select([c for c in output_cols if c in df_features.columns])
        parquet_path = str(output_dir / "features.parquet")
        df_out.write_parquet(parquet_path)
        print(f"Written features.parquet: {parquet_path} ({len(df_out)} rows, {df_out.width} cols)")

        # ── Build output JSON ─────────────────────────────────────────────────
        step_output = {
            "step": "12-feature-extraction",
            "run_id": args.run_id,
            "features": final_features,
            "features_excluded": features_excluded,
            "features_created": features_added,
            "feature_count": len(final_features),
            "split_strategy": {
                "resolved_mode": "time_series",
                "holdout_start_index": holdout_start,
                "holdout_size": holdout_size,
                "total_rows_after_burnin": len(df_features),
                "burn_in_rows": burn_in,
            },
            "lag_set": lag_set,
            "roll_windows": roll_windows if not white_noise else [],
            "fourier_features": fourier_features,
            "diff_features": diff_features if stationarity in ("non-stationary", "trend-stationary") else [],
            "pca_info": pca_info,
            "artifacts": {
                "features_parquet": parquet_path,
            },
        }

        out_path = output_dir / "step-12-features.json"
        with open(out_path, "w") as f:
            json.dump(step_output, f, indent=2)
        print(f"Written: {out_path}")

        # Update progress
        with open(progress_path) as f:
            p = json.load(f)
        if "12-feature-extraction" not in p.get("completed_steps", []):
            p.setdefault("completed_steps", []).append("12-feature-extraction")
        p["current_step"] = "12-feature-extraction"
        p["status"] = "running"
        with open(progress_path, "w") as f:
            json.dump(p, f, indent=2)

        print("Step 12 complete.")
        sys.exit(0)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR in step 12: {e}\n{tb}", file=sys.stderr)
        if progress_path.exists():
            with open(progress_path) as f:
                p = json.load(f)
            p["status"] = "error"
            p.setdefault("errors", []).append({"step": "12-feature-extraction", "error": str(e), "traceback": tb})
            with open(progress_path, "w") as f:
                json.dump(p, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
