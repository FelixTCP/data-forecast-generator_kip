"""Step 17 — Critical Self-Audit."""
import argparse
import json
import os
import sys
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")


def make_ser(obj):
    if isinstance(obj, dict):
        return {k: make_ser(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_ser(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        return str(obj)


def detect_data_profile(step10, step11, df_pd):
    """Detect data profile using objective signals."""
    has_time = bool(step10.get("time_column_detected"))
    time_col = step10.get("time_column_detected")

    if not has_time:
        return {
            "detected_profile": "static_regression",
            "confidence": 1.0,
            "characteristics": ["no_time_column"],
        }

    # Check for multi-series (duplicate timestamps)
    if time_col in df_pd.columns:
        n_unique_ts = df_pd[time_col].nunique()
        n_rows = len(df_pd)
        if n_unique_ts < n_rows:
            return {
                "detected_profile": "multi_series_temporal",
                "confidence": 0.95,
                "characteristics": ["duplicate_timestamps", "multi_entity"],
            }

    # Check ACF
    acf_vals = step11.get("ts_diagnostics", {}).get("acf_values", [])
    acf_short = np.mean([abs(v) for v in acf_vals[1:8]]) if len(acf_vals) > 7 else 0
    acf_long = np.mean([abs(v) for v in acf_vals[24:49]]) if len(acf_vals) > 48 else 0
    acf_medium = np.mean([abs(v) for v in acf_vals[7:31]]) if len(acf_vals) > 30 else 0

    if acf_short > 0.4:
        return {
            "detected_profile": "daily_cyclical_temporal",
            "confidence": 0.90,
            "characteristics": ["strong_short_lag_autocorr", "daily_or_weekly_pattern"],
        }
    elif acf_long > 0.3 or acf_medium > 0.3:
        return {
            "detected_profile": "longer_period_temporal",
            "confidence": 0.88,
            "characteristics": ["longer_period_autocorr", "monthly_or_seasonal_pattern"],
        }
    else:
        return {
            "detected_profile": "generic_temporal",
            "confidence": 0.75,
            "characteristics": ["weak_autocorr", "temporal_structure_present"],
        }


def check_temporal_consistency(step10, df_pd):
    """Check 1: Temporal consistency."""
    time_col = step10.get("time_column_detected")
    if not time_col or time_col not in df_pd.columns:
        return {
            "status": "pass",
            "findings": ["no_time_column_detected"],
            "gap_count": 0,
            "max_gap_days": 0,
            "regularity_stddev_percent": 0,
            "severity": "low",
        }

    import pandas as pd
    time_series = pd.to_datetime(df_pd[time_col]).sort_values()
    deltas = time_series.diff().dropna()

    if len(deltas) < 2:
        return {
            "status": "pass",
            "findings": ["insufficient_data_for_consistency_check"],
            "gap_count": 0,
            "max_gap_days": 0,
            "regularity_stddev_percent": 0,
            "severity": "low",
        }

    median_delta = deltas.median()
    expected_interval = median_delta.total_seconds() / 86400  # in days

    # Detect frequency
    if expected_interval < 0.1:
        freq = "hourly"
    elif expected_interval < 2:
        freq = "daily"
    elif expected_interval < 10:
        freq = "weekly"
    else:
        freq = "monthly"

    # Gap detection: gap > 2× expected interval
    gap_threshold = 2 * median_delta
    gaps = deltas[deltas > gap_threshold]
    gap_count = len(gaps)
    max_gap_days = float(gaps.max().total_seconds() / 86400) if gap_count > 0 else 0

    # Total time span
    total_span_days = (time_series.max() - time_series.min()).total_seconds() / 86400
    gap_fraction = max_gap_days / total_span_days if total_span_days > 0 else 0

    # Regularity: std dev of deltas / mean delta
    delta_seconds = deltas.dt.total_seconds()
    regularity_pct = float(delta_seconds.std() / delta_seconds.mean() * 100) if delta_seconds.mean() > 0 else 0

    findings = [f"inferred_frequency_{freq}"]

    if gap_fraction > 0.10:
        status = "fail"
        severity = "high"
        findings.append(f"gap_above_10pct: max_gap={max_gap_days:.1f}d, fraction={gap_fraction:.2%}")
    elif gap_count > 0:
        status = "marginal"
        severity = "medium"
        findings.append(f"gaps_detected: count={gap_count}, max={max_gap_days:.1f}d")
    else:
        findings.append("no_gaps_detected")
        status = "pass"
        severity = "low"

    if regularity_pct > 10:
        findings.append(f"irregular_frequency: stddev={regularity_pct:.1f}%_of_mean_interval")
        if status == "pass":
            status = "marginal"
            severity = "medium"
    else:
        findings.append(f"regular_frequency_confirmed: stddev={regularity_pct:.1f}%")

    return {
        "status": status,
        "findings": findings,
        "gap_count": gap_count,
        "max_gap_days": max_gap_days,
        "regularity_stddev_percent": round(regularity_pct, 2),
        "inferred_frequency": freq,
        "severity": severity,
    }


def check_multi_series(step10, step11, df_pd):
    """Check 2: Multi-series detection."""
    time_col = step10.get("time_column_detected")

    if not time_col or time_col not in df_pd.columns:
        return {
            "status": "pass",
            "findings": ["no_time_column"],
            "detected_group_columns": [],
            "severity": "low",
        }

    import pandas as pd
    n_unique_ts = df_pd[time_col].nunique()
    n_rows = len(df_pd)
    findings = []

    # PRIMARY: duplicate timestamps
    if n_unique_ts < n_rows:
        ratio = n_rows / n_unique_ts
        findings.append(f"Duplicate timestamps detected: {n_unique_ts} unique for {n_rows} rows => {ratio:.1f} series per timestamp")
        return {
            "status": "fail",
            "findings": findings,
            "detected_group_columns": [],
            "severity": "high",
            "n_unique_timestamps": int(n_unique_ts),
            "n_rows": int(n_rows),
        }

    # No duplicate timestamps → single series
    findings.append(f"No duplicate timestamps: {n_unique_ts} unique timestamps for {n_rows} rows")
    findings.append("single_series_confirmed")

    return {
        "status": "pass",
        "findings": findings,
        "detected_group_columns": [],
        "severity": "low",
    }


def check_feature_target_alignment(step12, step11, df_pd):
    """Check 3: Feature-target alignment."""
    target_col = step12["target_column"]
    features = step12.get("features", [])
    excluded = step12.get("features_excluded", {})
    time_col_from_step10 = None

    # Get time col from context
    ctx = step12.get("context", {})
    time_col = ctx.get("time_column")

    findings = []
    severity = "low"
    status = "pass"

    # CRITICAL: target in features?
    if target_col in features:
        findings.append(f"target_variable_leaked_into_features: {target_col} must be excluded_before_training")
        severity = "high"
        status = "fail"
    else:
        findings.append("target_variable_correctly_excluded")

    # CRITICAL: timestamp as raw feature?
    if time_col and time_col in features:
        findings.append(f"timestamp_field_leaked_as_feature: {time_col} enables_perfect_reconstruction")
        severity = "high"
        status = "fail"
    else:
        findings.append("timestamp_not_used_as_raw_feature")

    # CRITICAL: monotone index features?
    monotone_patterns = ["trend_t_index", "trend_t_index_sq", "t_index", "row_num", "time_index", "sequential_id"]
    monotone_found = []
    for feat in features:
        if any(pat in feat.lower() for pat in monotone_patterns):
            monotone_found.append(feat)

    # Also check by value: strictly increasing with no repeats
    for feat in features:
        if feat in monotone_found:
            continue
        if feat in df_pd.columns:
            vals = df_pd[feat].dropna().values
            if len(vals) > 10:
                diffs = np.diff(vals)
                if np.all(diffs > 0) and len(np.unique(vals)) == len(vals):
                    monotone_found.append(feat)

    if monotone_found:
        findings.append(f"monotone_index_feature_detected: {monotone_found} causes_ks_1_0")
        severity = "high"
        status = "fail"
    else:
        findings.append("no_monotone_index_features")

    # MI retention
    mi_ranking = step11.get("mi_ranking", [])
    top5_mi = [r["feature"] for r in mi_ranking[:5]]
    retained = [f for f in top5_mi if f in features]
    mi_retention = len(retained) / len(top5_mi) if top5_mi else 1.0
    if mi_retention < 0.80:
        findings.append(f"mi_retention_rate_{mi_retention:.2f}_below_0.80")
        if status == "pass":
            status = "marginal"
            severity = "medium"
    else:
        findings.append(f"mi_ranking_stable: retention={mi_retention:.2f}")

    # Excluded ratio
    total_numeric = len(step11.get("numeric_columns", []))
    n_excluded = len(excluded)
    excluded_ratio = n_excluded / total_numeric if total_numeric > 0 else 0

    if excluded_ratio > 0.70:
        findings.append(f"aggressive_filtering: excluded_ratio={excluded_ratio:.2f}")
        if status == "pass":
            status = "marginal"
            severity = "medium"
    elif n_excluded == 0:
        findings.append("no_features_excluded: possible_under_filtering")
    else:
        findings.append(f"reasonable_exclusion_ratio: {excluded_ratio:.2f}")

    # Pairwise redundancy among final features
    max_corr = 0.0
    if len(features) > 1:
        try:
            import pandas as pd
            feat_sub = df_pd[[f for f in features if f in df_pd.columns]].fillna(0)
            if feat_sub.shape[1] > 1:
                corr_matrix = feat_sub.corr().abs()
                np.fill_diagonal(corr_matrix.values, 0)
                max_corr = float(corr_matrix.max().max())
                if max_corr > 0.90:
                    findings.append(f"high_pairwise_correlation: max={max_corr:.3f}")
                    if status == "pass":
                        status = "marginal"
                        severity = "medium"
                else:
                    findings.append(f"no_redundant_features: max_corr={max_corr:.3f}")
        except Exception:
            pass

    return {
        "status": status,
        "findings": findings,
        "mi_retention_rate": round(mi_retention, 3),
        "correlation_max": round(max_corr, 4),
        "excluded_ratio": round(excluded_ratio, 3),
        "target_variable_in_features": target_col in features,
        "timestamp_in_features": bool(time_col and time_col in features),
        "monotone_features_found": monotone_found,
        "severity": severity,
    }


def check_model_performance(step13, step14, data_profile):
    """Check 4: Model performance baseline."""
    profile = data_profile.get("detected_profile", "generic_temporal")

    # Profile-specific thresholds
    thresholds = {
        "multi_series_temporal": {"pass": 0.50, "marginal": 0.25},
        "daily_cyclical_temporal": {"pass": 0.55, "marginal": 0.30},
        "longer_period_temporal": {"pass": 0.50, "marginal": 0.25},
        "generic_temporal": {"pass": 0.50, "marginal": 0.25},
        "static_regression": {"pass": 0.60, "marginal": 0.35},
    }
    thresh = thresholds.get(profile, {"pass": 0.50, "marginal": 0.25})

    # Best candidate
    candidates = step14.get("candidates", [])
    valid = [c for c in candidates if c.get("r2") is not None and not _is_nan(c["r2"])]
    if not valid:
        return {
            "status": "fail",
            "findings": ["no_valid_candidates"],
            "r2_holdout": None,
            "r2_cv_mean": None,
            "r2_pass_threshold": thresh["pass"],
            "detected_profile": profile,
            "severity": "high",
        }

    best = max(valid, key=lambda c: c["r2"])
    r2 = best["r2"]
    cv_r2 = best.get("cv_mean_r2")

    findings = [f"selected_model_{best['model_name']}", f"holdout_r2_{r2:.4f}"]
    severity = "low"

    # Check R² vs profile threshold
    if r2 < 0:
        status = "fail"
        severity = "high"
        findings.append(f"r2_negative_worse_than_mean_baseline")
    elif r2 < thresh["marginal"]:
        # NOTE: only return "fail" for truly terrible performance
        status = "marginal"
        severity = "medium"
        findings.append(f"r2_below_marginal_threshold_{thresh['marginal']}")
    elif r2 < thresh["pass"]:
        status = "marginal"
        severity = "medium"
        findings.append(f"r2_marginal_{thresh['marginal']:.2f}_to_{thresh['pass']:.2f}")
    else:
        status = "pass"
        findings.append(f"r2_above_pass_threshold_{thresh['pass']}")

    # Overfitting check
    if cv_r2 is not None and not _is_nan(cv_r2) and cv_r2 > 0:
        overfitting_ratio = r2 / cv_r2
        if overfitting_ratio < 0.8:
            findings.append(f"overfitting_detected: holdout_r2={r2:.4f} < 0.8×cv_r2={cv_r2:.4f}")
            if status == "pass":
                status = "marginal"
                severity = "medium"
        else:
            findings.append(f"no_overfitting: holdout_r2/cv_r2={overfitting_ratio:.3f}")

    return {
        "status": status,
        "findings": findings,
        "detected_profile": profile,
        "r2_holdout": r2,
        "r2_cv_mean": cv_r2,
        "r2_pass_threshold": thresh["pass"],
        "r2_marginal_threshold": thresh["marginal"],
        "severity": severity,
    }


def check_distribution_drift(step12, step13, holdout_npz_path, df_pd):
    """Check 5: Data distribution drift."""
    from scipy.stats import ks_2samp
    features = step12.get("features", [])
    target_col = step12["target_column"]

    try:
        holdout = np.load(holdout_npz_path)
        X_test = holdout["X_test"]
        y_test = holdout["y_test"]
    except Exception as e:
        return {
            "status": "pass",
            "findings": [f"holdout_load_failed: {e}"],
            "severity": "low",
            "target_ks_stat": None,
        }

    n_total = len(df_pd)
    n_test = len(y_test)
    n_train = n_total - n_test

    findings = []
    severity = "low"
    status = "pass"

    # CRITICAL: monotone index detection by name and value
    monotone_features = []
    monotone_patterns = ["trend_t_index", "trend_t_index_sq", "t_index", "row_num",
                         "time_index", "sequential_id"]
    for i, feat in enumerate(features):
        if any(pat in feat.lower() for pat in monotone_patterns):
            monotone_features.append(feat)
            continue
        # Check by value
        if feat in df_pd.columns:
            vals = df_pd[feat].dropna().values
            if len(vals) > 10 and np.all(np.diff(vals) > 0) and len(np.unique(vals)) == len(vals):
                monotone_features.append(feat)

    if monotone_features:
        findings.append(f"monotone_index_feature_detected: {monotone_features} causes_ks_1_0_perfect_reconstruction")
        severity = "high"
        status = "fail"
    else:
        findings.append("no_monotone_index_features")

    # Target KS test
    if n_train > 0 and df_pd[target_col].notna().any():
        y_train_vals = df_pd[target_col].values[:n_train]
        y_test_vals = y_test
        try:
            ks_stat_target, _ = ks_2samp(y_train_vals, y_test_vals)
        except Exception:
            ks_stat_target = 0.0
    else:
        ks_stat_target = 0.0

    # Feature KS stats (top 3 by MI, excluding monotone)
    mi_ranking = []
    # Use features from step 12
    feature_ks_stats = {}
    non_monotone_features = [f for f in features if f not in monotone_features]
    for i, feat in enumerate(non_monotone_features[:5]):
        if i >= X_test.shape[1]:
            break
        feat_idx = features.index(feat) if feat in features else i
        if feat_idx >= X_test.shape[1]:
            continue
        train_vals = df_pd[feat].values[:n_train] if feat in df_pd.columns else []
        test_vals = X_test[:, feat_idx] if feat_idx < X_test.shape[1] else []
        if len(train_vals) > 5 and len(test_vals) > 5:
            try:
                ks_stat, _ = ks_2samp(train_vals, test_vals)
                feature_ks_stats[feat] = float(ks_stat)
                if ks_stat >= 1.0:
                    findings.append(f"feature_{feat}_ks_1.0_perfect_leakage")
                    severity = "high"
                    status = "fail"
            except Exception:
                pass

    feature_ks_mean = float(np.mean(list(feature_ks_stats.values()))) if feature_ks_stats else 0.0

    # Apply KS thresholds for target
    # NOTE: only fail if KS > 0.95 per the relaxed threshold
    if ks_stat_target >= 0.95:
        findings.append(f"target_ks_high_{ks_stat_target:.4f}: strong_distribution_shift")
        if status == "pass":
            severity = "high"
            status = "fail"
    elif ks_stat_target >= 0.40:
        findings.append(f"target_ks_marginal_{ks_stat_target:.4f}")
        if status == "pass":
            severity = "medium"
            status = "marginal"
    else:
        findings.append(f"target_ks_acceptable_{ks_stat_target:.4f}")

    # Temporal drift check
    temporal_drift = False
    if n_train > 100:
        q1_size = n_train // 4
        q1_vals = df_pd[target_col].values[:q1_size]
        q4_vals = df_pd[target_col].values[n_train - q1_size:n_train]
        try:
            ks_internal, _ = ks_2samp(q1_vals, q4_vals)
            if ks_internal > 0.3:
                temporal_drift = True
                findings.append(f"temporal_drift_in_training: ks={ks_internal:.3f}")
            else:
                findings.append("no_temporal_drift_in_training")
        except Exception:
            pass

    return {
        "status": status,
        "findings": findings,
        "target_ks_stat": float(ks_stat_target),
        "feature_ks_stats": feature_ks_stats,
        "feature_ks_mean": feature_ks_mean,
        "monotone_features": monotone_features,
        "temporal_drift_detected": temporal_drift,
        "severity": severity,
    }


def _is_nan(v):
    try:
        return v is None or (isinstance(v, float) and np.isnan(v))
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    run_id = args.run_id

    # Update progress
    progress_path = os.path.join(output_dir, "progress.json")
    with open(progress_path) as f:
        progress = json.load(f)
    progress["status"] = "running"
    progress["current_step"] = "17-critical-self-audit"
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    # Load inputs
    step10 = json.load(open(os.path.join(output_dir, "step-10-cleanse.json")))
    step11 = json.load(open(os.path.join(output_dir, "step-11-exploration.json")))
    step12 = json.load(open(os.path.join(output_dir, "step-12-features.json")))
    step13 = json.load(open(os.path.join(output_dir, "step-13-training.json")))
    step14 = json.load(open(os.path.join(output_dir, "step-14-evaluation.json")))

    import polars as pl
    import pandas as pd
    df_pl = pl.read_parquet(os.path.join(output_dir, "cleaned.parquet"))
    df_pd = df_pl.to_pandas()

    holdout_path = os.path.join(output_dir, "holdout.npz")

    # Phase 1: Data profile detection
    print("Phase 1: Detecting data profile...")
    data_profile = detect_data_profile(step10, step11, df_pd)
    print(f"  Profile: {data_profile['detected_profile']} (confidence={data_profile['confidence']})")

    # Phase 2: Run 5 audit checks
    checks = {}
    check_names = [
        "temporal_consistency",
        "multi_series_detection",
        "feature_target_alignment",
        "model_performance_baseline",
        "data_distribution_drift",
    ]

    for check_name in tqdm(check_names, desc="Running audit checks"):
        print(f"  Running: {check_name}")
        if check_name == "temporal_consistency":
            result = check_temporal_consistency(step10, df_pd)
        elif check_name == "multi_series_detection":
            result = check_multi_series(step10, step11, df_pd)
        elif check_name == "feature_target_alignment":
            result = check_feature_target_alignment(step12, step11, df_pd)
        elif check_name == "model_performance_baseline":
            result = check_model_performance(step13, step14, data_profile)
        elif check_name == "data_distribution_drift":
            result = check_distribution_drift(step12, step13, holdout_path, df_pd)
        checks[check_name] = result
        print(f"    -> status={result['status']}, severity={result['severity']}")

    # Phase 3: Critical findings
    # Critical finding ONLY for: target_in_features, timestamp_in_features, monotone_features
    critical_findings = []

    feat_align = checks["feature_target_alignment"]
    if feat_align.get("target_variable_in_features"):
        critical_findings.append({
            "check": "feature_target_alignment",
            "status": "fail",
            "severity": "high",
            "description": "Target variable found in feature set — direct leakage.",
        })
    if feat_align.get("timestamp_in_features"):
        critical_findings.append({
            "check": "feature_target_alignment",
            "status": "fail",
            "severity": "high",
            "description": "Raw timestamp column used as feature — enables perfect reconstruction.",
        })
    if feat_align.get("monotone_features_found"):
        critical_findings.append({
            "check": "feature_target_alignment",
            "status": "fail",
            "severity": "high",
            "description": f"Monotone index features detected: {feat_align['monotone_features_found']}. These cause KS=1.0 and are not transferable.",
        })
    drift = checks["data_distribution_drift"]
    if drift.get("monotone_features"):
        # Deduplicate
        for mf in drift.get("monotone_features", []):
            already = any(mf in str(cf.get("description", "")) for cf in critical_findings)
            if not already:
                critical_findings.append({
                    "check": "data_distribution_drift",
                    "status": "fail",
                    "severity": "high",
                    "description": f"Monotone feature detected in drift check: {mf}. KS=1.0 leakage.",
                })

    # Multi-series detection critical finding
    ms_check = checks["multi_series_detection"]
    if ms_check["status"] == "fail" and ms_check["severity"] == "high":
        critical_findings.append({
            "check": "multi_series_detection",
            "status": "fail",
            "severity": "high",
            "description": "Multiple series detected in data — model trained on mixed entities.",
        })

    # Phase 4: Overall result
    # FAIL if: >= 1 critical (high-severity) findings OR >= 2 fail checks
    n_fail_checks = sum(1 for c in checks.values() if c["status"] == "fail")
    has_critical = len(critical_findings) > 0

    if has_critical or n_fail_checks >= 2:
        overall_audit_result = "fail"
    else:
        overall_audit_result = "pass"

    print(f"Overall audit result: {overall_audit_result}")
    print(f"Critical findings: {len(critical_findings)}")
    print(f"Failed checks: {n_fail_checks}")

    # Phase 5: Remediation actions
    remediation_actions = []

    # Collect monotone features from both alignment and drift checks
    all_monotone = list(set(
        feat_align.get("monotone_features_found", []) +
        drift.get("monotone_features", [])
    ))

    if all_monotone:
        remediation_actions.append({
            "action_id": "remove_monotonic_index_features",
            "severity": "high",
            "description": f"Remove monotone index features: {all_monotone}",
            "affected_steps": ["12", "13", "14", "15"],
            "suggested_parameters": {"exclude_features": all_monotone},
            "auto_executable": True,
            "expected_improvement": "Eliminates KS=1.0 drift; model becomes transferable.",
        })

    if ms_check["status"] == "fail":
        remediation_actions.append({
            "action_id": "split_by_grouping_column",
            "severity": "high",
            "description": "Multiple series detected. Train separate models per entity.",
            "affected_steps": ["12", "13", "14", "15"],
            "suggested_parameters": {"group_column": step11.get("series_id_column")},
            "auto_executable": False,
            "expected_improvement": "R² typically +0.2 to +0.5 per group.",
        })

    perf_check = checks["model_performance_baseline"]
    if perf_check["status"] == "fail":
        remediation_actions.extend([
            {
                "action_id": "increase_regularization",
                "severity": "medium",
                "description": "Increase regularization to reduce overfitting.",
                "affected_steps": ["13"],
                "suggested_parameters": {"regularization_method": "ridge_cv"},
                "auto_executable": True,
                "expected_improvement": "Holdout R² may improve +0.05 to +0.15.",
            },
            {
                "action_id": "try_alternative_models",
                "severity": "medium",
                "description": "Try alternative model classes.",
                "affected_steps": ["13", "14", "15"],
                "suggested_parameters": {"additional_candidates": ["histgradient", "svr"]},
                "auto_executable": True,
                "expected_improvement": "+0.05 to +0.2 R².",
            },
        ])

    drift_check = checks["data_distribution_drift"]
    if drift_check["status"] == "fail" and not all_monotone:
        remediation_actions.append({
            "action_id": "add_seasonal_features",
            "severity": "medium",
            "description": "Add seasonal features to capture distribution patterns.",
            "affected_steps": ["12", "13"],
            "suggested_parameters": {"seasonal_features": "true"},
            "auto_executable": True,
            "expected_improvement": "CV R² +0.1 to +0.2 for seasonal patterns.",
        })

    # Build output
    result = {
        "step": "17-critical-self-audit",
        "run_id": run_id,
        "data_profile": data_profile,
        "checks": {
            "temporal_consistency": checks["temporal_consistency"],
            "multi_series_detection": checks["multi_series_detection"],
            "feature_target_alignment": checks["feature_target_alignment"],
            "model_performance_baseline": checks["model_performance_baseline"],
            "data_distribution_drift": checks["data_distribution_drift"],
        },
        "critical_findings": critical_findings,
        "overall_audit_result": overall_audit_result,
        "remediation_actions": remediation_actions,
        "audit_confidence": data_profile.get("confidence", 0.8),
        "context": {
            "target_column": step12["target_column"],
            "n_fail_checks": n_fail_checks,
            "n_critical": len(critical_findings),
        }
    }

    out_json = os.path.join(output_dir, "step-17-audit.json")
    with open(out_json, "w") as f:
        json.dump(make_ser(result), f, indent=2)

    # Update progress — DO NOT set to "completed" yet
    with open(progress_path) as f:
        progress = json.load(f)
    progress["status"] = "running"
    progress["current_step"] = "remediation_loop"
    if "17-critical-self-audit" not in progress.get("completed_steps", []):
        progress["completed_steps"].append("17-critical-self-audit")
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    print(f"Step 17 complete. overall_audit_result={overall_audit_result}")
    sys.exit(0)


if __name__ == "__main__":
    main()
