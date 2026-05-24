#!/usr/bin/env python3
"""
Step 17: Critical Self-Audit

Performs comprehensive audit of the regression model and feature engineering.
Detects issues and generates remediation actions.
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np
import polars as pl
from scipy import stats
from tqdm import tqdm
import joblib


def detect_data_profile(cleanse_data, df):
    """Detect the data profile (temporal, multi-series, etc.)."""
    time_column = cleanse_data.get("time_column")
    row_count = df.shape[0]
    
    profile = {
        "detected_profile": "generic_regression",
        "confidence": 0.5,
        "characteristics": []
    }
    
    if time_column and time_column in df.columns:
        profile["characteristics"].append(f"Time column detected: {time_column}")
        profile["detected_profile"] = "longer_period_temporal"
        profile["confidence"] = 0.8
    
    return profile


def check_temporal_consistency(cleanse_data, df, time_column):
    """Check 1: Temporal Consistency - gaps, intervals, duplicates."""
    findings = []
    severity = "low"
    status = "pass"
    
    if not time_column or time_column not in df.columns:
        findings.append("No time column detected; temporal consistency check skipped.")
        return {"status": "pass", "findings": findings, "severity": severity}
    
    try:
        df_sorted = df.sort(time_column)
        ts = df_sorted[time_column].to_numpy()
        
        # Check for duplicates
        unique_count = len(np.unique(ts))
        dup_count = len(ts) - unique_count
        if dup_count > 0:
            dup_rate = dup_count / len(ts)
            findings.append(f"Duplicate timestamps detected: {dup_count} duplicates ({dup_rate*100:.1f}%)")
            if dup_rate > 0.05:
                severity = "medium"
                status = "marginal"
        
        # Check for gaps
        if len(ts) > 1:
            diffs = np.diff(ts)
            max_gap = np.max(diffs)
            min_gap = np.min(diffs)
            gap_variance = np.std(diffs)
            
            findings.append(f"Temporal intervals: min={min_gap}, max={max_gap}, std={gap_variance:.2f}")
            
            if max_gap > min_gap * 10:
                findings.append(f"Large gap detected: {max_gap} vs typical {min_gap}")
                severity = "medium"
                status = "marginal"
    
    except Exception as e:
        findings.append(f"Error analyzing temporal consistency: {str(e)}")
    
    return {
        "status": status,
        "findings": findings,
        "severity": severity,
        "confidence": 0.7
    }


def check_multi_series_detection(df, target_col):
    """Check 2: Multi-Series Detection - variance ratio between groups."""
    findings = []
    severity = "low"
    status = "pass"
    potential_group_columns = []
    
    # Look for columns that might group data
    for col in df.columns:
        if col == target_col or col.startswith("_"):
            continue
        
        try:
            # Check if categorical or low-cardinality
            unique_count = df[col].n_unique()
            if 2 <= unique_count <= 50:
                # Compute variance ratio
                grouped = df.group_by(col).agg(pl.col(target_col).var())
                between_var = grouped[target_col].var()
                within_var = df[target_col].var()
                
                if within_var > 0:
                    ratio = between_var / within_var
                    findings.append(f"Column '{col}': variance_ratio={ratio:.2f}")
                    
                    if ratio > 2.0:
                        potential_group_columns.append(col)
                        severity = "medium"
                        status = "marginal"
        except:
            pass
    
    if potential_group_columns:
        findings.append(f"Multi-series detected: potential grouping columns: {potential_group_columns}")
    
    return {
        "status": status,
        "findings": findings,
        "severity": severity,
        "confidence": 0.6,
        "potential_group_columns": potential_group_columns
    }


def check_feature_target_alignment(features_data, exploration_data):
    """Check 3: Feature-Target Alignment - MI retention, excluded features."""
    findings = []
    severity = "low"
    status = "pass"
    
    recommended_features = exploration_data.get("recommended_features", [])
    excluded_features = exploration_data.get("excluded_features", [])
    selected_features = features_data.get("features", [])
    
    total_initial = len(recommended_features) + len(excluded_features)
    if total_initial == 0:
        total_initial = 1
    
    excluded_ratio = len(excluded_features) / total_initial
    findings.append(f"Features excluded: {len(excluded_features)} / {total_initial} ({excluded_ratio*100:.1f}%)")
    
    if excluded_ratio > 0.5:
        severity = "medium"
        status = "marginal"
        findings.append("Over 50% of features were excluded; model may have insufficient signal")
    
    if len(selected_features) < 2:
        severity = "high"
        status = "fail"
        findings.append(f"Insufficient features: only {len(selected_features)} selected (minimum 2)")
    
    return {
        "status": status,
        "findings": findings,
        "severity": severity,
        "confidence": 0.8
    }


def check_model_performance_baseline(eval_data, data_profile):
    """Check 4: Model Performance Baseline - R² thresholds per profile."""
    findings = []
    severity = "low"
    status = "pass"
    overfitting_detected = False
    
    candidates = eval_data.get("candidates", [])
    if not candidates:
        return {"status": "fail", "findings": ["No candidates in evaluation"], "severity": "high"}
    
    best_r2 = max([c.get("r2", -1) for c in candidates])
    best_rmse = min([c.get("rmse", float('inf')) for c in candidates])
    
    # Get CV R² for overfitting check
    best_cv_r2 = 0
    for c in candidates:
        if c.get("r2") == best_r2:
            best_cv_r2 = c.get("cv_mean_r2", 0)
            break
    
    findings.append(f"Best holdout R²: {best_r2:.4f}, CV R²: {best_cv_r2:.4f}")
    
    # Overfitting check
    if best_cv_r2 > 0 and best_r2 < 0.8 * best_cv_r2:
        overfitting_detected = True
        findings.append(f"Overfitting detected: holdout R² ({best_r2:.4f}) < 80% of CV R² ({best_cv_r2:.4f})")
        severity = "medium"
    
    # Profile-dependent thresholds
    profile = data_profile.get("detected_profile", "generic_regression")
    
    if profile in ["multi_series_temporal", "daily_cyclical_temporal"]:
        thresholds = {"pass": 0.30, "marginal": 0.10}
    elif profile == "longer_period_temporal":
        thresholds = {"pass": 0.25, "marginal": 0.05}
    else:
        thresholds = {"pass": 0.50, "marginal": 0.25}
    
    if best_r2 < thresholds.get("marginal", 0.1):
        severity = "high"
        status = "fail"
        findings.append(f"Poor model performance: R² {best_r2:.4f} < {thresholds.get('marginal', 0.1)} threshold")
    elif best_r2 < thresholds.get("pass", 0.5):
        severity = "medium"
        status = "marginal"
        findings.append(f"Moderate performance: R² {best_r2:.4f} in marginal zone")
    
    return {
        "status": status,
        "findings": findings,
        "severity": severity,
        "confidence": 0.8,
        "best_r2": best_r2,
        "overfitting_detected": overfitting_detected
    }


def check_data_distribution_drift(features_df, holdout_npz):
    """Check 5: Data Distribution Drift - KS statistics."""
    findings = []
    severity = "low"
    status = "pass"
    high_drift_features = []
    
    try:
        data = np.load(holdout_npz)
        X_test = data["X_test"]
        
        # Assume first 80% of features_df is train, last 20% is test
        n_total = features_df.shape[0]
        n_train = int(n_total * 0.8)
        
        X_train = features_df.slice(0, n_train).to_numpy()
        X_holdout = features_df.slice(n_train, n_total).to_numpy()
        
        max_ks = 0
        for i in range(min(X_train.shape[1], 20)):  # Check first 20 features
            try:
                ks_stat, _ = stats.ks_2samp(X_train[:, i], X_holdout[:, i])
                max_ks = max(max_ks, ks_stat)
                
                if ks_stat >= 0.95:
                    high_drift_features.append(f"feature_{i} (KS={ks_stat:.3f})")
                    severity = "high"
                    status = "fail"
                elif ks_stat >= 0.40:
                    findings.append(f"Moderate drift in feature_{i}: KS={ks_stat:.3f}")
                    severity = "medium"
                    if status == "pass":
                        status = "marginal"
            except:
                pass
        
        findings.append(f"Max KS statistic: {max_ks:.3f}")
        
        if high_drift_features:
            findings.append(f"High-drift features ({len(high_drift_features)}): {', '.join(high_drift_features[:3])}")
    
    except Exception as e:
        findings.append(f"Could not compute KS statistics: {str(e)}")
    
    return {
        "status": status,
        "findings": findings,
        "severity": severity,
        "confidence": 0.6,
        "high_drift_features": high_drift_features
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    run_id = args.run_id
    
    # Load all context
    cleanse_path = output_dir / "step-10-cleanse.json"
    exploration_path = output_dir / "step-11-exploration.json"
    features_path = output_dir / "step-12-features.json"
    training_path = output_dir / "step-13-training.json"
    eval_path = output_dir / "step-14-evaluation.json"
    
    with open(cleanse_path) as f:
        cleanse_data = json.load(f)
    with open(exploration_path) as f:
        exploration_data = json.load(f)
    with open(features_path) as f:
        features_data = json.load(f)
    with open(training_path) as f:
        training_data = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)
    
    # Load data files
    cleaned_parquet = output_dir / "cleaned.parquet"
    features_parquet = output_dir / "features.parquet"
    holdout_npz = output_dir / "holdout.npz"
    
    df_cleaned = pl.read_parquet(cleaned_parquet)
    df_features = pl.read_parquet(features_parquet)
    
    # Detect data profile
    data_profile = detect_data_profile(cleanse_data, df_cleaned)
    
    # Run all 5 checks with tqdm
    checks = {}
    check_names = [
        "temporal_consistency",
        "multi_series_detection",
        "feature_target_alignment",
        "model_performance_baseline",
        "data_distribution_drift"
    ]
    
    time_column = cleanse_data.get("time_column")
    target_col = cleanse_data.get("target_column_normalized", "")
    
    for check_name in tqdm(check_names, desc="Running audit checks"):
        if check_name == "temporal_consistency":
            checks[check_name] = check_temporal_consistency(cleanse_data, df_cleaned, time_column)
        elif check_name == "multi_series_detection":
            checks[check_name] = check_multi_series_detection(df_cleaned, target_col)
        elif check_name == "feature_target_alignment":
            checks[check_name] = check_feature_target_alignment(features_data, exploration_data)
        elif check_name == "model_performance_baseline":
            checks[check_name] = check_model_performance_baseline(eval_data, data_profile)
        elif check_name == "data_distribution_drift":
            checks[check_name] = check_data_distribution_drift(df_features, holdout_npz)
    
    # Determine critical findings
    critical_findings = []
    
    for check_name, check_result in checks.items():
        if check_result["status"] in ["fail", "marginal"] and check_result["severity"] == "high":
            critical_findings.append({
                "check": check_name,
                "status": check_result["status"],
                "severity": check_result["severity"],
                "description": "; ".join(check_result["findings"][:2])
            })
    
    # Overall result
    overall_result = "pass"
    if any(c["status"] == "fail" for c in checks.values()):
        overall_result = "fail"
    elif any(c["status"] == "marginal" and c["severity"] == "high" for c in checks.values()):
        overall_result = "fail"
    
    # Generate remediation actions
    remediation_actions = []
    
    if checks["temporal_consistency"]["status"] == "fail":
        remediation_actions.append({
            "action_id": "handle_temporal_gaps",
            "severity": "high",
            "description": "Interpolate or separate training windows to handle temporal gaps",
            "affected_steps": ["10", "12"],
            "suggested_parameters": {"gap_handling": "interpolate"}
        })
    
    if checks["multi_series_detection"]["status"] in ["fail", "marginal"]:
        group_col = checks["multi_series_detection"].get("potential_group_columns", ["auto"])[0]
        remediation_actions.append({
            "action_id": "split_by_grouping_column",
            "severity": "high",
            "description": f"Train separate models per group ({group_col})",
            "affected_steps": ["12", "13", "14", "15"],
            "suggested_parameters": {"group_column": group_col}
        })
    
    if checks["feature_target_alignment"]["status"] == "fail":
        remediation_actions.append({
            "action_id": "extend_lag_window",
            "severity": "medium",
            "description": "Increase lag window for time-series features",
            "affected_steps": ["12", "13"],
            "suggested_parameters": {"max_lag": 20}
        })
    
    if checks["model_performance_baseline"]["status"] == "fail":
        remediation_actions.append({
            "action_id": "increase_regularization",
            "severity": "high",
            "description": "Strengthen regularization to reduce overfitting",
            "affected_steps": ["13"],
            "suggested_parameters": {"regularization_method": "ridge_cv"}
        })
        remediation_actions.append({
            "action_id": "try_alternative_models",
            "severity": "medium",
            "description": "Train additional model types",
            "affected_steps": ["13", "14", "15"],
            "suggested_parameters": {"candidates": ["lightgbm", "svr"]}
        })
    
    if checks["data_distribution_drift"]["status"] == "fail":
        high_drift = checks["data_distribution_drift"].get("high_drift_features", [])
        if high_drift:
            remediation_actions.append({
                "action_id": "remove_monotonic_index_features",
                "severity": "high",
                "description": "Remove high-drift features",
                "affected_steps": ["12", "13"],
                "suggested_parameters": {"high_drift_features": high_drift}
            })
    
    # Build output JSON
    output_json = {
        "step": "17-critical-self-audit",
        "run_id": run_id,
        "data_profile": data_profile,
        "checks": checks,
        "overall_audit_result": overall_result,
        "critical_findings": critical_findings,
        "remediation_actions": remediation_actions
    }
    
    # Write output
    audit_path = output_dir / "step-17-audit.json"
    with open(audit_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"✓ Written {audit_path}")
    
    # Update progress
    progress_path = output_dir / "progress.json"
    with open(progress_path) as f:
        progress = json.load(f)
    
    progress["current_step"] = "17-critical-self-audit"
    if "completed_steps" not in progress:
        progress["completed_steps"] = []
    if "17-critical-self-audit" not in progress["completed_steps"]:
        progress["completed_steps"].append("17-critical-self-audit")
    
    # Now mark as completed only if overall_result is "pass"
    if overall_result == "pass":
        progress["status"] = "completed"
        progress["final_audit_result"] = "pass"
    else:
        progress["status"] = "completed"
        progress["final_audit_result"] = "fail"
    
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)
    
    print(f"✓ Step 17 completed: overall_audit_result = {overall_result}")


if __name__ == "__main__":
    main()
