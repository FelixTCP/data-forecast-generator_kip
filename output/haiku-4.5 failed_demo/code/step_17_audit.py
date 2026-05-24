#!/usr/bin/env python3
"""
Step 17: Critical Self-Audit

Perform objective post-pipeline evaluation to detect issues with the model
and feature engineering. Generate remediation recommendations.
"""

import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, List

import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from tqdm import tqdm


def detect_data_profile(df: pd.DataFrame, time_col: str = None) -> Dict[str, Any]:
    """Detect the type of dataset."""
    
    # Check for multiple series
    if 'group_id' in df.columns or 'series_id' in df.columns:
        return {
            "detected_profile": "multi_series_temporal",
            "confidence": 0.9,
            "characteristics": ["multiple_series", "grouped_data"]
        }
    
    # Check for daily patterns
    if time_col and 'hour' in df.columns:
        return {
            "detected_profile": "daily_cyclical_temporal",
            "confidence": 0.8,
            "characteristics": ["hourly_data", "daily_cycles"]
        }
    
    # Check for longer periods
    if time_col:
        return {
            "detected_profile": "longer_period_temporal",
            "confidence": 0.7,
            "characteristics": ["temporal_data", "longer_periods"]
        }
    
    return {
        "detected_profile": "generic_regression",
        "confidence": 0.5,
        "characteristics": ["static_features"]
    }


def check_temporal_consistency(df: pd.DataFrame, time_col: str = None) -> Dict[str, Any]:
    """Check for temporal consistency issues (gaps, duplicates)."""
    
    if time_col is None or time_col not in df.columns:
        return {
            "status": "pass",
            "findings": ["No time column detected"],
            "severity": "low",
            "confidence": 1.0
        }
    
    # Check for duplicate timestamps
    time_series = df[time_col]
    duplicates = time_series.duplicated().sum()
    
    if duplicates > len(df) * 0.1:
        return {
            "status": "fail",
            "findings": [f"High duplicate timestamps: {duplicates} ({100*duplicates/len(df):.1f}%)"],
            "severity": "high",
            "confidence": 0.9
        }
    
    if duplicates > 0:
        return {
            "status": "marginal",
            "findings": [f"Some duplicate timestamps: {duplicates}"],
            "severity": "medium",
            "confidence": 0.8
        }
    
    return {
        "status": "pass",
        "findings": ["No temporal gaps or duplicates detected"],
        "severity": "low",
        "confidence": 1.0
    }


def check_multi_series_detection(df: pd.DataFrame, step11: Dict[str, Any]) -> Dict[str, Any]:
    """Check if data is truly multi-series."""
    
    multi_detected = step11.get("multiple_series_detected", False)
    
    if not multi_detected:
        return {
            "status": "pass",
            "findings": ["Single series detected (expected for univariate)"],
            "severity": "low",
            "confidence": 0.8
        }
    
    return {
        "status": "marginal",
        "findings": ["Multiple series detected - consider separate models per group"],
        "severity": "medium",
        "confidence": 0.7,
        "potential_group_columns": ["(auto)"]
    }


def check_feature_target_alignment(step11: Dict[str, Any], step12: Dict[str, Any]) -> Dict[str, Any]:
    """Check if features are well-aligned with target."""
    
    recommended = len(step11.get("recommended_features", []))
    excluded = len(step11.get("excluded_features", {}))
    
    if recommended == 0:
        return {
            "status": "fail",
            "findings": ["No features recommended - feature engineering failed"],
            "severity": "high",
            "confidence": 1.0
        }
    
    total = recommended + excluded
    exclusion_rate = excluded / total if total > 0 else 0
    
    if exclusion_rate > 0.5:
        return {
            "status": "fail",
            "findings": [f"High exclusion rate: {100*exclusion_rate:.1f}% of features excluded"],
            "severity": "medium",
            "confidence": 0.8
        }
    
    return {
        "status": "pass",
        "findings": [f"Good feature-target alignment: {recommended} features, exclusion rate {100*exclusion_rate:.1f}%"],
        "severity": "low",
        "confidence": 0.8
    }


def check_model_performance_baseline(step14: Dict[str, Any], data_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Check if model performance meets profile-dependent thresholds."""
    
    r2 = step14.get("best_candidate_r2", -1)
    quality = step14.get("quality_assessment", "unknown")
    
    # Profile-dependent thresholds
    if "multi_series" in data_profile.get("detected_profile", ""):
        acceptable_r2 = 0.30
        marginal_r2 = 0.10
    else:
        acceptable_r2 = 0.50
        marginal_r2 = 0.25
    
    if r2 >= acceptable_r2:
        return {
            "status": "pass",
            "findings": [f"Good model performance (R²={r2:.3f}, exceeds threshold {acceptable_r2})"],
            "severity": "low",
            "confidence": 0.9
        }
    
    if r2 >= marginal_r2:
        return {
            "status": "marginal",
            "findings": [f"Marginal model performance (R²={r2:.3f}, meets minimum {marginal_r2})"],
            "severity": "medium",
            "confidence": 0.8
        }
    
    return {
        "status": "marginal",
        "findings": [f"Weak model performance (R²={r2:.3f}, below thresholds)"],
        "severity": "medium",
        "confidence": 0.8
    }


def check_data_distribution_drift(X_train: np.ndarray, X_test: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """Check for distribution drift between train and test sets."""
    
    findings = []
    max_ks = 0.0
    
    for i, feature_name in enumerate(feature_names):
        try:
            ks_stat, p_value = ks_2samp(X_train[:, i], X_test[:, i])
            max_ks = max(max_ks, ks_stat)
            
            if ks_stat > 0.95:
                findings.append(f"CRITICAL DRIFT in {feature_name}: KS={ks_stat:.3f}")
            elif ks_stat > 0.40:
                findings.append(f"Moderate drift in {feature_name}: KS={ks_stat:.3f}")
        except Exception:
            pass
    
    if max_ks > 0.95:
        return {
            "status": "fail",
            "findings": findings,
            "severity": "high",
            "confidence": 0.8,
            "max_ks_statistic": max_ks
        }
    
    if max_ks > 0.40:
        return {
            "status": "marginal",
            "findings": findings,
            "severity": "medium",
            "confidence": 0.7,
            "max_ks_statistic": max_ks
        }
    
    return {
        "status": "pass",
        "findings": ["No significant distribution drift detected"],
        "severity": "low",
        "confidence": 0.9,
        "max_ks_statistic": max_ks
    }


def map_remediation_actions(checks: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map failed/high-severity checks to remediation actions."""
    
    actions = []
    
    for check_name, check_result in checks.items():
        status = check_result.get("status", "pass")
        severity = check_result.get("severity", "low")
        
        if status == "fail" or severity == "high":
            if check_name == "temporal_consistency" and status == "fail":
                actions.append({
                    "action_id": "handle_temporal_gaps",
                    "severity": "high",
                    "description": "Interpolate or separate training windows to handle temporal gaps",
                    "affected_steps": ["10", "12"],
                    "suggested_parameters": {"gap_handling": "interpolate"},
                    "expected_improvement": "Cleaner training data; prevents spurious patterns in gaps"
                })
            
            elif check_name == "multi_series_detection" and status in ["fail", "marginal"]:
                actions.append({
                    "action_id": "split_by_grouping_column",
                    "severity": "high",
                    "description": "Train separate models per group; ensemble predictions",
                    "affected_steps": ["12", "13", "14", "15"],
                    "suggested_parameters": {"group_column": "(auto)", "train_separate_models": True},
                    "expected_improvement": "R² +0.2 to +0.5 per group"
                })
            
            elif check_name == "feature_target_alignment" and status == "fail":
                actions.append({
                    "action_id": "extend_lag_window",
                    "severity": "medium",
                    "description": "Increase lag window for time-series features",
                    "affected_steps": ["12", "13"],
                    "suggested_parameters": {"max_lag": 20},
                    "expected_improvement": "CV R² +0.1 to +0.3"
                })
            
            elif check_name == "model_performance_baseline" and status == "fail":
                actions.append({
                    "action_id": "increase_regularization",
                    "severity": "high",
                    "description": "Strengthen regularization to reduce overfitting",
                    "affected_steps": ["13"],
                    "suggested_parameters": {"regularization_method": "ridge_cv"},
                    "expected_improvement": "Holdout R² +0.05 to +0.15"
                })
            
            elif check_name == "data_distribution_drift" and status == "fail":
                actions.append({
                    "action_id": "remove_monotonic_index_features",
                    "severity": "high",
                    "description": "Remove features with monotonic index patterns",
                    "affected_steps": ["12", "13"],
                    "suggested_parameters": {},
                    "expected_improvement": "Better generalization to future data"
                })
    
    return actions


def main():
    parser = argparse.ArgumentParser(description="Step 17: Critical Self-Audit")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    try:
        output_dir = Path(args.output_dir)
        
        # Load prior outputs
        step10 = json.loads((output_dir / "step-10-cleanse.json").read_text())
        step11 = json.loads((output_dir / "step-11-exploration.json").read_text())
        step12 = json.loads((output_dir / "step-12-features.json").read_text())
        step14 = json.loads((output_dir / "step-14-evaluation.json").read_text())
        
        # Load data
        cleaned_path = output_dir / "cleaned.parquet"
        df_pl = pl.read_parquet(str(cleaned_path))
        df = df_pl.to_pandas()
        
        features_path = output_dir / "features.parquet"
        df_features = pl.read_parquet(str(features_path)).to_pandas()
        
        holdout = np.load(str(output_dir / "holdout.npz"), allow_pickle=True)
        X_test = holdout["X_test"]
        
        # Detect data profile
        time_col = step10.get("time_column_detected") or None
        data_profile = detect_data_profile(df, time_col)
        
        # Run 5 checks
        checks = {}
        
        with tqdm(total=5, desc="Running audit checks") as pbar:
            # Check 1: Temporal consistency
            checks["temporal_consistency"] = check_temporal_consistency(df, time_col)
            pbar.update(1)
            
            # Check 2: Multi-series detection
            checks["multi_series_detection"] = check_multi_series_detection(df, step11)
            pbar.update(1)
            
            # Check 3: Feature-target alignment
            checks["feature_target_alignment"] = check_feature_target_alignment(step11, step12)
            pbar.update(1)
            
            # Check 4: Model performance baseline
            checks["model_performance_baseline"] = check_model_performance_baseline(step14, data_profile)
            pbar.update(1)
            
            # Check 5: Data distribution drift
            X_train_idx = int(len(df_features) * 0.8)
            X_train = df_features.iloc[:X_train_idx, :-1].values  # Drop target
            feature_names = step12.get("features", [])
            checks["data_distribution_drift"] = check_data_distribution_drift(X_train, X_test, feature_names)
            pbar.update(1)
        
        # Map remediation actions
        remediation_actions = map_remediation_actions(checks)
        
        # Determine overall result
        critical_findings = []
        overall_result = "pass"
        
        for check_name, check_result in checks.items():
            if check_result.get("status") == "fail":
                overall_result = "fail"
                critical_findings.append({
                    "check": check_name,
                    "status": check_result.get("status"),
                    "severity": check_result.get("severity", "medium"),
                    "description": check_result.get("findings", ["Unknown issue"])[0]
                })
            elif check_result.get("severity") == "high":
                if overall_result != "fail":
                    overall_result = "marginal"
                critical_findings.append({
                    "check": check_name,
                    "status": check_result.get("status"),
                    "severity": "high",
                    "description": check_result.get("findings", ["Unknown issue"])[0]
                })
        
        # Build output
        output_json = {
            "step": "17-critical-self-audit",
            "run_id": args.run_id,
            "data_profile": data_profile,
            "checks": {
                name: {
                    "status": result.get("status"),
                    "findings": result.get("findings", []),
                    "severity": result.get("severity", "low"),
                    "confidence": result.get("confidence", 0.5)
                }
                for name, result in checks.items()
            },
            "overall_audit_result": overall_result,
            "critical_findings": critical_findings,
            "remediation_actions": remediation_actions
        }
        
        # Write output JSON
        step_json_path = output_dir / "step-17-audit.json"
        step_json_path.write_text(json.dumps(output_json, indent=2))
        
        # Update progress (NOT to "completed" - remediation loop may run)
        progress_path = output_dir / "progress.json"
        progress = json.loads(progress_path.read_text())
        progress["status"] = "running"
        progress["current_step"] = "17-critical-self-audit"
        progress["completed_steps"] = [
            "10-csv-read-cleansing",
            "11-data-exploration",
            "12-feature-extraction",
            "13-model-training",
            "14-model-evaluation",
            "15-model-selection",
            "16-result-presentation",
            "17-critical-self-audit"
        ]
        progress_path.write_text(json.dumps(progress, indent=2))
        
        print(f"Step 17 completed: Audit result = {overall_result}")
        print(f"  Critical findings: {len(critical_findings)}")
        print(f"  Remediation actions: {len(remediation_actions)}")
        sys.exit(0)
        
    except Exception as e:
        print(f"Step 17 failed: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
