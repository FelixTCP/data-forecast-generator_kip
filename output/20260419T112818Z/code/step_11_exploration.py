#!/usr/bin/env python3
"""
Step 11: Data Exploration
Analyzes data quality, computes MI ranking, detects redundant features,
analyzes lags for time-series data, and produces recommended features list.
"""

import json
import argparse
from pathlib import Path
import polars as pl
import numpy as np
import sys
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf


def run_step_11(output_dir: str) -> dict:
    """
    Explore data: MI ranking, redundancy, variance, lag analysis.
    
    Returns step output dict.
    """
    output_path = Path(output_dir)
    
    # Load step 10 outputs
    step_10_path = output_path / "step-10-cleanse.json"
    if not step_10_path.exists():
        sys.exit("Step 10 output not found")
    
    with open(step_10_path) as f:
        step_10 = json.load(f)
    
    cleaned_parquet = output_path / "cleaned.parquet"
    if not cleaned_parquet.exists():
        sys.exit(f"Cleaned parquet not found: {cleaned_parquet}")
    
    df = pl.read_parquet(cleaned_parquet)
    
    target_column = step_10["target_column_normalized"]
    time_column = step_10["time_column_detected"]
    
    # Get numeric columns (exclude target for now)
    numeric_cols = [
        col for col in df.columns
        if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
    ]
    
    # Try to coerce String columns that might be numeric
    for col in df.columns:
        if df[col].dtype == pl.String and col not in numeric_cols and col != time_column:
            try:
                # Try to convert to float
                df = df.with_columns(pl.col(col).str.to_float().alias(col))
                if col not in numeric_cols:
                    numeric_cols.append(col)
            except:
                pass
    
    if target_column not in numeric_cols:
        numeric_cols.append(target_column)
    
    if len(numeric_cols) == 0:
        sys.exit("No numeric columns found")
    
    # Get target and features
    X = df.select([c for c in numeric_cols if c != target_column]).to_numpy()
    y = df[target_column].to_numpy()
    feature_names = [c for c in numeric_cols if c != target_column]
    
    # 1. Near-zero variance filter
    low_variance_cols = []
    for col in feature_names:
        col_data = df[col].to_numpy()
        if not np.isfinite(col_data).all():
            col_data = col_data[np.isfinite(col_data)]
        if len(col_data) > 0:
            var_scaled = np.var(col_data) / (np.max(col_data) - np.min(col_data) + 1e-10) ** 2
            if var_scaled < 1e-4:
                low_variance_cols.append(col)
    
    # 2. Mutual Information ranking with noise baseline
    np.random.seed(42)
    
    # Compute MI for noise
    noise_mi_scores = []
    for _ in range(5):
        noise = np.random.randn(len(y))
        try:
            mi = mutual_info_regression(noise.reshape(-1, 1), y, random_state=42)
            noise_mi_scores.append(float(mi[0]))
        except:
            pass
    
    noise_mi_baseline = np.mean(noise_mi_scores) if noise_mi_scores else 0.001
    
    # Compute MI for real features
    mi_scores = []
    try:
        mi_array = mutual_info_regression(X, y, random_state=42)
        for feat_name, mi_val in zip(feature_names, mi_array):
            mi_scores.append({
                "feature": feat_name,
                "mi_score": float(mi_val),
                "below_noise_baseline": bool(float(mi_val) <= noise_mi_baseline)
            })
    except Exception as e:
        sys.exit(f"MI computation failed: {e}")
    
    # Sort by MI descending
    mi_scores.sort(key=lambda x: x["mi_score"], reverse=True)
    
    # 3. Pairwise correlation & redundancy detection
    redundant_cols = set()
    if len(feature_names) > 1:
        for i, feat_i in enumerate(feature_names):
            for feat_j in feature_names[i+1:]:
                try:
                    corr, _ = pearsonr(df[feat_i].to_numpy(), df[feat_j].to_numpy())
                    if abs(corr) >= 0.90:
                        # Find MI scores
                        mi_i = next((m["mi_score"] for m in mi_scores if m["feature"] == feat_i), 0)
                        mi_j = next((m["mi_score"] for m in mi_scores if m["feature"] == feat_j), 0)
                        if mi_j < mi_i:
                            redundant_cols.add(feat_j)
                        else:
                            redundant_cols.add(feat_i)
                except:
                    pass
    
    # 4. Time-series lag analysis
    significant_lags = []
    useful_lag_features = []
    
    if time_column is not None:
        try:
            # Autocorrelation of target
            y_data = df[target_column].to_numpy()
            acf_result = acf(y_data, nlags=min(24, len(y_data) // 4), fft=False)
            for lag in range(1, len(acf_result)):
                if abs(acf_result[lag]) > 0.1:
                    significant_lags.append(lag)
            
            # Cross-correlation for lag features
            for feat in feature_names:
                if feat in low_variance_cols or feat in redundant_cols:
                    continue
                feat_data = df[feat].to_numpy()
                for lag in range(4):
                    if lag <= len(feat_data) - 1:
                        xcorr = np.corrcoef(feat_data[:-lag or None], y_data[lag:] if lag > 0 else y_data)[0, 1]
                        if abs(xcorr) > 0.15:
                            useful_lag_features.append({
                                "feature": feat,
                                "lag": lag,
                                "xcorr": float(xcorr)
                            })
        except Exception as e:
            pass  # Continue even if lag analysis fails
    
    # Build recommended features list
    recommended_features = [
        m["feature"] for m in mi_scores
        if m["feature"] not in low_variance_cols
        and m["feature"] not in redundant_cols
        and not m["below_noise_baseline"]
    ]
    
    # If all filtered out, loosen threshold by 50%
    if len(recommended_features) == 0:
        loosened_threshold = noise_mi_baseline * 0.5
        recommended_features = [
            m["feature"] for m in mi_scores
            if m["feature"] not in low_variance_cols
            and m["feature"] not in redundant_cols
            and m["mi_score"] > loosened_threshold
        ]
    
    if len(recommended_features) == 0:
        recommended_features = [m["feature"] for m in mi_scores if m["feature"] not in low_variance_cols][:5]
    
    # Build excluded features dict
    excluded_features = {}
    for feat in feature_names:
        if feat not in recommended_features:
            if feat in low_variance_cols:
                excluded_features[feat] = "low_variance"
            elif feat in redundant_cols:
                excluded_features[feat] = "redundant"
            elif any(m["feature"] == feat and m["below_noise_baseline"] for m in mi_scores):
                excluded_features[feat] = "below_noise_baseline"
    
    # Build target candidates
    target_candidates = []
    for col in numeric_cols:
        if col != target_column:
            target_candidates.append({
                "column": col,
                "null_rate": step_10["null_rate"].get(col, 0),
                "mi_with_target": next((m["mi_score"] for m in mi_scores if m["feature"] == col), 0)
            })
    target_candidates.sort(key=lambda x: x["mi_with_target"], reverse=True)
    
    # Build step output
    step_output = {
        "step": "11-data-exploration",
        "shape": {
            "rows": df.height,
            "columns": df.width
        },
        "numeric_columns": numeric_cols,
        "high_cardinality": [col for col in df.columns if col == time_column],
        "low_variance_columns": low_variance_cols,
        "mi_ranking": mi_scores,
        "noise_mi_baseline": float(noise_mi_baseline),
        "redundant_columns": list(redundant_cols),
        "recommended_features": recommended_features,
        "excluded_features": excluded_features,
        "target_candidates": target_candidates[:5],
        "time_series": {
            "time_column": time_column,
            "significant_lags": significant_lags,
            "useful_lag_features": useful_lag_features
        }
    }
    
    # Save step JSON
    step_json_path = output_path / "step-11-exploration.json"
    with open(step_json_path, "w") as f:
        json.dump(step_output, f, indent=2)
    
    # Update progress
    progress_path = output_path / "progress.json"
    with open(progress_path, "r") as f:
        progress = json.load(f)
    progress["status"] = "running"
    progress["current_step"] = "12-feature-extraction"
    progress["completed_steps"].append("11-data-exploration")
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)
    
    return step_output


if __name__ == "__main__":
    print("Step 11 starting...", flush=True)
    parser = argparse.ArgumentParser(description="Step 11: Data Exploration")
    parser.add_argument("--output-dir", required=True, help="Output directory for artifacts")
    
    args = parser.parse_args()
    print(f"Output dir: {args.output_dir}", flush=True)
    
    try:
        result = run_step_11(args.output_dir)
        print(json.dumps(result, indent=2), flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"Error in Step 11: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
