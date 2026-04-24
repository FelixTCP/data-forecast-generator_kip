#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from runtime_utils import mark_step_error, mark_step_start, mark_step_success, read_json, update_code_audit, write_json


STEP_NAME = "11-data-exploration"


def explore(output_dir: Path, run_id: str) -> None:
    mark_step_start(output_dir, STEP_NAME)
    try:
        step01 = read_json(output_dir / "step-01-cleanse.json")
        df = pl.read_parquet(output_dir / "cleaned.parquet")
        target = step01["target_column_normalized"]
        time_column = step01.get("time_column")

        pdf = df.to_pandas()
        for column in pdf.columns:
            if column == time_column:
                continue
            if pdf[column].dtype == object:
                pdf[column] = pd.to_numeric(pdf[column], errors="ignore")

        numeric_columns = [
            column
            for column in pdf.columns
            if column != target and pd.api.types.is_numeric_dtype(pdf[column])
        ]
        target_series = pd.to_numeric(pdf[target], errors="coerce")
        valid_mask = target_series.notna()
        pdf = pdf.loc[valid_mask].reset_index(drop=True)
        target_series = target_series.loc[valid_mask].astype(float)

        low_variance_columns: list[str] = []
        for column in numeric_columns:
            values = pd.to_numeric(pdf[column], errors="coerce")
            values = values.fillna(values.median())
            spread = values.max() - values.min()
            scaled_variance = 0.0 if spread == 0 else float(values.var(ddof=0) / (spread**2))
            if scaled_variance < 1e-4:
                low_variance_columns.append(column)

        high_cardinality = [
            column
            for column in pdf.columns
            if column not in {target, time_column}
            and not pd.api.types.is_numeric_dtype(pdf[column])
            and pdf[column].nunique(dropna=True) > 50
        ]

        X_numeric = pd.DataFrame(
            {
                column: pd.to_numeric(pdf[column], errors="coerce").fillna(pd.to_numeric(pdf[column], errors="coerce").median())
                for column in numeric_columns
            }
        )
        mi_values = mutual_info_regression(X_numeric, target_series, random_state=42)
        rng = np.random.default_rng(42)
        noise_scores = []
        for index in range(5):
            noise = rng.normal(size=len(target_series))
            score = mutual_info_regression(noise.reshape(-1, 1), target_series, random_state=42)[0]
            noise_scores.append(float(score))
            X_numeric[f"__noise_{index}"] = noise
        noise_baseline = float(np.mean(noise_scores))

        mi_lookup = {column: float(score) for column, score in zip(numeric_columns, mi_values)}
        mi_ranking = [
            {
                "feature": column,
                "mi_score": mi_lookup[column],
                "below_noise_baseline": mi_lookup[column] <= noise_baseline,
            }
            for column in sorted(numeric_columns, key=lambda item: mi_lookup[item], reverse=True)
        ]

        excluded_features: dict[str, str] = {}
        for column in low_variance_columns:
            excluded_features[column] = "low_variance"
        for column in high_cardinality:
            excluded_features[column] = "high_cardinality"
        for item in mi_ranking:
            if item["below_noise_baseline"]:
                excluded_features.setdefault(item["feature"], "below_noise_baseline")

        corr_matrix = X_numeric[numeric_columns].corr().fillna(0.0)
        max_pair = None
        max_corr = 0.0
        redundant_columns: list[str] = []
        for i, left in enumerate(numeric_columns):
            for right in numeric_columns[i + 1 :]:
                corr = float(corr_matrix.loc[left, right])
                if abs(corr) > abs(max_corr):
                    max_corr = corr
                    max_pair = [left, right]
                if abs(corr) >= 0.90:
                    drop = left if mi_lookup[left] < mi_lookup[right] else right
                    redundant_columns.append(drop)
                    excluded_features.setdefault(drop, "redundant")

        leakage_suspects = []
        for column in numeric_columns:
            corr = X_numeric[column].corr(target_series)
            if pd.notna(corr) and abs(float(corr)) > 0.98:
                leakage_suspects.append(column)
                excluded_features[column] = "leakage_suspect"

        significant_lags: list[int] = []
        useful_lag_features: list[dict[str, float | int | str]] = []
        multiple_series_detected = False
        ts_characteristics = {
            "trend_detected": False,
            "seasonality_detected": False,
            "stationarity": "unknown",
            "white_noise": False,
        }
        model_recommendations = ["Ridge", "RandomForest", "GradientBoosting"]
        if time_column:
            if pdf[time_column].duplicated().any():
                multiple_series_detected = True
            ordered = pdf.sort_values(time_column).reset_index(drop=True)
            y_ordered = pd.to_numeric(ordered[target], errors="coerce").astype(float)
            max_lag = min(24, max(1, len(ordered) // 4))
            for lag in range(1, max_lag + 1):
                corr = y_ordered.autocorr(lag=lag)
                if pd.notna(corr) and abs(float(corr)) > 0.10:
                    significant_lags.append(lag)
            for column in numeric_columns:
                feature = pd.to_numeric(ordered[column], errors="coerce").astype(float)
                for lag in range(0, 4):
                    shifted = feature.shift(lag)
                    joined = pd.concat([shifted, y_ordered], axis=1).dropna()
                    if len(joined) < 3:
                        continue
                    corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
                    if pd.notna(corr) and abs(float(corr)) > 0.15:
                        useful_lag_features.append(
                            {"feature": column, "lag": lag, "xcorr": float(corr)}
                        )
            clean_target = y_ordered.dropna()
            if len(clean_target) > 20:
                try:
                    adf_p = float(adfuller(clean_target.to_numpy())[1])
                    ts_characteristics["stationarity"] = "stationary" if adf_p < 0.05 else "non-stationary"
                except Exception:
                    pass
                try:
                    lb = acorr_ljungbox(clean_target.to_numpy(), lags=[min(10, len(clean_target) - 1)], return_df=True)
                    ts_characteristics["white_noise"] = bool(lb["lb_pvalue"].iloc[0] > 0.05)
                except Exception:
                    pass
                try:
                    period = 144 if len(clean_target) >= 288 else max(2, min(24, len(clean_target) // 4))
                    decomposition = seasonal_decompose(clean_target, model="additive", period=period, extrapolate_trend="freq")
                    trend_strength = np.nanstd(decomposition.trend) / max(np.nanstd(clean_target), 1e-9)
                    seasonal_strength = np.nanstd(decomposition.seasonal) / max(np.nanstd(clean_target), 1e-9)
                    ts_characteristics["trend_detected"] = bool(trend_strength > 0.10)
                    ts_characteristics["seasonality_detected"] = bool(seasonal_strength > 0.10)
                except Exception:
                    pass
            model_recommendations = []
            if ts_characteristics["seasonality_detected"] and ts_characteristics["stationarity"] == "non-stationary":
                model_recommendations.append("SARIMA")
            if multiple_series_detected:
                model_recommendations.append("XGBoost")
            model_recommendations.extend(["Ridge", "RandomForest", "GradientBoosting"])

        recommended_features = [
            column for column in numeric_columns if column not in excluded_features
        ]
        if not recommended_features:
            relaxed_threshold = noise_baseline * 0.5
            recommended_features = [
                item["feature"] for item in mi_ranking if item["mi_score"] > relaxed_threshold
            ]
        target_candidates = [
            {
                "column": target,
                "reason": "runtime_input_explicit_target",
                "variance": float(target_series.var()),
            }
        ]
        client_summary = (
            f"Target `{target}` shows "
            f"{'time structure' if time_column else 'no explicit time structure'}; "
            f"{len(excluded_features)} features were excluded for noise, redundancy, or leakage risk."
        )

        payload = {
            "step": STEP_NAME,
            "run_id": run_id,
            "shape": {"rows": int(len(pdf)), "columns": int(len(pdf.columns))},
            "numeric_columns": numeric_columns,
            "high_cardinality": high_cardinality,
            "low_variance_columns": low_variance_columns,
            "time_series_detected": bool(time_column),
            "time_column": time_column,
            "multiple_series_detected": multiple_series_detected,
            "time_series_characteristics": ts_characteristics,
            "model_recommendations": model_recommendations[:3],
            "mi_ranking": mi_ranking,
            "noise_mi_baseline": noise_baseline,
            "redundant_columns": sorted(set(redundant_columns)),
            "correlation_matrix_summary": {"max_pair": max_pair, "max_corr": float(max_corr)},
            "significant_lags": significant_lags,
            "useful_lag_features": useful_lag_features,
            "recommended_features": recommended_features,
            "excluded_features": excluded_features,
            "target_candidates": target_candidates,
            "client_facing_summary": client_summary,
            "context": {"leakage_suspects": leakage_suspects},
        }
        write_json(output_dir / "step-11-exploration.json", payload)
        mark_step_success(output_dir, STEP_NAME)
        update_code_audit(output_dir, Path(__file__).resolve().parent)
    except Exception as exc:
        mark_step_error(output_dir, STEP_NAME, str(exc))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    explore(Path(args.output_dir), args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
