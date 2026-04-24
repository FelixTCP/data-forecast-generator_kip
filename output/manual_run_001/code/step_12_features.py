#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor

from runtime_utils import mark_step_error, mark_step_start, mark_step_success, read_json, update_code_audit, write_json


STEP_NAME = "12-feature-extraction"


def build_features(output_dir: Path, run_id: str, split_mode: str) -> None:
    mark_step_start(output_dir, STEP_NAME)
    try:
        step01 = read_json(output_dir / "step-01-cleanse.json")
        step11 = read_json(output_dir / "step-11-exploration.json")
        df = pl.read_parquet(output_dir / "cleaned.parquet")
        target = step01["target_column_normalized"]
        time_column = step11.get("time_column")
        recommended = list(step11["recommended_features"])
        excluded = set(step11["excluded_features"].keys())
        if any(feature in excluded for feature in recommended):
            raise ValueError("Recommended features unexpectedly include excluded features.")

        work = df.select([col for col in df.columns if col in set(recommended) | {target} | ({time_column} if time_column else set())])
        created_features: list[dict[str, str]] = []

        if time_column:
            work = work.sort(time_column)
            work = work.with_columns(
                [
                    pl.col(time_column).dt.year().alias("year"),
                    pl.col(time_column).dt.month().alias("month"),
                    pl.col(time_column).dt.weekday().alias("day_of_week"),
                    pl.col(time_column).dt.hour().alias("hour"),
                ]
            )
            for name in ["year", "month", "day_of_week", "hour"]:
                created_features.append({"name": name, "reason": "time decomposition from detected time column"})

        useful_lag_features = step11.get("useful_lag_features", [])
        for item in useful_lag_features:
            feature = item["feature"]
            lag = int(item["lag"])
            if lag <= 0 or feature not in recommended:
                continue
            name = f"{feature}_lag_{lag}"
            work = work.with_columns(pl.col(feature).shift(lag).alias(name))
            created_features.append({"name": name, "reason": f"useful_lag_feature from step 11 xcorr={item['xcorr']:.3f}"})

        significant_lags = [int(lag) for lag in step11.get("significant_lags", [])]
        for lag in significant_lags:
            name = f"{target}_lag_{lag}"
            work = work.with_columns(pl.col(target).shift(lag).alias(name))
            created_features.append({"name": name, "reason": f"target lag from step 11 significant lag {lag}"})

        top_two = significant_lags[:2]
        for window in top_two:
            mean_name = f"{target}_rolling_mean_{window}"
            expressions = [pl.col(target).shift(1).rolling_mean(window).alias(mean_name)]
            std_name = f"{target}_rolling_std_{window}"
            if window >= 2:
                expressions.append(pl.col(target).shift(1).rolling_std(window).alias(std_name))
            work = work.with_columns(expressions)
            created_features.append({"name": mean_name, "reason": f"causal rolling mean with shift(1), window={window}"})
            if window >= 2:
                created_features.append({"name": std_name, "reason": f"causal rolling std with shift(1), window={window}"})

        rows_before = work.height
        work = work.drop_nulls()
        rows_dropped = rows_before - work.height
        if work.height <= 0:
            raise ValueError("Feature engineering dropped all rows.")

        feature_columns = [column for column in work.columns if column not in {target, time_column}]
        if len(feature_columns) < 2:
            raise ValueError("Fewer than two features remain after engineering.")

        features_df = work.select(feature_columns)
        y = work[target].to_numpy()
        leakage_candidates = []
        for column in feature_columns:
            corr = np.corrcoef(work[column].to_numpy(), y)[0, 1]
            if np.isfinite(corr) and abs(float(corr)) > 0.99:
                leakage_candidates.append({"feature": column, "correlation": float(corr)})
        reconstruction_probe_r2 = None
        if leakage_candidates:
            X_probe = work.select([item["feature"] for item in leakage_candidates]).to_numpy()
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_probe, y)
            reconstruction_probe_r2 = float(rf.score(X_probe, y))
        leakage_status = "pass"
        if leakage_candidates and reconstruction_probe_r2 and reconstruction_probe_r2 > 0.999:
            leakage_status = "fail"
        leakage_audit = {
            "step": STEP_NAME,
            "status": leakage_status,
            "threshold": 0.99,
            "leakage_candidates": leakage_candidates,
            "reconstruction_probe_r2": reconstruction_probe_r2,
        }
        write_json(output_dir / "leakage_audit.json", leakage_audit)
        if leakage_status == "fail":
            raise RuntimeError("Leakage audit failed; refusing to write step-12 artifacts.")

        work.write_parquet(output_dir / "features.parquet")
        payload = {
            "step": STEP_NAME,
            "run_id": run_id,
            "target_column": target,
            "time_column": time_column,
            "features": feature_columns,
            "features_excluded": {feature: reason for feature, reason in step11["excluded_features"].items()},
            "created_features": created_features,
            "rows_dropped_by_lags": rows_dropped,
            "split_strategy": {
                "requested_mode": split_mode,
                "resolved_mode": "time_series" if split_mode == "time_series" or (split_mode == "auto" and time_column) else "random",
            },
            "artifacts": {"features_parquet": str(output_dir / "features.parquet")},
            "leakage_audit": leakage_audit,
        }
        write_json(output_dir / "step-12-features.json", payload)
        mark_step_success(output_dir, STEP_NAME)
        update_code_audit(output_dir, Path(__file__).resolve().parent)
    except Exception as exc:
        mark_step_error(output_dir, STEP_NAME, str(exc))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--split-mode", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    build_features(Path(args.output_dir), args.run_id, args.split_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
