from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from tqdm import tqdm

RANDOM_STATE = 42


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def update_progress(
    progress_path: Path,
    run_id: str,
    csv_path: str,
    target_column: str,
    current_step: str,
    completed_steps: list[str],
    errors: list[str],
    status: str,
) -> None:
    write_json(
        progress_path,
        {
            "run_id": run_id,
            "csv_path": csv_path,
            "target_column": target_column,
            "status": status,
            "current_step": current_step,
            "completed_steps": completed_steps,
            "errors": errors,
        },
    )


def is_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Decimal,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 12 feature extraction")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split-mode", default="auto", choices=["auto", "random", "time_series"])
    return parser.parse_args()


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return 0.0
    xv = x[mask]
    yv = y[mask]
    if float(np.std(xv)) <= 1e-12 or float(np.std(yv)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(xv, yv)[0, 1])


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    step10 = read_json(output_dir / "step-10-cleanse.json")
    step11 = read_json(output_dir / "step-11-exploration.json")

    context = step10["context"]
    target = context["target_column"]
    time_column = context.get("time_column")

    progress_path = output_dir / "progress.json"
    progress = read_json(progress_path)
    completed_steps = [str(s) for s in progress.get("completed_steps", [])]
    errors = [str(e) for e in progress.get("errors", [])]

    update_progress(
        progress_path,
        args.run_id,
        progress["csv_path"],
        target,
        "12-feature-extraction",
        completed_steps,
        errors,
        "running",
    )

    try:
        recommended = step11.get("recommended_features")
        if not isinstance(recommended, list) or not recommended:
            raise ValueError("step-11-exploration.json has empty/missing recommended_features")

        excluded_from_step11 = dict(step11.get("excluded_features", {}))
        useful_lag_features = list(step11.get("useful_lag_features", []))
        significant_lags = [int(v) for v in step11.get("significant_lags", []) if int(v) > 0]

        df = pl.read_parquet(step10["artifacts"]["cleaned_parquet"])
        if target not in df.columns:
            raise ValueError(f"Target column not found: {target}")

        if time_column and time_column in df.columns:
            df = df.sort(time_column)

        features_excluded = {k: f"{v} (step 11)" for k, v in excluded_from_step11.items()}
        created_features: list[dict[str, str]] = []

        selected_base = [c for c in recommended if c in df.columns and c != target]
        missing_recommended = [c for c in recommended if c not in df.columns]
        for col in missing_recommended:
            features_excluded[col] = "missing_from_cleaned_parquet"

        if time_column and time_column in df.columns:
            time_dtype = df.schema[time_column]
            t = pl.col(time_column)
            # Date columns do not support hour directly; cast to datetime first.
            t_hour = t.cast(pl.Datetime("us")) if time_dtype == pl.Date else t
            df = df.with_columns(
                [
                    t.dt.year().alias("year"),
                    t.dt.month().alias("month"),
                    t.dt.weekday().alias("day_of_week"),
                    t_hour.dt.hour().alias("hour"),
                ]
            )
            created_features.extend(
                [
                    {"name": "year", "reason": "time decomposition"},
                    {"name": "month", "reason": "time decomposition"},
                    {"name": "day_of_week", "reason": "time decomposition"},
                    {"name": "hour", "reason": "time decomposition"},
                ]
            )

        lag_exprs: list[pl.Expr] = []
        for item in tqdm(useful_lag_features, desc="step12: feature lags", unit="lag"):
            feature = str(item.get("feature", ""))
            lag = int(item.get("lag", 0))
            xcorr = float(item.get("xcorr", 0.0))
            if feature not in selected_base or feature not in df.columns or lag <= 0:
                continue
            name = f"{feature}_lag_{lag}"
            lag_exprs.append(pl.col(feature).shift(lag).alias(name))
            created_features.append(
                {
                    "name": name,
                    "reason": f"useful_lag_feature: {feature} lag={lag}, xcorr={xcorr:.3f}",
                }
            )

        for lag in tqdm(significant_lags, desc="step12: target lags", unit="lag"):
            name = f"target_lag_{lag}"
            lag_exprs.append(pl.col(target).shift(lag).alias(name))
            created_features.append(
                {
                    "name": name,
                    "reason": f"significant_lag={lag} from step 11",
                }
            )

        top_lags = sorted(set(significant_lags))[:2]
        for window in tqdm(top_lags, desc="step12: rolling", unit="window"):
            mean_name = f"target_rolling_mean_{window}"
            std_name = f"target_rolling_std_{window}"
            lag_exprs.append(pl.col(target).shift(1).rolling_mean(window_size=window).alias(mean_name))
            lag_exprs.append(pl.col(target).shift(1).rolling_std(window_size=window).alias(std_name))
            created_features.append({"name": mean_name, "reason": f"rolling mean at lag-window={window}"})
            created_features.append({"name": std_name, "reason": f"rolling std at lag-window={window}"})

        if lag_exprs:
            df = df.with_columns(lag_exprs)
        else:
            context.setdefault("notes", []).append("No lag features created - no significant autocorrelation detected.")

        candidate_columns_raw = selected_base + [c["name"] for c in created_features] + ["year", "month", "day_of_week", "hour"]
        candidate_columns = list(dict.fromkeys([c for c in candidate_columns_raw if c in df.columns]))

        pre_drop_rows = int(df.height)
        applied_lags = [
            int(item.get("lag", 0))
            for item in useful_lag_features
            if int(item.get("lag", 0)) > 0
        ] + [int(v) for v in significant_lags if int(v) > 0]
        max_lag_applied = max(applied_lags) if applied_lags else 0

        # Drop only the leading rows that are structurally undefined due to shift-based features.
        if max_lag_applied > 0:
            df = df.slice(max_lag_applied)

        df = df.drop_nulls(subset=[target])
        rows_dropped_by_lag = pre_drop_rows - int(df.height)

        numeric_features: list[str] = []
        dropped_non_numeric: list[str] = []
        for col in tqdm(candidate_columns, desc="step12: numeric filter", unit="column"):
            if col not in df.columns:
                continue
            if is_numeric_dtype(df.schema[col]):
                non_null_count = int(df.select(pl.col(col).is_not_null().sum()).item())
                if non_null_count == 0:
                    features_excluded[col] = "all_null_after_construction"
                    continue
                numeric_features.append(col)
            else:
                dropped_non_numeric.append(col)
                features_excluded[col] = "non_numeric_after_construction"

        for col in dropped_non_numeric:
            features_excluded[col] = "non_numeric_after_construction"

        y_arr = df.select(pl.col(target).cast(pl.Float64, strict=False)).to_numpy().reshape(-1)
        leakage_flags: list[str] = []
        clean_features: list[str] = []
        for col in tqdm(numeric_features, desc="step12: leakage guard", unit="feature"):
            x_arr = df.select(pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0)).to_numpy().reshape(-1)
            corr = abs(safe_corr(x_arr, y_arr))
            if corr > 0.99:
                features_excluded[col] = f"potential_leakage_abs_corr_{corr:.4f}"
                leakage_flags.append(col)
                continue
            clean_features.append(col)

        if target in clean_features:
            clean_features.remove(target)

        if len(clean_features) < 2:
            raise ValueError(
                f"Too few features after filtering ({len(clean_features)}). Check step 11 filters and leakage exclusions."
            )

        disallowed = set(excluded_from_step11.keys())
        overlap = sorted(set(clean_features).intersection(disallowed))
        if overlap:
            raise ValueError(f"Leakage guard failed: step11 excluded features reintroduced: {overlap}")

        split_resolved = "time_series" if (args.split_mode == "auto" and time_column) else args.split_mode
        if split_resolved not in {"time_series", "random"}:
            split_resolved = "random"

        final_df = df.select([pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in clean_features + [target]])

        features_path = output_dir / "features.parquet"
        final_df.write_parquet(features_path)

        context["features"] = clean_features
        context["split_strategy"] = {
            "requested_mode": args.split_mode,
            "resolved_mode": split_resolved,
            "time_column": time_column,
            "random_state": RANDOM_STATE,
        }
        context.setdefault("artifacts", {})["features_parquet"] = str(features_path)

        leakage_audit = {
            "status": "pass" if not leakage_flags else "warning",
            "checks": ["pairwise_corr"],
            "details": [
                f"Excluded {name} for abs(target_corr)>0.99" for name in leakage_flags
            ],
        }

        step_output = {
            "step": "12-feature-extraction",
            "features": clean_features,
            "features_excluded": features_excluded,
            "created_features": created_features,
            "rows_dropped_by_lag": int(rows_dropped_by_lag),
            "leakage_flags": leakage_flags,
            "leakage_audit": leakage_audit,
            "split_strategy": context["split_strategy"],
            "artifacts": {"features_parquet": str(features_path)},
            "context": context,
        }

        write_json(output_dir / "step-12-features.json", step_output)

        if "12-feature-extraction" not in completed_steps:
            completed_steps.append("12-feature-extraction")

        update_progress(
            progress_path,
            args.run_id,
            progress["csv_path"],
            target,
            "12-feature-extraction",
            completed_steps,
            errors,
            "running",
        )
        return 0
    except Exception as exc:
        errors.append(str(exc))
        update_progress(
            progress_path,
            args.run_id,
            progress["csv_path"],
            target,
            "12-feature-extraction",
            completed_steps,
            errors,
            "error",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())