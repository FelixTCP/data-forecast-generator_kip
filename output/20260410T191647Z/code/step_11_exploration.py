from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.feature_selection import mutual_info_regression
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


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return 0.0
    x = a[mask]
    y = b[mask]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 11 data exploration")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    step10 = read_json(output_dir / "step-10-cleanse.json")
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
        "11-data-exploration",
        completed_steps,
        errors,
        "running",
    )

    try:
        df = pl.read_parquet(step10["artifacts"]["cleaned_parquet"])
        if target not in df.columns:
            raise ValueError(f"Target column not found: {target}")

        numeric_columns = [c for c, d in zip(df.columns, df.dtypes, strict=False) if is_numeric_dtype(d)]
        if target not in numeric_columns:
            raise ValueError("Target column is not numeric after cleansing")

        feature_candidates = [c for c in numeric_columns if c != target]
        if not feature_candidates:
            raise ValueError("No numeric candidate features available for exploration")

        cardinality: dict[str, int] = {}
        for col in tqdm(df.columns, desc="step11: cardinality", unit="column"):
            cardinality[col] = int(df.select(pl.col(col).n_unique()).item())

        low_variance_columns: list[str] = []
        numeric_summary: dict[str, dict[str, float | None]] = {}
        for col in tqdm(feature_candidates, desc="step11: variance", unit="feature"):
            stats = df.select(
                [
                    pl.col(col).mean().alias("mean"),
                    pl.col(col).std().alias("std"),
                    pl.col(col).min().alias("min"),
                    pl.col(col).max().alias("max"),
                    pl.col(col).is_null().mean().alias("null_rate"),
                ]
            ).to_dicts()[0]
            numeric_summary[col] = {k: (None if v is None else float(v)) for k, v in stats.items()}

            v_min = stats["min"]
            v_max = stats["max"]
            if v_min is None or v_max is None or float(v_max) - float(v_min) <= 1e-12:
                low_variance_columns.append(col)
                continue

            scaled = (
                (pl.col(col).cast(pl.Float64, strict=False) - float(v_min)) / (float(v_max) - float(v_min))
            ).fill_null(0.0)
            var_scaled = float(df.select(scaled.var()).item() or 0.0)
            if var_scaled < 1e-4:
                low_variance_columns.append(col)

        y = df.select(pl.col(target).cast(pl.Float64, strict=False).fill_null(0.0)).to_numpy().reshape(-1)
        x = df.select(
            [
                pl.col(c).cast(pl.Float64, strict=False).fill_null(pl.col(c).median()).fill_null(0.0).alias(c)
                for c in feature_candidates
            ]
        ).to_numpy()

        mi_scores = mutual_info_regression(x, y, random_state=RANDOM_STATE)

        rng = np.random.default_rng(RANDOM_STATE)
        noise_scores: list[float] = []
        for _ in tqdm(range(5), desc="step11: noise-mi", unit="noise"):
            noise_col = rng.normal(size=df.height)
            score = mutual_info_regression(noise_col.reshape(-1, 1), y, random_state=RANDOM_STATE)[0]
            noise_scores.append(float(score))
        noise_mi_baseline = float(np.mean(noise_scores)) if noise_scores else 0.0

        mi_lookup = {f: float(s) for f, s in zip(feature_candidates, mi_scores, strict=False)}

        excluded_features: dict[str, str] = {}
        for col in low_variance_columns:
            excluded_features[col] = "low_variance"

        for feature, score in tqdm(mi_lookup.items(), desc="step11: mi-filter", unit="feature"):
            if score <= noise_mi_baseline:
                excluded_features.setdefault(feature, "below_noise_baseline")

        corr_matrix = df.select(
            [
                pl.col(c).cast(pl.Float64, strict=False).fill_null(pl.col(c).median()).fill_null(0.0).alias(c)
                for c in feature_candidates
            ]
        ).to_numpy()

        redundant_columns: list[str] = []
        max_pair = ["", ""]
        max_corr = 0.0
        for i, j in tqdm(list(combinations(range(len(feature_candidates)), 2)), desc="step11: redundancy", unit="pair"):
            c1 = feature_candidates[i]
            c2 = feature_candidates[j]
            corr = safe_corr(corr_matrix[:, i], corr_matrix[:, j])
            if abs(corr) > abs(max_corr):
                max_pair = [c1, c2]
                max_corr = corr
            if abs(corr) >= 0.90:
                loser = c1 if mi_lookup.get(c1, 0.0) <= mi_lookup.get(c2, 0.0) else c2
                excluded_features.setdefault(loser, f"redundant_with_{c2 if loser == c1 else c1}")
                redundant_columns.append(loser)

        recommended_features = [f for f in feature_candidates if f not in excluded_features]
        if not recommended_features:
            relaxed_baseline = noise_mi_baseline * 0.5
            for f, s in mi_lookup.items():
                if f in excluded_features and excluded_features[f] == "below_noise_baseline" and s > relaxed_baseline:
                    excluded_features.pop(f)
            recommended_features = [f for f in feature_candidates if f not in excluded_features]
            context.setdefault("notes", []).append("Loosened MI noise threshold by 50% to avoid empty recommendations.")

        if not recommended_features:
            best = max(feature_candidates, key=lambda c: mi_lookup.get(c, -1.0))
            excluded_features.pop(best, None)
            recommended_features = [best]
            context.setdefault("notes", []).append("Fallback to best MI feature to avoid empty recommendations.")

        mi_ranking = [
            {
                "feature": f,
                "mi_score": float(mi_lookup[f]),
                "below_noise_baseline": bool(mi_lookup[f] <= noise_mi_baseline),
            }
            for f in sorted(feature_candidates, key=lambda name: mi_lookup[name], reverse=True)
        ]

        significant_lags: list[int] = []
        useful_lag_features: list[dict[str, Any]] = []

        if time_column and time_column in df.columns:
            max_lag = int(min(24, max(1, df.height // 4)))
            target_arr = y
            for lag in tqdm(range(1, max_lag + 1), desc="step11: target-acf", unit="lag"):
                acf = safe_corr(target_arr[lag:], target_arr[:-lag])
                if abs(acf) > 0.1:
                    significant_lags.append(lag)

            for feature in tqdm(feature_candidates, desc="step11: feature-xcorr", unit="feature"):
                arr = df.select(
                    pl.col(feature).cast(pl.Float64, strict=False).fill_null(pl.col(feature).median()).fill_null(0.0)
                ).to_numpy().reshape(-1)
                for lag in range(0, 4):
                    if lag == 0:
                        xcorr = safe_corr(arr, target_arr)
                    else:
                        xcorr = safe_corr(arr[:-lag], target_arr[lag:])
                    if abs(xcorr) > 0.15:
                        useful_lag_features.append(
                            {
                                "feature": feature,
                                "lag": int(lag),
                                "xcorr": float(xcorr),
                            }
                        )

        target_candidates: list[dict[str, Any]] = []
        numeric_cols_all = list(numeric_columns)
        for col in tqdm(numeric_cols_all, desc="step11: target-candidates", unit="column"):
            null_rate = float(df.select(pl.col(col).is_null().mean()).item() or 0.0)
            std = float(df.select(pl.col(col).cast(pl.Float64, strict=False).std()).item() or 0.0)
            avg_abs_corr = 0.0
            peers = [c for c in numeric_cols_all if c != col]
            if peers:
                vals = []
                for peer in peers[:10]:
                    a = df.select(pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0)).to_numpy().reshape(-1)
                    b = df.select(pl.col(peer).cast(pl.Float64, strict=False).fill_null(0.0)).to_numpy().reshape(-1)
                    vals.append(abs(safe_corr(a, b)))
                avg_abs_corr = float(np.mean(vals)) if vals else 0.0
            target_candidates.append(
                {
                    "column": col,
                    "null_rate": null_rate,
                    "std": std,
                    "avg_abs_corr": avg_abs_corr,
                    "is_user_target": col == target,
                }
            )

        target_candidates = sorted(
            target_candidates,
            key=lambda r: (r["null_rate"], -r["std"], -r["avg_abs_corr"]),
        )[:5]

        step_output = {
            "step": "11-data-exploration",
            "shape": {"rows": int(df.height), "columns": int(df.width)},
            "numeric_columns": numeric_columns,
            "high_cardinality": [c for c in df.columns if cardinality[c] > max(100, int(0.5 * max(1, df.height)))],
            "low_variance_columns": sorted(set(low_variance_columns)),
            "mi_ranking": mi_ranking,
            "noise_mi_baseline": float(noise_mi_baseline),
            "redundant_columns": sorted(set(redundant_columns)),
            "correlation_matrix_summary": {"max_pair": max_pair, "max_corr": float(max_corr)},
            "significant_lags": sorted(set(significant_lags)),
            "useful_lag_features": useful_lag_features,
            "recommended_features": recommended_features,
            "excluded_features": excluded_features,
            "target_candidates": target_candidates,
            "time_series_detected": bool(time_column),
            "time_column": time_column,
            "context": context,
        }

        write_json(output_dir / "step-11-exploration.json", step_output)

        if "11-data-exploration" not in completed_steps:
            completed_steps.append("11-data-exploration")

        update_progress(
            progress_path,
            args.run_id,
            progress["csv_path"],
            target,
            "11-data-exploration",
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
            "11-data-exploration",
            completed_steps,
            errors,
            "error",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())