#!/usr/bin/env python3
"""Step 11: data exploration and feature candidacy gating."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.feature_selection import mutual_info_regression

STEP_NAME = "11-data-exploration"
STEP01_JSON = "step-01-cleanse.json"
STEP11_JSON = "step-11-exploration.json"
PROGRESS_FILE = "progress.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _update_progress(
    output_dir: Path,
    *,
    run_id: str,
    csv_path: str,
    target_column: str,
    status: str,
    current_step: str,
    error: str | None = None,
    mark_completed: bool = False,
) -> None:
    path = output_dir / PROGRESS_FILE
    progress = _load_json(path)
    if not progress:
        progress = {
            "run_id": run_id,
            "csv_path": csv_path,
            "target_column": target_column,
            "status": "running",
            "current_step": current_step,
            "completed_steps": [],
            "errors": [],
        }

    progress["run_id"] = run_id
    progress["csv_path"] = csv_path
    progress["target_column"] = target_column
    progress["status"] = status
    progress["current_step"] = current_step

    completed = progress.setdefault("completed_steps", [])
    if mark_completed and current_step not in completed:
        completed.append(current_step)

    if error:
        progress.setdefault("errors", []).append({"step": current_step, "error": error})

    _write_json(path, progress)


def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype in pl.NUMERIC_DTYPES


def _to_filled_float_array(df: pl.DataFrame, col: str) -> np.ndarray:
    arr = df[col].cast(pl.Float64).to_numpy()
    if arr.size == 0:
        return arr
    if np.isnan(arr).all():
        return np.zeros_like(arr)
    med = float(np.nanmedian(arr))
    return np.where(np.isnan(arr), med, arr)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return 0.0
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return 0.0
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) == 0.0 or np.std(bb) == 0.0:
        return 0.0
    corr = float(np.corrcoef(aa, bb)[0, 1])
    if math.isnan(corr):
        return 0.0
    return corr


def _compute_target_candidates(df: pl.DataFrame, numeric_columns: list[str]) -> list[dict[str, Any]]:
    if not numeric_columns:
        return []

    variances: dict[str, float] = {}
    avg_corr: dict[str, float] = {}
    for col in numeric_columns:
        arr = _to_filled_float_array(df, col)
        variances[col] = float(np.nanvar(arr))

    for col in numeric_columns:
        corrs = []
        arr = _to_filled_float_array(df, col)
        for other in numeric_columns:
            if other == col:
                continue
            corrs.append(abs(_safe_corr(arr, _to_filled_float_array(df, other))))
        avg_corr[col] = float(np.mean(corrs)) if corrs else 0.0

    max_var = max(variances.values()) if variances else 1.0
    max_corr = max(avg_corr.values()) if avg_corr else 1.0
    if max_var <= 0:
        max_var = 1.0
    if max_corr <= 0:
        max_corr = 1.0

    candidates = []
    for col in numeric_columns:
        null_rate = float(df.select(pl.col(col).is_null().mean()).item())
        var_norm = variances[col] / max_var
        corr_norm = avg_corr[col] / max_corr
        score = (1.0 - null_rate) * 0.4 + var_norm * 0.3 + corr_norm * 0.3
        candidates.append(
            {
                "column": col,
                "score": round(float(score), 6),
                "null_rate": round(null_rate, 6),
                "variance": round(variances[col], 6),
                "avg_abs_corr": round(avg_corr[col], 6),
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:5]


def run_step(output_dir: Path, run_id: str) -> dict[str, Any]:
    step01 = _load_json(output_dir / STEP01_JSON)
    if not step01:
        raise FileNotFoundError("Missing step-01-cleanse.json. Run step 01 first.")

    cleaned_path = Path(step01["artifacts"]["cleaned_parquet"])
    df = pl.read_parquet(cleaned_path)

    target = step01["target_column_normalized"]
    time_column = step01.get("time_column")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in cleaned data.")

    row_count = df.height
    if row_count <= 10:
        raise ValueError(f"Row count too small for exploration: {row_count}")

    numeric_columns = [c for c, d in df.schema.items() if _is_numeric(d)]
    if target not in numeric_columns:
        raise ValueError("Target column is not numeric after cleansing.")

    feature_columns = [c for c in numeric_columns if c != target]
    if not feature_columns:
        raise ValueError("No numeric feature columns available for MI analysis.")

    low_variance_columns: list[str] = []
    scaled_variances: dict[str, float] = {}
    for col in feature_columns:
        arr = _to_filled_float_array(df, col)
        if arr.size < 2:
            low_variance_columns.append(col)
            scaled_variances[col] = 0.0
            continue
        cmin = float(np.min(arr))
        cmax = float(np.max(arr))
        if cmax - cmin == 0:
            scaled_var = 0.0
        else:
            scaled = (arr - cmin) / (cmax - cmin)
            scaled_var = float(np.var(scaled))
        scaled_variances[col] = scaled_var
        if scaled_var < 1e-4:
            low_variance_columns.append(col)

    y = _to_filled_float_array(df, target)
    X = np.column_stack([_to_filled_float_array(df, col) for col in feature_columns])

    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_map = {col: float(score) for col, score in zip(feature_columns, mi_scores, strict=True)}

    rng = np.random.default_rng(42)
    noise = rng.normal(loc=0.0, scale=1.0, size=(row_count, 5))
    noise_mi = mutual_info_regression(noise, y, random_state=42)
    noise_mi_baseline = float(np.mean(noise_mi))

    redundant_columns: set[str] = set()
    max_pair: list[str] = []
    max_corr = 0.0

    for i, left in enumerate(feature_columns):
        left_arr = _to_filled_float_array(df, left)
        for j in range(i + 1, len(feature_columns)):
            right = feature_columns[j]
            corr = abs(_safe_corr(left_arr, _to_filled_float_array(df, right)))
            if corr > max_corr:
                max_corr = corr
                max_pair = [left, right]
            if corr >= 0.90:
                left_mi = mi_map.get(left, 0.0)
                right_mi = mi_map.get(right, 0.0)
                redundant_columns.add(left if left_mi < right_mi else right)

    excluded_features: dict[str, str] = {}
    for col in low_variance_columns:
        excluded_features[col] = "low_variance"
    for col, score in mi_map.items():
        if score <= noise_mi_baseline:
            reason = "below_noise_baseline"
            excluded_features[col] = f"{excluded_features[col]};{reason}" if col in excluded_features else reason
    for col in sorted(redundant_columns):
        reason = "redundant"
        excluded_features[col] = f"{excluded_features[col]};{reason}" if col in excluded_features else reason

    recommended_features = [c for c in feature_columns if c not in excluded_features]

    notes: list[str] = []
    if not recommended_features:
        relaxed = noise_mi_baseline * 0.5
        notes.append(
            "All features were filtered. Relaxed MI noise threshold by 50% per guardrail."
        )
        excluded_features = {}
        for col in low_variance_columns:
            excluded_features[col] = "low_variance"
        for col in sorted(redundant_columns):
            reason = "redundant"
            excluded_features[col] = f"{excluded_features[col]};{reason}" if col in excluded_features else reason
        for col, score in mi_map.items():
            if score <= relaxed:
                reason = "below_noise_baseline_relaxed"
                excluded_features[col] = (
                    f"{excluded_features[col]};{reason}" if col in excluded_features else reason
                )
        recommended_features = [c for c in feature_columns if c not in excluded_features]

    if not recommended_features:
        sorted_by_mi = sorted(feature_columns, key=lambda c: mi_map.get(c, 0.0), reverse=True)
        recommended_features = sorted_by_mi[: min(3, len(sorted_by_mi))]
        notes.append("Fallback activated: selected top-MI features to keep recommended_features non-empty.")

    mi_ranking = [
        {
            "feature": feature,
            "mi_score": round(mi_map[feature], 8),
            "below_noise_baseline": mi_map[feature] <= noise_mi_baseline,
            "scaled_variance": round(scaled_variances.get(feature, 0.0), 8),
        }
        for feature in sorted(feature_columns, key=lambda c: mi_map[c], reverse=True)
    ]

    time_series_detected = bool(time_column and time_column in df.columns)
    significant_lags: list[int] = []
    useful_lag_features: list[dict[str, Any]] = []

    if time_series_detected:
        working = df
        if working.schema[time_column] not in (pl.Date, pl.Datetime):
            if working.schema[time_column] == pl.Utf8:
                working = working.with_columns(
                    pl.col(time_column).str.strptime(pl.Datetime, strict=False, exact=False)
                )
        if working.schema[time_column] in (pl.Date, pl.Datetime):
            working = working.sort(time_column)

        y_sorted = _to_filled_float_array(working, target)
        max_lag = min(24, max(1, len(y_sorted) // 4))
        for lag in range(1, max_lag + 1):
            if lag >= len(y_sorted):
                break
            ac = _safe_corr(y_sorted[lag:], y_sorted[:-lag])
            if abs(ac) > 0.1:
                significant_lags.append(lag)

        for feature in recommended_features:
            x_sorted = _to_filled_float_array(working, feature)
            for lag in range(0, 4):
                if lag == 0:
                    xcorr = _safe_corr(x_sorted, y_sorted)
                else:
                    if lag >= len(y_sorted):
                        continue
                    xcorr = _safe_corr(x_sorted[:-lag], y_sorted[lag:])
                if abs(xcorr) > 0.15:
                    useful_lag_features.append(
                        {
                            "feature": feature,
                            "lag": lag,
                            "xcorr": round(float(xcorr), 6),
                        }
                    )

    high_cardinality = [
        col
        for col, dtype in df.schema.items()
        if not _is_numeric(dtype)
        and float(df[col].n_unique()) / max(row_count, 1) > 0.5
    ]

    target_candidates = _compute_target_candidates(df, numeric_columns)

    context = dict(step01.get("context", {}))
    context["features"] = recommended_features
    context["time_column"] = time_column
    context.setdefault("notes", [])
    context["notes"].extend(notes)
    context.setdefault("artifacts", {})
    context["artifacts"]["step_11_json"] = str((output_dir / STEP11_JSON).resolve())

    payload = {
        "step": STEP_NAME,
        "shape": {"rows": row_count, "columns": df.width},
        "numeric_columns": numeric_columns,
        "high_cardinality": high_cardinality,
        "low_variance_columns": sorted(low_variance_columns),
        "mi_ranking": mi_ranking,
        "noise_mi_baseline": noise_mi_baseline,
        "redundant_columns": sorted(redundant_columns),
        "correlation_matrix_summary": {
            "max_pair": max_pair,
            "max_corr": round(float(max_corr), 6),
        },
        "significant_lags": significant_lags,
        "useful_lag_features": useful_lag_features,
        "recommended_features": recommended_features,
        "excluded_features": excluded_features,
        "target_candidates": target_candidates,
        "time_series_detected": time_series_detected,
        "time_column": time_column,
        "filter_drop_counts": {
            "low_variance": len(low_variance_columns),
            "below_noise_baseline": sum(
                1 for item in mi_ranking if item["below_noise_baseline"]
            ),
            "redundant": len(redundant_columns),
        },
        "context": context,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 11 data exploration")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    step01 = _load_json(output_dir / STEP01_JSON)
    csv_path = step01.get("csv_path", "")
    target_column = step01.get("target_column_normalized", "")

    _update_progress(
        output_dir,
        run_id=args.run_id,
        csv_path=csv_path,
        target_column=target_column,
        status="running",
        current_step=STEP_NAME,
    )

    try:
        payload = run_step(output_dir, args.run_id)
        _write_json(output_dir / STEP11_JSON, payload)

        _update_progress(
            output_dir,
            run_id=args.run_id,
            csv_path=csv_path,
            target_column=target_column,
            status="running",
            current_step=STEP_NAME,
            mark_completed=True,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        err = f"{exc}\n{traceback.format_exc()}"
        _update_progress(
            output_dir,
            run_id=args.run_id,
            csv_path=csv_path,
            target_column=target_column,
            status="failed",
            current_step=STEP_NAME,
            error=err,
        )
        print(err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
