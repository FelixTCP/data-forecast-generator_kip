#!/usr/bin/env python3
"""Step 12: leakage-safe feature extraction."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

STEP_NAME = "12-feature-extraction"
STEP01_JSON = "step-01-cleanse.json"
STEP11_JSON = "step-11-exploration.json"
STEP12_JSON = "step-12-features.json"
LEAKAGE_AUDIT_JSON = "leakage_audit.json"
FEATURES_PARQUET = "features.parquet"
PROGRESS_FILE = "progress.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


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


def _normalize_name(name: str) -> str:
    return "_".join(name.strip().lower().split())


def _resolve_split_mode(requested: str, time_column: str | None) -> str:
    if requested == "auto":
        return "time_series" if time_column else "random"
    if requested not in {"time_series", "random"}:
        raise ValueError(f"Invalid split mode: {requested}")
    return requested


def _to_numpy_filled(df: pl.DataFrame, col: str) -> np.ndarray:
    arr = df[col].cast(pl.Float64).to_numpy()
    if arr.size == 0:
        return arr
    if np.isnan(arr).all():
        return np.zeros_like(arr)
    med = float(np.nanmedian(arr))
    return np.where(np.isnan(arr), med, arr)


def _run_linear_probe(
    frame: pl.DataFrame,
    target: str,
    candidate_features: list[str],
    split_mode: str,
) -> dict[str, Any]:
    n = frame.height
    if n < 20:
        return {"status": "skip", "reason": "insufficient_rows_for_probe"}

    indices = np.arange(n)
    if split_mode == "time_series":
        split_idx = max(int(n * 0.8), 1)
        split_idx = min(split_idx, n - 1)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
    else:
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

    y = _to_numpy_filled(frame, target)
    diagnostics: list[dict[str, Any]] = []

    single_sets = [[f] for f in candidate_features[:10]]
    pair_sets = [
        list(combo)
        for combo in itertools.combinations(candidate_features[:6], 2)
    ]
    triple_sets = [
        list(combo)
        for combo in itertools.combinations(candidate_features[:5], 3)
    ]

    for subset in single_sets + pair_sets + triple_sets:
        X = frame.select(subset).to_numpy()
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = float(r2_score(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        record = {
            "subset": subset,
            "r2": round(r2, 8),
            "rmse": round(rmse, 8),
        }
        diagnostics.append(record)

        if r2 > 0.995 or rmse < 1e-6:
            return {
                "status": "fail",
                "trigger": "linear_probe_reconstruction",
                "record": record,
                "diagnostics": diagnostics,
            }

    return {
        "status": "pass",
        "trigger": None,
        "diagnostics": diagnostics,
    }


def run_step(output_dir: Path, run_id: str, target_column_arg: str | None, split_mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    step01 = _load_json(output_dir / STEP01_JSON)
    step11 = _load_json(output_dir / STEP11_JSON)

    if not step01:
        raise FileNotFoundError("Missing step-01-cleanse.json.")
    if not step11:
        raise FileNotFoundError("Missing step-11-exploration.json.")

    cleaned_path = Path(step01["artifacts"]["cleaned_parquet"])
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Missing cleaned parquet at {cleaned_path}")

    target = step01["target_column_normalized"]
    if target_column_arg is not None:
        normalized_arg = _normalize_name(target_column_arg)
        if normalized_arg != target:
            raise ValueError(
                f"Target mismatch: step01 target='{target}', arg target='{normalized_arg}'"
            )

    time_column = step11.get("time_column") or step01.get("time_column")
    resolved_mode = _resolve_split_mode(split_mode, time_column)

    recommended = step11.get("recommended_features")
    if not isinstance(recommended, list) or not recommended:
        raise ValueError(
            "step-11-exploration.json has empty or missing recommended_features. Fix step 11 before step 12."
        )

    excluded_features: dict[str, str] = dict(step11.get("excluded_features", {}))

    df = pl.read_parquet(cleaned_path)
    if time_column and time_column in df.columns:
        dtype = df.schema[time_column]
        if dtype not in (pl.Date, pl.Datetime) and dtype == pl.Utf8:
            df = df.with_columns(
                pl.col(time_column).str.strptime(pl.Datetime, strict=False, exact=False)
            )
        if df.schema[time_column] in (pl.Date, pl.Datetime):
            df = df.sort(time_column)

    selected_base = [c for c in recommended if c in df.columns and c != target]
    missing_recommended = [c for c in recommended if c not in df.columns]
    for col in missing_recommended:
        excluded_features[col] = "missing_in_cleaned_data"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not present in cleaned parquet.")

    select_cols = list(dict.fromkeys(selected_base + [target] + ([time_column] if time_column and time_column in df.columns else [])))
    out = df.select(select_cols)

    created_features: list[dict[str, Any]] = []
    expressions: list[pl.Expr] = []

    if time_column and time_column in out.columns and out.schema[time_column] in (pl.Date, pl.Datetime):
        expressions.extend(
            [
                pl.col(time_column).dt.year().alias("year"),
                pl.col(time_column).dt.month().alias("month"),
                pl.col(time_column).dt.weekday().alias("day_of_week"),
            ]
        )
        if out.schema[time_column] == pl.Datetime:
            expressions.append(pl.col(time_column).dt.hour().alias("hour"))
        else:
            expressions.append(pl.lit(0).cast(pl.Int64).alias("hour"))

        created_features.extend(
            [
                {"name": "year", "reason": "time decomposition"},
                {"name": "month", "reason": "time decomposition"},
                {"name": "day_of_week", "reason": "time decomposition"},
                {"name": "hour", "reason": "time decomposition"},
            ]
        )

    useful_lag_features = step11.get("useful_lag_features", [])
    for item in useful_lag_features:
        feature = item.get("feature")
        lag = int(item.get("lag", 0))
        if feature not in selected_base:
            continue
        if lag <= 0:
            continue
        name = f"{feature}_lag_{lag}"
        expressions.append(pl.col(feature).shift(lag).alias(name))
        created_features.append(
            {
                "name": name,
                "reason": f"useful_lag_feature: {feature} lag={lag} xcorr={item.get('xcorr')}",
            }
        )

    significant_lags = sorted({int(x) for x in step11.get("significant_lags", []) if int(x) >= 1})
    for lag in significant_lags:
        name = f"target_lag_{lag}"
        expressions.append(pl.col(target).shift(lag).alias(name))
        created_features.append(
            {
                "name": name,
                "reason": f"significant_lag={lag} from step 11",
            }
        )

    if significant_lags:
        top_windows = significant_lags[:2]
        for window in top_windows:
            w = max(int(window), 1)
            mean_name = f"target_shift1_roll_mean_w{w}"
            expressions.append(
                pl.col(target)
                .shift(1)
                .rolling_mean(window_size=w, min_samples=w)
                .alias(mean_name)
            )
            created_features.append(
                {
                    "name": mean_name,
                    "reason": f"causal rolling mean with window={w}",
                }
            )
            if w >= 2:
                std_name = f"target_shift1_roll_std_w{w}"
                expressions.append(
                    pl.col(target)
                    .shift(1)
                    .rolling_std(window_size=w, min_samples=w)
                    .alias(std_name)
                )
                created_features.append(
                    {
                        "name": std_name,
                        "reason": f"causal rolling std with window={w}",
                    }
                )

    if expressions:
        out = out.with_columns(expressions)

    rows_before_drop = out.height
    out = out.drop_nulls()
    rows_dropped_by_lag = rows_before_drop - out.height
    if out.height == 0:
        raise ValueError(
            "All rows were dropped after lag/rolling feature construction. Check engineered feature null patterns."
        )

    non_numeric_columns = [
        col
        for col, dtype in out.schema.items()
        if dtype not in pl.NUMERIC_DTYPES
    ]

    for col in non_numeric_columns:
        if col != time_column:
            excluded_features[col] = "non_numeric_after_construction"

    numeric_cols = [
        col
        for col, dtype in out.schema.items()
        if dtype in pl.NUMERIC_DTYPES
    ]
    out = out.select(numeric_cols)

    if target not in out.columns:
        raise ValueError("Target column dropped during numeric filtering.")

    features = [c for c in out.columns if c != target]
    if not features:
        raise ValueError("No features survived construction.")

    banned_from_step11 = set(step11.get("excluded_features", {}).keys())
    leaked_back = [f for f in features if f in banned_from_step11]
    if leaked_back:
        raise ValueError(f"Leakage guard triggered: excluded step11 features reintroduced: {leaked_back}")

    leakage_flags: list[dict[str, Any]] = []
    keep_features: list[str] = []
    y = _to_numpy_filled(out, target)
    for feature in features:
        x = _to_numpy_filled(out, feature)
        corr = abs(_safe_corr(x, y))
        if corr > 0.99:
            leakage_flags.append(
                {
                    "feature": feature,
                    "check": "pairwise_corr",
                    "corr_with_target": round(corr, 8),
                    "action": "excluded",
                }
            )
            excluded_features[feature] = "leakage_pairwise_corr_gt_0.99"
        else:
            keep_features.append(feature)

    forbidden_target_rolling = [f for f in keep_features if f.startswith("target_rolling_")]
    for feature in forbidden_target_rolling:
        leakage_flags.append(
            {
                "feature": feature,
                "check": "target_rolling_name_guard",
                "action": "excluded",
            }
        )
        excluded_features[feature] = "forbidden_unverified_target_rolling_feature"
        keep_features.remove(feature)

    if len(keep_features) < 2:
        raise ValueError(
            "Fewer than 2 features survived filters and leakage checks. Revisit step 11 thresholds."
        )

    out = out.select(keep_features + [target])

    target_derived = [f for f in keep_features if f.startswith("target_")]
    leakage_probe = {
        "status": "pass",
        "checks": ["pairwise_corr", "linear_probe_reconstruction"],
        "details": [],
    }

    if target_derived:
        probe_result = _run_linear_probe(out, target, target_derived, resolved_mode)
        if probe_result["status"] == "fail":
            leakage_probe["status"] = "fail"
            leakage_probe["details"] = [probe_result]
        else:
            leakage_probe["details"] = probe_result.get("diagnostics", [])[:10]

    leakage_audit = {
        "step": STEP_NAME,
        "run_id": run_id,
        "status": leakage_probe["status"],
        "checks": leakage_probe["checks"],
        "pairwise_flags": leakage_flags,
        "details": leakage_probe["details"],
    }

    if leakage_probe["status"] != "pass":
        return {}, leakage_audit

    features_path = (output_dir / FEATURES_PARQUET).resolve()
    out.write_parquet(features_path)

    context = dict(step11.get("context", {}))
    context["features"] = keep_features
    context.setdefault("split_strategy", {})
    context["split_strategy"] = {
        "requested_mode": split_mode,
        "resolved_mode": resolved_mode,
        "time_column": time_column,
        "random_state": 42,
    }
    context.setdefault("artifacts", {})
    context["artifacts"]["features_parquet"] = str(features_path)
    context.setdefault("notes", [])
    if not significant_lags:
        context["notes"].append("No lag features created - no significant autocorrelation detected.")

    payload = {
        "step": STEP_NAME,
        "features": keep_features,
        "features_excluded": excluded_features,
        "created_features": created_features,
        "rows_dropped_by_lag": rows_dropped_by_lag,
        "leakage_flags": leakage_flags,
        "leakage_audit": {
            "status": leakage_probe["status"],
            "checks": leakage_probe["checks"],
            "details": leakage_probe["details"],
        },
        "split_strategy": {
            "requested_mode": split_mode,
            "resolved_mode": resolved_mode,
            "time_column": time_column,
            "random_state": 42,
        },
        "artifacts": {
            "features_parquet": str(features_path),
            "leakage_audit": str((output_dir / LEAKAGE_AUDIT_JSON).resolve()),
            "step_12_json": str((output_dir / STEP12_JSON).resolve()),
        },
        "context": context,
    }

    return payload, leakage_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 12 feature extraction")
    parser.add_argument("--target-column", required=False)
    parser.add_argument("--split-mode", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    step01 = _load_json(output_dir / STEP01_JSON)
    csv_path = step01.get("csv_path", "")
    target = step01.get("target_column_normalized", "")

    _update_progress(
        output_dir,
        run_id=args.run_id,
        csv_path=csv_path,
        target_column=target,
        status="running",
        current_step=STEP_NAME,
    )

    try:
        payload, leakage_audit = run_step(
            output_dir=output_dir,
            run_id=args.run_id,
            target_column_arg=args.target_column,
            split_mode=args.split_mode,
        )
        _write_json(output_dir / LEAKAGE_AUDIT_JSON, leakage_audit)

        if not payload:
            raise RuntimeError(
                "Leakage suspected during feature extraction. See leakage_audit.json for diagnostics."
            )

        _write_json(output_dir / STEP12_JSON, payload)

        _update_progress(
            output_dir,
            run_id=args.run_id,
            csv_path=csv_path,
            target_column=target,
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
            target_column=target,
            status="failed",
            current_step=STEP_NAME,
            error=err,
        )
        print(err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
