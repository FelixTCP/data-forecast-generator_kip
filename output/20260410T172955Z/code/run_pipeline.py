from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

RANDOM_STATE = 42


@dataclass
class PipelineContext:
    dataset_id: str
    target_column: str
    time_column: str | None
    features: list[str]
    split_strategy: dict[str, Any]
    model_candidates: list[dict[str, Any]]
    metrics: dict[str, float]
    artifacts: dict[str, str]
    notes: list[str] = field(default_factory=list)


def normalize_name(name: str) -> str:
    return "_".join(name.strip().lower().split())


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
    status: str,
    current_step: str,
    completed_steps: list[str],
    errors: list[str],
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


def detect_time_column(df: pl.DataFrame, target_column: str) -> str | None:
    for col, dtype in zip(df.columns, df.dtypes, strict=False):
        if col == target_column:
            continue
        if dtype in {pl.Date, pl.Datetime, pl.Time}:
            return col
    for col in df.columns:
        if col == target_column:
            continue
        lower_col = col.lower()
        if "date" in lower_col or "time" in lower_col or "timestamp" in lower_col:
            return col
    return None


def row_hashes_for_code(code_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for py_file in sorted(code_dir.rglob("*.py")):
        digest = hashlib.sha256(py_file.read_bytes()).hexdigest()
        hashes[str(py_file.relative_to(code_dir))] = digest
    return hashes


def step_10_cleanse(
    csv_path: Path,
    output_dir: Path,
    context: PipelineContext,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    try:
        df_raw = pl.read_csv(csv_path, try_parse_dates=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV '{csv_path}': {exc}") from exc

    original_columns = df_raw.columns
    normalized_columns = [normalize_name(c) for c in original_columns]
    rename_map = dict(zip(original_columns, normalized_columns, strict=True))

    fixes: list[dict[str, Any]] = []
    if original_columns != normalized_columns:
        fixes.append(
            {
                "type": "normalized_column_names",
                "before_after": rename_map,
            }
        )

    df = df_raw.rename(rename_map)

    string_types = {t for t in (getattr(pl, "String", None), getattr(pl, "Utf8", None)) if t is not None}
    for col in tqdm(df.columns, desc="Step10: type coercion", unit="column"):
        dtype = df.schema[col]
        if dtype not in string_types:
            continue

        trimmed_expr = pl.col(col).str.strip_chars()
        non_empty_count = int(
            df.select(
                trimmed_expr.is_not_null().and_(trimmed_expr != "").sum()
            ).item()
        )
        if non_empty_count == 0:
            continue

        numeric_success = int(
            df.select(
                trimmed_expr.cast(pl.Float64, strict=False).is_not_null().sum()
            ).item()
        )
        numeric_ratio = numeric_success / non_empty_count
        if numeric_ratio >= 0.95:
            df = df.with_columns(trimmed_expr.cast(pl.Float64, strict=False).alias(col))
            fixes.append(
                {
                    "type": "coerced_numeric_string_column",
                    "column": col,
                    "success_ratio": numeric_ratio,
                }
            )
            continue

        if any(token in col for token in ("date", "time", "timestamp")):
            dt_success = int(
                df.select(
                    trimmed_expr.str.strptime(pl.Datetime, strict=False).is_not_null().sum()
                ).item()
            )
            dt_ratio = dt_success / non_empty_count
            if dt_ratio >= 0.95:
                df = df.with_columns(
                    trimmed_expr.str.strptime(pl.Datetime, strict=False).alias(col)
                )
                fixes.append(
                    {
                        "type": "coerced_datetime_string_column",
                        "column": col,
                        "success_ratio": dt_ratio,
                    }
                )

    null_rate: dict[str, float] = {}
    dtype_map: dict[str, str] = {}
    for col in tqdm(df.columns, desc="Step10: profiling columns", unit="column"):
        null_rate[col] = float(df.select(pl.col(col).is_null().mean()).item())
        dtype_map[col] = str(df.schema[col])

    duplicate_rows = int(df.is_duplicated().sum())

    clean_path = output_dir / "cleaned.parquet"
    df.write_parquet(clean_path)

    context.time_column = detect_time_column(df, context.target_column)
    context.artifacts["cleaned_parquet"] = str(clean_path)

    step_output = {
        "step": "10-csv-read-cleansing",
        "row_count_before": int(df_raw.height),
        "row_count_after": int(df.height),
        "column_count": int(df.width),
        "target_column_normalized": context.target_column,
        "time_column_detected": context.time_column,
        "null_rate": null_rate,
        "inferred_dtypes": dtype_map,
        "duplicate_rows": duplicate_rows,
        "applied_fixes": fixes,
        "context": asdict(context),
    }
    write_json(output_dir / "step-10-cleanse.json", step_output)
    return df, step_output


def step_11_exploration(
    df: pl.DataFrame,
    output_dir: Path,
    context: PipelineContext,
) -> dict[str, Any]:
    numeric_columns = [
        col for col, dtype in zip(df.columns, df.dtypes, strict=False) if is_numeric_dtype(dtype)
    ]

    cardinality: dict[str, int] = {}
    for col in tqdm(df.columns, desc="Step11: cardinality scan", unit="column"):
        cardinality[col] = int(df.select(pl.col(col).n_unique()).item())

    high_cardinality = [
        c for c in df.columns if cardinality[c] > max(100, int(0.5 * max(1, df.height)))
    ]

    numeric_summary: dict[str, dict[str, float | None]] = {}
    for col in tqdm(numeric_columns, desc="Step11: numeric summary", unit="column"):
        s = df.select(
            [
                pl.col(col).mean().alias("mean"),
                pl.col(col).std().alias("std"),
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
            ]
        ).to_dicts()[0]
        numeric_summary[col] = {
            key: (None if value is None else float(value)) for key, value in s.items()
        }

    correlation_preview: list[dict[str, float | str]] = []
    for left, right in tqdm(
        list(combinations(numeric_columns[:8], 2)),
        desc="Step11: correlation preview",
        unit="pair",
    ):
        corr = df.select(pl.corr(pl.col(left), pl.col(right))).item()
        if corr is not None:
            correlation_preview.append(
                {"left": left, "right": right, "pearson_corr": float(corr)}
            )

    target_candidates = sorted(
        [
            {
                "column": col,
                "null_rate": float(df.select(pl.col(col).is_null().mean()).item()),
                "std": float(df.select(pl.col(col).std()).item() or 0.0),
            }
            for col in numeric_columns
        ],
        key=lambda x: (x["null_rate"], -x["std"]),
    )[:5]

    time_series_detected = context.time_column is not None

    step_output = {
        "step": "11-data-exploration",
        "shape": {"rows": int(df.height), "columns": int(df.width)},
        "numeric_columns": numeric_columns,
        "high_cardinality": high_cardinality,
        "cardinality": cardinality,
        "numeric_summary": numeric_summary,
        "correlation_preview": correlation_preview,
        "target_candidates": target_candidates,
        "time_series_detected": time_series_detected,
        "time_column": context.time_column,
        "context": asdict(context),
    }
    write_json(output_dir / "step-11-exploration.json", step_output)
    return step_output


def step_12_features(
    df: pl.DataFrame,
    split_mode: str,
    output_dir: Path,
    context: PipelineContext,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    target = context.target_column
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found after normalization: {df.columns}")

    created_features: list[dict[str, str]] = []

    df_features = df
    if context.time_column and context.time_column in df_features.columns:
        df_features = df_features.sort(context.time_column)
        time_col = pl.col(context.time_column)
        date_parts = [
            time_col.dt.year().alias(f"{context.time_column}_year"),
            time_col.dt.month().alias(f"{context.time_column}_month"),
            time_col.dt.day().alias(f"{context.time_column}_day"),
            time_col.dt.weekday().alias(f"{context.time_column}_weekday"),
            time_col.dt.ordinal_day().alias(f"{context.time_column}_ordinal_day"),
        ]
        df_features = df_features.with_columns(date_parts)
        created_features.extend(
            [
                {
                    "feature": f"{context.time_column}_{suffix}",
                    "reason": "Date decomposition for seasonal signal",
                }
                for suffix in ["year", "month", "day", "weekday", "ordinal_day"]
            ]
        )

        base_numeric = [
            c
            for c, dt in zip(df_features.columns, df_features.dtypes, strict=False)
            if c not in {target, context.time_column} and is_numeric_dtype(dt)
        ]
        lag_cols = base_numeric[: min(5, len(base_numeric))]
        lag_expressions = []
        for c in lag_cols:
            lag_expressions.append(pl.col(c).shift(1).alias(f"{c}_lag1"))
            lag_expressions.append(pl.col(c).rolling_mean(window_size=3).alias(f"{c}_rollmean3"))
            created_features.append(
                {
                    "feature": f"{c}_lag1",
                    "reason": "Short-term temporal dependency",
                }
            )
            created_features.append(
                {
                    "feature": f"{c}_rollmean3",
                    "reason": "Local trend smoothing",
                }
            )

        if lag_expressions:
            df_features = df_features.with_columns(lag_expressions)

        # Drop raw datetime column so downstream sklearn receives numeric matrix only.
        df_features = df_features.drop(context.time_column)

    feature_columns = [c for c in df_features.columns if c != target]
    feature_columns = [
        c
        for c, dt in zip(feature_columns, [df_features.schema[c] for c in feature_columns], strict=False)
        if is_numeric_dtype(dt)
    ]

    split_strategy = {
        "requested_mode": split_mode,
        "resolved_mode": "time_series"
        if split_mode == "auto" and context.time_column is not None
        else ("time_series" if split_mode == "time_series" else "random"),
        "time_column": context.time_column,
        "random_state": RANDOM_STATE,
    }

    context.features = feature_columns
    context.split_strategy = split_strategy

    features_path = output_dir / "features.parquet"
    df_features.write_parquet(features_path)
    context.artifacts["features_parquet"] = str(features_path)

    step_output = {
        "step": "12-feature-extraction",
        "target_column": target,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "created_features": created_features,
        "split_strategy": split_strategy,
        "context": asdict(context),
    }
    write_json(output_dir / "step-12-features.json", step_output)
    return df_features, step_output


def make_candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "ridge",
            "complexity": 1,
            "estimator": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        },
        {
            "name": "gradient_boosting",
            "complexity": 3,
            "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
        },
        {
            "name": "random_forest",
            "complexity": 4,
            "estimator": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        },
    ]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def step_13_training(
    df_features: pl.DataFrame,
    output_dir: Path,
    context: PipelineContext,
) -> dict[str, Any]:
    target = context.target_column
    feature_columns = context.features

    if not feature_columns:
        raise ValueError("No usable numeric feature columns found after feature extraction.")

    df_features = df_features.with_columns(
        pl.col(target).cast(pl.Float64, strict=False).alias(target)
    ).filter(pl.col(target).is_not_null())

    if df_features.height < 20:
        raise ValueError("Not enough rows with valid target values for training.")

    mode = context.split_strategy["resolved_mode"]
    if mode == "time_series" and context.time_column:
        split_idx = int(df_features.height * 0.8)
        train_df = df_features.slice(0, split_idx)
        test_df = df_features.slice(split_idx, df_features.height - split_idx)
    else:
        idx = np.arange(df_features.height)
        train_idx, test_idx = train_test_split(
            idx,
            test_size=0.2,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
        train_df = df_features.take(pl.Series(train_idx))
        test_df = df_features.take(pl.Series(test_idx))

    if train_df.height == 0 or test_df.height == 0:
        idx = np.arange(df_features.height)
        train_idx, test_idx = train_test_split(
            idx,
            test_size=0.2,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
        train_df = df_features.take(pl.Series(train_idx))
        test_df = df_features.take(pl.Series(test_idx))

    numeric_feature_exprs = [
        pl.col(c).cast(pl.Float64, strict=False).fill_null(float("nan")).alias(c)
        for c in feature_columns
    ]
    x_train = train_df.select(numeric_feature_exprs).to_numpy().astype(np.float64)
    x_test = test_df.select(numeric_feature_exprs).to_numpy().astype(np.float64)
    y_train = (
        train_df.select(pl.col(target).cast(pl.Float64, strict=False).fill_null(float("nan")))
        .to_numpy()
        .reshape(-1)
        .astype(np.float64)
    )
    y_test = (
        test_df.select(pl.col(target).cast(pl.Float64, strict=False).fill_null(float("nan")))
        .to_numpy()
        .reshape(-1)
        .astype(np.float64)
    )

    train_mask = np.isfinite(y_train)
    test_mask = np.isfinite(y_test)
    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    if x_test.shape[0] == 0:
        raise ValueError("Holdout set became empty after target validation.")

    holdout_path = output_dir / "holdout.npz"
    np.savez(holdout_path, x_test=x_test, y_test=y_test)
    context.artifacts["holdout_npz"] = str(holdout_path)

    candidates = make_candidates()
    context.model_candidates = [
        {"name": c["name"], "complexity": c["complexity"]} for c in candidates
    ]

    if mode == "time_series":
        n_splits = max(2, min(5, len(y_train) // 50))
        cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    records: list[dict[str, Any]] = []
    candidate_model_paths: dict[str, str] = {}

    for candidate in tqdm(candidates, desc="Step13: model training", unit="model"):
        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", candidate["estimator"]),
            ]
        )
        pipe.fit(x_train, y_train)

        pred = pipe.predict(x_test)
        model_r2 = float(r2_score(y_test, pred))
        model_rmse = rmse(y_test, pred)
        model_mae = float(mean_absolute_error(y_test, pred))

        cv_scores = cross_val_score(pipe, x_train, y_train, cv=cv, scoring="r2", n_jobs=1)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        model_path = output_dir / f"candidate-{candidate['name']}.joblib"
        joblib.dump(pipe, model_path)
        candidate_model_paths[candidate["name"]] = str(model_path)

        records.append(
            {
                "model_name": candidate["name"],
                "complexity": candidate["complexity"],
                "params": candidate["estimator"].get_params(deep=False),
                "holdout_metrics": {
                    "r2": model_r2,
                    "rmse": model_rmse,
                    "mae": model_mae,
                },
                "cv_scores_r2": [float(v) for v in cv_scores],
                "cv_mean_r2": cv_mean,
                "cv_std_r2": cv_std,
                "artifact_path": str(model_path),
            }
        )

    best_for_artifact = max(records, key=lambda r: r["cv_mean_r2"])
    best_model_path = Path(candidate_model_paths[best_for_artifact["model_name"]])
    final_model = joblib.load(best_model_path)
    final_model_path = output_dir / "model.joblib"
    joblib.dump(final_model, final_model_path)
    context.artifacts["model_joblib"] = str(final_model_path)

    context.metrics = {
        "best_cv_mean_r2": float(best_for_artifact["cv_mean_r2"]),
        "best_holdout_r2": float(best_for_artifact["holdout_metrics"]["r2"]),
    }

    step_output = {
        "step": "13-model-training",
        "split_mode": mode,
        "train_rows": int(train_df.height),
        "test_rows": int(test_df.height),
        "feature_count": len(feature_columns),
        "records": records,
        "best_model_for_artifact": best_for_artifact["model_name"],
        "model_artifact": str(final_model_path),
        "context": asdict(context),
    }
    write_json(output_dir / "step-13-training.json", step_output)
    return step_output


def step_14_evaluation(output_dir: Path, context: PipelineContext) -> dict[str, Any]:
    training = read_json(output_dir / "step-13-training.json")
    holdout = np.load(output_dir / "holdout.npz", allow_pickle=True)
    x_test = holdout["x_test"]
    y_test = holdout["y_test"]

    candidate_reports: list[dict[str, Any]] = []
    for rec in tqdm(training["records"], desc="Step14: model evaluation", unit="model"):
        model = joblib.load(rec["artifact_path"])
        pred = model.predict(x_test)
        residuals = y_test - pred

        candidate_reports.append(
            {
                "model_name": rec["model_name"],
                "complexity": rec["complexity"],
                "metrics": {
                    "r2": float(r2_score(y_test, pred)),
                    "rmse": rmse(y_test, pred),
                    "mae": float(mean_absolute_error(y_test, pred)),
                },
                "cv_scores_r2": rec["cv_scores_r2"],
                "cv_mean_r2": rec["cv_mean_r2"],
                "cv_std_r2": rec["cv_std_r2"],
                "residual_summary": {
                    "mean": float(np.mean(residuals)),
                    "std": float(np.std(residuals)),
                    "p05": float(np.quantile(residuals, 0.05)),
                    "p95": float(np.quantile(residuals, 0.95)),
                },
                "residual_notes": [
                    "Residual center near zero is preferred.",
                    "Wide residual spread indicates unstable predictions on holdout.",
                ],
                "artifact_path": rec["artifact_path"],
            }
        )

    step_output = {
        "step": "14-model-evaluation",
        "candidate_reports": candidate_reports,
        "context": asdict(context),
    }
    write_json(output_dir / "step-14-evaluation.json", step_output)
    return step_output


def normalize(values: list[float], invert: bool = False) -> list[float]:
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-12:
        return [1.0 for _ in values]
    scaled = [(v - vmin) / (vmax - vmin) for v in values]
    return [1.0 - x for x in scaled] if invert else scaled


def step_15_selection(output_dir: Path, context: PipelineContext) -> dict[str, Any]:
    evaluation = read_json(output_dir / "step-14-evaluation.json")
    reports = evaluation["candidate_reports"]

    r2_vals = [float(r["metrics"]["r2"]) for r in reports]
    rmse_vals = [float(r["metrics"]["rmse"]) for r in reports]
    mae_vals = [float(r["metrics"]["mae"]) for r in reports]
    stability_vals = [max(0.0, 1.0 - float(r["cv_std_r2"])) for r in reports]

    r2_n = normalize(r2_vals)
    rmse_n = normalize(rmse_vals, invert=True)
    mae_n = normalize(mae_vals, invert=True)
    stability_n = normalize(stability_vals)

    ranking: list[dict[str, Any]] = []
    for idx, report in enumerate(reports):
        weighted = 0.5 * r2_n[idx] + 0.25 * rmse_n[idx] + 0.15 * mae_n[idx] + 0.1 * stability_n[idx]
        ranking.append(
            {
                "model_name": report["model_name"],
                "complexity": int(report["complexity"]),
                "weighted_score": float(weighted),
                "metrics": report["metrics"],
                "cv_mean_r2": float(report["cv_mean_r2"]),
                "cv_std_r2": float(report["cv_std_r2"]),
                "artifact_path": report["artifact_path"],
            }
        )

    ranking.sort(key=lambda r: (-r["weighted_score"], r["complexity"]))
    selected = ranking[0]

    step_output = {
        "step": "15-model-selection",
        "selection_rule": {
            "r2_weight": 0.5,
            "rmse_weight": 0.25,
            "mae_weight": 0.15,
            "stability_weight": 0.1,
            "tie_breaker": "lower complexity",
        },
        "ranking": ranking,
        "selected_model": selected,
        "rationale": [
            "Model selected using weighted normalized score across R2/RMSE/MAE/stability.",
            "Tie-breaker favors lower complexity for similar weighted score.",
        ],
        "context": asdict(context),
    }
    write_json(output_dir / "step-15-selection.json", step_output)
    return step_output


def step_16_report(output_dir: Path, context: PipelineContext) -> Path:
    cleanse = read_json(output_dir / "step-10-cleanse.json")
    evaluation = read_json(output_dir / "step-14-evaluation.json")
    selection = read_json(output_dir / "step-15-selection.json")

    selected = selection["selected_model"]

    candidates_block = "\n".join(
        [
            f"- {r['model_name']}: R2={r['metrics']['r2']:.4f}, RMSE={r['metrics']['rmse']:.4f}, MAE={r['metrics']['mae']:.4f}, CV_STD={r['cv_std_r2']:.4f}"
            for r in evaluation["candidate_reports"]
        ]
    )

    report_text = (
        f"# Forecasting Run Report ({datetime.now(UTC).isoformat()})\n\n"
        "## 1. Problem + selected target\n"
        f"Regression forecasting for target column `{context.target_column}` using CSV `{context.dataset_id}`.\n\n"
        "## 2. Data quality summary\n"
        f"Rows: {cleanse['row_count_after']}, Columns: {cleanse['column_count']}, "
        f"Duplicate rows: {cleanse['duplicate_rows']}.\n"
        f"Detected time column: {cleanse['time_column_detected']}.\n\n"
        "## 3. Candidate models + scores\n"
        f"{candidates_block}\n\n"
        "## 4. Selected model rationale\n"
        f"Selected model: {selected['model_name']} (weighted_score={selected['weighted_score']:.4f}).\n"
        "Selection used weighted normalized ranking with complexity tie-break.\n\n"
        "## 5. Risks and caveats\n"
        "- Holdout performance may shift on future data drift.\n"
        "- If temporal ordering changes in source systems, retraining is required.\n"
        "- Residual variance suggests periodic recalibration checks.\n\n"
        "## 6. Next iteration recommendations\n"
        "- Add domain-specific lag windows and holiday/event flags.\n"
        "- Benchmark additional regularized and boosting variants.\n"
        "- Add backtesting slices for stronger temporal robustness checks.\n"
    )

    report_path = output_dir / "step-16-report.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agentic pipeline end-to-end")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split-mode", default="auto", choices=["auto", "random", "time_series"])
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--continue-mode", default="false", choices=["true", "false"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    code_dir = Path(args.code_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    normalized_target = normalize_name(args.target_column)

    context = PipelineContext(
        dataset_id=csv_path.name,
        target_column=normalized_target,
        time_column=None,
        features=[],
        split_strategy={},
        model_candidates=[],
        metrics={},
        artifacts={},
        notes=[],
    )

    progress_path = output_dir / "progress.json"
    completed_steps: list[str] = []
    errors: list[str] = []

    step_items = [
        ("10-csv-read-cleansing", "10-csv-read-cleansing"),
        ("11-data-exploration", "11-data-exploration"),
        ("12-feature-extraction", "12-feature-extraction"),
        ("13-model-training", "13-model-training"),
        ("14-model-evaluation", "14-model-evaluation"),
        ("15-model-selection", "15-model-selection"),
        ("16-result-presentation", "16-result-presentation"),
    ]

    try:
        update_progress(
            progress_path,
            args.run_id,
            args.csv_path,
            normalized_target,
            "running",
            "10-csv-read-cleansing",
            completed_steps,
            errors,
        )

        # Step 10
        df, _ = step_10_cleanse(csv_path=csv_path, output_dir=output_dir, context=context)
        completed_steps.append(step_items[0][1])
        update_progress(
            progress_path,
            args.run_id,
            args.csv_path,
            normalized_target,
            "running",
            "11-data-exploration",
            completed_steps,
            errors,
        )

        # Step 11
        step_11_exploration(df=df, output_dir=output_dir, context=context)
        completed_steps.append(step_items[1][1])
        update_progress(
            progress_path,
            args.run_id,
            args.csv_path,
            normalized_target,
            "running",
            "12-feature-extraction",
            completed_steps,
            errors,
        )

        # Step 12
        df_features, _ = step_12_features(df=df, split_mode=args.split_mode, output_dir=output_dir, context=context)
        completed_steps.append(step_items[2][1])
        update_progress(
            progress_path,
            args.run_id,
            args.csv_path,
            normalized_target,
            "running",
            "13-model-training",
            completed_steps,
            errors,
        )

        # Step 13
        step_13_training(df_features=df_features, output_dir=output_dir, context=context)
        completed_steps.append(step_items[3][1])
        update_progress(
            progress_path,
            args.run_id,
            args.csv_path,
            normalized_target,
            "running",
            "14-model-evaluation",
            completed_steps,
            errors,
        )

        # Step 14
        step_14_evaluation(output_dir=output_dir, context=context)
        completed_steps.append(step_items[4][1])
        update_progress(
            progress_path,
            args.run_id,
            args.csv_path,
            normalized_target,
            "running",
            "15-model-selection",
            completed_steps,
            errors,
        )

        # Step 15
        step_15_selection(output_dir=output_dir, context=context)
        completed_steps.append(step_items[5][1])
        update_progress(
            progress_path,
            args.run_id,
            args.csv_path,
            normalized_target,
            "running",
            "16-result-presentation",
            completed_steps,
            errors,
        )

        # Step 16
        step_16_report(output_dir=output_dir, context=context)
        completed_steps.append(step_items[6][1])

        audit = {
            "run_id": args.run_id,
            "code_dir": str(code_dir),
            "python_files": row_hashes_for_code(code_dir),
            "steps": {
                step_name: sorted(row_hashes_for_code(code_dir).keys()) for _, step_name in step_items
            },
        }
        write_json(output_dir / "code_audit.json", audit)

        update_progress(
            progress_path,
            args.run_id,
            args.csv_path,
            normalized_target,
            "completed",
            "16-result-presentation",
            completed_steps,
            errors,
        )

        print("Pipeline run completed successfully.")
        print(f"Artifacts directory: {output_dir}")
        return 0

    except Exception as exc:
        errors.append(str(exc))
        update_progress(
            progress_path,
            args.run_id,
            args.csv_path,
            normalized_target,
            "failed",
            "error",
            completed_steps,
            errors,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
