from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
        "12-feature-extraction",
        completed_steps,
        errors,
        "running",
    )

    try:
        df = pl.read_parquet(step10["artifacts"]["cleaned_parquet"])
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in cleaned parquet")

        df_features = df
        created_features: list[dict[str, str]] = []

        if time_column and time_column in df_features.columns:
            df_features = df_features.sort(time_column)
            tcol = pl.col(time_column)
            date_parts = [
                tcol.dt.year().alias(f"{time_column}_year"),
                tcol.dt.month().alias(f"{time_column}_month"),
                tcol.dt.weekday().alias(f"{time_column}_weekday"),
                tcol.dt.hour().alias(f"{time_column}_hour"),
            ]
            df_features = df_features.with_columns(date_parts)
            created_features.extend(
                [
                    {"feature": f"{time_column}_year", "reason": "calendar seasonality"},
                    {"feature": f"{time_column}_month", "reason": "monthly seasonality"},
                    {"feature": f"{time_column}_weekday", "reason": "weekly cycles"},
                    {"feature": f"{time_column}_hour", "reason": "intraday pattern"},
                ]
            )

            numeric_candidates = [
                c
                for c, dtype in zip(df_features.columns, df_features.dtypes, strict=False)
                if c not in {target, time_column} and is_numeric_dtype(dtype)
            ]
            lag_base = numeric_candidates[: min(5, len(numeric_candidates))]
            lag_exprs: list[pl.Expr] = []
            for col in tqdm(lag_base, desc="step12: temporal features", unit="column"):
                lag_exprs.append(pl.col(col).shift(1).alias(f"{col}_lag1"))
                lag_exprs.append(pl.col(col).rolling_mean(window_size=3).alias(f"{col}_rollmean3"))
                created_features.append({"feature": f"{col}_lag1", "reason": "lag dependency"})
                created_features.append({"feature": f"{col}_rollmean3", "reason": "local trend"})

            if lag_exprs:
                df_features = df_features.with_columns(lag_exprs)

            df_features = df_features.drop(time_column)

        numeric_features = [
            c
            for c, dtype in zip(df_features.columns, df_features.dtypes, strict=False)
            if c != target and is_numeric_dtype(dtype)
        ]

        split_resolved = "time_series" if (args.split_mode == "auto" and time_column) else args.split_mode
        if split_resolved not in {"time_series", "random"}:
            split_resolved = "random"

        context["features"] = numeric_features
        context["split_strategy"] = {
            "requested_mode": args.split_mode,
            "resolved_mode": split_resolved,
            "time_column": time_column,
            "random_state": RANDOM_STATE,
        }

        features_path = output_dir / "features.parquet"
        df_features.write_parquet(features_path)
        context.setdefault("artifacts", {})["features_parquet"] = str(features_path)

        step_output = {
            "step": "12-feature-extraction",
            "target_column": target,
            "features": numeric_features,
            "feature_count": len(numeric_features),
            "created_features": created_features,
            "split_strategy": context["split_strategy"],
            "context": context,
            "artifacts": {"features_parquet": str(features_path)},
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
