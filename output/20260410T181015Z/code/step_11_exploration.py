from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm


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
    parser = argparse.ArgumentParser(description="Step 11 data exploration")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    step10 = read_json(output_dir / "step-10-cleanse.json")
    context = step10["context"]
    progress_path = output_dir / "progress.json"
    progress = read_json(progress_path)

    completed_steps = [str(s) for s in progress.get("completed_steps", [])]
    errors = [str(e) for e in progress.get("errors", [])]

    update_progress(
        progress_path,
        args.run_id,
        progress["csv_path"],
        context["target_column"],
        "11-data-exploration",
        completed_steps,
        errors,
        "running",
    )

    try:
        df = pl.read_parquet(step10["artifacts"]["cleaned_parquet"])

        numeric_columns = [
            col for col, dtype in zip(df.columns, df.dtypes, strict=False) if is_numeric_dtype(dtype)
        ]

        cardinality: dict[str, int] = {}
        for col in tqdm(df.columns, desc="step11: cardinality", unit="column"):
            cardinality[col] = int(df.select(pl.col(col).n_unique()).item())

        numeric_summary: dict[str, dict[str, float | None]] = {}
        for col in tqdm(numeric_columns, desc="step11: numeric summary", unit="column"):
            stats = df.select(
                [
                    pl.col(col).mean().alias("mean"),
                    pl.col(col).std().alias("std"),
                    pl.col(col).min().alias("min"),
                    pl.col(col).max().alias("max"),
                    pl.col(col).is_null().mean().alias("null_rate"),
                ]
            ).to_dicts()[0]
            numeric_summary[col] = {
                k: (None if v is None else float(v)) for k, v in stats.items()
            }

        corr_pairs = list(combinations(numeric_columns[:8], 2))
        correlation_preview: list[dict[str, float | str]] = []
        for left, right in tqdm(corr_pairs, desc="step11: correlations", unit="pair"):
            corr = df.select(pl.corr(pl.col(left), pl.col(right))).item()
            if corr is not None:
                correlation_preview.append({"left": left, "right": right, "pearson_corr": float(corr)})

        target_candidates = sorted(
            [
                {
                    "column": col,
                    "null_rate": float(df.select(pl.col(col).is_null().mean()).item()),
                    "std": float(df.select(pl.col(col).std()).item() or 0.0),
                }
                for col in numeric_columns
            ],
            key=lambda item: (item["null_rate"], -item["std"]),
        )[:5]

        step_output = {
            "step": "11-data-exploration",
            "shape": {"rows": int(df.height), "columns": int(df.width)},
            "numeric_columns": numeric_columns,
            "high_cardinality": [
                c for c in df.columns if cardinality[c] > max(100, int(0.5 * max(1, df.height)))
            ],
            "cardinality": cardinality,
            "numeric_summary": numeric_summary,
            "correlation_preview": correlation_preview,
            "target_candidates": target_candidates,
            "time_series_detected": context.get("time_column") is not None,
            "context": context,
        }

        write_json(output_dir / "step-11-exploration.json", step_output)

        if "11-data-exploration" not in completed_steps:
            completed_steps.append("11-data-exploration")

        update_progress(
            progress_path,
            args.run_id,
            progress["csv_path"],
            context["target_column"],
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
            context["target_column"],
            "11-data-exploration",
            completed_steps,
            errors,
            "error",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
