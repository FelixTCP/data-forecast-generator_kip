from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm


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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_name(name: str) -> str:
    return "_".join(name.strip().lower().split())


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


def detect_time_column(df: pl.DataFrame, target_column: str) -> str | None:
    for col, dtype in zip(df.columns, df.dtypes, strict=False):
        if col == target_column:
            continue
        if dtype in {pl.Date, pl.Datetime, pl.Time}:
            return col
    for col in df.columns:
        if col == target_column:
            continue
        col_lower = col.lower()
        if "date" in col_lower or "time" in col_lower or "timestamp" in col_lower:
            return col
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 10 CSV read and cleansing")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-column", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"

    target_normalized = normalize_name(args.target_column)

    progress = read_json(progress_path) if progress_path.exists() else {
        "run_id": args.run_id,
        "csv_path": args.csv_path,
        "target_column": target_normalized,
        "status": "running",
        "current_step": "10-csv-read-cleansing",
        "completed_steps": [],
        "errors": [],
    }
    completed_steps = [str(s) for s in progress.get("completed_steps", [])]
    errors = [str(e) for e in progress.get("errors", [])]

    update_progress(
        progress_path,
        args.run_id,
        args.csv_path,
        target_normalized,
        "10-csv-read-cleansing",
        completed_steps,
        errors,
        "running",
    )

    try:
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df_raw = pl.read_csv(csv_path, try_parse_dates=True)
        original_columns = list(df_raw.columns)
        normalized_columns = [normalize_name(c) for c in original_columns]
        rename_map = dict(zip(original_columns, normalized_columns, strict=True))

        fixes: list[dict[str, Any]] = []
        if original_columns != normalized_columns:
            fixes.append(
                {
                    "type": "normalized_column_names",
                    "count": len(original_columns),
                    "before_after": rename_map,
                }
            )

        df = df_raw.rename(rename_map)

        string_types = {t for t in (getattr(pl, "String", None), getattr(pl, "Utf8", None)) if t is not None}
        for col in tqdm(df.columns, desc="step10: type coercion", unit="column"):
            dtype = df.schema[col]
            if dtype not in string_types:
                continue

            trimmed = pl.col(col).str.strip_chars()
            non_empty = int(df.select(trimmed.is_not_null().and_(trimmed != "").sum()).item())
            if non_empty == 0:
                continue

            numeric_success = int(df.select(trimmed.cast(pl.Float64, strict=False).is_not_null().sum()).item())
            ratio = numeric_success / non_empty
            if ratio >= 0.95:
                df = df.with_columns(trimmed.cast(pl.Float64, strict=False).alias(col))
                fixes.append(
                    {
                        "type": "coerced_numeric_string_column",
                        "column": col,
                        "success_ratio": ratio,
                    }
                )
                continue

            if "date" in col or "time" in col or "timestamp" in col:
                dt_success = int(
                    df.select(trimmed.str.strptime(pl.Datetime, strict=False).is_not_null().sum()).item()
                )
                dt_ratio = dt_success / non_empty
                if dt_ratio >= 0.95:
                    df = df.with_columns(trimmed.str.strptime(pl.Datetime, strict=False).alias(col))
                    fixes.append(
                        {
                            "type": "coerced_datetime_string_column",
                            "column": col,
                            "success_ratio": dt_ratio,
                        }
                    )

        null_rate: dict[str, float] = {}
        inferred_dtypes: dict[str, str] = {}
        for col in tqdm(df.columns, desc="step10: profiling", unit="column"):
            null_rate[col] = float(df.select(pl.col(col).is_null().mean()).item())
            inferred_dtypes[col] = str(df.schema[col])

        duplicate_rows = int(df.is_duplicated().sum())

        if target_normalized not in df.columns:
            raise ValueError(f"Target column '{target_normalized}' not found after normalization")

        clean_path = output_dir / "cleaned.parquet"
        df.write_parquet(clean_path)

        context = PipelineContext(
            dataset_id=Path(args.csv_path).name,
            target_column=target_normalized,
            time_column=detect_time_column(df, target_normalized),
            features=[],
            split_strategy={},
            model_candidates=[],
            metrics={},
            artifacts={"cleaned_parquet": str(clean_path)},
            notes=[],
        )

        step_output = {
            "step": "10-csv-read-cleansing",
            "row_count_before": int(df_raw.height),
            "row_count_after": int(df.height),
            "column_count": int(df.width),
            "target_column_original": args.target_column,
            "target_column_normalized": target_normalized,
            "time_column_detected": context.time_column,
            "null_rate": null_rate,
            "inferred_dtypes": inferred_dtypes,
            "duplicate_rows": duplicate_rows,
            "applied_fixes": fixes,
            "context": asdict(context),
            "artifacts": {"cleaned_parquet": str(clean_path)},
        }
        write_json(output_dir / "step-10-cleanse.json", step_output)

        if "10-csv-read-cleansing" not in completed_steps:
            completed_steps.append("10-csv-read-cleansing")

        update_progress(
            progress_path,
            args.run_id,
            args.csv_path,
            target_normalized,
            "10-csv-read-cleansing",
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
            args.csv_path,
            target_normalized,
            "10-csv-read-cleansing",
            completed_steps,
            errors,
            "error",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
