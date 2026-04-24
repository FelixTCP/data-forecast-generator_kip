#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from runtime_utils import (
    mark_step_error,
    mark_step_start,
    mark_step_success,
    normalize_name,
    read_json,
    update_code_audit,
    write_json,
)


STEP_NAME = "01-csv-read-cleansing"


def maybe_numeric(series: pl.Series) -> tuple[pl.Series, dict]:
    stripped = series.cast(pl.String).str.strip_chars()
    converted = stripped.cast(pl.Float64, strict=False)
    non_null = max(stripped.len() - stripped.is_null().sum(), 1)
    parsed = converted.is_not_null().sum()
    ratio = parsed / non_null
    meta = {"parsed_ratio": ratio, "dtype_before": str(series.dtype)}
    if ratio >= 0.95:
        return converted.alias(series.name), meta | {"converted": True}
    return stripped.alias(series.name), meta | {"converted": False}


def cleanse(csv_path: Path, output_dir: Path, run_id: str, target_column: str) -> None:
    mark_step_start(output_dir, STEP_NAME)
    try:
        report_md = output_dir / "step-00_data_profile_report.md"
        recommendations = report_md.read_text(encoding="utf-8") if report_md.exists() else ""
        lf = pl.scan_csv(csv_path, try_parse_dates=True)
        original_columns = lf.collect_schema().names()
        renamed = {column: normalize_name(column) for column in original_columns}
        lf = lf.rename(renamed)
        df = lf.collect()

        original_name_map = {normalize_name(key): key for key in original_columns}
        coercions: dict[str, dict] = {}
        transformed_columns: list[pl.Series] = []
        for column in df.columns:
            series = df[column]
            if series.dtype == pl.String:
                new_series, meta = maybe_numeric(series)
                coercions[column] = meta
                transformed_columns.append(new_series)
            else:
                transformed_columns.append(series)
        df = pl.DataFrame(transformed_columns)

        target_normalized = normalize_name(target_column)
        if target_normalized not in df.columns:
            raise ValueError(f"Normalized target '{target_normalized}' not found in columns {df.columns}")

        time_column = None
        for column, dtype in df.schema.items():
            if dtype in (pl.Date, pl.Datetime):
                time_column = column
                break
        if time_column is None:
            for column in df.columns:
                if "date" in column or "time" in column:
                    time_column = column
                    break

        duplicates = int(df.is_duplicated().sum())
        rows_before = df.height
        null_target_before = int(df[target_normalized].is_null().sum())
        if null_target_before:
            df = df.filter(pl.col(target_normalized).is_not_null())
        rows_after = df.height

        null_rate = {
            column: float(df.select(pl.col(column).is_null().mean()).item())
            for column in df.columns
        }
        quality = {
            "step": STEP_NAME,
            "run_id": run_id,
            "csv_path": str(csv_path),
            "source_profile_used": report_md.exists(),
            "profile_recommendations_excerpt": recommendations.splitlines()[:12],
            "original_columns": original_columns,
            "normalized_columns": df.columns,
            "original_name_map": original_name_map,
            "row_count_before": rows_before,
            "row_count_after": rows_after,
            "duplicate_rows_detected": duplicates,
            "target_column_requested": target_column,
            "target_column_normalized": target_normalized,
            "time_column": time_column,
            "null_rate": null_rate,
            "schema": {column: str(dtype) for column, dtype in df.schema.items()},
            "string_numeric_coercions": coercions,
            "changes": [
                {"action": "rename_columns", "count": len(original_columns)},
                {"action": "drop_null_target_rows", "count": rows_before - rows_after},
            ],
            "artifacts": {"cleaned_parquet": str(output_dir / "cleaned.parquet")},
        }
        df.write_parquet(output_dir / "cleaned.parquet")
        write_json(output_dir / "step-01-cleanse.json", quality)
        mark_step_success(output_dir, STEP_NAME)
        update_code_audit(output_dir, Path(__file__).resolve().parent)
    except Exception as exc:
        mark_step_error(output_dir, STEP_NAME, str(exc))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    cleanse(Path(args.csv_path), Path(args.output_dir), args.run_id, args.target_column)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
