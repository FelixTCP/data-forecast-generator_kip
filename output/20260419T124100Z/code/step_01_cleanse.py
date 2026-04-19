#!/usr/bin/env python3
"""Step 01: CSV read and cleansing with strict audit logging."""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Any

import polars as pl

STEP_NAME = "01-csv-read-cleansing"
STEP00_JSON = "step-00_profiler.json"
STEP01_JSON = "step-01-cleanse.json"
CLEANED_PARQUET = "cleaned.parquet"
PROGRESS_FILE = "progress.json"


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    dtype_name = str(dtype)
    return dtype_name.startswith(("Int", "UInt", "Float", "Decimal"))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "column"


def _build_unique_name_map(columns: list[str]) -> dict[str, str]:
    used: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for col in columns:
        base = _normalize_name(col)
        count = used.get(base, 0)
        final = base if count == 0 else f"{base}_{count + 1}"
        used[base] = count + 1
        mapping[col] = final
    return mapping


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


def _coerce_numeric_like_columns(df: pl.DataFrame, target_column: str) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    fixes: list[dict[str, Any]] = []
    out = df
    for col, dtype in out.schema.items():
        if col == target_column:
            continue
        if dtype != pl.Utf8:
            continue

        series = out[col]
        non_null_count = int(series.is_not_null().sum())
        if non_null_count == 0:
            continue

        parsed = out.select(
            pl.col(col)
            .str.replace_all(r"\s+", "")
            .str.replace_all(",", ".")
            .cast(pl.Float64, strict=False)
            .alias(col)
        )[col]

        parsed_non_null = int(parsed.is_not_null().sum())
        parse_ratio = parsed_non_null / non_null_count if non_null_count else 0.0

        if parse_ratio >= 0.9 and parsed_non_null >= 10:
            out = out.with_columns(parsed.alias(col))
            fixes.append(
                {
                    "column": col,
                    "fix": "coerced_numeric_like_string_to_float64",
                    "parse_ratio": round(parse_ratio, 6),
                }
            )

    return out, fixes


def _detect_time_column(df: pl.DataFrame) -> tuple[pl.DataFrame, str | None, str | None]:
    for col, dtype in df.schema.items():
        if dtype in (pl.Date, pl.Datetime):
            return df, col, "already_datetime_dtype"

    for col in df.columns:
        lowered = col.lower()
        if "date" not in lowered and "time" not in lowered:
            continue
        if df.schema[col] != pl.Utf8:
            continue

        parsed = df.select(
            pl.col(col).str.strptime(pl.Datetime, strict=False, exact=False).alias(col)
        )[col]
        ratio = float(parsed.is_not_null().sum()) / max(df.height, 1)
        if ratio >= 0.8:
            out = df.with_columns(parsed.alias(col))
            return out, col, f"parsed_utf8_to_datetime_ratio={ratio:.3f}"

    return df, None, None


def run_step(csv_path: Path, target_column: str, output_dir: Path, run_id: str) -> dict[str, Any]:
    step00_payload = _load_json(output_dir / STEP00_JSON)

    lf = pl.scan_csv(str(csv_path), try_parse_dates=True)
    original_columns = list(lf.collect_schema().names())
    name_map = _build_unique_name_map(original_columns)
    normalized_columns = [name_map[c] for c in original_columns]

    if any(src != dst for src, dst in name_map.items()):
        lf = lf.rename(name_map)

    df = lf.collect()
    row_count_before = int(df.height)
    col_count_before = len(original_columns)

    target_column_normalized = _normalize_name(target_column)
    if target_column_normalized not in df.columns:
        raise ValueError(
            f"Normalized target column '{target_column_normalized}' not found. Available columns: {df.columns}"
        )

    applied_fixes: list[dict[str, Any]] = []

    df, numeric_fixes = _coerce_numeric_like_columns(df, target_column_normalized)
    applied_fixes.extend(numeric_fixes)

    if not _is_numeric_dtype(df.schema[target_column_normalized]):
        cast_target = df.select(
            pl.col(target_column_normalized)
            .cast(pl.Utf8)
            .str.replace_all(r"\s+", "")
            .str.replace_all(",", ".")
            .cast(pl.Float64, strict=False)
            .alias(target_column_normalized)
        )[target_column_normalized]
        non_null_target = int(df[target_column_normalized].is_not_null().sum())
        cast_non_null = int(cast_target.is_not_null().sum())
        ratio = cast_non_null / max(non_null_target, 1)
        if ratio < 0.9:
            raise ValueError(
                f"Target column '{target_column_normalized}' could not be reliably cast to numeric (ratio={ratio:.3f})."
            )
        df = df.with_columns(cast_target.alias(target_column_normalized))
        applied_fixes.append(
            {
                "column": target_column_normalized,
                "fix": "cast_target_to_float64",
                "parse_ratio": round(ratio, 6),
            }
        )

    df, time_column, time_column_detection = _detect_time_column(df)
    if time_column is not None:
        applied_fixes.append(
            {
                "column": time_column,
                "fix": "time_column_detected",
                "detection": time_column_detection,
            }
        )

    null_rate = {
        c: float(df.select(pl.col(c).is_null().mean()).item())
        for c in df.columns
    }

    duplicate_rows = int(df.height - df.unique().height) if df.height > 0 else 0
    row_count_after = int(df.height)

    cleaned_path = (output_dir / CLEANED_PARQUET).resolve()
    df.write_parquet(cleaned_path)

    context = dict(step00_payload.get("context", {}))
    context.setdefault("dataset_id", csv_path.stem)
    context["target_column"] = target_column_normalized
    context["time_column"] = time_column
    context.setdefault("features", [])
    context.setdefault("split_strategy", {})
    context.setdefault("model_candidates", [])
    context.setdefault("metrics", {})
    context.setdefault("artifacts", {})
    context["artifacts"]["cleaned_parquet"] = str(cleaned_path)
    context.setdefault("notes", [])
    context["notes"].append("Step 01 completed with strict non-dropping policy.")

    payload = {
        "step": STEP_NAME,
        "run_id": run_id,
        "csv_path": str(csv_path),
        "row_count_before": row_count_before,
        "row_count_after": row_count_after,
        "column_count_before": col_count_before,
        "column_count_after": df.width,
        "dropped_rows": 0,
        "dropped_columns": [],
        "original_columns": original_columns,
        "normalized_columns": normalized_columns,
        "column_name_mapping": name_map,
        "target_column_original": target_column,
        "target_column_normalized": target_column_normalized,
        "time_column": time_column,
        "null_rate": null_rate,
        "dtype_report": {c: str(t) for c, t in df.schema.items()},
        "duplicate_rows": duplicate_rows,
        "applied_fixes": applied_fixes,
        "artifacts": {
            "cleaned_parquet": str(cleaned_path),
            "step_01_json": str((output_dir / STEP01_JSON).resolve()),
        },
        "context": context,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 01 CSV read and cleansing")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _update_progress(
        output_dir,
        run_id=args.run_id,
        csv_path=args.csv_path,
        target_column=args.target_column,
        status="running",
        current_step=STEP_NAME,
    )

    try:
        payload = run_step(Path(args.csv_path), args.target_column, output_dir, args.run_id)
        _write_json(output_dir / STEP01_JSON, payload)

        _update_progress(
            output_dir,
            run_id=args.run_id,
            csv_path=args.csv_path,
            target_column=payload["target_column_normalized"],
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
            csv_path=args.csv_path,
            target_column=args.target_column,
            status="failed",
            current_step=STEP_NAME,
            error=err,
        )
        print(err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
