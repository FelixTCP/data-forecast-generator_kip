#!/usr/bin/env python3
"""Step 00: Lightweight CSV profiling without full in-memory load."""

from __future__ import annotations

import argparse
import csv
import json
import random
import traceback
from collections import deque
from pathlib import Path
from typing import Any

STEP_NAME = "00-pre-exploration"
PROFILER_JSON = "step-00_profiler.json"
PROFILE_REPORT = "step-00_data_profile_report.md"
PROGRESS_FILE = "progress.json"


MISSING_TOKENS = {"", "na", "n/a", "null", "none", "nan", "?", "-"}
SUMMARY_HINTS = ("total", "summary", "average", "avg", "sum")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _normalize_token(value: str) -> str:
    return value.strip().lower()


def _detect_delimiter(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        sample = fh.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except csv.Error:
        return ","


def _update_progress(
    output_dir: Path,
    *,
    run_id: str,
    csv_path: str,
    status: str,
    current_step: str,
    target_column: str | None = None,
    error: str | None = None,
    mark_completed: bool = False,
) -> None:
    path = output_dir / PROGRESS_FILE
    progress = _load_json(path)
    if not progress:
        progress = {
            "run_id": run_id,
            "csv_path": csv_path,
            "target_column": target_column or "",
            "status": "running",
            "current_step": current_step,
            "completed_steps": [],
            "errors": [],
        }

    progress["run_id"] = run_id
    progress["csv_path"] = csv_path
    if target_column is not None:
        progress["target_column"] = target_column
    progress["status"] = status
    progress["current_step"] = current_step

    completed = progress.setdefault("completed_steps", [])
    if mark_completed and current_step not in completed:
        completed.append(current_step)

    if error:
        progress.setdefault("errors", []).append({"step": current_step, "error": error})

    _write_json(path, progress)


def _reservoir_append(sample: list[list[str]], row: list[str], seen: int, k: int, rng: random.Random) -> None:
    if len(sample) < k:
        sample.append(row)
        return
    idx = rng.randint(0, seen)
    if idx < k:
        sample[idx] = row


def profile_csv(csv_path: Path) -> dict[str, Any]:
    delimiter = _detect_delimiter(csv_path)
    rng = random.Random(42)

    head_rows: list[list[str]] = []
    tail_rows: deque[list[str]] = deque(maxlen=5)
    sample_rows: list[list[str]] = []

    inconsistent_rows: list[dict[str, Any]] = []
    missing_counts: dict[str, int] = {}
    footer_suspects: list[dict[str, Any]] = []

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV is empty.") from exc

        expected_cols = len(header)
        row_count = 0

        for idx, row in enumerate(reader, start=2):
            row_count += 1
            if len(head_rows) < 5:
                head_rows.append(row)
            tail_rows.append(row)
            _reservoir_append(sample_rows, row, row_count - 1, 10, rng)

            if len(row) != expected_cols and len(inconsistent_rows) < 10:
                inconsistent_rows.append(
                    {
                        "line": idx,
                        "expected_columns": expected_cols,
                        "found_columns": len(row),
                    }
                )

            for cell in row:
                token = _normalize_token(cell)
                if token in MISSING_TOKENS:
                    missing_counts[token] = missing_counts.get(token, 0) + 1

        for offset, row in enumerate(list(tail_rows), start=max(2, row_count - len(tail_rows) + 2)):
            first_cell = _normalize_token(row[0]) if row else ""
            if any(hint in first_cell for hint in SUMMARY_HINTS):
                footer_suspects.append({"line": offset, "value": row[0] if row else ""})

    recommendations: list[str] = []
    if inconsistent_rows:
        recommendations.append(
            "Rows with inconsistent column counts were found. Configure strict CSV parsing and inspect malformed lines."
        )
    if footer_suspects:
        recommendations.append(
            "Potential footer rows detected. Add explicit footer filtering before model steps."
        )
    if missing_counts:
        recommendations.append(
            "Normalize missing value tokens (for example NA/NONE/?/-) to null during cleansing."
        )
    recommendations.append("Normalize column names and preserve original names in metadata.")
    recommendations.append("Use polars scan_csv with try_parse_dates enabled and log all cast fixes.")

    return {
        "step": STEP_NAME,
        "csv_path": str(csv_path),
        "delimiter": delimiter,
        "header": header,
        "row_count": row_count,
        "column_count": len(header),
        "head": head_rows,
        "tail": list(tail_rows),
        "sample": sample_rows,
        "anomalies": {
            "inconsistent_rows": inconsistent_rows,
            "inconsistent_row_count": len(inconsistent_rows),
            "missing_value_tokens": missing_counts,
            "footer_suspects": footer_suspects,
        },
        "recommended_cleansing_steps": recommendations,
    }


def write_report(output_path: Path, profile: dict[str, Any]) -> None:
    anomalies = profile["anomalies"]
    lines = [
        "# Data Profile Report",
        "",
        "## File Metadata",
        f"- CSV path: {profile['csv_path']}",
        f"- Delimiter: `{profile['delimiter']}`",
        f"- Row count (excluding header): {profile['row_count']}",
        f"- Column count: {profile['column_count']}",
        "",
        "## Columns",
    ]
    lines.extend(f"- {col}" for col in profile["header"])

    lines.extend(
        [
            "",
            "## Structural Anomalies",
            f"- Inconsistent rows captured: {anomalies['inconsistent_row_count']}",
            f"- Potential footer rows: {len(anomalies['footer_suspects'])}",
            f"- Missing token variants observed: {len(anomalies['missing_value_tokens'])}",
            "",
            "## Recommended Polars Cleansing Steps",
        ]
    )
    lines.extend(f"- {rec}" for rec in profile["recommended_cleansing_steps"])

    lines.extend(
        [
            "",
            "## Sampling Preview",
            "### Head (first 5 rows)",
            "```",
            json.dumps(profile["head"], indent=2),
            "```",
            "",
            "### Tail (last 5 rows)",
            "```",
            json.dumps(profile["tail"], indent=2),
            "```",
            "",
            "### Random Sample (10 rows)",
            "```",
            json.dumps(profile["sample"], indent=2),
            "```",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 00 pre-exploration profiler")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _update_progress(
        output_dir,
        run_id=args.run_id,
        csv_path=str(csv_path),
        status="running",
        current_step=STEP_NAME,
    )

    try:
        profile = profile_csv(csv_path)
        context = {
            "dataset_id": csv_path.stem,
            "target_column": "",
            "time_column": None,
            "features": [],
            "split_strategy": {},
            "model_candidates": [],
            "metrics": {},
            "artifacts": {
                "step_00_profiler": str((output_dir / PROFILER_JSON).resolve()),
                "step_00_report": str((output_dir / PROFILE_REPORT).resolve()),
            },
            "notes": [],
        }
        profile["context"] = context

        _write_json(output_dir / PROFILER_JSON, profile)
        write_report(output_dir / PROFILE_REPORT, profile)

        _update_progress(
            output_dir,
            run_id=args.run_id,
            csv_path=str(csv_path),
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
            csv_path=str(csv_path),
            status="failed",
            current_step=STEP_NAME,
            error=err,
        )
        print(err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
