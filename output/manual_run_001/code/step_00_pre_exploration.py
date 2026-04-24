#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, deque
from pathlib import Path

from runtime_utils import mark_step_error, mark_step_start, mark_step_success, update_code_audit, write_json


STEP_NAME = "00-pre-exploration"


def profile_csv(csv_path: Path, output_dir: Path, run_id: str) -> None:
    mark_step_start(output_dir, STEP_NAME)
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delimiter = dialect.delimiter
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader)

            head_rows: list[list[str]] = []
            tail_rows: deque[list[str]] = deque(maxlen=5)
            reservoir: list[list[str]] = []
            line_widths: Counter[int] = Counter()
            missing_markers: Counter[str] = Counter()
            row_count = 0
            random.seed(42)

            for row in reader:
                row_count += 1
                line_widths[len(row)] += 1
                if row_count <= 5:
                    head_rows.append(row)
                tail_rows.append(row)
                if len(reservoir) < 10:
                    reservoir.append(row)
                else:
                    pick = random.randint(1, row_count)
                    if pick <= 10:
                        reservoir[pick - 1] = row
                for value in row:
                    stripped = value.strip().lower()
                    if stripped in {"", "na", "n/a", "null", "none", "nan"}:
                        missing_markers[stripped or "<empty>"] += 1

        anomalies: list[str] = []
        expected_width = len(header)
        inconsistent_rows = sum(count for width, count in line_widths.items() if width != expected_width)
        if inconsistent_rows:
            anomalies.append(
                f"{inconsistent_rows} rows have delimiter count mismatches relative to the {expected_width}-column header."
            )
        if tail_rows:
            last_row = list(tail_rows)[-1]
            if len(last_row) != expected_width:
                anomalies.append("Trailing footer or malformed summary row suspected in file tail.")

        profiler = {
            "step": STEP_NAME,
            "run_id": run_id,
            "csv_path": str(csv_path),
            "delimiter": delimiter,
            "total_rows": row_count,
            "total_columns": len(header),
            "header": header,
            "head": head_rows,
            "tail": list(tail_rows),
            "random_sample": reservoir,
            "line_width_histogram": dict(sorted(line_widths.items())),
            "missing_value_markers": dict(missing_markers),
            "anomalies": anomalies,
        }
        write_json(output_dir / "step-00_profiler.json", profiler)

        report_lines = [
            "# Data Profile Report",
            "",
            "## File metadata",
            f"- Path: `{csv_path}`",
            f"- Rows: {row_count}",
            f"- Columns: {len(header)}",
            f"- Delimiter: `{delimiter}`",
            "",
            "## Structural observations",
        ]
        if anomalies:
            report_lines.extend(f"- {item}" for item in anomalies)
        else:
            report_lines.append("- No delimiter-width anomalies detected in the streamed scan.")
        report_lines.extend(
            [
                "",
                "## Recommended Polars cleansing steps",
                "- Normalize headers to lowercase underscore names while keeping an original-name map.",
                "- Load with `polars.scan_csv(..., try_parse_dates=True)` and collect once after lazy transforms.",
                "- Attempt numeric coercion for string columns dominated by numeric-looking values.",
                "- Record null-rate, duplicate count, and any dropped rows explicitly in the step 01 audit.",
                "- Preserve and parse the detected time column for downstream time-series profiling.",
                "",
                "## Header",
                f"`{header}`",
            ]
        )
        (output_dir / "step-00_data_profile_report.md").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )
        mark_step_success(output_dir, STEP_NAME)
        update_code_audit(output_dir, Path(__file__).resolve().parent)
    except Exception as exc:
        mark_step_error(output_dir, STEP_NAME, str(exc))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    profile_csv(Path(args.csv_path), Path(args.output_dir), args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
