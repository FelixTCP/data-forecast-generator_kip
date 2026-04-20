#!/usr/bin/env python3
"""Step 16: result presentation report generation."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

STEP_NAME = "16-result-presentation"
STEP01_JSON = "step-01-cleanse.json"
STEP11_JSON = "step-11-exploration.json"
STEP14_JSON = "step-14-evaluation.json"
STEP15_JSON = "step-15-selection.json"
REPORT_MD = "step-16-report.md"
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


def _ranking_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| model_name | status | eligible | r2 | rmse | mae | cv_std_r2 | weighted_score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model_name} | {status} | {eligible} | {r2} | {rmse} | {mae} | {cv_std_r2} | {weighted_score} |".format(
                model_name=row.get("model_name", ""),
                status=row.get("status", ""),
                eligible=row.get("eligible", ""),
                r2=round(float(row.get("r2", 0.0)), 6) if row.get("r2") is not None else "",
                rmse=round(float(row.get("rmse", 0.0)), 6) if row.get("rmse") is not None else "",
                mae=round(float(row.get("mae", 0.0)), 6) if row.get("mae") is not None else "",
                cv_std_r2=round(float(row.get("cv_std_r2", 0.0)), 6) if row.get("cv_std_r2") is not None else "",
                weighted_score=(
                    round(float(row.get("weighted_score", 0.0)), 6)
                    if row.get("weighted_score") is not None
                    else ""
                ),
            )
        )
    return "\n".join(lines)


def build_report(output_dir: Path) -> str:
    step01 = _load_json(output_dir / STEP01_JSON)
    step11 = _load_json(output_dir / STEP11_JSON)
    step14 = _load_json(output_dir / STEP14_JSON)
    step15 = _load_json(output_dir / STEP15_JSON)

    if not step15:
        raise FileNotFoundError("Missing step-15-selection.json.")

    target = step01.get("target_column_normalized", "unknown_target")
    quality_flag = step15.get("quality_flag", "unknown")
    selected_model = step15.get("selected_model")
    rationale = step15.get("rationale", "No rationale available.")
    ranking = step15.get("full_ranking", [])

    target_stats = step14.get("target_stats", {})
    recommendations = [
        "Review excluded features and MI thresholds from step 11 to recover potentially useful predictors.",
        "Run additional hyperparameter search for the top 2 ranked model families.",
        "Backtest across multiple chronological splits to validate stability over different periods.",
    ]

    if quality_flag in {"subpar", "subpar_after_expansion", "no_viable_candidate", "leakage_suspected"}:
        recommendations.insert(
            0,
            "Treat this run as non-production. Improve feature engineering and rerun from step 11 or 12 before deployment.",
        )

    warning = ""
    if quality_flag in {"leakage_suspected", "subpar", "subpar_after_expansion", "no_viable_candidate"}:
        warning = (
            "WARNING: This run is not production-usable as-is. "
            f"Quality flag is '{quality_flag}', so predictions should be treated as diagnostic only until remediation."
        )

    report_lines = [
        "# Forecasting Run Report",
        "",
        "## 1. Problem + selected target",
        f"This pipeline run performs regression forecasting for target column '{target}'.",
        f"Selected model: {selected_model if selected_model else 'none selected'}.",
        "",
        "## 2. Data quality summary",
        f"Rows after cleansing: {step01.get('row_count_after', 'n/a')}",
        f"Columns after cleansing: {step01.get('column_count_after', 'n/a')}",
        f"Detected time column: {step01.get('time_column', 'none')}",
        f"Recommended features from exploration: {len(step11.get('recommended_features', []))}",
        f"Low variance columns flagged: {len(step11.get('low_variance_columns', []))}",
        f"Redundant columns flagged: {len(step11.get('redundant_columns', []))}",
        "",
        "## 3. Candidate models + scores table",
        _ranking_table(ranking),
        "",
        "## 4. Selected model rationale",
        rationale,
        "",
        "## 5. Risks and caveats",
        f"Quality flag: {quality_flag}",
        f"Target holdout mean/std/min/max: {target_stats.get('mean')}, {target_stats.get('std')}, {target_stats.get('min')}, {target_stats.get('max')}",
        warning or "No major risk flags were raised beyond normal forecasting uncertainty.",
        "",
        "## 6. Next iteration recommendations",
    ]
    report_lines.extend(f"- {item}" for item in recommendations)

    return "\n".join(report_lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 16 report generation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    progress = _load_json(output_dir / PROGRESS_FILE)
    csv_path = progress.get("csv_path", "")
    target = progress.get("target_column", "")

    _update_progress(
        output_dir,
        run_id=args.run_id,
        csv_path=csv_path,
        target_column=target,
        status="running",
        current_step=STEP_NAME,
    )

    try:
        report_text = build_report(output_dir)
        report_path = output_dir / REPORT_MD
        report_path.write_text(report_text, encoding="utf-8")

        _update_progress(
            output_dir,
            run_id=args.run_id,
            csv_path=csv_path,
            target_column=target,
            status="completed",
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
