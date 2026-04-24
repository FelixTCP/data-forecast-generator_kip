#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from runtime_utils import mark_status, mark_step_error, mark_step_start, mark_step_success, read_json, update_code_audit


STEP_NAME = "16-result-presentation"


def build_report(output_dir: Path, run_id: str) -> None:
    mark_step_start(output_dir, STEP_NAME)
    try:
        step00 = read_json(output_dir / "step-00_profiler.json")
        step01 = read_json(output_dir / "step-01-cleanse.json")
        step11 = read_json(output_dir / "step-11-exploration.json")
        step12 = read_json(output_dir / "step-12-features.json")
        step13 = read_json(output_dir / "step-13-training.json")
        step14 = read_json(output_dir / "step-14-evaluation.json")
        step15 = read_json(output_dir / "step-15-selection.json")
        leakage = read_json(output_dir / "leakage_audit.json")

        quality = step15["quality_flag"]
        production_usable = quality == "acceptable"
        warning = ""
        if quality in {"marginal", "subpar_after_expansion", "subpar", "no_viable_candidate"}:
            warning = f"Warning: this run is not production-usable without follow-up work because quality_flag={quality}."
        report = "\n".join(
            [
                "# Forecasting Run Report",
                "",
                "## 1. Problem + selected target",
                f"- Run ID: `{run_id}`",
                f"- CSV path: `{step00['csv_path']}`",
                f"- Selected target: `{step01['target_column_normalized']}`",
                f"- Production usable: `{production_usable}`",
                "",
                "## 2. Data quality summary",
                f"- Rows after cleansing: {step01['row_count_after']}",
                f"- Time column: `{step01['time_column']}`",
                f"- Duplicate rows detected: {step01['duplicate_rows_detected']}",
                f"- Null-rate keys tracked: {len(step01['null_rate'])}",
                f"- Excluded features in exploration: {len(step11['excluded_features'])}",
                "",
                "## 3. Candidate models + scores table",
                "| Model | R2 | RMSE | MAE |",
                "| --- | ---: | ---: | ---: |",
                *[
                    f"| {item['model_name']} | {item['r2']:.4f} | {item['rmse']:.4f} | {item['mae']:.4f} |"
                    for item in step14["candidates"] + step14.get("expansion_candidates", [])
                ],
                "",
                "## 4. Selected model rationale",
                f"- Selected model: `{step15['selected_model']}`",
                f"- Quality flag: `{quality}`",
                f"- Rationale: {step15['rationale']}",
                "",
                "## 5. Risks and caveats",
                f"- Leakage audit status: `{leakage['status']}`",
                f"- Evaluation quality: `{step14['quality_assessment']}`",
                f"- {warning or 'No critical warning beyond standard validation risk.'}",
                "",
                "## 6. Next iteration recommendations",
                "- Validate the selected model on a fresh temporal slice before any deployment.",
                "- Revisit lag and rolling-window choices if business use needs stronger recall on peaks.",
                "- Add domain-specific external signals if the current covariates cap achievable quality.",
                "",
            ]
        )
        (output_dir / "step-16-report.md").write_text(report + "\n", encoding="utf-8")
        mark_step_success(output_dir, STEP_NAME)
        mark_status(output_dir, "completed")
        update_code_audit(output_dir, Path(__file__).resolve().parent)
    except Exception as exc:
        mark_step_error(output_dir, STEP_NAME, str(exc))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    build_report(Path(args.output_dir), args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
