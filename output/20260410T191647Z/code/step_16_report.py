from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 16 result report")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    step10 = read_json(output_dir / "step-10-cleanse.json")
    step14 = read_json(output_dir / "step-14-evaluation.json")
    step15 = read_json(output_dir / "step-15-selection.json")
    context = step15["context"]

    progress_path = output_dir / "progress.json"
    progress = read_json(progress_path)
    completed_steps = [str(s) for s in progress.get("completed_steps", [])]
    errors = [str(e) for e in progress.get("errors", [])]

    update_progress(
        progress_path,
        args.run_id,
        progress["csv_path"],
        context["target_column"],
        "16-result-presentation",
        completed_steps,
        errors,
        "running",
    )

    try:
        lines: list[str] = []
        lines.append(f"# Forecasting Report ({datetime.now(UTC).isoformat()})")
        lines.append("")
        lines.append("## 1. Problem + selected target")
        lines.append(
            f"This run predicts '{context['target_column']}' from dataset '{context['dataset_id']}' using a per-step regression workflow."
        )
        lines.append(
            f"Model selection quality flag: {step15.get('quality_flag', 'unknown')} and selected model: {step15.get('selected_model')}."
        )
        lines.append("")

        lines.append("## 2. Data quality summary")
        lines.append(
            f"Rows after cleansing: {step10['row_count_after']}; columns: {step10['column_count']}; duplicates: {step10['duplicate_rows']}."
        )
        lines.append(f"Detected time column: {step10['time_column_detected']}.")
        lines.append("Top null-rate columns:")
        top_nulls = sorted(step10["null_rate"].items(), key=lambda item: item[1], reverse=True)[:10]
        for col, rate in top_nulls:
            lines.append(f"- {col}: {rate:.4f}")
        lines.append("")

        lines.append("## 3. Candidate models + scores table")
        lines.append("| Model | R2 | RMSE | MAE | CV Std | Ineligible |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in tqdm(step15.get("full_ranking", []), desc="step16: table", unit="row"):
            lines.append(
                f"| {row['model_name']} | {row['r2']:.4f} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['cv_std_r2']:.4f} | {row.get('ineligible', False)} |"
            )
        lines.append("")

        lines.append("## 4. Selected model rationale")
        lines.append(str(step15.get("rationale", "No rationale was generated.")))
        if step15.get("quality_flag") in {"subpar", "subpar_after_expansion", "no_viable_candidate"}:
            lines.append(
                "Quality note: The achieved predictive quality is below the preferred threshold and should be treated as exploratory."
            )
        lines.append("")

        lines.append("## 5. Risks and caveats")
        lines.append("- Holdout quality can degrade under distribution shift in weather and occupancy patterns.")
        lines.append("- Lag features may leak weak short-term autocorrelation that does not generalize across seasons.")
        lines.append("- Negative-R2 candidates were marked ineligible and excluded from winner consideration.")
        lines.append("- If quality flag is subpar, downstream decisions should not rely on this model without retraining.")
        lines.append("")

        lines.append("## 6. Next iteration recommendations")
        lines.append("- Add richer exogenous features (holiday calendar, occupancy proxies, weather interactions).")
        lines.append("- Run rolling-origin backtests to stress temporal robustness beyond one holdout split.")
        lines.append("- Tune expanded candidates using constrained search spaces and monotonic constraints where valid.")

        report_path = output_dir / "step-16-report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")

        if "16-result-presentation" not in completed_steps:
            completed_steps.append("16-result-presentation")

        update_progress(
            progress_path,
            args.run_id,
            progress["csv_path"],
            context["target_column"],
            "16-result-presentation",
            completed_steps,
            errors,
            "completed",
        )
        return 0
    except Exception as exc:
        errors.append(str(exc))
        update_progress(
            progress_path,
            args.run_id,
            progress["csv_path"],
            context["target_column"],
            "16-result-presentation",
            completed_steps,
            errors,
            "error",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())