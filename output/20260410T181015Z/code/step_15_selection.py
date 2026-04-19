from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def normalize(values: list[float], inverse: bool = False) -> list[float]:
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-12:
        base = [1.0 for _ in values]
    else:
        base = [(v - lo) / (hi - lo) for v in values]
    if inverse:
        return [1.0 - v for v in base]
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 15 model selection")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    step14 = read_json(output_dir / "step-14-evaluation.json")
    context = step14["context"]

    progress_path = output_dir / "progress.json"
    progress = read_json(progress_path)
    completed_steps = [str(s) for s in progress.get("completed_steps", [])]
    errors = [str(e) for e in progress.get("errors", [])]

    update_progress(
        progress_path,
        args.run_id,
        progress["csv_path"],
        context["target_column"],
        "15-model-selection",
        completed_steps,
        errors,
        "running",
    )

    try:
        reports = step14["candidate_reports"]

        r2_n = normalize([float(r["r2"]) for r in reports])
        rmse_n = normalize([float(r["rmse"]) for r in reports], inverse=True)
        mae_n = normalize([float(r["mae"]) for r in reports], inverse=True)
        stability_n = normalize([max(0.0, 1.0 - float(r["cv_std_r2"])) for r in reports])

        ranking: list[dict[str, Any]] = []
        for idx, report in enumerate(reports):
            weighted_score = (
                0.5 * r2_n[idx]
                + 0.25 * rmse_n[idx]
                + 0.15 * mae_n[idx]
                + 0.1 * stability_n[idx]
            )
            ranking.append(
                {
                    "model_name": report["model_name"],
                    "complexity": int(report["complexity"]),
                    "weighted_score": float(weighted_score),
                    "r2": float(report["r2"]),
                    "rmse": float(report["rmse"]),
                    "mae": float(report["mae"]),
                    "cv_std_r2": float(report["cv_std_r2"]),
                    "artifact_path": report["artifact_path"],
                }
            )

        ranking.sort(key=lambda item: (-item["weighted_score"], item["complexity"]))
        selected = ranking[0]

        step_output = {
            "step": "15-model-selection",
            "selection_rule": {
                "r2_weight": 0.5,
                "rmse_weight": 0.25,
                "mae_weight": 0.15,
                "stability_weight": 0.1,
                "tie_breaker": "lower complexity",
            },
            "ranking": ranking,
            "selected_model": selected["model_name"],
            "selected_model_details": selected,
            "rationale": (
                f"Selected {selected['model_name']} with top weighted score "
                f"{selected['weighted_score']:.4f}; tie-breaker favors lower complexity."
            ),
            "context": context,
        }

        write_json(output_dir / "step-15-selection.json", step_output)

        if "15-model-selection" not in completed_steps:
            completed_steps.append("15-model-selection")

        update_progress(
            progress_path,
            args.run_id,
            progress["csv_path"],
            context["target_column"],
            "15-model-selection",
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
            "15-model-selection",
            completed_steps,
            errors,
            "error",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
