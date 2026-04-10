from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 14 model evaluation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    step13 = read_json(output_dir / "step-13-training.json")
    context = step13["context"]

    progress_path = output_dir / "progress.json"
    progress = read_json(progress_path)
    completed_steps = [str(s) for s in progress.get("completed_steps", [])]
    errors = [str(e) for e in progress.get("errors", [])]

    update_progress(
        progress_path,
        args.run_id,
        progress["csv_path"],
        context["target_column"],
        "14-model-evaluation",
        completed_steps,
        errors,
        "running",
    )

    try:
        holdout = np.load(output_dir / "holdout.npz")
        x_test = holdout["X_test"]
        y_test = holdout["y_test"]

        candidate_reports: list[dict[str, Any]] = []
        for candidate in tqdm(step13["candidates"], desc="step14: evaluate", unit="model"):
            model = joblib.load(candidate["artifact_path"])
            preds = model.predict(x_test)
            residuals = y_test - preds

            r2 = float(r2_score(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            for value, metric in ((r2, "r2"), (rmse, "rmse"), (mae, "mae")):
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite metric {metric} for model {candidate['model_name']}")

            candidate_reports.append(
                {
                    "model_name": candidate["model_name"],
                    "complexity": candidate["complexity"],
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                    "cv_mean_r2": candidate["cv_mean_r2"],
                    "cv_std_r2": candidate["cv_std_r2"],
                    "residual_note": {
                        "mean_residual": float(np.mean(residuals)),
                        "max_abs_error": float(np.max(np.abs(residuals))),
                    },
                    "artifact_path": candidate["artifact_path"],
                }
            )

        step_output = {
            "step": "14-model-evaluation",
            "candidate_reports": candidate_reports,
            "context": context,
        }
        write_json(output_dir / "step-14-evaluation.json", step_output)

        if "14-model-evaluation" not in completed_steps:
            completed_steps.append("14-model-evaluation")

        update_progress(
            progress_path,
            args.run_id,
            progress["csv_path"],
            context["target_column"],
            "14-model-evaluation",
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
            "14-model-evaluation",
            completed_steps,
            errors,
            "error",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
