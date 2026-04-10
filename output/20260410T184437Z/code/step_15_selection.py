from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    parser = argparse.ArgumentParser(description="Step 15 model selection")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def normalize(values: list[float], inverse: bool = False) -> list[float]:
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-12:
        normalized = [1.0 for _ in values]
    else:
        normalized = [(v - lo) / (hi - lo) for v in values]
    return [1.0 - v for v in normalized] if inverse else normalized


def complexity_order(name: str) -> int:
    order = {
        "ridge": 1,
        "elastic_net": 2,
        "hist_gradient_boosting": 3,
        "random_forest": 4,
        "gradient_boosting": 5,
        "svr_rbf": 6,
        "xgboost": 7,
    }
    return order.get(name, 99)


def quality_flag(best_r2: float, eval_quality: str) -> str:
    if eval_quality == "subpar_after_expansion":
        return "subpar_after_expansion"
    if best_r2 >= 0.50:
        return "acceptable"
    if best_r2 >= 0.25:
        return "marginal"
    return "subpar"


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
        base_candidates = list(step14.get("candidate_reports", []))
        expanded = list(step14.get("expansion_candidates", []))
        all_candidates = base_candidates + expanded
        if not all_candidates:
            raise ValueError("No candidate reports found in step 14 output")

        eligible = [c for c in all_candidates if float(c.get("r2", -999.0)) >= 0.0]
        full_ranking: list[dict[str, Any]] = []

        if not eligible:
            for c in all_candidates:
                full_ranking.append(
                    {
                        "model_name": c["model_name"],
                        "r2": float(c["r2"]),
                        "rmse": float(c["rmse"]),
                        "mae": float(c["mae"]),
                        "cv_std_r2": float(c.get("cv_std_r2", 0.0)),
                        "weighted_score": 0.0,
                        "ineligible": True,
                        "ineligible_reason": "r2_below_zero",
                        "complexity": complexity_order(c["model_name"]),
                        "artifact_path": c.get("artifact_path", ""),
                    }
                )

            step_output = {
                "step": "15-model-selection",
                "quality_flag": "no_viable_candidate",
                "selected_model": None,
                "selected_model_details": None,
                "weighted_score": None,
                "message": "All candidates are below mean-baseline. Revisit feature engineering or expand model classes.",
                "rationale": "All evaluated candidates had negative R2 on holdout, so none has predictive value beyond mean baseline.",
                "full_ranking": full_ranking,
                "context": context,
            }
        else:
            r2_n = normalize([float(c["r2"]) for c in eligible])
            rmse_n = normalize([float(c["rmse"]) for c in eligible], inverse=True)
            mae_n = normalize([float(c["mae"]) for c in eligible], inverse=True)
            stability_n = normalize([max(0.0, 1.0 - float(c.get("cv_std_r2", 0.0))) for c in eligible])

            scored: list[dict[str, Any]] = []
            for idx, c in enumerate(tqdm(eligible, desc="step15: scoring", unit="model")):
                weighted = 0.5 * r2_n[idx] + 0.25 * rmse_n[idx] + 0.15 * mae_n[idx] + 0.1 * stability_n[idx]
                scored.append(
                    {
                        "model_name": c["model_name"],
                        "r2": float(c["r2"]),
                        "rmse": float(c["rmse"]),
                        "mae": float(c["mae"]),
                        "cv_std_r2": float(c.get("cv_std_r2", 0.0)),
                        "weighted_score": float(weighted),
                        "ineligible": False,
                        "complexity": complexity_order(c["model_name"]),
                        "artifact_path": c.get("artifact_path", ""),
                    }
                )

            for c in all_candidates:
                if float(c.get("r2", -1.0)) < 0:
                    scored.append(
                        {
                            "model_name": c["model_name"],
                            "r2": float(c["r2"]),
                            "rmse": float(c["rmse"]),
                            "mae": float(c["mae"]),
                            "cv_std_r2": float(c.get("cv_std_r2", 0.0)),
                            "weighted_score": 0.0,
                            "ineligible": True,
                            "ineligible_reason": "r2_below_zero",
                            "complexity": complexity_order(c["model_name"]),
                            "artifact_path": c.get("artifact_path", ""),
                        }
                    )

            scored.sort(key=lambda r: (-float(r["weighted_score"]), int(r["complexity"])))
            best = scored[0]

            qflag = quality_flag(float(best["r2"]), str(step14.get("quality_assessment", "subpar")))
            step_output = {
                "step": "15-model-selection",
                "quality_flag": qflag,
                "selected_model": best["model_name"],
                "selected_model_details": best,
                "weighted_score": float(best["weighted_score"]),
                "rationale": (
                    f"{best['model_name']} achieved the highest weighted score under the configured blend of R2, RMSE, MAE, and stability. "
                    f"The tie-breaker favors lower complexity, which helps reduce overfitting risk when scores are close."
                ),
                "full_ranking": scored,
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