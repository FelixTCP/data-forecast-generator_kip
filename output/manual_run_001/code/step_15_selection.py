#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from runtime_utils import mark_step_error, mark_step_start, mark_step_success, read_json, update_code_audit, write_json


STEP_NAME = "15-model-selection"
COMPLEXITY = {
    "ridge": 0,
    "elastic_net": 1,
    "hist_gradient_boosting": 2,
    "random_forest": 3,
    "gradient_boosting": 4,
    "svr": 5,
    "xgboost": 6,
}


def normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    low, high = float(np.min(values)), float(np.max(values))
    if high == low:
        return np.full_like(values, 0.5, dtype=float)
    return (values - low) / (high - low)


def select_model(output_dir: Path, run_id: str) -> None:
    mark_step_start(output_dir, STEP_NAME)
    try:
        evaluation = read_json(output_dir / "step-14-evaluation.json")
        quality = evaluation["quality_assessment"]
        if quality == "leakage_suspected":
            raise RuntimeError("Evaluation marked leakage_suspected; refusing to select a model.")
        all_candidates = evaluation["candidates"] + evaluation.get("expansion_candidates", [])
        eligible = [item for item in all_candidates if item["r2"] >= 0]
        ranking = []
        if not eligible:
            payload = {
                "step": STEP_NAME,
                "run_id": run_id,
                "selected_model": None,
                "weighted_score": None,
                "rationale": "All candidates are below mean-baseline. Revisit feature engineering or broaden the expansion round.",
                "quality_flag": "no_viable_candidate",
                "full_ranking": [
                    {**item, "eligible": False, "weighted_score": None, "reason": "R2 < 0"} for item in all_candidates
                ],
            }
            write_json(output_dir / "step-15-selection.json", payload)
            mark_step_success(output_dir, STEP_NAME)
            update_code_audit(output_dir, Path(__file__).resolve().parent)
            return

        r2 = np.array([item["r2"] for item in eligible], dtype=float)
        rmse = np.array([item["rmse"] for item in eligible], dtype=float)
        mae = np.array([item["mae"] for item in eligible], dtype=float)
        stability = np.array([1 - float(item["cv_std_r2"] or 0.0) for item in eligible], dtype=float)
        scores = 0.50 * normalize(r2) + 0.25 * (1 - normalize(rmse)) + 0.15 * (1 - normalize(mae)) + 0.10 * normalize(stability)
        for item, score in zip(eligible, scores):
            ranking.append({**item, "eligible": True, "weighted_score": float(score)})
        ranking.sort(
            key=lambda row: (
                -row["weighted_score"],
                COMPLEXITY.get(row["model_name"], 999),
            )
        )
        winner = ranking[0]
        full_ranking = ranking + [
            {**item, "eligible": False, "weighted_score": None, "reason": "R2 < 0"} for item in all_candidates if item["r2"] < 0
        ]
        rationale = (
            f"{winner['model_name']} ranks first because it leads the weighted score across accuracy, error, and stability. "
            f"It balances R2={winner['r2']:.4f}, RMSE={winner['rmse']:.4f}, and MAE={winner['mae']:.4f} better than the remaining eligible candidates."
        )
        payload = {
            "step": STEP_NAME,
            "run_id": run_id,
            "selected_model": winner["model_name"],
            "weighted_score": float(winner["weighted_score"]),
            "rationale": rationale,
            "quality_flag": quality if quality in {"acceptable", "marginal", "subpar_after_expansion"} else "subpar",
            "full_ranking": full_ranking,
        }
        write_json(output_dir / "step-15-selection.json", payload)
        mark_step_success(output_dir, STEP_NAME)
        update_code_audit(output_dir, Path(__file__).resolve().parent)
    except Exception as exc:
        mark_step_error(output_dir, STEP_NAME, str(exc))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    select_model(Path(args.output_dir), args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
