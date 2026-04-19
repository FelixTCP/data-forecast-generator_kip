#!/usr/bin/env python3
"""Step 15: transparent model selection with quality gating."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

STEP_NAME = "15-model-selection"
STEP14_JSON = "step-14-evaluation.json"
STEP15_JSON = "step-15-selection.json"
PROGRESS_FILE = "progress.json"

COMPLEXITY_ORDER = {
    "ridge": 1,
    "elastic_net": 2,
    "hist_gradient_boosting": 3,
    "random_forest": 4,
    "gradient_boosting": 5,
    "svr_rbf": 6,
    "xgboost": 7,
}


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


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin == 0:
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def run_step(output_dir: Path) -> tuple[dict[str, Any], bool, str | None]:
    step14 = _load_json(output_dir / STEP14_JSON)
    if not step14:
        raise FileNotFoundError("Missing step-14-evaluation.json.")

    qa = step14.get("quality_assessment")
    if qa == "leakage_suspected":
        payload = {
            "step": STEP_NAME,
            "quality_flag": "leakage_suspected",
            "selected_model": None,
            "weighted_score": None,
            "rationale": "Selection halted because evaluation marked leakage_suspected. Metrics are invalid for production use.",
            "full_ranking": [],
            "diagnostics": step14.get("leakage_probe", {}),
            "context": step14.get("context", {}),
        }
        return payload, True, "Selection hard-stopped due to leakage_suspected"

    base_candidates = step14.get("candidates", [])
    expansion_candidates = step14.get("expansion_candidates", [])
    all_candidates = base_candidates + expansion_candidates

    ranking_rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []

    for item in all_candidates:
        status = item.get("status", "ok")
        if status != "ok":
            ranking_rows.append(
                {
                    "model_name": item.get("model_name", "unknown"),
                    "status": status,
                    "eligible": False,
                    "ineligible_reason": item.get("error", "candidate_failed"),
                    "r2": item.get("r2"),
                    "rmse": item.get("rmse"),
                    "mae": item.get("mae"),
                    "cv_std_r2": item.get("cv_std_r2"),
                    "weighted_score": None,
                }
            )
            continue

        r2 = float(item["r2"])
        row = {
            "model_name": item["model_name"],
            "status": "ok",
            "r2": r2,
            "rmse": float(item["rmse"]),
            "mae": float(item["mae"]),
            "cv_std_r2": float(item.get("cv_std_r2", 0.0)),
            "eligible": r2 >= 0.0,
            "weighted_score": None,
        }
        if r2 < 0.0:
            row["ineligible_reason"] = "r2_below_mean_baseline"
        else:
            eligible.append(dict(item))
        ranking_rows.append(row)

    if not eligible:
        payload = {
            "step": STEP_NAME,
            "selected_model": None,
            "weighted_score": None,
            "rationale": "All candidates are below mean-baseline. Revisit feature engineering (step 12) or expand model classes (step 14 expansion).",
            "full_ranking": ranking_rows,
            "quality_flag": "no_viable_candidate",
            "context": step14.get("context", {}),
        }
        return payload, False, None

    r2_vals = [float(c["r2"]) for c in eligible]
    rmse_vals = [float(c["rmse"]) for c in eligible]
    mae_vals = [float(c["mae"]) for c in eligible]
    stability_vals = [max(0.0, 1.0 - float(c.get("cv_std_r2", 0.0))) for c in eligible]

    r2_norm = _minmax(r2_vals)
    rmse_inv = [1.0 - x for x in _minmax(rmse_vals)]
    mae_inv = [1.0 - x for x in _minmax(mae_vals)]
    stab_norm = _minmax(stability_vals)

    scored: list[dict[str, Any]] = []
    for i, cand in enumerate(eligible):
        weighted = 0.50 * r2_norm[i] + 0.25 * rmse_inv[i] + 0.15 * mae_inv[i] + 0.10 * stab_norm[i]
        rec = {
            **cand,
            "weighted_score": float(weighted),
            "complexity_order": COMPLEXITY_ORDER.get(cand["model_name"], 99),
        }
        scored.append(rec)

    scored.sort(
        key=lambda c: (
            -c["weighted_score"],
            COMPLEXITY_ORDER.get(c["model_name"], 99),
            -float(c["r2"]),
        )
    )
    winner = scored[0]

    for row in ranking_rows:
        for scored_row in scored:
            if row.get("model_name") == scored_row.get("model_name"):
                row["weighted_score"] = round(float(scored_row["weighted_score"]), 8)
                break

    best_r2 = float(winner["r2"])
    if qa in {"acceptable", "marginal", "subpar_after_expansion"}:
        quality_flag = qa
    elif best_r2 >= 0.50:
        quality_flag = "acceptable"
    elif best_r2 >= 0.25:
        quality_flag = "marginal"
    else:
        quality_flag = "subpar"

    rationale = (
        f"Model '{winner['model_name']}' achieved the highest weighted score ({winner['weighted_score']:.4f}) under the configured multi-metric rule. "
        f"Its holdout profile balances R2={winner['r2']:.4f}, RMSE={winner['rmse']:.4f}, and MAE={winner['mae']:.4f} with stability bonus from CV variance."
    )

    payload = {
        "step": STEP_NAME,
        "selected_model": winner["model_name"],
        "weighted_score": float(winner["weighted_score"]),
        "rationale": rationale,
        "full_ranking": ranking_rows,
        "quality_flag": quality_flag,
        "context": step14.get("context", {}),
    }

    return payload, False, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 15 model selection")
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
        payload, hard_stop, reason = run_step(output_dir)
        _write_json(output_dir / STEP15_JSON, payload)

        if hard_stop:
            raise RuntimeError(reason or "Hard stop in step 15")

        _update_progress(
            output_dir,
            run_id=args.run_id,
            csv_path=csv_path,
            target_column=target,
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
