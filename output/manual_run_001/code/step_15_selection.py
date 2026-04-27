#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def quality_label(r2: float) -> str:
    if r2 >= 0.50:
        return "strong"
    if r2 >= 0.25:
        return "usable"
    if r2 >= 0:
        return "weak"
    return "below_mean_baseline"


def candidate_analysis(candidate: dict, selected_model: str | None) -> str:
    name = candidate["model_name"]
    r2 = float(candidate["r2"])
    rmse = float(candidate["rmse"])
    mae = float(candidate["mae"])
    cv_std = float(candidate.get("cv_std_r2") or 0.0)
    naive_r2 = candidate.get("naive_baseline_r2")
    baseline_note = ""
    if naive_r2 is not None:
        delta = r2 - float(naive_r2)
        if delta >= 0:
            baseline_note = f" It beats the naive lag baseline by {delta:.4f} R2."
        else:
            baseline_note = f" It trails the naive lag baseline by {abs(delta):.4f} R2."
    stability_note = "stable" if cv_std < 0.08 else "less stable across folds"
    selection_note = " This is the selected model." if name == selected_model else ""
    return (
        f"{name} is {quality_label(r2)} with R2={r2:.4f}, RMSE={rmse:.4f}, "
        f"and MAE={mae:.4f}; it is {stability_note}.{baseline_note}{selection_note}"
    )


def baseline_summary(evaluation: dict) -> dict:
    candidates = evaluation.get("candidates", []) + evaluation.get("expansion_candidates", [])
    naive_rows = [item for item in candidates if item.get("naive_baseline_r2") is not None]
    if not naive_rows:
        return {
            "mean_baseline": "R2=0 represents predicting the mean target value.",
            "naive_lag_baseline": None,
        }
    first = naive_rows[0]
    return {
        "mean_baseline": "R2=0 represents predicting the mean target value.",
        "naive_lag_baseline": {
            "r2": first.get("naive_baseline_r2"),
            "rmse": first.get("naive_baseline_rmse"),
            "mae": first.get("naive_baseline_mae"),
            "source": "Step 14 holdout naive baseline, usually y_hat_t = y_(t-1) when available.",
        },
    }


def markdown_table(rows: list[dict]) -> str:
    header = "| Model | Eligible | Weighted score | R2 | RMSE | MAE | CV std | Note |\n"
    sep = "|---|---:|---:|---:|---:|---:|---:|---|\n"
    body = []
    for row in rows:
        body.append(
            "| {model} | {eligible} | {score} | {r2:.4f} | {rmse:.4f} | {mae:.4f} | {cv:.4f} | {note} |".format(
                model=row["model_name"],
                eligible="yes" if row.get("eligible") else "no",
                score=f"{row['weighted_score']:.4f}" if row.get("weighted_score") is not None else "-",
                r2=float(row["r2"]),
                rmse=float(row["rmse"]),
                mae=float(row["mae"]),
                cv=float(row.get("cv_std_r2") or 0.0),
                note=row.get("reason", ""),
            )
        )
    return header + sep + "\n".join(body) + "\n"


def write_metrics_plot(output_dir: Path, ranking: list[dict]) -> Path:
    path = output_dir / "step-15-model-selection-metrics.png"
    names = [row["model_name"] for row in ranking]
    r2 = [float(row["r2"]) for row in ranking]
    rmse = [float(row["rmse"]) for row in ranking]
    mae = [float(row["mae"]) for row in ranking]

    x = np.arange(len(names))
    width = 0.25
    _, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x, r2, color="#2563eb")
    axes[0].axhline(0, color="#475569", linewidth=1)
    axes[0].set_title("Holdout R2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right")
    axes[0].set_ylabel("R2")

    axes[1].bar(x - width / 2, rmse, width=width, label="RMSE", color="#f97316")
    axes[1].bar(x + width / 2, mae, width=width, label="MAE", color="#16a34a")
    axes[1].set_title("Holdout error")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def write_selection_report(
    output_dir: Path,
    payload: dict,
    evaluation: dict,
    baselines: dict,
    analyses: list[dict],
    plot_path: Path | None,
) -> Path:
    report_path = output_dir / "step-15-model-selection-report.md"
    target_stats = evaluation.get("target_stats", {})
    selected = payload.get("selected_model") or "no viable candidate"
    lines = [
        "# Step 15 - Model Selection Technical Report",
        "",
        "## Inputs",
        "",
        "- Source: `step-14-evaluation.json`",
        f"- Quality assessment from Step 14: `{evaluation.get('quality_assessment')}`",
        f"- Selected model: `{selected}`",
        "",
        "## Baselines",
        "",
        f"- Mean baseline: {baselines['mean_baseline']}",
    ]
    naive = baselines.get("naive_lag_baseline")
    if naive:
        lines.extend(
            [
                f"- Naive lag baseline R2: `{float(naive['r2']):.4f}`",
                f"- Naive lag baseline RMSE: `{float(naive['rmse']):.4f}`",
                f"- Naive lag baseline MAE: `{float(naive['mae']):.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Target Context",
            "",
            f"- Mean: `{float(target_stats.get('mean', 0.0)):.4f}`",
            f"- Std: `{float(target_stats.get('std', 0.0)):.4f}`",
            f"- Min: `{float(target_stats.get('min', 0.0)):.4f}`",
            f"- Max: `{float(target_stats.get('max', 0.0)):.4f}`",
            "",
            "## Candidate Ranking",
            "",
            markdown_table(payload["full_ranking"]),
            "",
            "## Why Models Worked Or Struggled",
            "",
        ]
    )
    for item in analyses:
        lines.append(f"- {item['analysis']}")
    lines.extend(["", "## Selection Rationale", "", payload["rationale"]])
    if plot_path:
        lines.extend(["", "## Plots", "", f"![Model selection metrics]({plot_path.name})"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def select_model(output_dir: Path, run_id: str) -> None:
    mark_step_start(output_dir, STEP_NAME)
    try:
        evaluation = read_json(output_dir / "step-14-evaluation.json")
        quality = evaluation["quality_assessment"]
        if quality == "leakage_suspected":
            raise RuntimeError("Evaluation marked leakage_suspected; refusing to select a model.")
        all_candidates = evaluation["candidates"] + evaluation.get("expansion_candidates", [])
        baselines = baseline_summary(evaluation)
        eligible = [item for item in all_candidates if item["r2"] >= 0]
        ranking = []
        if not eligible:
            analyses = [
                {**item, "analysis": candidate_analysis(item, None)}
                for item in all_candidates
            ]
            payload = {
                "step": STEP_NAME,
                "run_id": run_id,
                "selected_model": None,
                "weighted_score": None,
                "rationale": "All candidates are below mean-baseline. Revisit feature engineering or broaden the expansion round.",
                "quality_flag": "no_viable_candidate",
                "baselines": baselines,
                "candidate_analysis": analyses,
                "full_ranking": [
                    {**item, "eligible": False, "weighted_score": None, "reason": "R2 < 0"} for item in all_candidates
                ],
                "artifacts": {},
            }
            plot_path = write_metrics_plot(output_dir, payload["full_ranking"]) if all_candidates else None
            report_path = write_selection_report(output_dir, payload, evaluation, baselines, analyses, plot_path)
            payload["artifacts"] = {
                "selection_report_md": str(report_path),
                "selection_metrics_png": str(plot_path) if plot_path else None,
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
        analyses = [
            {**item, "analysis": candidate_analysis(item, winner["model_name"])}
            for item in full_ranking
        ]
        plot_path = write_metrics_plot(output_dir, full_ranking)
        payload = {
            "step": STEP_NAME,
            "run_id": run_id,
            "selected_model": winner["model_name"],
            "weighted_score": float(winner["weighted_score"]),
            "rationale": rationale,
            "quality_flag": quality if quality in {"acceptable", "marginal", "subpar_after_expansion"} else "subpar",
            "baselines": baselines,
            "candidate_analysis": analyses,
            "full_ranking": full_ranking,
            "artifacts": {
                "selection_report_md": str(output_dir / "step-15-model-selection-report.md"),
                "selection_metrics_png": str(plot_path),
            },
        }
        write_selection_report(output_dir, payload, evaluation, baselines, analyses, plot_path)
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
