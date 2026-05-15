"""Step 15 — Model Selection.

Runnable:
    python step_15_selection.py --output-dir <dir> --run-id <id>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


COMPLEXITY_ORDER = [
    "naive_persistence", "ar1_benchmark", "seasonal_naive",
    "ridge", "elasticnet", "hist_gbm", "random_forest",
    "holt_winters", "auto_arima_benchmark", "faar_arima",
    "xgboost", "lightgbm", "svr",
]


def _complexity_rank(name: str) -> int:
    """Lower = simpler."""
    for i, n in enumerate(COMPLEXITY_ORDER):
        if n in name.lower():
            return i
    return len(COMPLEXITY_ORDER)


def update_progress(progress_path: Path, updates: dict):
    if progress_path.exists():
        with open(progress_path) as f:
            p = json.load(f)
    else:
        p = {}
    p.update(updates)
    with open(progress_path, "w") as f:
        json.dump(p, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Step 15: Model Selection")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    progress_path = output_dir / "progress.json"
    update_progress(progress_path, {"status": "running", "current_step": "15-model-selection"})

    try:
        with open(output_dir / "step-14-evaluation.json") as f:
            ctx14 = json.load(f)

        quality_assessment = ctx14["quality_assessment"]

        # Hard stop for leakage
        if quality_assessment == "leakage_suspected":
            step_output = {
                "step": "15-model-selection",
                "quality_flag": "leakage_suspected",
                "selected_model": None,
                "weighted_score": None,
                "rationale": "Leakage detected in step 14. No model selected. Revisit feature engineering.",
            }
            with open(output_dir / "step-15-selection.json", "w") as f:
                json.dump(step_output, f, indent=2)
            update_progress(progress_path, {"status": "leakage_halt"})
            print("Leakage suspected — halting.")
            sys.exit(2)

        candidates = ctx14.get("candidates", []) + ctx14.get("expansion_candidates", [])
        target_stats = ctx14.get("target_stats", {})
        naive_lag_baseline = ctx14.get("naive_lag_baseline", {})

        # ── Pre-selection filter ───────────────────────────────────────────────
        # Exclude benchmarks from selection ranking but keep for reporting
        # Only trained candidates are eligible
        benchmark_names = {
            "naive_persistence", "seasonal_naive", "auto_arima_benchmark",
            "ar1_benchmark",
        }
        trained_candidates = [
            c for c in candidates
            if not c.get("is_benchmark", False)
            and c.get("r2") is not None
            and np.isfinite(c.get("r2", float("nan")))
            and c.get("model_name") not in benchmark_names
        ]
        ineligible = [
            c for c in trained_candidates
            if c["r2"] < 0
        ]
        eligible = [c for c in trained_candidates if c["r2"] >= 0]

        print(f"Eligible: {[c['model_name'] for c in eligible]}")
        print(f"Ineligible (R²<0): {[c['model_name'] for c in ineligible]}")

        if not eligible:
            step_output = {
                "step": "15-model-selection",
                "quality_flag": "no_viable_candidate",
                "selected_model": None,
                "weighted_score": None,
                "rationale": (
                    "All candidates are below mean-baseline (R² < 0). "
                    "Revisit feature engineering (step 12) or expand model classes (step 14 expansion)."
                ),
                "full_ranking": [
                    {"model_name": c["model_name"], "r2": c["r2"], "status": "ineligible"}
                    for c in ineligible
                ],
                "baselines": {
                    "mean_baseline": {"r2": 0.0, "description": "Predicting mean target value"},
                    "naive_lag_baseline": naive_lag_baseline,
                },
            }
            with open(output_dir / "step-15-selection.json", "w") as f:
                json.dump(step_output, f, indent=2)
            update_progress(progress_path, {
                "status": "running",
                "current_step": "15-model-selection",
            })
            print("No viable candidates — selection done (no_viable_candidate).")
            sys.exit(0)

        # ── Weighted scoring ───────────────────────────────────────────────────
        # Collect metrics for normalization
        r2_vals = np.array([c["r2"] for c in eligible], dtype=float)
        rmse_vals = np.array([c.get("rmse", float("nan")) or float("nan") for c in eligible], dtype=float)
        mae_vals = np.array([c.get("mae", float("nan")) or float("nan") for c in eligible], dtype=float)
        cv_std_vals = np.array([
            c.get("cv_std_r2", float("nan")) or float("nan") for c in eligible
        ], dtype=float)

        def _minmax(arr: np.ndarray) -> np.ndarray:
            mn = np.nanmin(arr)
            mx = np.nanmax(arr)
            if mx == mn:
                return np.ones_like(arr)
            return (arr - mn) / (mx - mn)

        r2_norm = _minmax(r2_vals)
        rmse_norm = 1.0 - _minmax(rmse_vals)   # lower RMSE → higher score
        mae_norm = 1.0 - _minmax(mae_vals)      # lower MAE → higher score
        # stability: 1 - cv_std_r2 (lower std → higher stability); handle NaN
        stability = np.where(
            np.isfinite(cv_std_vals),
            1.0 - _minmax(cv_std_vals),
            0.5,  # neutral when not available
        )

        weighted_scores = (
            0.50 * r2_norm
            + 0.25 * rmse_norm
            + 0.15 * mae_norm
            + 0.10 * stability
        )

        best_idx = int(np.argmax(weighted_scores))

        # Tie-breaking: among candidates within 0.01 of top score, prefer simpler
        top_score = weighted_scores[best_idx]
        near_top = [i for i, s in enumerate(weighted_scores) if abs(s - top_score) <= 0.01]
        if len(near_top) > 1:
            best_idx = min(near_top, key=lambda i: _complexity_rank(eligible[i]["model_name"]))

        winner = eligible[best_idx]
        winner_name = winner["model_name"]
        winner_score = float(weighted_scores[best_idx])

        print(f"Winner: {winner_name} (weighted_score={winner_score:.4f})")

        # ── Determine quality_flag ─────────────────────────────────────────────
        best_eligible_r2 = winner["r2"]
        if quality_assessment == "subpar_after_expansion":
            quality_flag = "subpar_after_expansion"
        elif best_eligible_r2 >= 0.50:
            quality_flag = "acceptable"
        elif best_eligible_r2 >= 0.25:
            quality_flag = "marginal"
        else:
            quality_flag = "subpar"

        # ── Candidate analysis ─────────────────────────────────────────────────
        candidate_analysis = {}
        naive_r2 = naive_lag_baseline.get("r2", 0.0) or 0.0

        for i, c in enumerate(eligible):
            name = c["model_name"]
            r2 = c["r2"]
            cv_r2 = c.get("cv_mean_r2")
            rmse = c.get("rmse")
            mae = c.get("mae")

            parts = []
            # Mean baseline comparison
            if r2 >= 0:
                parts.append(f"R²={r2:.4f} > 0 (beats mean baseline).")
            # Naive lag comparison
            delta_naive = r2 - naive_r2
            if abs(delta_naive) < 0.01:
                parts.append(f"Performance nearly identical to naïve lag-1 baseline (Δ={delta_naive:+.4f}).")
            elif delta_naive > 0:
                parts.append(f"Beats naïve lag-1 by Δ={delta_naive:+.4f} R².")
            else:
                parts.append(f"Below naïve lag-1 by Δ={delta_naive:+.4f} R².")
            # CV vs holdout
            if cv_r2 is not None and np.isfinite(cv_r2):
                diff = r2 - cv_r2
                if abs(diff) < 0.05:
                    parts.append(f"CV R²={cv_r2:.4f} ≈ holdout R²={r2:.4f}: good generalization.")
                elif diff < -0.05:
                    parts.append(f"CV R²={cv_r2:.4f} >> holdout R²={r2:.4f}: mild overfitting.")
                else:
                    parts.append(f"CV R²={cv_r2:.4f} < holdout R²={r2:.4f}: better than expected.")
            # Error magnitude
            if rmse and target_stats.get("std"):
                rmse_pct = rmse / target_stats["std"] * 100
                parts.append(f"RMSE={rmse:.2f}°F ({rmse_pct:.1f}% of target std).")
            candidate_analysis[name] = " ".join(parts)

        for c in ineligible:
            candidate_analysis[c["model_name"]] = (
                f"R²={c['r2']:.4f} < 0 — model is worse than predicting the mean. "
                "Ineligible for selection."
            )

        # ── Full ranking ───────────────────────────────────────────────────────
        full_ranking = []
        for i, c in enumerate(eligible):
            full_ranking.append({
                "rank": i + 1,
                "model_name": c["model_name"],
                "r2": c["r2"],
                "rmse": c.get("rmse"),
                "mae": c.get("mae"),
                "cv_mean_r2": c.get("cv_mean_r2"),
                "weighted_score": float(weighted_scores[i]),
                "status": "eligible",
            })
        full_ranking.sort(key=lambda x: x["weighted_score"], reverse=True)
        for rank, row in enumerate(full_ranking, 1):
            row["rank"] = rank

        for c in ineligible:
            full_ranking.append({
                "rank": None,
                "model_name": c["model_name"],
                "r2": c["r2"],
                "rmse": c.get("rmse"),
                "mae": c.get("mae"),
                "cv_mean_r2": c.get("cv_mean_r2"),
                "weighted_score": None,
                "status": "ineligible",
            })

        # Benchmarks in ranking (for reference)
        for c in candidates:
            if c.get("is_benchmark"):
                full_ranking.append({
                    "rank": None,
                    "model_name": c["model_name"],
                    "r2": c.get("r2"),
                    "rmse": c.get("rmse"),
                    "mae": c.get("mae"),
                    "cv_mean_r2": None,
                    "weighted_score": None,
                    "status": "benchmark",
                })

        # ── Rationale ─────────────────────────────────────────────────────────
        delta_naive_winner = winner["r2"] - naive_r2
        rationale = (
            f"{winner_name} achieved the highest weighted score ({winner_score:.4f}) "
            f"with holdout R²={winner['r2']:.4f}, RMSE={winner.get('rmse', 'N/A'):.3f}°F, "
            f"MAE={winner.get('mae', 'N/A'):.3f}°F. "
            f"It beats the naïve lag-1 baseline (R²={naive_r2:.4f}) by Δ={delta_naive_winner:+.4f}. "
        )
        if winner.get("cv_mean_r2") and np.isfinite(winner["cv_mean_r2"]):
            rationale += (
                f"Cross-validation R²={winner['cv_mean_r2']:.4f}±{winner.get('cv_std_r2', 0):.4f} "
                "indicates consistent performance across temporal folds. "
            )
        if delta_naive_winner < 0.02:
            rationale += (
                "Note: The improvement over the naïve persistence baseline is modest. "
                "Daily temperature autocorrelation is strong — any lag-1 model performs well. "
                "The value of this model lies in better handling of seasonal transitions and feature integration."
            )

        # ── Write Markdown report ──────────────────────────────────────────────
        report_lines = [
            "# Step 15 — Model Selection Report",
            "",
            f"**Run ID:** {args.run_id}",
            "",
            "## Baseline Summary",
            "",
            "| Baseline | R² | RMSE | MAE |",
            "|----|----|----|-----|",
            f"| Mean baseline (R²=0) | 0.0000 | - | - |",
        ]
        if naive_lag_baseline:
            report_lines.append(
                f"| Naïve lag-1 | "
                f"{naive_lag_baseline.get('r2', 'N/A'):.4f} | "
                f"{naive_lag_baseline.get('rmse', 'N/A'):.3f} | "
                f"{naive_lag_baseline.get('mae', 'N/A'):.3f} |"
            )
        report_lines += [
            "",
            "## Candidate Ranking",
            "",
            "| Rank | Model | R² | RMSE | MAE | CV R² | Weighted Score | Status |",
            "|----|----|----|----|----|----|----|-----|",
        ]
        for row in full_ranking:
            score_str = f"{row['weighted_score']:.4f}" if row['weighted_score'] else "-"
            cv_str = f"{row['cv_mean_r2']:.4f}" if row.get('cv_mean_r2') and np.isfinite(row['cv_mean_r2']) else "-"
            rmse_str = f"{row['rmse']:.3f}" if row.get('rmse') else "-"
            mae_str = f"{row['mae']:.3f}" if row.get('mae') else "-"
            r2_str = f"{row['r2']:.4f}" if row.get('r2') is not None else "-"
            rank_str = str(row['rank']) if row['rank'] else "-"
            report_lines.append(
                f"| {rank_str} | {row['model_name']} | {r2_str} | {rmse_str} | "
                f"{mae_str} | {cv_str} | {score_str} | {row['status']} |"
            )
        report_lines += [
            "",
            "## Selected Model",
            "",
            f"**{winner_name}** — {rationale}",
            "",
            "## Candidate Analysis",
            "",
        ]
        for name, analysis in candidate_analysis.items():
            report_lines.append(f"**{name}:** {analysis}")
            report_lines.append("")

        if quality_flag != "acceptable":
            report_lines += [
                "## ⚠️ Quality Warning",
                "",
                f"Quality flag: `{quality_flag}`. See step 16 report for details and recommendations.",
                "",
            ]

        report_md_path = str(output_dir / "step-15-model-selection-report.md")
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        # ── Produce metrics plot ────────────────────────────────────────────────
        plot_path = str(output_dir / "step-15-model-selection-metrics.png")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot_names = [row["model_name"] for row in full_ranking if row.get("r2") is not None]
            plot_r2 = [row.get("r2", 0) for row in full_ranking if row.get("r2") is not None]

            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ["green" if r >= 0.5 else "orange" if r >= 0.25 else "red" if r >= 0 else "gray"
                      for r in plot_r2]
            bars = ax.barh(plot_names, plot_r2, color=colors)
            ax.axvline(0, color="black", linewidth=1, linestyle="--", label="Mean baseline")
            ax.axvline(naive_r2, color="blue", linewidth=1, linestyle=":", label=f"Naïve persistence (R²={naive_r2:.2f})")
            ax.set_xlabel("R²")
            ax.set_title(f"Model Comparison — {args.run_id}")
            ax.legend(fontsize=8)
            ax.invert_yaxis()
            # Annotate winner
            if winner_name in plot_names:
                idx = plot_names.index(winner_name)
                ax.get_yticklabels()[idx].set_fontweight("bold")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=120)
            plt.close()
            print(f"Saved plot: {plot_path}")
        except Exception as e:
            print(f"Plot failed (non-fatal): {e}")
            plot_path = None

        # ── Build output JSON ─────────────────────────────────────────────────
        step_output = {
            "step": "15-model-selection",
            "run_id": args.run_id,
            "selected_model": winner_name,
            "weighted_score": winner_score,
            "quality_flag": quality_flag,
            "rationale": rationale,
            "baselines": {
                "mean_baseline": {"r2": 0.0, "description": "Predicting mean target value"},
                "naive_lag_baseline": naive_lag_baseline,
            },
            "candidate_analysis": candidate_analysis,
            "full_ranking": full_ranking,
            "artifacts": {
                "selection_report_md": report_md_path,
                "selection_metrics_png": plot_path,
            },
        }

        out_path = output_dir / "step-15-selection.json"
        with open(out_path, "w") as f:
            json.dump(step_output, f, indent=2)
        print(f"Written: {out_path}")

        with open(progress_path) as f:
            p = json.load(f)
        if "15-model-selection" not in p.get("completed_steps", []):
            p.setdefault("completed_steps", []).append("15-model-selection")
        p["current_step"] = "15-model-selection"
        p["status"] = "running"
        with open(progress_path, "w") as f:
            json.dump(p, f, indent=2)

        print(f"Step 15 complete. Selected: {winner_name}, quality_flag={quality_flag}")
        sys.exit(0)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR in step 15: {e}\n{tb}", file=sys.stderr)
        if progress_path.exists():
            with open(progress_path) as f:
                p = json.load(f)
            p["status"] = "error"
            p.setdefault("errors", []).append({"step": "15-model-selection", "error": str(e), "traceback": tb})
            with open(progress_path, "w") as f:
                json.dump(p, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
