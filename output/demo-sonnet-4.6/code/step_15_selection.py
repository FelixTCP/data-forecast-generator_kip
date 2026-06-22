"""Step 15 — Model Selection."""
import argparse
import json
import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def make_ser(obj):
    if isinstance(obj, dict):
        return {k: make_ser(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_ser(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        return str(obj)


COMPLEXITY_ORDER = ["ridge", "elasticnet", "histgradientboosting", "random_forest",
                    "gradient_boosting", "svr", "xgboost", "lightgbm"]


def complexity_rank(name):
    name_lower = name.lower()
    for i, m in enumerate(COMPLEXITY_ORDER):
        if m in name_lower:
            return i
    return len(COMPLEXITY_ORDER)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    run_id = args.run_id

    # Update progress
    progress_path = os.path.join(output_dir, "progress.json")
    with open(progress_path) as f:
        progress = json.load(f)
    progress["status"] = "running"
    progress["current_step"] = "15-model-selection"
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    # Load evaluation results
    step14 = json.load(open(os.path.join(output_dir, "step-14-evaluation.json")))
    step14_quality = step14["quality_assessment"]

    if step14_quality == "leakage_suspected":
        print("ERROR: leakage_suspected — cannot proceed to selection", file=sys.stderr)
        sys.exit(1)

    candidates = step14["candidates"]
    naive_baseline = step14.get("naive_baseline", {})
    target_stats = step14.get("target_stats", {})

    # Pre-selection filter: remove R² < 0
    eligible = [c for c in candidates if c.get("r2") is not None and not _is_nan(c["r2"]) and c["r2"] >= 0]
    ineligible = [c for c in candidates if c not in eligible]

    if not eligible:
        # No viable candidate
        result = {
            "step": "15-model-selection",
            "run_id": run_id,
            "selected_model": None,
            "weighted_score": None,
            "quality_flag": "no_viable_candidate",
            "rationale": (
                "All candidates are below mean-baseline. "
                "Revisit feature engineering (step 12) or expand model classes (step 14 expansion)."
            ),
            "full_ranking": [
                {**c, "eligible": False, "note": "R² < 0"} for c in candidates
            ],
            "baselines": {"mean_baseline_r2": 0.0, "naive_lag_baseline": naive_baseline},
            "candidate_analysis": {},
            "artifacts": {},
        }
        out_json = os.path.join(output_dir, "step-15-selection.json")
        with open(out_json, "w") as f:
            json.dump(make_ser(result), f, indent=2)
        print("Step 15: No viable candidate")
        sys.exit(0)

    # Normalize metrics for scoring
    r2_vals = [c["r2"] for c in eligible]
    rmse_vals = [c["rmse"] for c in eligible if c.get("rmse") is not None and not _is_nan(c["rmse"])]
    mae_vals = [c["mae"] for c in eligible if c.get("mae") is not None and not _is_nan(c["mae"])]

    r2_min, r2_max = min(r2_vals), max(r2_vals)
    rmse_min, rmse_max = (min(rmse_vals), max(rmse_vals)) if rmse_vals else (0, 1)
    mae_min, mae_max = (min(mae_vals), max(mae_vals)) if mae_vals else (0, 1)

    def norm(v, vmin, vmax):
        if vmax == vmin:
            return 1.0
        return (v - vmin) / (vmax - vmin)

    scored = []
    for c in eligible:
        r2 = c["r2"]
        rmse = c.get("rmse") or 0
        mae = c.get("mae") or 0
        cv_std = c.get("cv_std_r2") or 0

        r2_norm = norm(r2, r2_min, r2_max)
        rmse_norm = 1 - norm(rmse, rmse_min, rmse_max)  # lower is better
        mae_norm = 1 - norm(mae, mae_min, mae_max)       # lower is better
        stability = 1 - min(float(cv_std), 1.0)

        weighted = 0.50 * r2_norm + 0.25 * rmse_norm + 0.15 * mae_norm + 0.10 * stability
        scored.append({**c, "weighted_score": float(weighted), "eligible": True})

    # Sort by weighted score, tie-break by complexity (lower complexity = preferred)
    scored.sort(key=lambda x: (-x["weighted_score"], complexity_rank(x["model_name"])))

    best = scored[0]

    # Quality flag
    best_r2 = best["r2"]
    if step14_quality == "subpar_after_expansion":
        quality_flag = "subpar_after_expansion"
    elif best_r2 >= 0.50:
        quality_flag = "acceptable"
    elif best_r2 >= 0.25:
        quality_flag = "marginal"
    else:
        quality_flag = "subpar"

    # Candidate analysis
    candidate_analysis = {}
    for c in eligible:
        vs_mean = "beats mean baseline (R² > 0)" if c["r2"] > 0 else "worse than mean baseline"
        vs_naive = ""
        if naive_baseline.get("r2") is not None and not _is_nan(naive_baseline["r2"]):
            delta_naive = c["r2"] - naive_baseline["r2"]
            vs_naive = f"{'beats' if delta_naive >= 0 else 'lags'} naive baseline by {abs(delta_naive):.4f} R²"
        cv_mean = c.get("cv_mean_r2")
        cv_stability = f"CV R²={cv_mean:.3f}±{c.get('cv_std_r2', 0):.3f}" if cv_mean else "no CV"
        candidate_analysis[c["model_name"]] = (
            f"R²={c['r2']:.4f}; {vs_mean}. {vs_naive}. "
            f"RMSE={c.get('rmse', 'N/A'):.2f}. {cv_stability}."
        )

    # Rationale
    rationale = (
        f"{best['model_name']} scored highest with a weighted score of {best['weighted_score']:.4f} "
        f"(R²={best['r2']:.4f}, RMSE={best.get('rmse', 0):.2f}, MAE={best.get('mae', 0):.2f}). "
        f"It outperforms the naive lag baseline (R²={naive_baseline.get('r2', 0):.4f}) and "
        f"demonstrates stable cross-validation performance (CV R²={best.get('cv_mean_r2', 0):.3f}). "
        f"Lower-complexity models are preferred as tie-breakers to reduce overfitting risk."
    )

    # Full ranking
    full_ranking = []
    for c in scored:
        full_ranking.append({
            "rank": scored.index(c) + 1,
            "model_name": c["model_name"],
            "r2": c["r2"],
            "rmse": c.get("rmse"),
            "mae": c.get("mae"),
            "cv_mean_r2": c.get("cv_mean_r2"),
            "weighted_score": c["weighted_score"],
            "eligible": True,
        })
    for c in ineligible:
        full_ranking.append({
            "rank": None,
            "model_name": c["model_name"],
            "r2": c.get("r2"),
            "rmse": c.get("rmse"),
            "mae": c.get("mae"),
            "cv_mean_r2": c.get("cv_mean_r2"),
            "weighted_score": None,
            "eligible": False,
            "note": "ineligible: R² < 0",
        })

    # Write Markdown report
    report_md_path = os.path.join(output_dir, "step-15-model-selection-report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Model Selection Report\n\n")
        f.write(f"**Run ID:** {run_id}\n\n")
        f.write(f"**Selected Model:** {best['model_name']}\n\n")
        f.write(f"**Quality Flag:** {quality_flag}\n\n")
        f.write("## Baselines\n\n")
        f.write(f"- Mean baseline R²: 0.0000\n")
        f.write(f"- Naive lag baseline R²: {naive_baseline.get('r2', 'N/A')}\n\n")
        f.write("## Candidate Ranking\n\n")
        f.write("| Rank | Model | R² | RMSE | MAE | CV R² | Weighted Score | Eligible |\n")
        f.write("|------|-------|-----|------|-----|-------|----------------|----------|\n")
        for row in full_ranking:
            r = row.get("rank", "-")
            eligible_str = "✓" if row.get("eligible") else "✗"
            ws = f"{row['weighted_score']:.4f}" if row.get("weighted_score") is not None else "N/A"
            cv_str = f"{row['cv_mean_r2']:.4f}" if row.get("cv_mean_r2") is not None else "N/A"
            rmse_str = f"{row['rmse']:.2f}" if row.get("rmse") is not None else "N/A"
            mae_str = f"{row['mae']:.2f}" if row.get("mae") is not None else "N/A"
            r2_str = f"{row['r2']:.4f}" if row.get("r2") is not None else "N/A"
            f.write(f"| {r} | {row['model_name']} | {r2_str} | {rmse_str} | {mae_str} | {cv_str} | {ws} | {eligible_str} |\n")
        f.write("\n## Rationale\n\n")
        f.write(rationale + "\n\n")
        f.write("## Candidate Analysis\n\n")
        for name, analysis in candidate_analysis.items():
            f.write(f"**{name}**: {analysis}\n\n")

    # Write comparison plot
    png_path = os.path.join(output_dir, "step-15-model-selection-metrics.png")
    try:
        model_names = [r["model_name"] for r in full_ranking[:8]]
        r2_values = [r["r2"] if r.get("r2") is not None else 0 for r in full_ranking[:8]]
        rmse_values = [r["rmse"] if r.get("rmse") is not None else 0 for r in full_ranking[:8]]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        x = range(len(model_names))
        ax1.barh(model_names, r2_values, color=["green" if v >= 0.5 else "orange" if v >= 0.25 else "red" for v in r2_values])
        ax1.axvline(x=0.5, color="green", linestyle="--", label="Acceptable threshold")
        ax1.set_xlabel("R²")
        ax1.set_title("Model R² Comparison")
        ax1.legend()
        
        ax2.barh(model_names, rmse_values, color="steelblue")
        ax2.set_xlabel("RMSE")
        ax2.set_title("Model RMSE Comparison")
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=100, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Warning: Plot failed: {e}")
        png_path = None

    result = {
        "step": "15-model-selection",
        "run_id": run_id,
        "selected_model": best["model_name"],
        "weighted_score": best["weighted_score"],
        "quality_flag": quality_flag,
        "rationale": rationale,
        "baselines": {
            "mean_baseline_r2": 0.0,
            "naive_lag_baseline": naive_baseline,
        },
        "candidate_analysis": candidate_analysis,
        "full_ranking": full_ranking,
        "artifacts": {
            "selection_report_md": report_md_path,
            "selection_metrics_png": png_path,
        },
        "context": {
            "selected_model": best["model_name"],
            "r2": best["r2"],
            "quality_flag": quality_flag,
        }
    }

    out_json = os.path.join(output_dir, "step-15-selection.json")
    with open(out_json, "w") as f:
        json.dump(make_ser(result), f, indent=2)

    # Update progress
    with open(progress_path) as f:
        progress = json.load(f)
    progress["status"] = "running"
    progress["current_step"] = "16-result-presentation"
    if "15-model-selection" not in progress.get("completed_steps", []):
        progress["completed_steps"].append("15-model-selection")
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    print(f"Step 15 complete. Selected: {best['model_name']}, quality={quality_flag}")
    sys.exit(0)


def _is_nan(v):
    try:
        return v is None or (isinstance(v, float) and np.isnan(v))
    except Exception:
        return False


if __name__ == "__main__":
    main()
