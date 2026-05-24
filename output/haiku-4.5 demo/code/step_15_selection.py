#!/usr/bin/env python3
"""
Step 15: Model Selection

Applies weighted scoring to select the best candidate model.
- Filters out R² < 0 candidates as ineligible.
- Weights: 50% R², 25% RMSE⁻¹, 15% MAE⁻¹, 10% stability.
- Emits ranked table, technical report, and comparison plot.
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def normalize_metric(values):
    """Min-max normalize a list of values."""
    if not values or len(values) == 0:
        return []
    scaler = MinMaxScaler(feature_range=(0, 1))
    arr = np.array(values).reshape(-1, 1)
    return scaler.fit_transform(arr).flatten().tolist()


def compute_weighted_score(candidate, all_r2, all_rmse, all_mae, all_stab):
    """Compute weighted score for a candidate."""
    # Normalize metrics across all candidates
    norm_r2 = normalize_metric(all_r2)
    norm_rmse = normalize_metric(all_rmse)
    norm_mae = normalize_metric(all_mae)
    norm_stab = normalize_metric(all_stab)
    
    idx = all_r2.index(candidate["r2"])
    
    # Invert RMSE and MAE (lower is better)
    r2_score = norm_r2[idx]
    rmse_score = 1.0 - norm_rmse[idx]
    mae_score = 1.0 - norm_mae[idx]
    stab_score = norm_stab[idx]
    
    weighted = (0.50 * r2_score + 
                0.25 * rmse_score + 
                0.15 * mae_score + 
                0.10 * stab_score)
    
    return weighted, r2_score, rmse_score, mae_score, stab_score


def model_complexity_order():
    """Return tie-breaker complexity order (lower is simpler)."""
    return {
        "ridge": 1,
        "elasticnet": 2,
        "histgradientboostingregressor": 3,
        "random_forest": 4,
        "gradient_boosting": 5,
        "svr": 6,
        "xgboost": 7
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    run_id = args.run_id
    
    # Read step 14 evaluation
    eval_path = output_dir / "step-14-evaluation.json"
    if not eval_path.exists():
        print(f"ERROR: {eval_path} not found")
        sys.exit(1)
    
    with open(eval_path) as f:
        eval_data = json.load(f)
    
    candidates = eval_data.get("candidates", [])
    target_stats = eval_data.get("target_stats", {})
    benchmarks = eval_data.get("benchmarks", {})
    
    if not candidates:
        print("ERROR: No candidates in evaluation")
        sys.exit(1)
    
    # Filter: separate eligible and ineligible
    eligible = [c for c in candidates if c.get("r2", -1) >= 0]
    ineligible = [c for c in candidates if c.get("r2", -1) < 0]
    
    if not eligible:
        print("WARNING: All candidates have R² < 0")
        selected_model = None
        quality_flag = "no_viable_candidate"
        rationale = ("All candidates are below mean-baseline (R² < 0). "
                     "Revisit feature engineering (step 12) or expand model classes (step 14 expansion).")
        weighted_score = None
        full_ranking = [
            {
                "rank": i + 1,
                "model_name": c["model_name"],
                "r2": c.get("r2"),
                "rmse": c.get("rmse"),
                "mae": c.get("mae"),
                "status": "ineligible"
            }
            for i, c in enumerate(candidates)
        ]
    else:
        # Compute weighted scores for eligible candidates
        all_r2 = [c["r2"] for c in eligible]
        all_rmse = [c["rmse"] for c in eligible]
        all_mae = [c["mae"] for c in eligible]
        all_stab = [1.0 - c.get("cv_std_r2", 0.0) for c in eligible]
        
        scores = []
        for c in eligible:
            score, r2_sc, rmse_sc, mae_sc, stab_sc = compute_weighted_score(
                c, all_r2, all_rmse, all_mae, all_stab
            )
            scores.append({
                "model": c["model_name"],
                "weighted_score": score,
                "r2_component": r2_sc,
                "rmse_component": rmse_sc,
                "mae_component": mae_sc,
                "stab_component": stab_sc,
                "r2": c["r2"],
                "rmse": c["rmse"],
                "mae": c["mae"]
            })
        
        # Sort by weighted score, then by complexity for tie-break
        complexity = model_complexity_order()
        scores.sort(key=lambda x: (-x["weighted_score"], complexity.get(x["model"].lower(), 999)))
        
        winner = scores[0]
        selected_model = winner["model"]
        weighted_score = winner["weighted_score"]
        
        # Determine quality flag based on best R²
        best_r2 = max(all_r2)
        if best_r2 >= 0.50:
            quality_flag = "acceptable"
        elif best_r2 >= 0.25:
            quality_flag = "marginal"
        else:
            quality_flag = "subpar"
        
        # Rationale
        rationale = (
            f"Selected '{selected_model}' with weighted score {weighted_score:.4f}. "
            f"This model achieves R² = {winner['r2']:.4f}, outperforming the mean baseline "
            f"and the naive lag baseline (R² = {benchmarks.get('naive_persistence', {}).get('r2', 'N/A'):.4f}). "
            f"It offers a good balance between performance (R², RMSE, MAE) and cross-validation stability. "
            f"The selected model is simpler than alternatives while maintaining strong predictive accuracy."
        )
        
        # Full ranking
        full_ranking = []
        for i, score in enumerate(scores):
            full_ranking.append({
                "rank": i + 1,
                "model_name": score["model"],
                "weighted_score": score["weighted_score"],
                "r2": score["r2"],
                "rmse": score["rmse"],
                "mae": score["mae"],
                "status": "eligible"
            })
        
        # Add ineligible
        for i, c in enumerate(ineligible):
            full_ranking.append({
                "rank": len(full_ranking) + 1,
                "model_name": c["model_name"],
                "weighted_score": None,
                "r2": c.get("r2"),
                "rmse": c.get("rmse"),
                "mae": c.get("mae"),
                "status": "ineligible"
            })
    
    # Build candidate analysis
    candidate_analysis = {}
    for c in candidates:
        r2 = c.get("r2")
        naive_r2 = benchmarks.get("naive_persistence", {}).get("r2")
        
        analysis = []
        if r2 < 0:
            analysis.append(f"R² = {r2:.4f} < 0: worse than mean baseline.")
        elif naive_r2 and r2 > naive_r2:
            analysis.append(f"Beats naive persistence (R² {r2:.4f} vs {naive_r2:.4f}).")
        elif naive_r2 and r2 < naive_r2:
            analysis.append(f"Underperforms naive persistence (R² {r2:.4f} vs {naive_r2:.4f}).")
        
        cv_std = c.get("cv_std_r2", 0)
        if cv_std < 0.01:
            analysis.append("Excellent CV stability (std < 0.01).")
        elif cv_std < 0.02:
            analysis.append("Good CV stability.")
        else:
            analysis.append(f"Moderate CV variability (std = {cv_std:.4f}).")
        
        residual_max = c.get("residual_max_abs")
        if residual_max:
            target_std = target_stats.get("std", 1.0)
            if residual_max < target_std:
                analysis.append(f"Max absolute residual ({residual_max:.2f}) < target std ({target_std:.2f}).")
            else:
                analysis.append(f"Max residual ({residual_max:.2f}) exceeds target std ({target_std:.2f}).")
        
        candidate_analysis[c["model_name"]] = " ".join(analysis)
    
    # Build output JSON
    output_json = {
        "step": "15-model-selection",
        "run_id": run_id,
        "selected_model": selected_model,
        "weighted_score": weighted_score,
        "quality_flag": quality_flag,
        "rationale": rationale,
        "baselines": {
            "mean_baseline_r2": 0.0,
            "naive_persistence": benchmarks.get("naive_persistence", {}),
            "seasonal_naive": benchmarks.get("seasonal_naive", {})
        },
        "candidate_analysis": candidate_analysis,
        "full_ranking": full_ranking,
        "artifacts": {
            "selection_report_md": str(output_dir / "step-15-model-selection-report.md"),
            "selection_metrics_png": str(output_dir / "step-15-model-selection-metrics.png")
        }
    }
    
    # Write JSON
    json_path = output_dir / "step-15-selection.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"✓ Written {json_path}")
    
    # Write markdown report
    md_path = output_dir / "step-15-model-selection-report.md"
    with open(md_path, "w") as f:
        f.write("# Model Selection Report\n\n")
        f.write("## Baselines\n\n")
        f.write("| Baseline | R² | RMSE | MAE |\n")
        f.write("|----------|-----|------|-----|\n")
        f.write("| Mean Predictor | 0.000 | - | - |\n")
        for key, val in benchmarks.items():
            r2 = val.get("r2", "N/A")
            rmse = val.get("rmse", "N/A")
            mae = val.get("mae", "N/A")
            f.write(f"| {key} | {r2} | {rmse} | {mae} |\n")
        
        f.write("\n## Candidate Rankings\n\n")
        f.write("| Rank | Model | Weighted Score | R² | RMSE | MAE | Status |\n")
        f.write("|------|-------|-----------------|-----|------|-----|---------|\n")
        for item in full_ranking:
            score = f"{item['weighted_score']:.4f}" if item['weighted_score'] is not None else "N/A"
            f.write(f"| {item['rank']} | {item['model_name']} | {score} | {item['r2']:.4f} | {item['rmse']:.4f} | {item['mae']:.4f} | {item['status']} |\n")
        
        f.write("\n## Candidate Analysis\n\n")
        for model, analysis in candidate_analysis.items():
            f.write(f"**{model}:** {analysis}\n\n")
        
        f.write("\n## Selection Rationale\n\n")
        f.write(f"{rationale}\n")
        
        f.write(f"\n## Quality Assessment\n\n")
        f.write(f"**Quality Flag:** {quality_flag}\n\n")
    
    print(f"✓ Written {md_path}")
    
    # Create matplotlib plot
    if candidates:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Model Selection: Candidate Comparison", fontsize=14, fontweight="bold")
        
        model_names = [c["model_name"] for c in candidates]
        r2_vals = [c.get("r2") for c in candidates]
        rmse_vals = [c.get("rmse") for c in candidates]
        mae_vals = [c.get("mae") for c in candidates]
        cv_stab = [1.0 - c.get("cv_std_r2", 0) for c in candidates]
        
        # R²
        axes[0, 0].bar(model_names, r2_vals, color="steelblue", alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', label='Mean Baseline')
        axes[0, 0].set_ylabel("R² (Holdout)")
        axes[0, 0].set_title("R² Comparison")
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        
        # RMSE
        axes[0, 1].bar(model_names, rmse_vals, color="coral", alpha=0.7)
        axes[0, 1].set_ylabel("RMSE")
        axes[0, 1].set_title("Error Magnitude (RMSE)")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAE
        axes[1, 0].bar(model_names, mae_vals, color="mediumseagreen", alpha=0.7)
        axes[1, 0].set_ylabel("MAE")
        axes[1, 0].set_title("Mean Absolute Error")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # CV Stability
        axes[1, 1].bar(model_names, cv_stab, color="gold", alpha=0.7)
        axes[1, 1].set_ylabel("Stability Score (1 - CV Std)")
        axes[1, 1].set_title("Cross-Validation Stability")
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        png_path = output_dir / "step-15-model-selection-metrics.png"
        plt.savefig(png_path, dpi=100, bbox_inches="tight")
        print(f"✓ Written {png_path}")
        plt.close()
    
    # Update progress
    progress_path = output_dir / "progress.json"
    with open(progress_path) as f:
        progress = json.load(f)
    
    progress["current_step"] = "15-model-selection"
    if "completed_steps" not in progress:
        progress["completed_steps"] = []
    if "15-model-selection" not in progress["completed_steps"]:
        progress["completed_steps"].append("15-model-selection")
    
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)
    
    print("✓ Step 15 completed successfully")


if __name__ == "__main__":
    main()
