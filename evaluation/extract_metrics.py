#!/usr/bin/env python3
"""
Extract metrics from output folders in the data-forecast-generator pipeline.

Iterates over timestamped output folders and collects metrics from:
  - progress.json: error rate, status, completed steps
  - step-13-training.json: best model and performance metrics (R², RMSE, MAE)
  - step-15-selection.json: quality flags and model ranking
  - stats.json: LLM model name and running time (when available)

Outputs a JSONL file (one JSON object per line) with all extracted metrics.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


def extract_metrics_from_folder(folder_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract metrics from a single output folder.
    
    Args:
        folder_path: Path to the timestamped output folder (e.g., 20260520T073850Z)
    
    Returns:
        Dictionary of extracted metrics, or None if extraction fails
    """
    metrics = {
        "timestamp": folder_path.name,
        "timestamp_parsed": None,
        "csv_path": None,
        "target_column": None,
        "status": None,
        "error_count": 0,
        "error_rate": 0.0,
        "completed_steps": [],
        "best_model": None,
        "best_r2": None,
        "best_rmse": None,
        "best_mae": None,
        "model_candidates": [],
        "quality_flag": None,
        "llm_model_name": None,
        "running_time_sec": None,
        "tokens_generated": None,
        "tokens_used": None,
    }
    
    # Parse timestamp from folder name (ISO 8601 format: 20260520T073850Z)
    try:
        metrics["timestamp_parsed"] = datetime.strptime(
            folder_path.name, "%Y%m%dT%H%M%SZ"
        ).isoformat()
    except ValueError:
        print(f"  Warning: Could not parse timestamp from folder name: {folder_path.name}")
    
    # 1. Extract from progress.json
    progress_file = folder_path / "progress.json"
    if progress_file.exists():
        try:
            with open(progress_file, "r") as f:
                progress = json.load(f)
            metrics["csv_path"] = progress.get("csv_path")
            metrics["target_column"] = progress.get("target_column")
            metrics["status"] = progress.get("status")
            metrics["completed_steps"] = progress.get("completed_steps", [])
            
            errors = progress.get("errors", [])
            metrics["error_count"] = len(errors)
            # Error rate: number of errors / number of completed steps (if available)
            total_steps = len(progress.get("completed_steps", []))
            if total_steps > 0:
                metrics["error_rate"] = metrics["error_count"] / total_steps
        except Exception as e:
            print(f"  Warning: Error reading progress.json: {e}")
    else:
        print(f"  Warning: progress.json not found in {folder_path.name}")
    
    # 2. Extract from step-13-training.json
    training_file = folder_path / "step-13-training.json"
    if training_file.exists():
        try:
            with open(training_file, "r") as f:
                training = json.load(f)
            
            best_model = training.get("best_model")
            metrics["best_model"] = best_model
            
            # Extract best scores
            best_score = training.get("best_score", {})
            metrics["best_r2"] = best_score.get("r2")
            metrics["best_rmse"] = best_score.get("rmse")
            metrics["best_mae"] = best_score.get("mae")
            
            # List of candidate models
            metrics["model_candidates"] = training.get("model_candidates", [])
        except Exception as e:
            print(f"  Warning: Error reading step-13-training.json: {e}")
    else:
        print(f"  Warning: step-13-training.json not found in {folder_path.name}")
    
    # 3. Extract from step-15-selection.json
    selection_file = folder_path / "step-15-selection.json"
    if selection_file.exists():
        try:
            with open(selection_file, "r") as f:
                selection = json.load(f)
            metrics["quality_flag"] = selection.get("quality_flag")
        except Exception as e:
            print(f"  Warning: Error reading step-15-selection.json: {e}")
    else:
        print(f"  Info: step-15-selection.json not found in {folder_path.name}")
    
    # 4. Extract from stats.json (if available)
    stats_file = folder_path / "stats.json"
    if stats_file.exists():
        try:
            with open(stats_file, "r") as f:
                stats = json.load(f)
            metrics["llm_model_name"] = stats.get("llm_model_name")
            metrics["running_time_sec"] = stats.get("running_time_sec")
            metrics["tokens_generated"] = stats.get("tokens_generated")
            metrics["tokens_used"] = stats.get("tokens_used")
        except Exception as e:
            print(f"  Warning: Error reading stats.json: {e}")
    else:
        print(f"  Info: stats.json not found in {folder_path.name} (not yet generated)")
    
    return metrics


def extract_all_metrics(output_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract metrics from all timestamped folders in the output directory.
    
    Args:
        output_dir: Path to the output directory containing timestamped folders
    
    Returns:
        List of metric dictionaries, one per folder
    """
    metrics_list = []
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return metrics_list
    
    # Find all timestamped folders (directories matching YYYYMMDDTHHMMSSZ pattern)
    timestamped_folders = sorted([
        folder for folder in output_dir.iterdir()
        if folder.is_dir() and len(folder.name) == 16 and folder.name[8] == 'T'
    ])
    
    print(f"\nFound {len(timestamped_folders)} output folder(s) to process:")
    for folder in timestamped_folders:
        print(f"\nProcessing: {folder.name}")
        metrics = extract_metrics_from_folder(folder)
        if metrics:
            metrics_list.append(metrics)
            print(f"  ✓ Extracted metrics for {folder.name}")
        else:
            print(f"  ✗ Failed to extract metrics for {folder.name}")
    
    return metrics_list


def save_metrics_jsonl(metrics_list: List[Dict[str, Any]], output_file: Path) -> None:
    """
    Save metrics as JSONL (one JSON object per line).
    
    Args:
        metrics_list: List of metric dictionaries
        output_file: Path to output JSONL file
    """
    try:
        with open(output_file, "w") as f:
            for metrics in metrics_list:
                f.write(json.dumps(metrics) + "\n")
        print(f"\n✓ Saved {len(metrics_list)} metric records to: {output_file}")
    except Exception as e:
        print(f"Error saving metrics to {output_file}: {e}")


def save_metrics_summary_json(metrics_list: List[Dict[str, Any]], output_file: Path) -> None:
    """
    Save metrics as a single JSON file (array format) for easier viewing.
    
    Args:
        metrics_list: List of metric dictionaries
        output_file: Path to output JSON file
    """
    try:
        with open(output_file, "w") as f:
            json.dump(metrics_list, f, indent=2)
        print(f"✓ Saved metrics summary to: {output_file}")
    except Exception as e:
        print(f"Error saving summary to {output_file}: {e}")


def main():
    """Main entry point."""
    # Define paths
    current_dir = Path(__file__).parent
    output_dir = current_dir.parent / "output"
    
    metrics_jsonl = current_dir / "metrics.jsonl"
    metrics_json = current_dir / "metrics.json"
    
    print("=" * 70)
    print("Data Forecast Generator - Metrics Extraction")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    
    # Extract metrics from all folders
    metrics_list = extract_all_metrics(output_dir)
    
    if not metrics_list:
        print("\nNo metrics could be extracted.")
        return
    
    # Save as JSONL
    save_metrics_jsonl(metrics_list, metrics_jsonl)
    
    # Save as JSON summary
    save_metrics_summary_json(metrics_list, metrics_json)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Total runs processed: {len(metrics_list)}")
    
    successful_runs = [m for m in metrics_list if m["status"] == "completed"]
    print(f"Successful runs: {len(successful_runs)}")
    
    if successful_runs:
        avg_r2 = sum(m["best_r2"] for m in successful_runs if m["best_r2"] is not None) / len([m for m in successful_runs if m["best_r2"] is not None])
        print(f"Average Best R²: {avg_r2:.4f}")
        
        avg_rmse = sum(m["best_rmse"] for m in successful_runs if m["best_rmse"] is not None) / len([m for m in successful_runs if m["best_rmse"] is not None])
        print(f"Average Best RMSE: {avg_rmse:.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
