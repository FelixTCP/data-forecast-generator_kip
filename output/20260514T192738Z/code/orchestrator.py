#!/usr/bin/env python3
"""
Orchestrator: Thin wrapper that calls all pipeline steps in order.
Supports resume mode via progress.json.
"""
import json
import subprocess
import sys
from pathlib import Path
from argparse import ArgumentParser


def run_orchestrator(csv_path, target_column, output_dir, run_id, continue_mode=False):
    """
    Execute all pipeline steps in order.
    Supports resume from last completed step.
    """
    output_dir_path = Path(output_dir)
    code_dir = output_dir_path / "code"
    progress_path = output_dir_path / "progress.json"
    
    # Load or initialize progress
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)
    else:
        progress = {
            "run_id": run_id,
            "csv_path": csv_path,
            "target_column": target_column,
            "status": "running",
            "current_step": None,
            "completed_steps": [],
            "errors": [],
        }
    
    # Define all steps
    steps = [
        ("10-csv-read-cleansing", "step_10_cleanse.py", ["--csv-path", csv_path]),
        ("11-data-exploration", "step_11_exploration.py", ["--target-column", target_column]),
        ("12-feature-extraction", "step_12_features.py", []),
        ("13-model-training", "step_13_training.py", []),
        ("14-model-evaluation", "step_14_evaluation.py", []),
        ("15-model-selection", "step_15_selection.py", []),
        ("16-result-presentation", "step_16_report.py", []),
    ]
    
    # Execute steps
    for step_name, script_name, extra_args in steps:
        if continue_mode and step_name in progress.get("completed_steps", []):
            print(f"ℹ️  Step {step_name} already complete, skipping...")
            continue
        
        progress["current_step"] = step_name
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Executing: {step_name}")
        print(f"{'='*70}")
        
        script_path = code_dir / script_name
        cmd = [
            "python", str(script_path),
            "--output-dir", str(output_dir_path),
            "--run-id", run_id,
            *extra_args,
        ]
        
        result = subprocess.run(cmd, cwd=str(output_dir_path))
        
        if result.returncode != 0:
            error_msg = f"Step {step_name} failed with exit code {result.returncode}"
            progress["errors"].append(error_msg)
            progress["status"] = "error"
            with open(progress_path, 'w') as f:
                json.dump(progress, f, indent=2)
            print(f"\n✗ {error_msg}")
            return False
    
    print(f"\n{'='*70}")
    print(f"✓ ALL STEPS COMPLETE")
    print(f"{'='*70}")
    
    return True


def main():
    parser = ArgumentParser(description="Regression Forecasting Pipeline Orchestrator")
    parser.add_argument("--csv-path", required=True, help="Path to input CSV")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed step")
    
    args = parser.parse_args()
    
    success = run_orchestrator(
        csv_path=args.csv_path,
        target_column=args.target_column,
        output_dir=args.output_dir,
        run_id=args.run_id,
        continue_mode=args.resume,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
