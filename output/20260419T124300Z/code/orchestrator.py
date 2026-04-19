#!/usr/bin/env python3
"""
Orchestrator: Run pipeline steps sequentially with resume capability.
"""

import json
import os
import sys
import subprocess
import argparse
from pathlib import Path

def load_progress(output_dir: str) -> dict:
    """Load progress.json."""
    progress_path = os.path.join(output_dir, "progress.json")
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            return json.load(f)
    return {
        "completed_steps": [],
        "current_step": None,
        "status": "running",
        "errors": []
    }

def run_step(step_name: str, step_script: str, args: list, code_dir: str, output_dir: str) -> bool:
    """Run a single step script."""
    step_path = os.path.join(code_dir, step_script)
    
    if not os.path.exists(step_path):
        print(f"✗ {step_name}: Script not found: {step_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"[ORCHESTRATOR] Running {step_name}...")
    print(f"{'='*60}")
    
    cmd = ["python", step_path] + args
    
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)
        print(f"✓ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {step_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {step_name} failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Pipeline Orchestrator")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--csv-path", help="CSV path (for step 01)")
    parser.add_argument("--target-column", required=True, help="Target column")
    parser.add_argument("--split-mode", default="auto", help="Split mode")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed step")
    parser.add_argument("--from-step", help="Start from specific step (e.g., 12)")
    
    args = parser.parse_args()
    
    code_dir = os.path.join(args.output_dir, "code")
    
    # Load progress
    progress = load_progress(args.output_dir)
    
    # Determine which steps to run
    steps_to_run = []
    
    if args.from_step:
        # Run from specific step
        start_idx = int(args.from_step)
        steps_to_run = list(range(start_idx, 17))
    elif args.resume:
        # Resume from last incomplete step
        if progress.get("status") == "completed":
            print("✓ Pipeline already completed")
            return 0
        
        completed = progress.get("completed_steps", [])
        for step_num in range(1, 17):
            step_name = f"{step_num:02d}"
            found = False
            for comp in completed:
                if step_name in comp:
                    found = True
                    break
            if not found:
                steps_to_run = list(range(step_num, 17))
                break
    else:
        # Run all steps
        steps_to_run = list(range(1, 17))
    
    print(f"[ORCHESTRATOR] Steps to execute: {steps_to_run}")
    print(f"[ORCHESTRATOR] Output directory: {args.output_dir}")
    
    # Step definitions: (step_number, name, script, args)
    step_configs = {
        12: ("12-Feature Extraction", "step_12_features.py", [
            "--target-column", args.target_column.lower(),
            "--split-mode", args.split_mode,
            "--output-dir", args.output_dir,
            "--run-id", args.run_id,
        ]),
        13: ("13-Model Training", "step_13_training.py", [
            "--output-dir", args.output_dir,
            "--run-id", args.run_id,
            "--target-column", args.target_column.lower(),
            "--split-mode", args.split_mode,
        ]),
        14: ("14-Model Evaluation", "step_14_evaluation.py", [
            "--output-dir", args.output_dir,
            "--run-id", args.run_id,
        ]),
        15: ("15-Model Selection", "step_15_selection.py", [
            "--output-dir", args.output_dir,
            "--run-id", args.run_id,
        ]),
        16: ("16-Result Presentation", "step_16_report.py", [
            "--output-dir", args.output_dir,
            "--run-id", args.run_id,
            "--csv-path", args.csv_path or "data/appliances_energy_prediction.csv",
            "--target-column", args.target_column,
        ]),
    }
    
    # Run steps
    failed_steps = []
    
    for step_num in steps_to_run:
        if step_num not in step_configs:
            print(f"⊘ Step {step_num} not configured")
            continue
        
        step_name, script, step_args = step_configs[step_num]
        
        # Check if step already completed (for resume)
        if args.resume:
            completed = progress.get("completed_steps", [])
            step_str = f"{step_num:02d}"
            if any(step_str in c for c in completed):
                print(f"⊘ {step_name}: Already completed (skipping)")
                continue
        
        success = run_step(step_name, script, step_args, code_dir, args.output_dir)
        
        if not success:
            failed_steps.append(step_name)
            print(f"\n✗ Pipeline halted at {step_name}")
            break
    
    # Final status
    print(f"\n{'='*60}")
    if not failed_steps:
        print("✓ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        return 0
    else:
        print(f"✗ PIPELINE FAILED at: {', '.join(failed_steps)}")
        print(f"{'='*60}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
