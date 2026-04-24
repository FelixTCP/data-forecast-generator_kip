#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from runtime_utils import bootstrap_progress, validate_existing_output


STEPS = [
    ("00-pre-exploration", "step_00_pre_exploration.py", "step-00_profiler.json"),
    ("01-csv-read-cleansing", "step_01_cleanse.py", "step-01-cleanse.json"),
    ("11-data-exploration", "step_11_exploration.py", "step-11-exploration.json"),
    ("12-feature-extraction", "step_12_features.py", "step-12-features.json"),
    ("13-model-training", "step_13_training.py", "step-13-training.json"),
    ("14-model-evaluation", "step_14_evaluation.py", "step-14-evaluation.json"),
    ("15-model-selection", "step_15_selection.py", "step-15-selection.json"),
    ("16-result-presentation", "step_16_report.py", None),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split-mode", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    code_dir = Path(__file__).resolve().parent
    bootstrap_progress(output_dir, args.run_id, args.csv_path, args.target_column, args.split_mode)

    for step_name, script_name, output_name in STEPS:
        if args.resume and output_name and validate_existing_output(output_dir / output_name, step_name):
            continue
        cmd = [sys.executable, str(code_dir / script_name), "--output-dir", str(output_dir), "--run-id", args.run_id]
        if script_name == "step_00_pre_exploration.py":
            cmd.extend(["--csv-path", args.csv_path])
        elif script_name == "step_01_cleanse.py":
            cmd.extend(["--csv-path", args.csv_path, "--target-column", args.target_column])
        elif script_name == "step_12_features.py":
            cmd.extend(["--target-column", args.target-column if False else args.target_column, "--split-mode", args.split_mode])
        elif script_name == "step_13_training.py":
            cmd.extend(["--split-mode", args.split_mode, "--target-column", args.target_column])
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
