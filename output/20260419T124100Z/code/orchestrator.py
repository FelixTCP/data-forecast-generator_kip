#!/usr/bin/env python3
"""Thin orchestrator for step-wise pipeline execution."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StepDef:
    name: str
    script: str
    output_file: str
    expected_step_value: str | None
    needs_csv_path: bool = False
    needs_target: bool = False
    needs_split_mode: bool = False


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _is_step_complete(step: StepDef, output_dir: Path) -> bool:
    out_path = output_dir / step.output_file
    if not out_path.exists():
        return False
    if out_path.suffix.lower() != ".json":
        return True
    payload = _load_json(out_path)
    if not payload:
        return False
    if step.expected_step_value is None:
        return True
    return payload.get("step") == step.expected_step_value


def _run_step(
    step: StepDef,
    code_dir: Path,
    csv_path: str,
    target_column: str,
    split_mode: str,
    output_dir: Path,
    run_id: str,
) -> int:
    cmd = [sys.executable, str(code_dir / step.script), "--output-dir", str(output_dir), "--run-id", run_id]
    if step.needs_csv_path:
        cmd.extend(["--csv-path", csv_path])
    if step.needs_target:
        cmd.extend(["--target-column", target_column])
    if step.needs_split_mode:
        cmd.extend(["--split-mode", split_mode])

    print(f"Running {step.name}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run agentic pipeline steps in order")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split-mode", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-step", action="append", default=[])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    code_dir = Path(args.code_dir)

    steps = [
        StepDef("00-pre-exploration", "step_00_pre_exploration.py", "step-00_profiler.json", "00-pre-exploration", needs_csv_path=True),
        StepDef("01-csv-read-cleansing", "step_01_cleanse.py", "step-01-cleanse.json", "01-csv-read-cleansing", needs_csv_path=True, needs_target=True),
        StepDef("11-data-exploration", "step_11_exploration.py", "step-11-exploration.json", "11-data-exploration"),
        StepDef("12-feature-extraction", "step_12_features.py", "step-12-features.json", "12-feature-extraction", needs_target=True, needs_split_mode=True),
        StepDef("13-model-training", "step_13_training.py", "step-13-training.json", "13-model-training", needs_target=True, needs_split_mode=True),
        StepDef("14-model-evaluation", "step_14_evaluation.py", "step-14-evaluation.json", "14-model-evaluation"),
        StepDef("15-model-selection", "step_15_selection.py", "step-15-selection.json", "15-model-selection"),
        StepDef("16-result-presentation", "step_16_report.py", "step-16-report.md", None),
    ]

    force_steps = set(args.force_step)
    for step in steps:
        if args.resume and step.name not in force_steps and _is_step_complete(step, output_dir):
            print(f"Step already complete, skipping due to --resume: {step.name}")
            continue

        exit_code = _run_step(
            step=step,
            code_dir=code_dir,
            csv_path=args.csv_path,
            target_column=args.target_column,
            split_mode=args.split_mode,
            output_dir=output_dir,
            run_id=args.run_id,
        )
        if exit_code != 0:
            print(f"Step failed with exit code {exit_code}: {step.name}")
            return exit_code

    print("Pipeline orchestrator run completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
