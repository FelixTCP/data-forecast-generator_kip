"""Orchestrator — runs all pipeline steps in order.

Usage:
    python orchestrator.py --csv-path <path> --target-column <col> \
                           --output-dir <dir> --run-id <id> [--resume]
                           [--force-step <NN>]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


STEPS = [
    {
        "name": "10-csv-read-cleansing",
        "script": "step_10_cleanse.py",
        "output_json": "step-10-cleanse.json",
        "extra_args": lambda args: [
            "--csv-path", args.csv_path,
            "--target-column", args.target_column,
        ],
    },
    {
        "name": "11-data-exploration",
        "script": "step_11_exploration.py",
        "output_json": "step-11-exploration.json",
        "extra_args": lambda args: [],
    },
    {
        "name": "12-feature-extraction",
        "script": "step_12_features.py",
        "output_json": "step-12-features.json",
        "extra_args": lambda args: [],
    },
    {
        "name": "13-model-training",
        "script": "step_13_training.py",
        "output_json": "step-13-training.json",
        "extra_args": lambda args: [],
    },
    {
        "name": "14-model-evaluation",
        "script": "step_14_evaluation.py",
        "output_json": "step-14-evaluation.json",
        "extra_args": lambda args: [],
    },
    {
        "name": "15-model-selection",
        "script": "step_15_selection.py",
        "output_json": "step-15-selection.json",
        "extra_args": lambda args: [],
    },
    {
        "name": "16-result-presentation",
        "script": "step_16_report.py",
        "output_json": "step-16-report.md",
        "extra_args": lambda args: [],
    },
]


def _step_is_complete(step: dict, output_dir: Path, completed_steps: list[str]) -> bool:
    """Check if a step can be skipped (resume mode)."""
    if step["name"] not in completed_steps:
        return False
    output_file = output_dir / step["output_json"]
    if not output_file.exists():
        return False
    if output_file.suffix == ".json":
        try:
            with open(output_file) as f:
                data = json.load(f)
            if "step" not in data:
                return False
        except Exception:
            return False
    return True


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _write_code_audit(code_dir: Path, output_dir: Path):
    audit = {"files": []}
    for py_file in sorted(code_dir.glob("*.py")):
        audit["files"].append({
            "filename": py_file.name,
            "path": str(py_file),
            "sha256": _file_hash(py_file),
            "size_bytes": py_file.stat().st_size,
        })
    audit_path = output_dir / "code_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Written code_audit.json ({len(audit['files'])} files)")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Orchestrator")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--force-step", type=str, default=None,
                        help="Force re-run of step NN even in resume mode")
    args = parser.parse_args()

    code_dir = Path(__file__).parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_path = output_dir / "progress.json"

    # Initialize or load progress
    if progress_path.exists() and args.resume:
        with open(progress_path) as f:
            progress = json.load(f)
        completed_steps = progress.get("completed_steps", [])
        print(f"Resuming from: completed_steps={completed_steps}")
    else:
        progress = {
            "run_id": args.run_id,
            "csv_path": args.csv_path,
            "target_column": args.target_column,
            "status": "running",
            "current_step": None,
            "completed_steps": [],
            "errors": [],
        }
        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)
        completed_steps = []

    # Run steps
    for step in STEPS:
        step_name = step["name"]
        script = code_dir / step["script"]

        # Resume / skip logic
        if args.resume and _step_is_complete(step, output_dir, completed_steps):
            if args.force_step and step_name.startswith(args.force_step):
                print(f"Force re-run: {step_name}")
            else:
                print(f"[SKIP] {step_name} — already complete")
                continue

        print(f"\n{'='*60}")
        print(f"[RUN] {step_name}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, str(script),
            "--output-dir", str(output_dir),
            "--run-id", args.run_id,
        ] + step["extra_args"](args)

        max_attempts = 3
        last_returncode = None

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                print(f"  Attempt {attempt}/{max_attempts}...")
            result = subprocess.run(cmd, capture_output=False)
            last_returncode = result.returncode

            if last_returncode == 0:
                print(f"[OK] {step_name}")
                if step_name not in completed_steps:
                    completed_steps.append(step_name)
                break
            elif last_returncode == 2:
                # Leakage halt or intentional stop
                print(f"[HALT] {step_name} returned exit code 2 — pipeline halted intentionally.")
                sys.exit(2)
            else:
                print(f"[FAIL] {step_name} exit code {last_returncode} (attempt {attempt})")

        if last_returncode != 0:
            print(f"\n[FATAL] Step {step_name} failed after {max_attempts} attempts. Halting.")
            with open(progress_path) as f:
                p = json.load(f)
            p["status"] = "error"
            p["current_step"] = step_name
            with open(progress_path, "w") as f:
                json.dump(p, f, indent=2)
            sys.exit(1)

    # Write code_audit.json
    _write_code_audit(code_dir, output_dir)

    print("\n" + "="*60)
    print("Pipeline COMPLETE")
    print("="*60)

    # Verify final progress
    if progress_path.exists():
        with open(progress_path) as f:
            p = json.load(f)
        print(f"Status: {p.get('status')}")
        print(f"Completed steps: {p.get('completed_steps')}")
    sys.exit(0)


if __name__ == "__main__":
    main()
