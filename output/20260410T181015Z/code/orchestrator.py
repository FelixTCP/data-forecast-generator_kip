from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
from tqdm import tqdm


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrator for per-step forecasting pipeline")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--split-mode", default="auto", choices=["auto", "random", "time_series"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-step", default="")
    return parser.parse_args()


def ensure_progress(output_dir: Path, run_id: str, csv_path: str, target_column: str) -> dict[str, Any]:
    path = output_dir / "progress.json"
    if path.exists():
        return read_json(path)
    progress = {
        "run_id": run_id,
        "csv_path": csv_path,
        "target_column": target_column,
        "status": "running",
        "current_step": "10-csv-read-cleansing",
        "completed_steps": [],
        "errors": [],
    }
    write_json(path, progress)
    return progress


def is_valid_step_output(path: Path, expected_step: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    return payload.get("step") == expected_step


def validate_step_10(output_dir: Path, target_column: str) -> None:
    j = read_json(output_dir / "step-10-cleanse.json")
    assert j["step"] == "10-csv-read-cleansing"
    assert int(j["row_count_after"]) > 0
    assert j["target_column_normalized"] == "_".join(target_column.strip().lower().split())
    assert isinstance(j.get("null_rate"), dict)
    assert Path(j["artifacts"]["cleaned_parquet"]).exists()


def validate_step_11(output_dir: Path) -> None:
    j = read_json(output_dir / "step-11-exploration.json")
    assert isinstance(j.get("numeric_columns"), list) and len(j["numeric_columns"]) > 0
    assert isinstance(j.get("target_candidates"), list) and len(j["target_candidates"]) > 0
    assert int(j["shape"]["rows"]) > 10


def validate_step_12(output_dir: Path) -> None:
    j = read_json(output_dir / "step-12-features.json")
    assert isinstance(j.get("features"), list) and len(j["features"]) > 0
    resolved = j["split_strategy"]["resolved_mode"]
    assert resolved in {"random", "time_series"}
    assert Path(j["artifacts"]["features_parquet"]).exists()


def validate_step_13(output_dir: Path) -> None:
    j = read_json(output_dir / "step-13-training.json")
    model_path = output_dir / "model.joblib"
    holdout_path = output_dir / "holdout.npz"
    assert model_path.exists()
    _ = joblib.load(model_path)
    assert holdout_path.exists()
    finite_r2 = any(math.isfinite(float(c["r2"])) for c in j["candidates"])
    assert finite_r2


def validate_step_14(output_dir: Path) -> None:
    j = read_json(output_dir / "step-14-evaluation.json")
    for c in j["candidate_reports"]:
        for key in ("r2", "rmse", "mae"):
            value = float(c[key])
            assert math.isfinite(value)


def validate_step_15(output_dir: Path) -> None:
    j = read_json(output_dir / "step-15-selection.json")
    assert j.get("selected_model")
    rationale = str(j.get("rationale", ""))
    assert len(rationale.split(".")) >= 1


def validate_step_16(output_dir: Path) -> None:
    report_path = output_dir / "step-16-report.md"
    assert report_path.exists()
    assert report_path.stat().st_size >= 500
    text = report_path.read_text(encoding="utf-8")
    required = [
        "## 1. Problem + selected target",
        "## 2. Data quality summary",
        "## 3. Candidate models + scores table",
        "## 4. Selected model rationale",
        "## 5. Risks and caveats",
        "## 6. Next iteration recommendations",
    ]
    for heading in required:
        assert heading in text

    progress = read_json(output_dir / "progress.json")
    assert progress.get("status") == "completed"


def step_hashes(code_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for pyf in sorted(code_dir.glob("*.py")):
        if pyf.name == "orchestrator.py":
            continue
        result[pyf.name] = hashlib.sha256(pyf.read_bytes()).hexdigest()
    return result


def run_step(step_script: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(step_script)] + args
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    code_dir = Path(args.code_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = ensure_progress(output_dir, args.run_id, args.csv_path, args.target_column)

    steps = [
        {
            "id": "10-csv-read-cleansing",
            "script": code_dir / "step_10_cleanse.py",
            "json": output_dir / "step-10-cleanse.json",
            "expected": "10-csv-read-cleansing",
            "validator": lambda: validate_step_10(output_dir, args.target_column),
            "args": [
                "--csv-path",
                args.csv_path,
                "--output-dir",
                str(output_dir),
                "--run-id",
                args.run_id,
                "--target-column",
                args.target_column,
            ],
        },
        {
            "id": "11-data-exploration",
            "script": code_dir / "step_11_exploration.py",
            "json": output_dir / "step-11-exploration.json",
            "expected": "11-data-exploration",
            "validator": lambda: validate_step_11(output_dir),
            "args": ["--output-dir", str(output_dir), "--run-id", args.run_id],
        },
        {
            "id": "12-feature-extraction",
            "script": code_dir / "step_12_features.py",
            "json": output_dir / "step-12-features.json",
            "expected": "12-feature-extraction",
            "validator": lambda: validate_step_12(output_dir),
            "args": [
                "--output-dir",
                str(output_dir),
                "--run-id",
                args.run_id,
                "--split-mode",
                args.split_mode,
            ],
        },
        {
            "id": "13-model-training",
            "script": code_dir / "step_13_training.py",
            "json": output_dir / "step-13-training.json",
            "expected": "13-model-training",
            "validator": lambda: validate_step_13(output_dir),
            "args": ["--output-dir", str(output_dir), "--run-id", args.run_id],
        },
        {
            "id": "14-model-evaluation",
            "script": code_dir / "step_14_evaluation.py",
            "json": output_dir / "step-14-evaluation.json",
            "expected": "14-model-evaluation",
            "validator": lambda: validate_step_14(output_dir),
            "args": ["--output-dir", str(output_dir), "--run-id", args.run_id],
        },
        {
            "id": "15-model-selection",
            "script": code_dir / "step_15_selection.py",
            "json": output_dir / "step-15-selection.json",
            "expected": "15-model-selection",
            "validator": lambda: validate_step_15(output_dir),
            "args": ["--output-dir", str(output_dir), "--run-id", args.run_id],
        },
        {
            "id": "16-result-presentation",
            "script": code_dir / "step_16_report.py",
            "json": output_dir / "step-16-report.md",
            "expected": "16-result-presentation",
            "validator": lambda: validate_step_16(output_dir),
            "args": ["--output-dir", str(output_dir), "--run-id", args.run_id],
        },
    ]

    force_step = args.force_step.strip()
    for step in tqdm(steps, desc="orchestrator: steps", unit="step"):
        already_completed = step["id"] in progress.get("completed_steps", [])
        valid_artifact = (
            is_valid_step_output(step["json"], step["expected"]) if step["json"].suffix == ".json" else step["json"].exists()
        )

        if args.resume and force_step != step["id"] and already_completed and valid_artifact:
            print(f"Step already complete, resuming: {step['id']}")
            continue

        attempt = 0
        while attempt < 3:
            attempt += 1
            try:
                run_step(step["script"], step["args"])
                step["validator"]()
                break
            except Exception as exc:
                print(f"Step failed: {step['id']} attempt={attempt} error={exc}")
                if attempt >= 3:
                    raise

    audit = {
        "run_id": args.run_id,
        "code_dir": str(code_dir),
        "steps": [
            "step_10_cleanse.py",
            "step_11_exploration.py",
            "step_12_features.py",
            "step_13_training.py",
            "step_14_evaluation.py",
            "step_15_selection.py",
            "step_16_report.py",
            "orchestrator.py",
        ],
        "sha256": step_hashes(code_dir),
    }
    write_json(output_dir / "code_audit.json", audit)
    print(f"Pipeline completed successfully. Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
