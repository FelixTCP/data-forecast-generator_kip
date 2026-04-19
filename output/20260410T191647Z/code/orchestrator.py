from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-step forecasting orchestrator")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--split-mode", default="auto", choices=["auto", "random", "time_series"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-step", default="")
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return "_".join(name.strip().lower().split())


def ensure_progress(output_dir: Path, run_id: str, csv_path: str, target_column: str) -> dict[str, Any]:
    progress_path = output_dir / "progress.json"
    if progress_path.exists():
        return read_json(progress_path)

    progress = {
        "run_id": run_id,
        "csv_path": csv_path,
        "target_column": normalize_name(target_column),
        "status": "running",
        "current_step": "10-csv-read-cleansing",
        "completed_steps": [],
        "errors": [],
    }
    write_json(progress_path, progress)
    return progress


def set_progress(output_dir: Path, **updates: Any) -> None:
    progress_path = output_dir / "progress.json"
    payload = read_json(progress_path)
    payload.update(updates)
    write_json(progress_path, payload)


def is_valid_step_output(path: Path, expected_step: str) -> bool:
    if not path.exists():
        return False
    if path.suffix != ".json":
        return True
    try:
        payload = read_json(path)
    except Exception:
        return False
    return payload.get("step") == expected_step


def run_step(script_path: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script_path)] + args
    subprocess.run(cmd, check=True)


def validate_step_10(output_dir: Path, target_column: str) -> None:
    payload = read_json(output_dir / "step-10-cleanse.json")
    assert payload.get("step") == "10-csv-read-cleansing"
    assert int(payload.get("row_count_after", 0)) > 0
    assert payload.get("target_column_normalized") == normalize_name(target_column)
    assert isinstance(payload.get("null_rate"), dict)
    cleaned = Path(payload["artifacts"]["cleaned_parquet"])
    assert cleaned.exists()


def validate_step_11(output_dir: Path) -> None:
    payload = read_json(output_dir / "step-11-exploration.json")
    assert payload.get("step") == "11-data-exploration"
    assert isinstance(payload.get("numeric_columns"), list) and len(payload["numeric_columns"]) > 0
    assert isinstance(payload.get("mi_ranking"), list) and len(payload["mi_ranking"]) > 0
    assert isinstance(payload.get("recommended_features"), list) and len(payload["recommended_features"]) > 0
    assert math.isfinite(float(payload.get("noise_mi_baseline", float("nan"))))
    assert isinstance(payload.get("target_candidates"), list) and len(payload["target_candidates"]) > 0
    assert int(payload["shape"]["rows"]) > 10


def validate_step_12(output_dir: Path) -> None:
    step12 = read_json(output_dir / "step-12-features.json")
    step11 = read_json(output_dir / "step-11-exploration.json")

    assert step12.get("step") == "12-feature-extraction"
    assert isinstance(step12.get("features"), list) and len(step12["features"]) > 0
    assert "features_excluded" in step12 and isinstance(step12["features_excluded"], dict)
    assert step12["split_strategy"]["resolved_mode"] in {"random", "time_series"}

    features_path = Path(step12["artifacts"]["features_parquet"])
    assert features_path.exists()

    excluded = set(step11.get("excluded_features", {}).keys())
    overlap = excluded.intersection(set(step12["features"]))
    assert not overlap, f"Excluded features were re-included: {sorted(overlap)}"


def validate_step_13(output_dir: Path) -> None:
    payload = read_json(output_dir / "step-13-training.json")
    assert payload.get("step") == "13-model-training"

    model_path = output_dir / "model.joblib"
    holdout_path = output_dir / "holdout.npz"
    assert model_path.exists()
    assert holdout_path.exists()

    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import joblib; m=joblib.load(r'{model_path}'); print(type(m))",
        ],
        check=True,
    )

    finite_r2 = any(math.isfinite(float(c.get("r2", float("nan")))) for c in payload.get("candidates", []))
    assert finite_r2


def validate_step_14(output_dir: Path) -> None:
    payload = read_json(output_dir / "step-14-evaluation.json")
    assert payload.get("step") == "14-model-evaluation"

    reports = payload.get("candidate_reports", [])
    assert isinstance(reports, list) and len(reports) > 0
    for c in reports:
        for key in ("r2", "rmse", "mae"):
            assert math.isfinite(float(c[key]))
        if float(c["r2"]) < 0:
            assert c.get("model_worse_than_mean_baseline") is True

    quality = payload.get("quality_assessment")
    assert quality in {"acceptable", "marginal", "subpar", "subpar_after_expansion"}
    assert isinstance(payload.get("target_stats"), dict)

    if quality == "subpar":
        diag = str(payload.get("expansion_diagnosis", "")).strip()
        assert diag
        expansion = payload.get("expansion_candidates", [])
        assert isinstance(expansion, list) and len(expansion) > 0


def validate_step_15(output_dir: Path) -> None:
    payload = read_json(output_dir / "step-15-selection.json")
    assert payload.get("step") == "15-model-selection"

    quality = payload.get("quality_flag")
    assert quality in {
        "acceptable",
        "marginal",
        "subpar",
        "subpar_after_expansion",
        "no_viable_candidate",
    }

    ranking = payload.get("full_ranking", [])
    assert isinstance(ranking, list) and len(ranking) > 0

    if quality == "no_viable_candidate":
        assert payload.get("selected_model") in {None, ""}
        assert str(payload.get("message", "")).strip()
    else:
        assert str(payload.get("selected_model", "")).strip()
        assert len(str(payload.get("rationale", "")).strip()) > 10


def validate_step_16(output_dir: Path) -> None:
    report_path = output_dir / "step-16-report.md"
    assert report_path.exists()
    assert report_path.stat().st_size >= 500

    report = report_path.read_text(encoding="utf-8")
    required = [
        "## 1. Problem + selected target",
        "## 2. Data quality summary",
        "## 3. Candidate models + scores table",
        "## 4. Selected model rationale",
        "## 5. Risks and caveats",
        "## 6. Next iteration recommendations",
    ]
    for heading in required:
        assert heading in report

    progress = read_json(output_dir / "progress.json")
    assert progress.get("status") == "completed"


def file_hashes(code_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for pyf in sorted(code_dir.glob("*.py")):
        hashes[pyf.name] = hashlib.sha256(pyf.read_bytes()).hexdigest()
    return hashes


def maybe_skip_step(resume: bool, force_step: str, step_id: str, output_json: Path, expected: str, completed: list[str]) -> bool:
    if not resume:
        return False
    if force_step and force_step == step_id:
        return False
    return (step_id in completed) and is_valid_step_output(output_json, expected)


def main() -> int:
    from tqdm import tqdm

    args = parse_args()
    output_dir = Path(args.output_dir)
    code_dir = Path(args.code_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = ensure_progress(output_dir, args.run_id, args.csv_path, args.target_column)
    completed = [str(s) for s in progress.get("completed_steps", [])]

    steps: list[dict[str, Any]] = [
        {
            "id": "10-csv-read-cleansing",
            "script": code_dir / "step_10_cleanse.py",
            "output": output_dir / "step-10-cleanse.json",
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
            "output": output_dir / "step-11-exploration.json",
            "expected": "11-data-exploration",
            "validator": lambda: validate_step_11(output_dir),
            "args": ["--output-dir", str(output_dir), "--run-id", args.run_id],
        },
        {
            "id": "12-feature-extraction",
            "script": code_dir / "step_12_features.py",
            "output": output_dir / "step-12-features.json",
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
            "output": output_dir / "step-13-training.json",
            "expected": "13-model-training",
            "validator": lambda: validate_step_13(output_dir),
            "args": ["--output-dir", str(output_dir), "--run-id", args.run_id],
        },
        {
            "id": "14-model-evaluation",
            "script": code_dir / "step_14_evaluation.py",
            "output": output_dir / "step-14-evaluation.json",
            "expected": "14-model-evaluation",
            "validator": lambda: validate_step_14(output_dir),
            "args": ["--output-dir", str(output_dir), "--run-id", args.run_id],
        },
        {
            "id": "15-model-selection",
            "script": code_dir / "step_15_selection.py",
            "output": output_dir / "step-15-selection.json",
            "expected": "15-model-selection",
            "validator": lambda: validate_step_15(output_dir),
            "args": ["--output-dir", str(output_dir), "--run-id", args.run_id],
        },
        {
            "id": "16-result-presentation",
            "script": code_dir / "step_16_report.py",
            "output": output_dir / "step-16-report.md",
            "expected": "16-result-presentation",
            "validator": lambda: validate_step_16(output_dir),
            "args": ["--output-dir", str(output_dir), "--run-id", args.run_id],
        },
    ]

    for step in tqdm(steps, desc="orchestrator: steps", unit="step"):
        if maybe_skip_step(
            resume=args.resume,
            force_step=args.force_step.strip(),
            step_id=step["id"],
            output_json=step["output"],
            expected=step["expected"],
            completed=completed,
        ):
            print(f"Step already complete, resuming from existing output: {step['id']}")
            continue

        set_progress(output_dir, current_step=step["id"], status="running")

        attempts = 0
        while attempts < 3:
            attempts += 1
            try:
                run_step(step["script"], step["args"])
                step["validator"]()
                if step["id"] not in completed:
                    completed.append(step["id"])
                    set_progress(output_dir, completed_steps=completed)
                break
            except Exception as exc:
                payload = read_json(output_dir / "progress.json")
                errors = [str(e) for e in payload.get("errors", [])]
                errors.append(f"{step['id']} attempt={attempts}: {exc}")
                set_progress(output_dir, errors=errors, status="error")
                if attempts >= 3:
                    raise
                print(f"Retrying {step['id']} after failure: {exc}")

    audit = {
        "run_id": args.run_id,
        "code_dir": str(code_dir),
        "steps": [p.name for p in sorted(code_dir.glob("*.py"))],
        "sha256": file_hashes(code_dir),
    }
    write_json(output_dir / "code_audit.json", audit)

    set_progress(output_dir, status="completed", current_step="16-result-presentation", completed_steps=completed)
    print(f"Pipeline completed successfully. Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())