from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
from tqdm import tqdm


@dataclass
class PipelineResult:
    run_id: str
    output_dir: str
    selected_model: str
    metrics: dict[str, float]
    artifacts: dict[str, str]


def _iso_run_id() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_step_prompts(path: Path) -> dict[str, str]:
    text = _read_text(path)
    prompts: dict[str, str] = {}

    current_key: str | None = None
    current_lines: list[str] = []
    in_block = False

    for line in text.splitlines():
        if line.startswith("## "):
            if current_key and current_lines:
                prompts[current_key] = "\n".join(current_lines).strip()
            current_key = line[3:].strip()
            current_lines = []
            in_block = False
            continue

        if line.strip() == "```markdown":
            in_block = True
            continue

        if line.strip() == "```" and in_block:
            in_block = False
            continue

        if in_block and current_key:
            current_lines.append(line)

    if current_key and current_lines:
        prompts[current_key] = "\n".join(current_lines).strip()

    return prompts


def _render_prompt(template: str, mapping: dict[str, str]) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace(f"{{{{{key}}}}}", value)
    return out


def _run_copilot_prompt(
    prompt: str,
    cwd: Path,
    *,
    copilot_model: str,
    reasoning_effort: str,
) -> str:
    cmd = [
        "copilot",
        "--allow-all-tools",
        "--allow-all-paths",
        "--allow-all-urls",
        "--no-ask-user",
        "--model",
        copilot_model,
        "--reasoning-effort",
        reasoning_effort,
        "-s",
        "-p",
        prompt,
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Copilot step failed\n"
            f"exit={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def _compute_command_hash(
    *,
    csv_path: str,
    target_column: str | None,
    split_mode: str,
    budget_mode: str,
    copilot_model: str,
    reasoning_effort: str,
) -> str:
    payload = {
        "csv_path": csv_path,
        "target_column": target_column or "",
        "split_mode": split_mode,
        "budget_mode": budget_mode,
        "copilot_model": copilot_model,
        "reasoning_effort": reasoning_effort,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _prepare_code_workspace(
    *,
    output_root: Path,
    command_hash: str,
    continue_run: bool,
    run_id: str,
) -> tuple[Path, dict[str, str | bool | None]]:
    code_root = output_root / ".agent_code" / command_hash
    workspace = code_root / "workspace"
    snapshots = code_root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)

    existed = workspace.exists()
    snapshot_path: Path | None = None
    if continue_run:
        workspace.mkdir(parents=True, exist_ok=True)
    else:
        if existed:
            snapshot_path = snapshots / f"{run_id}_pre_reset"
            if not snapshot_path.exists():
                shutil.copytree(workspace, snapshot_path)
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

    metadata_path = code_root / "metadata.json"
    _write_json(
        metadata_path,
        {
            "command_hash": command_hash,
            "last_run_id": run_id,
            "workspace": str(workspace),
            "continue_run": continue_run,
            "workspace_existed": existed,
            "snapshot_path": str(snapshot_path) if snapshot_path else None,
            "updated_at": datetime.now(tz=UTC).isoformat(),
        },
    )
    return workspace, {
        "workspace_existed": existed,
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "metadata_path": str(metadata_path),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_python_files(code_dir: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    if not code_dir.exists():
        return rows
    for path in sorted(code_dir.rglob("*.py")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(code_dir)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return rows


def _append_code_audit(audit_path: Path, *, step: str, code_dir: Path) -> None:
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["steps"].append(
        {
            "step": step,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "python_files": _collect_python_files(code_dir),
        }
    )
    _write_json(audit_path, payload)


def _update_progress(
    progress_path: Path,
    *,
    status: str,
    current_step: str,
    completed_steps: list[str],
    errors: list[str] | None = None,
) -> None:
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    payload["status"] = status
    payload["current_step"] = current_step
    payload["completed_steps"] = completed_steps
    payload["errors"] = errors or []
    _write_json(progress_path, payload)


def _validate_required_artifacts(run_dir: Path, code_dir: Path) -> None:
    required = [
        run_dir / "progress.json",
        run_dir / "code_audit.json",
        run_dir / "step-10-cleanse.json",
        run_dir / "step-11-exploration.json",
        run_dir / "step-12-features.json",
        run_dir / "step-13-training.json",
        run_dir / "model.joblib",
        run_dir / "step-14-evaluation.json",
        run_dir / "step-15-selection.json",
        run_dir / "step-16-report.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Pipeline completed without required artifacts. Missing:\n" + "\n".join(missing)
        )

    model_path = run_dir / "model.joblib"
    model_obj = joblib.load(model_path)
    model_candidate = model_obj
    if isinstance(model_obj, dict):
        for key in ("model", "estimator", "best_model", "pipeline"):
            if key in model_obj:
                model_candidate = model_obj[key]
                break
    if not hasattr(model_candidate, "predict"):
        raise ValueError(
            "model.joblib exists but does not contain a fitted predictor (missing .predict)."
        )

    if not _collect_python_files(code_dir):
        raise ValueError(f"No Python files were generated in hashed code workspace: {code_dir}")


def _validate_model_artifact_only(run_dir: Path) -> None:
    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")

    model_obj = joblib.load(model_path)
    model_candidate = model_obj
    if isinstance(model_obj, dict):
        for key in ("model", "estimator", "best_model", "pipeline"):
            if key in model_obj:
                model_candidate = model_obj[key]
                break
    if not hasattr(model_candidate, "predict"):
        raise ValueError(
            "model.joblib exists but does not contain a fitted predictor (missing .predict)."
        )


def run_pipeline(
    csv_path: str,
    output_dir: str,
    target_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    split_mode: str = "auto",
    budget_mode: str = "low",
    copilot_model: str | None = None,
    reasoning_effort: str | None = None,
    continue_run: bool = False,
) -> PipelineResult:
    del test_size
    del random_state

    budget_defaults = {
        "low": ("gpt-5-mini", "low"),
        "balanced": ("gpt-5.3-codex", "medium"),
        "high": ("gpt-5.3-codex", "high"),
    }
    if budget_mode not in budget_defaults:
        raise ValueError("budget_mode must be one of: low, balanced, high")
    default_model, default_reasoning = budget_defaults[budget_mode]
    effective_model = copilot_model or default_model
    effective_reasoning = reasoning_effort or default_reasoning

    root = Path.cwd()
    output_root = Path(output_dir)
    run_id = _iso_run_id()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    command_hash = _compute_command_hash(
        csv_path=csv_path,
        target_column=target_column,
        split_mode=split_mode,
        budget_mode=budget_mode,
        copilot_model=effective_model,
        reasoning_effort=effective_reasoning,
    )
    code_dir, code_meta = _prepare_code_workspace(
        output_root=output_root,
        command_hash=command_hash,
        continue_run=continue_run,
        run_id=run_id,
    )

    progress_path = run_dir / "progress.json"
    _write_json(
        progress_path,
        {
            "run_id": run_id,
            "csv_path": csv_path,
            "target_column": target_column,
            "status": "running",
            "current_step": "10-csv-read-cleansing",
            "completed_steps": [],
            "errors": [],
            "command_hash": command_hash,
            "code_dir": str(code_dir),
            "continue_run": continue_run,
        },
    )

    code_audit_path = run_dir / "code_audit.json"
    _write_json(
        code_audit_path,
        {
            "run_id": run_id,
            "command_hash": command_hash,
            "code_dir": str(code_dir),
            "continue_run": continue_run,
            "workspace_existed": code_meta["workspace_existed"],
            "snapshot_path": code_meta["snapshot_path"],
            "metadata_path": code_meta["metadata_path"],
            "steps": [],
        },
    )

    setup_prompt = _read_text(root / "docs/agentic-pipeline/setup-prompt.md")
    step_templates = _load_step_prompts(root / "docs/agentic-pipeline/step-prompts.md")

    mapping = {
        "CSV_PATH": csv_path,
        "TARGET_COLUMN": target_column or "",
        "OUTPUT_DIR": str(run_dir),
        "RUN_ID": run_id,
        "SPLIT_MODE": split_mode,
        "CODE_DIR": str(code_dir),
        "COMMAND_HASH": command_hash,
        "CONTINUE_MODE": "true" if continue_run else "false",
    }

    try:
        _run_copilot_prompt(
            (
                f"{setup_prompt}\n\n"
                "Runtime variables:\n"
                f"{json.dumps(mapping, indent=2)}\n\n"
                f"Execution profile:\n"
                f"- budget_mode: {budget_mode}\n"
                f"- model: {effective_model}\n"
                f"- reasoning_effort: {effective_reasoning}\n"
            ),
            cwd=root,
            copilot_model=effective_model,
            reasoning_effort=effective_reasoning,
        )
        _append_code_audit(code_audit_path, step="setup", code_dir=code_dir)

        step_order = [
            "10-csv-read-cleansing",
            "11-data-exploration",
            "12-feature-extraction",
            "13-model-training",
            "14-model-evaluation",
            "15-model-selection",
            "16-result-presentation",
        ]

        completed_steps: list[str] = []
        step_progress = tqdm(step_order, desc="Agentic pipeline", unit="step")
        for step in step_progress:
            step_progress.set_postfix_str(step)
            if step not in step_templates:
                raise ValueError(f"Missing step prompt template for {step}")

            _update_progress(
                progress_path,
                status="running",
                current_step=step,
                completed_steps=completed_steps,
            )

            step_prompt = _render_prompt(step_templates[step], mapping)
            full_prompt = (
                "Follow repository instructions and execute exactly this step.\n"
                "Do not ask user questions.\n"
                "Write and execute all generated python code under CODE_DIR only.\n"
                f"Code workspace: {code_dir}\n"
                "Use existing files and write outputs to requested paths.\n\n"
                "Read and follow:\n"
                "- docs/agentic-pipeline/contracts.md\n"
                "- docs/pipeline-framework/00-overview.md\n"
                "- the relevant step file under docs/pipeline-framework/\n\n"
                f"Step:\n{step_prompt}"
            )
            _run_copilot_prompt(
                full_prompt,
                cwd=root,
                copilot_model=effective_model,
                reasoning_effort=effective_reasoning,
            )
            completed_steps.append(step)
            _append_code_audit(code_audit_path, step=step, code_dir=code_dir)

        _update_progress(
            progress_path,
            status="completed",
            current_step="done",
            completed_steps=completed_steps,
        )

        try:
            _validate_required_artifacts(run_dir, code_dir)
        except Exception:  # noqa: BLE001
            repair_prompt = (
                "Repair invalid model artifact for this completed run.\n"
                "The current model.joblib is not loadable as a portable predictor.\n"
                "Recreate and persist a valid sklearn-compatible estimator/pipeline to:\n"
                f"{run_dir / 'model.joblib'}\n\n"
                "Constraints:\n"
                f"- CODE_DIR: {code_dir}\n"
                "- Use importable classes only (no custom classes defined under __main__).\n"
                "- The loaded object must expose .predict(...).\n"
                "- Reuse existing run outputs under this output directory when possible.\n"
                f"- Keep selected model consistent with {run_dir / 'step-15-selection.json'} when feasible.\n"
            )
            _run_copilot_prompt(
                repair_prompt,
                cwd=root,
                copilot_model=effective_model,
                reasoning_effort=effective_reasoning,
            )
            _append_code_audit(code_audit_path, step="repair-model-artifact", code_dir=code_dir)
            _validate_required_artifacts(run_dir, code_dir)

    except Exception as exc:  # noqa: BLE001
        _update_progress(
            progress_path,
            status="failed",
            current_step="failed",
            completed_steps=json.loads(progress_path.read_text(encoding="utf-8")).get(
                "completed_steps", []
            ),
            errors=[str(exc)],
        )
        raise

    selection_path = run_dir / "step-15-selection.json"
    evaluation_path = run_dir / "step-14-evaluation.json"
    report_path = run_dir / "step-16-report.md"

    selected_model = "unknown"
    metrics: dict[str, float] = {}

    if selection_path.exists():
        sel = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_model = str(sel.get("selected_model", selected_model))

    if evaluation_path.exists():
        ev = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if isinstance(ev.get("metrics"), dict):
            metrics = {
                k: float(v)
                for k, v in ev["metrics"].items()
                if isinstance(v, (int, float))
            }

    artifacts = {
        "progress": str(progress_path),
        "code_audit": str(code_audit_path),
        "code_dir": str(code_dir),
        "cleanse": str(run_dir / "step-10-cleanse.json"),
        "exploration": str(run_dir / "step-11-exploration.json"),
        "features": str(run_dir / "step-12-features.json"),
        "training": str(run_dir / "step-13-training.json"),
        "model": str(run_dir / "model.joblib"),
        "evaluation": str(evaluation_path),
        "selection": str(selection_path),
        "report": str(report_path),
    }

    return PipelineResult(
        run_id=run_id,
        output_dir=str(run_dir),
        selected_model=selected_model,
        metrics=metrics,
        artifacts=artifacts,
    )
