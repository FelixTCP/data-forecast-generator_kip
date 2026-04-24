#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_name(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    normalized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in normalized)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def bootstrap_progress(
    output_dir: Path,
    run_id: str,
    csv_path: str,
    target_column: str,
    split_mode: str,
) -> dict[str, Any]:
    progress_path = output_dir / "progress.json"
    payload = {
        "run_id": run_id,
        "csv_path": csv_path,
        "target_column": target_column,
        "status": "running",
        "current_step": None,
        "split_mode": split_mode,
        "completed_steps": [],
        "errors": [],
    }
    write_json(progress_path, payload)
    return payload


def load_progress(output_dir: Path) -> dict[str, Any]:
    return read_json(output_dir / "progress.json", default={})


def mark_step_start(output_dir: Path, step: str) -> dict[str, Any]:
    progress = load_progress(output_dir)
    progress["status"] = "running"
    progress["current_step"] = step
    write_json(output_dir / "progress.json", progress)
    return progress


def mark_step_success(output_dir: Path, step: str) -> dict[str, Any]:
    progress = load_progress(output_dir)
    completed = progress.setdefault("completed_steps", [])
    if step not in completed:
        completed.append(step)
    progress["current_step"] = step
    if progress.get("status") == "error":
        progress["status"] = "running"
    write_json(output_dir / "progress.json", progress)
    return progress


def mark_status(output_dir: Path, status: str) -> dict[str, Any]:
    progress = load_progress(output_dir)
    progress["status"] = status
    write_json(output_dir / "progress.json", progress)
    return progress


def mark_step_error(output_dir: Path, step: str, message: str) -> None:
    progress = load_progress(output_dir)
    progress["status"] = "error"
    progress["current_step"] = step
    errors = progress.setdefault("errors", [])
    errors.append({"step": step, "message": message, "timestamp": utc_now()})
    write_json(output_dir / "progress.json", progress)


def validate_existing_output(output_path: Path, expected_step: str) -> bool:
    if not output_path.exists():
        return False
    try:
        payload = read_json(output_path)
    except json.JSONDecodeError:
        return False
    return payload.get("step") == expected_step


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_code_audit(output_dir: Path, code_dir: Path) -> None:
    inventory: list[dict[str, Any]] = []
    for path in sorted(code_dir.glob("*.py")):
        inventory.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload = {
        "generated_at": utc_now(),
        "code_dir": str(code_dir),
        "files": inventory,
    }
    write_json(output_dir / "code_audit.json", payload)


def finite_float(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric == numeric and numeric not in (float("inf"), float("-inf"))
