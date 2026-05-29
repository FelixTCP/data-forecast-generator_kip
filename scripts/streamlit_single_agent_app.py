"""
Streamlit UI for Single Agent Pipeline — Professional data scientist dashboard.

Five views:
  Tab 1 — EDA       : stationarity, Hurst, ACF/PACF, MI, outliers, seasonality
  Tab 2 — Features  : feature groups, importances, PCA, correlation, data preview
  Tab 3 — Models    : filterable comparison of all trained candidates
  Tab 4 — Best Model: SHAP, residuals, detailed metrics
  Tab 5 — Report    : full step-16-report.md
  Tab 6 — Audit     : critical self-audit results, remediation actions
  Tab 7 — Judge     : compact customer-facing judgement
"""

from __future__ import annotations

import json
import html
import re
import subprocess
import textwrap
import time
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import polars as pl
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_UPLOAD_DIR = ROOT_DIR / "artifacts" / "ui_uploads"
DEFAULT_RUNS_DIR = ROOT_DIR / "output"

PIPELINE_STEPS = [
    "10-csv-read-cleansing",
    "11-data-exploration",
    "12-feature-extraction",
    "13-model-training",
    "14-model-evaluation",
    "15-model-selection",
    "16-result-presentation",
    "17-critical-self-audit",
    "18-llm-as-judge",
]

_BENCHMARK_NAMES = {"arima_benchmark", "kmeans_benchmark", "naive_persistence", "seasonal_naive",
                    "auto_arima_benchmark"}
_BENCHMARK_COLOR = "rgb(255, 165, 0)"
_BEST_COLOR = "rgb(0, 180, 80)"
_CANDIDATE_COLOR = "rgb(26, 118, 255)"
_NEG_COLOR = "rgb(200, 0, 0)"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_column_name(value: str) -> str:
    return re.sub(r"_+", "_", value.strip().lower().replace(" ", "_")).strip("_")

def _read_uploaded_dataframe(raw_bytes: bytes) -> pl.DataFrame:
    first_line = raw_bytes.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    separator = ";" if first_line.count(";") > first_line.count(",") else ","

    read_options = {
        "separator": separator,
        "try_parse_dates": True,
        "truncate_ragged_lines": True,
    }

    try:
        return pl.read_csv(BytesIO(raw_bytes), **read_options)
    except pl.exceptions.ComputeError:
        # Retry with a wider inference window so mixed integer/float columns are
        # detected as floating-point instead of being locked in as Int64.
        return pl.read_csv(
            BytesIO(raw_bytes),
            **read_options,
            infer_schema_length=10_000,
        )

# Keywords that strongly suggest a column is a good regression target
_TARGET_KEYWORDS = [
    "target", "sales", "revenue", "price", "amount", "total", "value",
    "demand", "quantity", "volume", "count", "cost", "profit", "income",
    "output", "production", "consumption", "forecast", "energy", "power",
    "load", "usage", "temperature", "temp", "close", "appliance", "y",
]

# Keywords that suggest a column should NOT be the target
_EXCLUDE_KEYWORDS = [
    "id", "index", "key", "uuid", "date", "time", "year", "month",
    "day", "hour", "minute", "second", "timestamp", "created", "updated",
]

_NUMERIC_DTYPES = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


def _coerce_series_to_numeric(series: pl.Series) -> tuple[pl.Series | None, float]:
    if series.dtype in _NUMERIC_DTYPES:
        numeric = series.cast(pl.Float64)
    else:
        numeric = (
            series.cast(pl.String, strict=False)
            .str.strip_chars()
            .str.replace_all(",", ".")
            .cast(pl.Float64, strict=False)
        )

    non_null = numeric.drop_nulls()
    if len(non_null) == 0:
        return None, 0.0

    return numeric, len(non_null) / len(series)


def _pairwise_abs_correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return 0.0

    left_values = left[mask]
    right_values = right[mask]
    if np.std(left_values) == 0 or np.std(right_values) == 0:
        return 0.0

    corr = np.corrcoef(left_values, right_values)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return abs(float(corr))


def _recommend_target_column(df: pl.DataFrame) -> str:
    """Heuristically recommend the most suitable regression target column."""
    if not df.columns:
        return ""

    n_rows = len(df)
    normalized_columns = {_normalize_column_name(col) for col in df.columns}
    has_ohlc_price_columns = len({"open", "high", "low", "close"} & normalized_columns) >= 2

    candidates: list[dict[str, object]] = []
    for col in df.columns:
        col_lower = col.lower()
        col_normalized = _normalize_column_name(col)

        numeric_series, parsed_ratio = _coerce_series_to_numeric(df[col])
        if numeric_series is None or parsed_ratio < 0.8:
            continue

        non_null = numeric_series.drop_nulls()

        # Skip columns with too few unique values (likely categorical/binary)
        n_unique = non_null.n_unique()
        if n_unique <= 2:
            continue

        # Skip ID-like columns (near 100 % unique) or excluded keywords
        if any(kw in col_lower for kw in _EXCLUDE_KEYWORDS):
            continue
        if n_rows > 10 and n_unique / n_rows > 0.99:
            continue

        score = 0.0

        # Boost for target-related keywords
        for kw in _TARGET_KEYWORDS:
            if kw == col_lower or kw in col_lower:
                score += 2.0
                break

        if has_ohlc_price_columns and col_normalized in {"open", "high", "low", "close"}:
            score += 1.0
        elif has_ohlc_price_columns and col_normalized == "volume":
            score -= 1.5

        # Use coefficient of variation as a signal of forecast-worthiness
        col_std = non_null.std() or 0.0
        col_mean = non_null.mean() or 0.0
        if col_mean != 0:
            score += min(abs(col_std / col_mean), 2.0)

        # Prefer columns that can actually be parsed reliably from CSV input.
        score += parsed_ratio

        # Slight preference for the last numeric column (often the target in many CSVs)
        if col == df.columns[-1]:
            score += 0.25

        candidates.append(
            {
                "column": col,
                "score": score,
                "values": numeric_series.to_numpy(),
            }
        )

    if not candidates:
        # Fallback: last column
        return df.columns[-1]

    for candidate in candidates:
        correlations = sorted(
            _pairwise_abs_correlation(candidate["values"], other["values"])
            for other in candidates
            if other["column"] != candidate["column"]
        )
        if correlations:
            candidate["score"] += sum(correlations[-3:]) / min(len(correlations), 3)

    candidates.sort(key=lambda item: (item["score"], item["column"]), reverse=True)
    return str(candidates[0]["column"])


def _render_single_agent_prompt(csv_path: Path, target_column: str,
                                 output_dir: Path, copilot_model: str) -> str:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    code_dir = output_dir / "code"
    return (
        "Run the custom agent `Single Agent Pipeline` end-to-end.\n\n"
        f"CSV path: {csv_path}\n"
        f"target={target_column}\n"
        f"OUTPUT_DIR={output_dir}\n"
        f"RUN_ID={run_id}\n"
        f"CODE_DIR={code_dir}\n"
        f"COPILOT_MODEL={copilot_model}\n"
        "CONTINUE_MODE=false\n\n"
        "Hard completion rule:\n"
        "- Do not stop after an intermediate step.\n"
        "- A successful run requires all steps 10 through 18 to complete.\n"
        "- Do not exit successfully until these files exist in OUTPUT_DIR: "
        "step-13-training.json, step-14-evaluation.json, step-15-selection.json, "
        "step-16-report.md, step-17-audit.json, step-18-judge.json, step-18-judge.md.\n"
        "- progress.json must end with status=\"completed\" and final_audit_result=\"pass\".\n"
        "- If any required artifact is missing, treat the run as incomplete and report the missing files.\n\n"
        "Follow exactly the contract in "
        "`@.github/agents/Single Agent Pipeline.agent.md`."
    )


def _save_uploaded_csv(uploaded_file, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / uploaded_file.name
    destination_path.write_bytes(uploaded_file.getvalue())
    return destination_path


def _find_copilot_cli() -> str | None:
    """Return the absolute path to the Copilot CLI binary, or None if not found.

    Probe order (most reliable first):
    1. System-wide installer EXE — works standalone, no VS Code required.
    2. 'copilot' on the current PATH — works inside VS Code terminals where
       the VS Code proxy BAT/shim is injected into PATH automatically.
    3. VS Code global-storage proxy BAT (absolute path) — last resort; only
       probed if VS Code is running as the proxy host.
    """
    import shutil, os, sys

    def _probe(binary: str) -> bool:
        """Run --version with stdin closed so interactive prompts never block."""
        try:
            r = subprocess.run(
                [binary, "--version"],
                stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=5,
            )
            return r.returncode == 0
        except Exception:
            return False

    # 1. System-wide installer locations (standalone EXE — preferred).
    if sys.platform == "win32":
        system_candidates = [
            Path(r"C:\Program Files\GitHub Copilot CLI\copilot.exe"),
            Path(r"C:\Program Files (x86)\GitHub Copilot CLI\copilot.exe"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "GitHub Copilot CLI" / "copilot.exe",
        ]
    else:
        system_candidates = [
            Path("/usr/local/bin/copilot"),
            Path(os.path.expanduser("~/.local/bin/copilot")),
        ]

    for candidate in system_candidates:
        if candidate.exists() and _probe(str(candidate)):
            return str(candidate)

    # 2. PATH lookup — works when launched from a VS Code terminal that has
    #    injected copilotCli into PATH, or when a real binary is on PATH.
    found = shutil.which("copilot")
    if found and not found.lower().endswith(".bat") and _probe(found):
        return found

    # 3. VS Code proxy BAT (absolute path) — only useful when VS Code is running
    #    as the proxy. Skip if PATH lookup already failed (BAT needs VS Code env).
    vscode_appdata = os.environ.get("APPDATA", "")
    if vscode_appdata:
        bat = Path(vscode_appdata) / "Code" / "User" / "globalStorage" \
              / "github.copilot-chat" / "copilotCli" / "copilot.bat"
        if bat.exists() and _probe(str(bat)):
            return str(bat)

    # Last: accept a PATH-found .bat if it responded OK (VS Code terminal)
    if found and _probe(found):
        return found

    return None


def _copilot_cli_available() -> bool:
    """Return True when any Copilot CLI binary is reachable (PATH or known paths)."""
    return _find_copilot_cli() is not None


def _start_pipeline_process(prompt: str, working_dir: Path,
                              model: str = "claude-haiku-4.5") -> subprocess.Popen[str]:
    cli = _find_copilot_cli()
    if cli is None:
        raise FileNotFoundError(
            "Copilot CLI not found. Install it via 'npm install -g @github/copilot-cli' "
            "or enable the Copilot Chat VS Code extension."
        )
    command = [cli, "--allow-all-tools", "--allow-all-paths",
               "--allow-all-urls", "--no-ask-user", "--model", model]
    if "gpt" in model.lower():
        command.extend(["--reasoning-effort", "low"])
    command.extend(["-s", "-p", prompt])
    return subprocess.Popen(command, cwd=working_dir,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _start_orchestrator_process(output_dir: Path, csv_path: Path,
                                 target_column: str, run_id: str) -> subprocess.Popen[str]:
    """Run orchestrator.py directly via Python (no Copilot CLI needed)."""
    orchestrator = output_dir / "code" / "orchestrator.py"
    if not orchestrator.exists():
        raise FileNotFoundError(
            f"orchestrator.py not found at {orchestrator}. "
            "Generate pipeline scripts first via VS Code Copilot Chat."
        )
    return subprocess.Popen(
        [
            "python", str(orchestrator),
            "--csv-path", str(csv_path),
            "--target-column", target_column,
            "--output-dir", str(output_dir),
            "--run-id", run_id,
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _format_elapsed(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    h, rem = divmod(seconds_int, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_step_label(step: str | None) -> str:
    if not step:
        return "waiting"
    return step.replace("-", " ").title()


def _metric_display_value(value: object, default: str = "unknown") -> str | int | float:
    """Convert arbitrary JSON values into types accepted by st.metric."""
    if value is None:
        return default
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value if value.strip() else default
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(items) if items else default
    if isinstance(value, dict):
        if not value:
            return default
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def _html_escape(value: object, default: str = "") -> str:
    """Escape arbitrary values for small HTML dashboard snippets."""
    if value is None:
        return default
    return html.escape(str(value))


def _html_block(value: str) -> str:
    """Dedent HTML before sending it to Streamlit markdown."""
    return textwrap.dedent(value).strip()


def _existing_runs() -> list[Path]:
    if not DEFAULT_RUNS_DIR.exists():
        return []
    return sorted(
        [p for p in DEFAULT_RUNS_DIR.iterdir() if p.is_dir() and (p / "progress.json").exists()],
        reverse=True,
    )


def _bar_color(name: str, best: str) -> str:
    if name == best:
        return _BEST_COLOR
    if name in _BENCHMARK_NAMES:
        return _BENCHMARK_COLOR
    return _CANDIDATE_COLOR


def _save_metadata(output_dir: Path, llm_model: str, elapsed_seconds: float) -> None:
    """Save minimal metadata: LLM model name and running time."""
    metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "llm_model_name": llm_model,
        "running_time_sec": elapsed_seconds,
    }
    try:
        meta_path = output_dir / "meta_data.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        st.warning(f"⚠️ Could not save metadata: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Progress and Self-Audit Remediation
# ─────────────────────────────────────────────────────────────────────────────

# Mapping of remediation action IDs to affected steps
_REMEDIATION_STEPS_MAP = {
    "remove_monotonic_index_features": [12, 13, 14, 15],
    "improve_model_performance": [12, 13, 14, 15],
    "extend_lag_window": [12, 13],
    "add_seasonal_features": [12, 13],
    "increase_regularization": [13],
    "try_alternative_models": [13, 14, 15],
    "use_time_series_split": [12, 13],
    "split_by_grouping_column": [12, 13, 14, 15],
    "handle_temporal_gaps": [10, 12],
    "remove_outliers_by_isolation": [10, 13],
}

# Human-readable descriptions for remediation actions
_REMEDIATION_DESCRIPTIONS = {
    "remove_monotonic_index_features": "Remove monotonic index features causing perfect data leakage (KS=1.0)",
    "improve_model_performance": "Improve low model performance with log transform and expanded model pool",
    "extend_lag_window": "Extend lag window for better temporal dependency capture",
    "add_seasonal_features": "Add seasonal features for detected cyclical patterns",
    "increase_regularization": "Increase regularization to prevent overfitting",
    "try_alternative_models": "Try alternative model types for better performance",
    "use_time_series_split": "Switch to temporal cross-validation for proper time-series evaluation",
    "split_by_grouping_column": "Train separate models per group (multi-series detected)",
    "handle_temporal_gaps": "Handle temporal gaps in time-series data",
    "remove_outliers_by_isolation": "Remove anomalous outliers affecting model",
}


def _parse_audit_results(output_dir: Path) -> dict | None:
    """
    Parse step-17-audit.json to extract remediation information.
    
    Returns dict with:
    - overall_audit_result: "pass" or "fail"
    - remediation_actions: list of action dicts
    - restart_step: minimum step to restart from (or None if pass)
    - critical_findings: human-readable findings
    - affected_checks: which checks failed
    """
    audit_file = output_dir / "step-17-audit.json"
    if not audit_file.exists():
        return None
    
    try:
        audit = _read_json(audit_file)
        if not audit:
            return None
        
        # Normalise checks: agent may emit a list or a dict
        raw_checks = audit.get("checks", {})
        if isinstance(raw_checks, list):
            checks_dict = {c.get("check", f"check_{i}"): c for i, c in enumerate(raw_checks)}
        else:
            checks_dict = raw_checks if isinstance(raw_checks, dict) else {}

        result = {
            "overall_audit_result": audit.get("overall_audit_result", "unknown"),
            "remediation_actions": audit.get("remediation_actions", []),
            "critical_findings": audit.get("critical_findings", []),
            "checks": checks_dict,
            "restart_step": None,
            "affected_steps": set(),
        }
        
        # Determine which steps need to restart based on remediation actions.
        # Prefer affected_steps embedded in the audit action; fall back to local map.
        if result["remediation_actions"]:
            for action in result["remediation_actions"]:
                action_id = action.get("action_id")
                embedded = action.get("affected_steps")
                if isinstance(embedded, list) and embedded:
                    result["affected_steps"].update(embedded)
                elif action_id in _REMEDIATION_STEPS_MAP:
                    result["affected_steps"].update(_REMEDIATION_STEPS_MAP[action_id])
        
        # Find the minimum affected step
        if result["affected_steps"]:
            result["restart_step"] = min(result["affected_steps"])
            result["affected_steps"] = sorted(list(result["affected_steps"]))
        
        return result
    except Exception as e:
        st.warning(f"Could not parse audit results: {e}")
        return None


# Step name → sentinel output file that confirms the step completed
_STEP_SENTINEL_FILES: dict[str, str] = {
    "10-csv-read-cleansing":   "step-10-cleanse.json",
    "11-data-exploration":     "step-11-exploration.json",
    "12-feature-extraction":   "step-12-features.json",
    "13-model-training":       "step-13-training.json",
    "14-model-evaluation":     "step-14-evaluation.json",
    "15-model-selection":      "step-15-selection.json",
    "16-result-presentation":  "step-16-report.md",
    "17-critical-self-audit":  "step-17-audit.json",
    "18-llm-as-judge":         "step-18-judge.json",
}


def _completed_steps_from_files(output_dir: Path) -> set[str]:
    """Derive which pipeline steps completed by checking for their output files.

    This is more reliable than reading completed_steps from progress.json because
    the orchestrator sometimes only flushes that list at the very end of the run.
    """
    return {
        step for step, sentinel in _STEP_SENTINEL_FILES.items()
        if (output_dir / sentinel).exists()
    }


# Step number → output JSON/artifact file names that must be deleted to force re-run
_STEP_OUTPUT_FILES: dict[int, list[str]] = {
    12: ["step-12-features.json", "features.parquet"],
    13: ["step-13-training.json", "model.joblib", "holdout.npz"],
    14: ["step-14-evaluation.json"],
    15: ["step-15-selection.json"],
    16: ["step-16-report.md"],
    17: ["step-17-audit.json"],
    18: ["step-18-judge.json", "step-18-judge.md"],
}

# AUTO action → env var to inject so the step scripts apply the remediation
_AUTO_ACTION_ENV: dict[str, dict[str, str]] = {
    "remove_monotonic_index_features": {},   # features passed separately via affected_features
    "improve_model_performance":  {"PIPELINE_FORCE_EXPANSION_MODELS": "true"},
    "extend_lag_window":          {"PIPELINE_MAX_LAG": "20"},
    "add_seasonal_features":      {"PIPELINE_SEASONAL_FEATURES": "true"},
    "increase_regularization":    {"PIPELINE_REGULARIZATION": "ridge_cv"},
    "try_alternative_models":     {"PIPELINE_EXTRA_MODELS": "lightgbm,svr,histgradient"},
    "use_time_series_split":      {"PIPELINE_SPLIT_MODE": "time_series"},
}


def _delete_step_outputs(output_dir: Path, steps: list[int]) -> list[str]:
    """Delete output files for the given step numbers. Returns list of deleted paths."""
    deleted = []
    for step in steps:
        for fname in _STEP_OUTPUT_FILES.get(step, []):
            p = output_dir / fname
            if p.exists():
                p.unlink()
                deleted.append(str(p))
    return deleted


def _trigger_auto_remediation(output_dir: Path) -> bool:
    """
    Trigger an automatic remediation pass directly from the Streamlit UI.

    Reads `step-17-audit.json`, deletes output files for all affected steps, and
    re-runs the orchestrator.py with `--resume` so only the deleted steps execute.
    Environment variables encoding the AUTO remediation parameters are injected.

    Returns True if the orchestrator was launched, False otherwise.
    """
    orchestrator = output_dir / "code" / "orchestrator.py"
    if not orchestrator.exists():
        st.error("orchestrator.py not found — cannot trigger remediation automatically.")
        return False

    audit_results = _parse_audit_results(output_dir)
    if not audit_results or audit_results["overall_audit_result"] != "fail":
        st.info("No audit failure detected — nothing to remediate.")
        return False

    # Collect AUTO actions only
    auto_actions = [
        a for a in audit_results.get("remediation_actions", [])
        if a.get("type") == "[AUTO]"
    ]
    if not auto_actions:
        st.warning(
            "Only MANUAL remediation actions detected. "
            "Automatic re-run is not possible — please review the actions below "
            "and supply the required parameters manually."
        )
        return False

    # Determine affected steps and env vars
    affected_steps: set[int] = set()
    extra_env: dict[str, str] = {}
    for action in auto_actions:
        action_id = action.get("action_id", "")
        steps_for_action = _REMEDIATION_STEPS_MAP.get(action_id, [])
        affected_steps.update(steps_for_action)
        extra_env.update(_AUTO_ACTION_ENV.get(action_id, {}))
        # Handle remove_monotonic_index_features: build exclude list from findings
        if action_id == "remove_monotonic_index_features":
            feats = [
                f.get("feature", "") for f in action.get("affected_features", []) if f.get("feature")
            ]
            if not feats:
                # Fall back to parsing critical_findings
                feats = [
                    cf.get("check", "") for cf in audit_results.get("critical_findings", [])
                    if "monotonic" in cf.get("description", "").lower()
                ]
            if feats:
                extra_env["PIPELINE_EXCLUDE_FEATURES"] = ",".join(feats)

    # Always include steps 16 and 17 so the report and audit are refreshed
    affected_steps.update([16, 17])

    deleted = _delete_step_outputs(output_dir, sorted(affected_steps))
    st.caption(f"Deleted {len(deleted)} artifact(s) for steps {sorted(affected_steps)}: {deleted}")

    progress_data = _read_json(output_dir / "progress.json") or {}
    csv_path = progress_data.get("csv_path", "")
    target_column = progress_data.get("target_column", "")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    if not csv_path or not target_column:
        st.error("Cannot determine CSV path or target column from progress.json.")
        return False

    import os
    env = {**os.environ, **extra_env}
    try:
        subprocess.Popen(
            [
                "python", str(orchestrator),
                "--csv-path", csv_path,
                "--target-column", target_column,
                "--output-dir", str(output_dir),
                "--run-id", run_id,
                "--resume",
            ],
            env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        return True
    except Exception as exc:
        st.error(f"Failed to launch orchestrator for remediation: {exc}")
        return False


def _render_live_status(output_dir: Path, started_at: float) -> dict | None:
    progress = _read_json(output_dir / "progress.json")
    elapsed = _format_elapsed(time.monotonic() - started_at)

    completed_count = 0
    current_step = None
    status = "running"
    errors: list[str] = []
    completed_set: set[str] = set()

    
    # Parse audit results if available (step 17)
    audit_results = _parse_audit_results(output_dir)
    remediation_triggered = False
    restart_step = None

    if progress:
        completed = progress.get("completed_steps", [])
        if isinstance(completed, list):
            completed_set = set(completed)
        current_step = progress.get("current_step")
        status = str(progress.get("status", "running"))
        raw_errors = progress.get("errors", [])
        if isinstance(raw_errors, list):
            errors = [str(e) for e in raw_errors]

    # Use output-file presence as the primary source of truth for completed steps.
    # The orchestrator often only flushes completed_steps to progress.json at the
    # very end of the run, so file-based detection gives real-time progress.
    file_completed_set = _completed_steps_from_files(output_dir)
    # Merge: take the union so remediation re-runs aren't lost
    completed_set = completed_set | file_completed_set
    completed_count = sum(1 for s in PIPELINE_STEPS if s in completed_set)

    # If pipeline is fully done, show all steps complete
    if status == "completed":
        completed_count = len(PIPELINE_STEPS)

    # Fix stale current_step: the orchestrator often leaves current_step
    # pointing to the last step it *started* (e.g. "12-feature-extraction")
    # even after later steps complete. Derive the real active step from
    # file presence and status instead.
    if status == "completed":
        current_step = PIPELINE_STEPS[-1]
    elif current_step and current_step in completed_set:
        # Step already finished — advance to first pending step
        remaining = [s for s in PIPELINE_STEPS if s not in completed_set]
        current_step = remaining[0] if remaining else PIPELINE_STEPS[-1]
    elif not current_step:
        # No current_step written yet — infer from first pending step
        remaining = [s for s in PIPELINE_STEPS if s not in completed_set]
        current_step = remaining[0] if remaining else PIPELINE_STEPS[-1]

    # Handle remediation: if audit failed and has actions, reset progress
    if audit_results and audit_results["overall_audit_result"] == "fail":
        remediation_triggered = True
        restart_step = audit_results.get("restart_step")
        
        # If orchestrator is actively remediating, keep progress as-is
        remediation_block = progress.get("remediation") if progress else None
        if remediation_block and status == "remediating":
            rem_iter = remediation_block.get("iteration", 1)
            max_iter = remediation_block.get("max_iterations", 3)
            steps_rerun = remediation_block.get("steps_rerun", [])
            progress_text += f" (Remediation {rem_iter}/{max_iter} — re-running steps {steps_rerun})"
        elif restart_step is not None and status not in ("remediating", "completed"):
            # Not yet remediating: show where it will restart
            completed_count = restart_step - 10  # Step 10 is first step
            status = "remediation"

    # Update progress bar with correct count
    progress_text = f"Completed {completed_count}/{len(PIPELINE_STEPS)} steps"
    if remediation_triggered and restart_step is not None:
        progress_text += f" (Restart at Step {restart_step})"
    
    st.progress(min(1.0, completed_count / len(PIPELINE_STEPS)), text=progress_text)

    c1, c2, c3, c4 = st.columns(4)
    
    # Update status display
    status_display = status.upper()
    if remediation_triggered:
        status_display = "🔄 REMEDIATION REQUIRED"
    c1.metric("Status", status_display)
    c2.metric("Step", _format_step_label(current_step))
    c3.metric("Elapsed", elapsed)
    cm = progress.get("current_model") if progress else None
    c4.metric("Model", cm or "—")

    # Display remediation information if triggered
    if remediation_triggered and audit_results:
        st.markdown("---")
        st.warning("🔄 **REMEDIATION TRIGGERED**")
        
        if restart_step is not None:
            affected_steps = audit_results.get("affected_steps", [])
            st.markdown(
                f"**Restart at Step {restart_step}** | "
                f"Affected steps: {', '.join(map(str, affected_steps))}"
            )
        
        # Display remediation actions
        if audit_results["remediation_actions"]:
            st.subheader("📋 Remediation Actions")
            for i, action in enumerate(audit_results["remediation_actions"], 1):
                action_id = action.get("action_id", "unknown")
                severity = action.get("severity", "medium")
                description = action.get("description", _REMEDIATION_DESCRIPTIONS.get(action_id, ""))
                action_type = action.get("type", "UNKNOWN")
                
                # Format action display
                emoji = "🟢" if action_type == "[AUTO]" else "🔴"
                col1, col2 = st.columns([1, 5])
                col1.write(emoji)
                col2.write(f"**{i}. {action_id}** ({action_type})")
                st.caption(description)
        
        # Display critical findings
        if audit_results["critical_findings"]:
            st.subheader("⚠️ Critical Findings")
            for finding in audit_results["critical_findings"]:
                check = finding.get("check", "unknown")
                desc = finding.get("description", "")
                sev = finding.get("severity", "medium")
                st.error(f"**{check}** ({sev.upper()}): {desc}")
        
        # Display failed checks
        if audit_results["checks"]:
            st.subheader("❌ Audit Check Results")
            for check_name, check_result in audit_results["checks"].items():
                check_status = check_result.get("status", "unknown")
                
                if check_status == "fail":
                    status_color = "🔴"
                elif check_status == "marginal":
                    status_color = "🟡"
                else:
                    status_color = "🟢"
                
                findings = check_result.get("findings", [])
                with st.expander(f"{status_color} {check_name}: {check_status.upper()}"):
                    if findings:
                        for finding in findings:
                            st.caption(f"• {finding}")
                    st.json(check_result)
        
        st.markdown("---")

    if progress:
        mp = progress.get("model_progress")
        if isinstance(mp, (int, float)):
            st.progress(min(1.0, max(0.0, float(mp))), text="Model training progress")
        completed_models = progress.get("completed_models", [])
        if completed_models:
            st.caption("✓ " + ", ".join(completed_models))
        with st.expander("Raw progress.json"):
            st.json(progress)

    if errors:
        st.error("\n".join(errors))

    return progress


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — EDA
# ─────────────────────────────────────────────────────────────────────────────

def _render_eda_tab(output_dir: Path) -> None:
    exp = _read_json(output_dir / "step-11-exploration.json")
    cleanse = _read_json(output_dir / "step-10-cleanse.json")

    if not exp:
        st.info("Step-11 exploration data not yet available.")
        return

    ts = exp.get("ts_diagnostics") or {}

    # ── Client summary ───────────────────────────────────────────────────────
    summary = exp.get("client_facing_summary")
    if summary:
        st.info(f"💬 {summary}")

    # ── Data Quality ─────────────────────────────────────────────────────────
    st.subheader("📋 Data Quality")
    if cleanse:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows (raw)", cleanse.get("row_count_initial", "—"))
        c2.metric("Rows (clean)", cleanse.get("row_count_after", "—"))
        c3.metric("Duplicates removed", cleanse.get("duplicate_rows_removed", 0))
        c4.metric("Columns", cleanse.get("column_count", "—"))

        null_rate = cleanse.get("null_rate", {})
        if null_rate:
            cols = list(null_rate.keys())
            vals = [null_rate[c] * 100 for c in cols]
            fig = go.Figure(go.Bar(x=cols, y=vals, marker_color="steelblue"))
            fig.update_layout(title="Null Rate per Column (%)",
                              xaxis_title="Column", yaxis_title="Null %",
                              template="plotly_white", height=300)
            st.plotly_chart(fig, use_container_width=True)

        outliers = cleanse.get("outliers", {})
        if outliers:
            st.markdown("**IQR Outlier Summary**")
            rows = []
            for col, info in outliers.items():
                rows.append({
                    "Column": col,
                    "Outlier Count": info.get("iqr_outlier_count", 0),
                    "Outlier %": f"{info.get('outlier_fraction', 0)*100:.2f}%",
                    "Lower Bound": f"{info.get('iqr_lower_bound', 0):.2f}",
                    "Upper Bound": f"{info.get('iqr_upper_bound', 0):.2f}",
                })
            st.dataframe(rows, use_container_width=True)

            # Box plot from cleaned.parquet
            parquet_path = output_dir / "cleaned.parquet"
            if parquet_path.exists():
                try:
                    df_clean = pl.read_parquet(parquet_path)
                    numeric_cols = [c for c in df_clean.columns
                                    if df_clean[c].dtype in (pl.Float64, pl.Float32,
                                                              pl.Int32, pl.Int64)][:12]
                    if numeric_cols:
                        with st.expander("Box Plots (outlier visualisation)", expanded=False):
                            df_pd = df_clean.select(numeric_cols).to_pandas()
                            fig_box = px.box(df_pd, points=False,
                                             title="Distribution of Numeric Columns")
                            fig_box.update_layout(template="plotly_white", height=400)
                            st.plotly_chart(fig_box, use_container_width=True)
                except Exception:
                    pass

    # ── Target time-series ───────────────────────────────────────────────────
    st.subheader("📈 Target Series")
    parquet_path = output_dir / "cleaned.parquet"
    prog = _read_json(output_dir / "progress.json") or {}
    target_col = prog.get("target_column") or exp.get("target_candidates", [{}])[0].get("column")
    time_col = exp.get("time_column")

    if parquet_path.exists() and target_col:
        try:
            df_clean = pl.read_parquet(parquet_path)
            if target_col in df_clean.columns:
                y_series = df_clean[target_col].to_numpy()
                x_axis = (df_clean[time_col].to_numpy()
                          if time_col and time_col in df_clean.columns
                          else np.arange(len(y_series)))
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(x=x_axis, y=y_series, mode="lines",
                                             name=target_col,
                                             line=dict(color="royalblue", width=1.5)))
                fig_ts.update_layout(title=f"Target: {target_col}",
                                     xaxis_title="Time", yaxis_title=target_col,
                                     template="plotly_white", height=300)
                st.plotly_chart(fig_ts, use_container_width=True)

                # Distribution
                fig_hist = go.Figure(go.Histogram(x=y_series, nbinsx=50,
                                                   marker_color="rgba(26,118,255,0.7)"))
                fig_hist.update_layout(title=f"Distribution of {target_col}",
                                       template="plotly_white", height=280)
                st.plotly_chart(fig_hist, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render target series: {e}")

    # ── Stationarity ─────────────────────────────────────────────────────────
    st.subheader("📊 Stationarity Tests (ADF + KPSS)")
    conclusion = ts.get("stationarity_conclusion", "unknown")
    color_map = {
        "stationary": "success",
        "non-stationary": "error",
        "trend-stationary": "warning",
        "ambiguous": "warning",
        "insufficient_data": "info",
    }
    badge = color_map.get(conclusion, "info")
    getattr(st, badge)(f"**Stationarity conclusion: {conclusion.upper()}**")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**ADF Test**")
        # accept both naming conventions (adf_stat / adf_statistic, adf_pval / adf_pvalue)
        adf_stat_val = ts.get("adf_statistic") or ts.get("adf_stat")
        adf_pval_val = ts.get("adf_pvalue") or ts.get("adf_pval")
        st.metric("ADF Statistic", f"{adf_stat_val:.4f}" if isinstance(adf_stat_val, (int, float)) else "N/A")
        st.metric("ADF p-value", f"{adf_pval_val:.4f}" if isinstance(adf_pval_val, (int, float)) else "N/A")
    with c2:
        st.markdown("**KPSS Test**")
        kpss_stat_val = ts.get("kpss_statistic") or ts.get("kpss_stat")
        kpss_pval_val = ts.get("kpss_pvalue") or ts.get("kpss_pval")
        st.metric("KPSS Statistic", f"{kpss_stat_val:.4f}" if isinstance(kpss_stat_val, (int, float)) else "N/A")
        st.metric("KPSS p-value", f"{kpss_pval_val:.4f}" if isinstance(kpss_pval_val, (int, float)) else "N/A")

    # ── Hurst Exponent ───────────────────────────────────────────────────────
    st.subheader("🧠 Memory Analysis — Hurst Exponent")
    hurst = ts.get("hurst_exponent")
    if hurst is not None:
        interp = ts.get("hurst_interpretation", "unknown")
        hurst_r2 = ts.get("hurst_r2_fit")
        c1, c2, c3 = st.columns(3)
        c1.metric("Hurst Exponent (H)", f"{hurst:.4f}")
        c2.metric("Interpretation", interp.replace("_", " ").title())
        c3.metric("R/S Fit R²", f"{hurst_r2:.4f}" if hurst_r2 else "—")

        # Gauge-style indicator
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=hurst,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Hurst Exponent H"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 0.45], "color": "lightcoral"},
                    {"range": [0.45, 0.55], "color": "lightyellow"},
                    {"range": [0.55, 0.75], "color": "lightgreen"},
                    {"range": [0.75, 1.0], "color": "mediumseagreen"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "value": 0.5},
            },
        ))
        fig_g.update_layout(height=280, template="plotly_white")
        st.plotly_chart(fig_g, use_container_width=True)

        _hurst_legend()
    else:
        st.info("Hurst exponent not available (series too short or not computed).")

    # ── White Noise / Ljung-Box ──────────────────────────────────────────────
    st.subheader("🎲 White Noise Test (Ljung-Box)")
    wn = ts.get("white_noise")
    lb_pvals = ts.get("ljung_box_pvalues", {})
    if wn is True:
        st.error("⚠️ Target series is WHITE NOISE — autocorrelation is not exploitable. "
                 "Only naive baselines are meaningful.")
    elif wn is False:
        st.success("✅ Target series has exploitable autocorrelation structure.")
    if lb_pvals:
        st.write("Ljung-Box p-values:")
        lb_cols = st.columns(len(lb_pvals))
        for i, (lag, pval) in enumerate(lb_pvals.items()):
            lb_cols[i].metric(f"Lag {lag}", f"{pval:.4f}",
                              delta="significant" if pval < 0.05 else "not significant",
                              delta_color="off")

    # ── ACF / PACF ───────────────────────────────────────────────────────────
    st.subheader("📉 ACF / PACF")
    acf_vals = ts.get("acf_values", [])
    pacf_vals = ts.get("pacf_values", [])
    acf_sig = set(ts.get("acf_significant_lags", []))
    pacf_sig = set(ts.get("pacf_significant_lags", []))

    if acf_vals and pacf_vals:
        lags = list(range(len(acf_vals)))
        n_obs = exp.get("shape", {}).get("rows", 1000)
        conf_bound = 2.0 / (n_obs ** 0.5)

        c1, c2 = st.columns(2)
        with c1:
            colors_acf = ["red" if i in acf_sig else "steelblue" for i in lags]
            fig_acf = go.Figure()
            for i, (lag, val) in enumerate(zip(lags, acf_vals)):
                fig_acf.add_trace(go.Bar(x=[lag], y=[val],
                                         marker_color=colors_acf[i],
                                         showlegend=False))
            fig_acf.add_hline(y=conf_bound, line_dash="dash", line_color="gray",
                               annotation_text="95% CI")
            fig_acf.add_hline(y=-conf_bound, line_dash="dash", line_color="gray")
            fig_acf.update_layout(title="ACF (red = significant)",
                                   xaxis_title="Lag", yaxis_title="Autocorrelation",
                                   template="plotly_white", height=300)
            st.plotly_chart(fig_acf, use_container_width=True)

        with c2:
            colors_pacf = ["red" if i in pacf_sig else "steelblue" for i in lags]
            fig_pacf = go.Figure()
            for i, (lag, val) in enumerate(zip(lags[1:], pacf_vals[1:]), start=1):
                fig_pacf.add_trace(go.Bar(x=[lag], y=[val],
                                           marker_color=colors_pacf[i],
                                           showlegend=False))
            fig_pacf.add_hline(y=conf_bound, line_dash="dash", line_color="gray",
                                annotation_text="95% CI")
            fig_pacf.add_hline(y=-conf_bound, line_dash="dash", line_color="gray")
            fig_pacf.update_layout(title="PACF (red = significant)",
                                    xaxis_title="Lag", yaxis_title="Partial Autocorrelation",
                                    template="plotly_white", height=300)
            st.plotly_chart(fig_pacf, use_container_width=True)

        suggested_p = ts.get("suggested_ar_order")
        suggested_q = ts.get("suggested_ma_order")
        if suggested_p is not None or suggested_q is not None:
            st.caption(f"Suggested orders → AR(p)={suggested_p}  MA(q)={suggested_q}  "
                       f"d={ts.get('suggested_d', '?')}")
    else:
        st.info("ACF/PACF values not available in step-11 output.")

    # ── Seasonality ──────────────────────────────────────────────────────────
    st.subheader("🌊 Seasonality")
    detected_periods = ts.get("detected_periods", [])
    primary = ts.get("primary_seasonal_period")
    trend_strength = ts.get("trend_strength")
    trend_detected = ts.get("trend_detected")
    freq = exp.get("detected_frequency", "unknown")

    c1, c2, c3 = st.columns(3)
    c1.metric("Detected Frequency", freq)
    c2.metric("Primary Seasonal Period", primary or "None")
    c3.metric("Trend Detected",
              "Yes" if trend_detected else "No",
              delta=f"strength={trend_strength:.2f}" if trend_strength else None,
              delta_color="off")

    if detected_periods:
        rows = []
        for p in detected_periods:
            if isinstance(p, dict):
                rows.append({
                    "Period": p.get("period"),
                    "Seasonal Strength": f"{p.get('seasonal_strength', 0):.3f}",
                    "Significant": "✅" if p.get("significant") else "❌",
                })
            else:
                # Pipeline emits plain ints (e.g. [7, 365])
                rows.append({
                    "Period": int(p),
                    "Seasonal Strength": "—",
                    "Significant": "✅",
                })
        st.dataframe(rows, use_container_width=True)

    # ── Mutual Information ───────────────────────────────────────────────────
    st.subheader("🔗 Mutual Information Ranking")
    _mi_raw = exp.get("mi_ranking", [])
    noise_baseline = exp.get("noise_mi_baseline")
    # Normalise: accept both [feature, score] tuples and {"feature":…, "mi_score":…} dicts
    mi_ranking: list[dict] = []
    for entry in _mi_raw:
        if isinstance(entry, (list, tuple)):
            mi_ranking.append({"feature": entry[0], "mi_score": entry[1]})
        elif isinstance(entry, dict):
            # Normalise key: accept mi_score, mi, score, importance, value
            score = (entry.get("mi_score") or entry.get("mi") or entry.get("score")
                     or entry.get("importance") or entry.get("value") or 0.0)
            mi_ranking.append({**entry, "mi_score": score})
        else:
            continue
    # Flag entries below noise baseline
    if noise_baseline is not None:
        for e in mi_ranking:
            if "below_noise_baseline" not in e:
                e["below_noise_baseline"] = (e.get("mi_score") or 0.0) <= noise_baseline

    if mi_ranking:
        mi_features = [e["feature"] for e in mi_ranking]
        mi_scores = [e["mi_score"] for e in mi_ranking]
        mi_colors = ["tomato" if e.get("below_noise_baseline") else "steelblue"
                     for e in mi_ranking]

        fig_mi = go.Figure(go.Bar(
            x=mi_scores[::-1], y=mi_features[::-1],
            orientation="h", marker_color=mi_colors[::-1]
        ))
        if noise_baseline is not None:
            fig_mi.add_vline(x=noise_baseline, line_dash="dash", line_color="red",
                             annotation_text="Noise baseline")
        fig_mi.update_layout(
            title="Mutual Information vs Target (red = below noise baseline)",
            xaxis_title="MI Score", template="plotly_white", height=max(300, len(mi_features) * 18)
        )
        st.plotly_chart(fig_mi, use_container_width=True)

        excluded = exp.get("excluded_features", {})
        if excluded:
            with st.expander(f"Excluded features ({len(excluded)})", expanded=False):
                if isinstance(excluded, dict):
                    for feat, reason in excluded.items():
                        st.write(f"- **{feat}**: {reason}")
                else:
                    for entry in excluded:
                        feat = entry.get("feature", "?") if isinstance(entry, dict) else str(entry)
                        reason = entry.get("reason", "") if isinstance(entry, dict) else ""
                        st.write(f"- **{feat}**: {reason}")

    # ── Model Class Recommendations ──────────────────────────────────────────
    st.subheader("🎯 Recommended Model Classes")
    recs = exp.get("model_class_recommendations", [])
    if recs:
        for r in recs:
            if isinstance(r, dict):
                st.markdown(f"**{r.get('model_class', '?')}** — {r.get('justification', '')}")
            else:
                # Pipeline emits plain strings (e.g. ["AR", "Ridge", ...])
                st.markdown(f"- {r}")
    else:
        st.info("Model class recommendations not available.")


def _hurst_legend() -> None:
    st.caption(
        "H < 0.45 → anti-persistent (mean-reverting) | "
        "H ≈ 0.5 → random walk | "
        "H 0.55–0.75 → mildly persistent | "
        "H > 0.75 → strongly persistent / trending"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

_GROUP_META: dict[str, dict] = {
    "A":     {"label": "Calendar",        "icon": "🗓️",  "color": "#4C9BE8",
              "desc": "Date/time structural signals (hour, day, month, elapsed, …)"},
    "B":     {"label": "Target Lags",     "icon": "⏱️",  "color": "#9B59B6",
              "desc": "Autoregressive features from past target values"},
    "C":     {"label": "Exogenous Lags",  "icon": "🔗",  "color": "#27AE60",
              "desc": "Lagged exogenous variables with significant cross-correlation"},
    "D":     {"label": "Differencing",    "icon": "📉",  "color": "#E67E22",
              "desc": "First/seasonal differences for stationarity"},
    "E":     {"label": "Rolling Stats",   "icon": "📊",  "color": "#16A085",
              "desc": "Rolling mean, std, min, max & EWM (all lag-shifted)"},
    "F":     {"label": "Fourier",         "icon": "🌊",  "color": "#E74C3C",
              "desc": "Harmonic sin/cos components capturing periodic seasonality"},
    "G":     {"label": "PCA Factors",     "icon": "🧩",  "color": "#F39C12",
              "desc": "Compressed exogenous signal via PCA (designed for FAAR models)"},
    "exog":  {"label": "Exogenous (raw)", "icon": "🔤",  "color": "#95A5A6",
              "desc": "Forward-known exogenous features"},
    "other": {"label": "Other",           "icon": "🔲",  "color": "#7F8C8D",
              "desc": "Features that did not match a known pattern"},
}

# Display order for groups
_GROUP_ORDER = ["A", "B", "C", "D", "E", "F", "G", "exog", "other"]


def _group_meta(g: str) -> dict:
    """Return display metadata for a group key, generating a fallback for unknown keys."""
    if g in _GROUP_META:
        return _GROUP_META[g]
    # Unknown group — generate a display label from the key itself
    return {"label": g.replace("_", " ").title(), "icon": "🔲", "color": "#7F8C8D", "desc": ""}


def _infer_feature_group(fname: str, target_col: str) -> str:
    """
    Infer a feature group letter purely from the feature name.

    Priority order (first match wins):
      G — PCA factors
      F — Fourier harmonics
      D — Differencing
      E — Rolling / EWM statistics
      B — Target lags  (prefix matches target, or name starts with y_lag)
      C — Exogenous lags
      A — Calendar / time-structural features
      other — anything else
    """
    fl = fname.lower()
    tc = (target_col or "").lower()

    if "pca_factor" in fl or "pca_component" in fl or re.match(r"^factor_\d+$", fl):
        return "G"
    if "fourier_" in fl or "_fourier" in fl:
        return "F"
    if "_diff_" in fl or fl.endswith("_diff") or re.search(r"_diff\d+$", fl):
        return "D"
    if "rolling_" in fl or "_rolling" in fl or "ewm_" in fl or "_ewm" in fl:
        return "E"
    # Lag features — distinguish target vs exogenous by prefix
    if "_lag_" in fl or re.search(r"_lag\d+$", fl):
        if fl.startswith("y_lag") or (tc and fl.startswith(tc + "_lag")):
            return "B"
        return "C"
    # Calendar / time-structural
    _CALENDAR_TOKENS = {
        "month", "year", "hour", "quarter", "season", "elapsed",
        "day_of", "day_of_week", "day_of_month", "day_of_year",
        "week_of", "week_of_year", "month_of", "time_index",
        "is_weekend", "is_month", "is_holiday", "t_elapsed", "trend",
    }
    if fl in ("month", "year", "day", "hour", "week", "quarter", "season"):
        return "A"
    if any(tok in fl for tok in _CALENDAR_TOKENS):
        return "A"
    # Last resort
    return "other"


def _parse_feature_groups(
    feature_names: list[str],
    features_created: dict,
    target_col: str,
) -> dict[str, list[str]]:
    """
    Group features by engineering type.

    Strategy (in priority order):
    1. If `features_created` carries group tags (group_A … group_G), use them.
    2. Otherwise fall back to name-based inference via `_infer_feature_group`.

    Returns {group_key: [feature_names]} preserving the canonical group order.
    """
    # --- path 1: tag-based (old schema) ---
    TAG_MAP = {
        "group_a": "A", "calendar": "A",
        "group_b": "B",
        "group_c": "C",
        "group_d": "D",
        "group_e": "E",
        "group_f": "F",
        "group_g": "G",
        "exogenous_forward": "exog",
        "exog": "exog",
    }
    if features_created:
        groups: dict[str, list[str]] = {}
        for feat, tag in features_created.items():
            tl = str(tag).lower()
            g = next((v for k, v in TAG_MAP.items() if k in tl), "other")
            groups.setdefault(g, []).append(feat)
        # Any feature in the feature_names list but missing from features_created
        # gets assigned by name inference
        tagged = set(features_created.keys())
        for feat in feature_names:
            if feat not in tagged:
                groups.setdefault(_infer_feature_group(feat, target_col), []).append(feat)
        return groups

    # --- path 2: name inference (new/minimal schema) ---
    groups = {}
    for feat in feature_names:
        g = _infer_feature_group(feat, target_col)
        groups.setdefault(g, []).append(feat)
    return groups


def _sorted_groups(groups: dict[str, list[str]]) -> list[str]:
    """Return group keys in canonical display order; unknown keys appended alphabetically."""
    known = [g for g in _GROUP_ORDER if g in groups]
    unknown = sorted(g for g in groups if g not in _GROUP_ORDER)
    return known + unknown


def _render_features_tab(output_dir: Path) -> None:
    feat_info = _read_json(output_dir / "step-12-features.json")
    leakage = _read_json(output_dir / "leakage_audit.json")
    features_parquet = output_dir / "features.parquet"

    if not feat_info:
        st.info("Feature extraction step not yet complete. Run the pipeline to generate features.")
        return

    features_created: dict = feat_info.get("features_created", {})
    # Accept both key spellings used by different pipeline versions
    feature_count: int = (
        feat_info.get("feature_count")
        or feat_info.get("features_count")
        or len(feat_info.get("features", []))
        or len(features_created)
    )
    features_excluded: dict = feat_info.get("features_excluded", {}) or {}
    # Normalise: excluded may be a list of strings rather than a dict
    if isinstance(features_excluded, list):
        features_excluded = {f: "excluded" for f in features_excluded}
    target_col: str = feat_info.get("target_column", "")
    # Feature name list: prefer the explicit 'features' key, fall back to features_created keys
    feature_names: list[str] = feat_info.get("features") or list(features_created.keys())

    # ── Split info — merge step-12 and step-13 (different schemas may omit fields) ──
    split_info: dict = feat_info.get("split_strategy", {}) or {}
    training_json = _read_json(output_dir / "step-13-training.json") or {}
    train_rows: int | None = (
        split_info.get("holdout_start_index")
        or split_info.get("train_row_count")
        or training_json.get("train_size")
    )
    holdout_rows: int | None = (
        split_info.get("holdout_size")
        or split_info.get("holdout_row_count")
        or training_json.get("test_size")
    )
    burn_in: int = split_info.get("burn_in_rows", 0) or 0
    # holdout_start_index for the data preview — reconstruct if missing
    holdout_start_idx: int | None = (
        split_info.get("holdout_start_index")
        or (train_rows if isinstance(train_rows, int) else None)
    )

    # ── Leakage info — accept inline key or separate leakage_audit.json ─────
    leakage_data: dict = (
        leakage
        or feat_info.get("leakage")
        or {}
    )

    # ── PCA info — accept nested pca_info or top-level pca_* keys ───────────
    pca_info: dict = feat_info.get("pca_info") or {}
    if not pca_info and feat_info.get("pca_n_components"):
        pca_info = {
            "pca_n_components": feat_info.get("pca_n_components"),
            "pca_explained_variance_ratio": feat_info.get("pca_explained_variance_ratio", []),
        }

    # ── Scaling metadata (new schema only) ───────────────────────────────────
    scaling_meta: dict = feat_info.get("scaling_metadata") or {}

    groups = _parse_feature_groups(feature_names, features_created, target_col)

    # ── Feature → group lookup ────────────────────────────────────────────────
    feat_to_group: dict[str, str] = {}
    for g, feats in groups.items():
        for f in feats:
            feat_to_group[f] = g

    ordered_groups = _sorted_groups(groups)

    # ── Overview KPIs ─────────────────────────────────────────────────────────
    st.subheader("🧱 Feature Engineering Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Features", feature_count)
    # Count only non-empty groups with meaningful content
    c2.metric("Feature Groups", len([g for g in groups if groups[g]]))
    c3.metric("Train Rows", f"{train_rows:,}" if isinstance(train_rows, int) else "—")
    c4.metric("Holdout Rows", f"{holdout_rows:,}" if isinstance(holdout_rows, int) else "—")
    c5.metric("Burn-in Dropped", f"{burn_in:,}")

    # Leakage badge
    if leakage_data:
        ls = leakage_data.get("status", "unknown")
        lcolor_map = {"pass": "success", "warn": "warning", "fail": "error"}
        # Accept different key names for violations
        violations = (
            leakage_data.get("pearson_violations")
            or leakage_data.get("leakage_candidates")
            or []
        )
        msg = (
            f"**Leakage Audit: {ls.upper()}** — No leakage detected ✓"
            if ls == "pass"
            else f"**Leakage Audit: {ls.upper()}** — violations: {violations}"
        )
        getattr(st, lcolor_map.get(ls, "info"))(msg)

    # Scaling info badge (new schema)
    if scaling_meta.get("scaler_used") and scaling_meta["scaler_used"] != "None":
        st.info(f"Scaler applied: **{scaling_meta['scaler_used']}** on {len(scaling_meta.get('features_scaled', []))} features")

    st.divider()

    # ── Feature Group Breakdown — donut + per-group badge lists ──────────────
    st.subheader("🎨 Feature Groups Breakdown")
    col_donut, col_groups = st.columns([1, 1.7])

    with col_donut:
        g_labels, g_counts, g_colors = [], [], []
        for g in ordered_groups:
            meta = _group_meta(g)
            g_labels.append(f"{meta['icon']} {meta['label']}")
            g_counts.append(len(groups[g]))
            g_colors.append(meta["color"])

        fig_donut = go.Figure(go.Pie(
            labels=g_labels,
            values=g_counts,
            hole=0.54,
            marker_colors=g_colors,
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>%{value} features (%{percent})<extra></extra>",
        ))
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=0, r=0),
            height=360,
            annotations=[dict(
                text=f"<b>{feature_count}</b><br>features",
                x=0.5, y=0.5, showarrow=False, font=dict(size=17),
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_groups:
        for g in ordered_groups:
            meta = _group_meta(g)
            feats = groups[g]
            g_header = g if len(g) == 1 else g.title()
            with st.expander(
                f"{meta['icon']} **Group {g_header} — {meta['label']}** &nbsp; `{len(feats)}` features"
                + (f" &nbsp;·&nbsp; _{meta['desc']}_" if meta.get("desc") else ""),
                expanded=False,
            ):
                badge_html = " ".join(
                    f'<span style="background:{meta["color"]}22;border:1px solid {meta["color"]}88;'
                    f'border-radius:5px;padding:2px 8px;font-size:12px;font-family:monospace;'
                    f'margin:2px 2px;display:inline-block">{f}</span>'
                    for f in feats
                )
                st.markdown(badge_html, unsafe_allow_html=True)

    st.divider()

    # ── Train / Holdout Split Visualisation ──────────────────────────────────
    st.subheader("📐 Train / Holdout Split")
    if isinstance(train_rows, int) and isinstance(holdout_rows, int) and train_rows > 0:
        fig_split = go.Figure()
        if burn_in:
            fig_split.add_trace(go.Bar(
                name="Burn-in (dropped)", x=[burn_in], y=["rows"], orientation="h",
                marker_color="#BDC3C7",
                hovertemplate=f"Burn-in: {burn_in:,} rows<extra></extra>",
            ))
        fig_split.add_trace(go.Bar(
            name="Training", x=[train_rows], y=["rows"], orientation="h",
            marker_color="#3498DB",
            hovertemplate=f"Training: {train_rows:,} rows<extra></extra>",
        ))
        fig_split.add_trace(go.Bar(
            name="Holdout (test)", x=[holdout_rows], y=["rows"], orientation="h",
            marker_color="#E74C3C",
            hovertemplate=f"Holdout: {holdout_rows:,} rows<extra></extra>",
        ))
        fig_split.update_layout(
            barmode="stack", height=120,
            margin=dict(t=5, b=5, l=5, r=5),
            template="plotly_white",
            legend=dict(orientation="h", y=1.5),
            xaxis_title="Row count",
        )
        st.plotly_chart(fig_split, use_container_width=True)
        pct = holdout_rows / (train_rows + holdout_rows) * 100
        cs1, cs2, cs3 = st.columns(3)
        cs1.metric("Training rows", f"{train_rows:,}")
        cs2.metric("Holdout rows", f"{holdout_rows:,}")
        cs3.metric("Holdout fraction", f"{pct:.1f}%")

    st.divider()

    # ── Model × Feature Group — importance heatmap + per-model bar charts ────
    st.subheader("🔬 Feature Importance by Model & Group")
    candidate_models = sorted(output_dir.glob("candidate-*.joblib"))
    model_importance_by_group: dict[str, dict[str, float]] = {}
    feat_importances_raw: dict[str, list[float]] = {}
    # Track the source of each model's importances for labelling
    importance_source: dict[str, str] = {}

    for cpath in candidate_models:
        mname = cpath.stem.replace("candidate-", "")
        try:
            mdl = joblib.load(str(cpath))
            imps: np.ndarray | None = None

            # Unwrap sklearn Pipeline — attributes live on the final estimator
            estimator = mdl.steps[-1][1] if hasattr(mdl, "steps") else mdl

            # Tree models: Gini impurity-based feature importances
            if hasattr(estimator, "feature_importances_"):
                raw = np.array(estimator.feature_importances_)
                if len(raw) == len(feature_names):
                    imps = raw
                    importance_source[mname] = "impurity"

            # Linear models (Ridge, ElasticNet, …): absolute coefficients, normalised
            elif hasattr(estimator, "coef_"):
                coef = np.array(estimator.coef_).ravel()
                if len(coef) == len(feature_names):
                    abs_coef = np.abs(coef)
                    total = abs_coef.sum()
                    imps = abs_coef / total if total > 0 else abs_coef
                    importance_source[mname] = "coef"

            if imps is not None:
                feat_importances_raw[mname] = list(imps)
                group_imp: dict[str, float] = {}
                for fi, fname in enumerate(feature_names):
                    g = feat_to_group.get(fname) or _infer_feature_group(fname, target_col)
                    group_imp[g] = group_imp.get(g, 0.0) + float(imps[fi])
                model_importance_by_group[mname] = group_imp
        except Exception:
            pass

    if model_importance_by_group:
        # Build x-axis labels that annotate the importance source
        _source_label = {"impurity": "🌲 impurity", "coef": "📐 |coef|"}
        model_names_sorted = list(model_importance_by_group.keys())
        x_labels = [
            f"{m}\n({_source_label.get(importance_source.get(m, ''), '?')})"
            for m in model_names_sorted
        ]

        all_groups_in_data = _sorted_groups(
            {g: [] for d in model_importance_by_group.values() for g in d}
        )
        y_labels = [
            f"{_group_meta(g)['icon']} {_group_meta(g)['label']}"
            for g in all_groups_in_data
        ]
        z_matrix = [
            [model_importance_by_group[m].get(g, 0.0) for m in model_names_sorted]
            for g in all_groups_in_data
        ]
        fig_heat = go.Figure(go.Heatmap(
            z=z_matrix,
            x=x_labels,
            y=y_labels,
            colorscale="Blues",
            text=[[f"{v:.3f}" for v in row] for row in z_matrix],
            texttemplate="%{text}",
            hovertemplate="Group: %{y}<br>Model: %{x}<br>Cumulative importance: %{z:.4f}<extra></extra>",
        ))
        fig_heat.update_layout(
            title=(
                "Cumulative feature importance per group — "
                "🌲 tree models: Gini impurity  |  📐 linear models: normalised |coef|"
            ),
            height=max(320, len(all_groups_in_data) * 54 + 80),
            template="plotly_white",
            xaxis_title="Model",
            yaxis_title="Feature Group",
            margin=dict(l=160, r=20, t=60, b=50),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("#### 📌 Top-20 Features per Model")
        for mname, imps in feat_importances_raw.items():
            src = importance_source.get(mname, "")
            src_note = "normalised |coef|" if src == "coef" else "Gini impurity"
            with st.expander(f"`{mname}` — top 20 features  ·  _{src_note}_", expanded=False):
                fi_pairs = sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True)[:20]
                top_names = [p[0] for p in fi_pairs]
                top_vals = [p[1] for p in fi_pairs]
                top_colors = [
                    _group_meta(feat_to_group.get(n) or _infer_feature_group(n, target_col))["color"]
                    for n in top_names
                ]
                top_labels = [
                    f"{_group_meta(feat_to_group.get(n) or _infer_feature_group(n, target_col))['icon']} {n}"
                    for n in top_names
                ]
                fig_fi = go.Figure(go.Bar(
                    x=top_vals, y=top_labels, orientation="h",
                    marker_color=top_colors,
                    text=[f"{v:.4f}" for v in top_vals], textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Importance: %{x:.5f}<extra></extra>",
                ))
                fig_fi.update_layout(
                    height=max(320, len(top_names) * 28 + 80),
                    template="plotly_white",
                    xaxis_title=f"Feature Importance ({src_note})",
                    margin=dict(l=20, r=70, t=20, b=20),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info(
            "No feature importances available yet. "
            "Once the pipeline has run, tree-model (impurity) and linear-model (|coef|) "
            "importances will appear here."
        )

    # ── PCA Variance Explained ────────────────────────────────────────────────
    if pca_info.get("pca_n_components"):
        st.divider()
        st.subheader("🧩 PCA Factor Analysis (Group G)")
        evr = pca_info.get("pca_explained_variance_ratio", [])
        n_comp = pca_info.get("pca_n_components", len(evr))
        if evr:
            cumulative = np.cumsum(evr).tolist()
            comp_labels = [f"PC{i + 1}" for i in range(len(evr))]
            fig_pca = go.Figure()
            fig_pca.add_trace(go.Bar(
                name="Individual", x=comp_labels, y=evr,
                marker_color="#F39C12",
                text=[f"{v:.1%}" for v in evr], textposition="outside",
            ))
            fig_pca.add_trace(go.Scatter(
                name="Cumulative", x=comp_labels, y=cumulative,
                mode="lines+markers", line=dict(color="#E74C3C", width=2),
                yaxis="y2",
            ))
            fig_pca.update_layout(
                title=f"PCA Explained Variance — {n_comp} components retained",
                yaxis=dict(title="Individual Variance", tickformat=".0%", range=[0, max(evr) * 1.35]),
                yaxis2=dict(
                    title="Cumulative Variance", overlaying="y", side="right",
                    tickformat=".0%", range=[0, 1.08],
                ),
                template="plotly_white", height=360,
                legend=dict(orientation="h", y=1.15),
                margin=dict(t=60, b=40),
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            cp1, cp2 = st.columns(2)
            cp1.metric("Components retained", n_comp)
            cp2.metric("Variance captured", f"{cumulative[-1]:.1%}")

    # ── Feature–Target Correlation ────────────────────────────────────────────
    if features_parquet.exists():
        st.divider()
        st.subheader("📈 Feature–Target Correlation (top 25)")
        try:
            df = pl.read_parquet(features_parquet)
            # Detect target column: use feat_info, or fallback to second column
            tgt = target_col or (df.columns[1] if len(df.columns) > 1 else None)
            if tgt and tgt in df.columns:
                tgt_arr = df[tgt].to_numpy(allow_copy=True).astype(float)
                corrs: dict[str, float] = {}
                for c in df.columns:
                    if c in (tgt, "date"):
                        continue
                    try:
                        arr = df[c].cast(pl.Float64).to_numpy(allow_copy=True)
                        mask = ~(np.isnan(arr) | np.isnan(tgt_arr))
                        if mask.sum() > 10:
                            corrs[c] = float(np.corrcoef(arr[mask], tgt_arr[mask])[0, 1])
                    except Exception:
                        pass
                sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:25]
                if sorted_corrs:
                    c_names = [p[0] for p in sorted_corrs]
                    c_vals = [p[1] for p in sorted_corrs]
                    c_colors = [
                        _group_meta(feat_to_group.get(n) or _infer_feature_group(n, target_col))["color"]
                        for n in c_names
                    ]
                    c_labels = [
                        f"{_group_meta(feat_to_group.get(n) or _infer_feature_group(n, target_col))['icon']} {n}"
                        for n in c_names
                    ]
                    fig_corr = go.Figure(go.Bar(
                        x=c_vals, y=c_labels, orientation="h",
                        marker_color=c_colors,
                        text=[f"{v:+.3f}" for v in c_vals], textposition="outside",
                        hovertemplate="<b>%{y}</b><br>Pearson r = %{x:+.4f}<extra></extra>",
                    ))
                    fig_corr.update_layout(
                        template="plotly_white",
                        height=max(380, len(c_names) * 28 + 80),
                        xaxis_title=f"Pearson r  with  {tgt}",
                        xaxis=dict(range=[-1.08, 1.08]),
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=20, r=70, t=20, b=20),
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    active_groups = _sorted_groups(
                        {g: [] for g in feat_to_group.values()}
                    )
                    st.caption(
                        "Colour = feature group: "
                        + "  ".join(
                            f'<span style="color:{_group_meta(g)["color"]};font-weight:bold">'
                            f'{_group_meta(g)["icon"]} {_group_meta(g)["label"]}</span>'
                            for g in active_groups
                        ),
                        unsafe_allow_html=True,
                    )

            # Data preview
            st.subheader("🗃️ Features Data Preview")
            hi = holdout_start_idx
            if isinstance(hi, int) and 0 < hi < len(df):
                split_caption = (
                    f"📦 **{len(df):,} rows × {df.width} columns** | "
                    f"Train: rows 0–{hi - 1} | Holdout: rows {hi}–{len(df) - 1}"
                )
            else:
                split_caption = f"📦 **{len(df):,} rows × {df.width} columns**"
            st.caption(split_caption)
            non_meta_cols = [c for c in df.columns if c not in ("date", tgt or "")]
            preview_cols = st.multiselect(
                "Select features to preview:",
                options=non_meta_cols,
                default=non_meta_cols[:min(8, len(non_meta_cols))],
                key="feat_preview_cols",
            )
            show_cols = (["date"] if "date" in df.columns else []) + ([tgt] if tgt else []) + preview_cols
            n_show = st.slider("Rows to display:", 5, 50, 10, key="feat_preview_rows")
            st.dataframe(
                df.select([c for c in show_cols if c in df.columns]).head(n_show).to_pandas(),
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Could not load features.parquet: {e}")

    # ── Excluded Features ────────────────────────────────────────────────────
    if features_excluded:
        st.divider()
        st.subheader("🗑️ Excluded Features")
        st.dataframe(
            [{"Feature": k, "Reason": v} for k, v in features_excluded.items()],
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────

def _collect_all_entries(training: dict | None, evaluation: dict | None) -> list[dict]:
    """Merge step-13 training + step-14 evaluation into one flat list.

    Step-13 schema (actual pipeline output):
      candidates: dict[model_name -> {r2, rmse, mae, cv_mean_r2, cv_std_r2, is_benchmark, type}]
      model_history: list[{model_name, status, fit_time_sec, ...}]
    Step-14 schema:
      candidates: list[{model_name, r2, rmse, mae, cv_mean_r2, cv_std_r2, mape, ...}]
    """
    entries: dict[str, dict] = {}

    if training:
        # Build a fit_time lookup from model_history (list of dicts)
        fit_times: dict[str, float] = {}
        statuses: dict[str, str] = {}
        for h in training.get("model_history", []):
            if isinstance(h, dict):
                n = h.get("model_name", "")
                if h.get("fit_time_sec") is not None:
                    fit_times[n] = h["fit_time_sec"]
                if h.get("status"):
                    statuses[n] = h["status"]

        raw_cands = training.get("candidates", {})

        # Support both dict (pipeline) and list (future/extended format)
        if isinstance(raw_cands, dict):
            items = raw_cands.items()
        else:
            # list of dicts with model_name key
            items = ((c.get("model_name", "unknown"), c) for c in raw_cands
                     if isinstance(c, dict))

        for name, c in items:
            is_bm = c.get("is_benchmark", False) or c.get("type") == "benchmark"
            tier = "benchmark" if is_bm else str(c.get("type", c.get("tier", "?")))
            entries[name] = {
                "model_name": name,
                "tier": tier,
                # step-13 uses r2/rmse/mae directly (already holdout metrics)
                "r2": c.get("r2") or c.get("holdout_r2"),
                "rmse": c.get("rmse") or c.get("holdout_rmse"),
                "mae": c.get("mae") or c.get("holdout_mae"),
                # step-13 uses cv_mean_r2 / cv_std_r2
                "cv_r2_mean": c.get("cv_mean_r2") or c.get("cv_r2_mean"),
                "cv_r2_std": c.get("cv_std_r2") or c.get("cv_r2_std"),
                "fit_time_sec": fit_times.get(name),
                "status": statuses.get(name, "ok"),
            }

    # Overlay/supplement with step-14 evaluation (always a list of dicts)
    if evaluation:
        for c in evaluation.get("candidates", []):
            if not isinstance(c, dict):
                continue
            name = c.get("model_name", "unknown")
            if name not in entries:
                entries[name] = {"model_name": name, "tier": "?"}
            entries[name].update({
                "r2": c.get("r2") or entries[name].get("r2"),
                "rmse": c.get("rmse") or entries[name].get("rmse"),
                "mae": c.get("mae") or entries[name].get("mae"),
                "cv_r2_mean": c.get("cv_mean_r2") or entries[name].get("cv_r2_mean"),
                "cv_r2_std": c.get("cv_std_r2") or entries[name].get("cv_r2_std"),
                "model_worse_than_mean": c.get("model_worse_than_mean_baseline"),
            })

    return list(entries.values())


def _render_model_comparison_tab(output_dir: Path) -> None:
    training = _read_json(output_dir / "step-13-training.json")
    evaluation = _read_json(output_dir / "step-14-evaluation.json")
    selection = _read_json(output_dir / "step-15-selection.json")

    if not training and not evaluation:
        st.info("Model training/evaluation data not yet available.")
        return

    all_entries = _collect_all_entries(training, evaluation)
    best_name = (selection or {}).get("selected_model") or \
                (training or {}).get("best_model_name") or \
                (training or {}).get("best_model")

    # ── Quality badge ────────────────────────────────────────────────────────
    qa = (evaluation or {}).get("quality_assessment", "unknown")
    badge_map = {"acceptable": "success", "marginal": "warning",
                 "subpar": "error", "subpar_after_expansion": "error",
                 "leakage_suspected": "error"}
    getattr(st, badge_map.get(qa, "info"))(
        f"**Quality assessment: {qa.upper()}**")
    if (evaluation or {}).get("expansion_diagnosis"):
        with st.expander("Expansion Diagnosis"):
            st.write(evaluation["expansion_diagnosis"])

    # ── Filter ───────────────────────────────────────────────────────────────
    all_names = [e["model_name"] for e in all_entries]
    visible = st.multiselect("Show / hide models:", all_names, default=all_names,
                              key="model_filter")
    filtered = [e for e in all_entries if e["model_name"] in visible]

    if not filtered:
        st.warning("No models selected.")
        return

    # ── R² bar chart ─────────────────────────────────────────────────────────
    st.subheader("📊 R² on Holdout")
    sorted_f = sorted(filtered, key=lambda x: (x.get("r2") or -999))
    names = [e["model_name"] for e in sorted_f]
    r2_vals = [e.get("r2") or 0.0 for e in sorted_f]
    colors = [_bar_color(n, best_name or "") for n in names]

    fig_r2 = go.Figure(go.Bar(
        x=r2_vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in r2_vals], textposition="outside",
    ))
    fig_r2.update_layout(
        title="Holdout R² (green = best candidate, orange = benchmarks)",
        xaxis_title="R²", template="plotly_white",
        height=max(300, len(names) * 30),
        xaxis=dict(range=[min(0.0, min(r2_vals) - 0.05),
                          min(1.05, max(r2_vals) + 0.12)]),
    )
    st.plotly_chart(fig_r2, use_container_width=True)

    # ── Error metrics ────────────────────────────────────────────────────────
    st.subheader("📉 RMSE & MAE on Holdout")
    rmse_vals = [e.get("rmse") or 0.0 for e in sorted_f]
    mae_vals = [e.get("mae") or 0.0 for e in sorted_f]

    fig_err = go.Figure()
    fig_err.add_trace(go.Bar(y=names, x=rmse_vals, name="RMSE",
                              orientation="h", marker_color="rgb(255,127,14)"))
    fig_err.add_trace(go.Bar(y=names, x=mae_vals, name="MAE",
                              orientation="h", marker_color="rgb(44,160,44)"))
    fig_err.update_layout(barmode="group", template="plotly_white",
                           title="Error Metrics (lower = better)",
                           xaxis_title="Error", height=max(300, len(names) * 30))
    st.plotly_chart(fig_err, use_container_width=True)

    # ── CV Stability ─────────────────────────────────────────────────────────
    cv_entries = [e for e in filtered if e.get("cv_r2_mean") is not None]
    if cv_entries:
        st.subheader("📈 CV Stability (mean ± std R²)")
        cv_names = [e["model_name"] for e in cv_entries]
        cv_means = [e["cv_r2_mean"] for e in cv_entries]
        cv_stds = [e.get("cv_r2_std") or 0.0 for e in cv_entries]

        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(x=cv_names, y=cv_means,
                                 error_y=dict(type="data", array=cv_stds, visible=True),
                                 marker_color="steelblue", name="CV R²"))
        fig_cv.update_layout(title="Cross-Validation R² (error bars = std dev)",
                              yaxis_title="CV R²", template="plotly_white", height=350)
        st.plotly_chart(fig_cv, use_container_width=True)

    # ── Full ranking table ───────────────────────────────────────────────────
    st.subheader("📋 Full Ranking Table")
    if selection and selection.get("full_ranking"):
        ranking = selection["full_ranking"]
        # Only show items in the current visible filter
        visible_ranking = [r for r in ranking if r.get("model_name") in visible]
        if visible_ranking:
            st.dataframe(visible_ranking, use_container_width=True)
    else:
        # Build our own from collected entries
        table_rows = []
        for e in sorted(filtered, key=lambda x: (x.get("r2") or -999), reverse=True):
            table_rows.append({
                "Model": e["model_name"],
                "Tier": e.get("tier", "?"),
                "Holdout R²": f"{e['r2']:.4f}" if e.get("r2") is not None else "—",
                "RMSE": f"{e['rmse']:.4f}" if e.get("rmse") is not None else "—",
                "MAE": f"{e['mae']:.4f}" if e.get("mae") is not None else "—",
                "CV R² mean": f"{e['cv_r2_mean']:.4f}" if e.get("cv_r2_mean") is not None else "—",
                "Fit time (s)": f"{e['fit_time_sec']:.1f}" if e.get("fit_time_sec") else "—",
                "Status": e.get("status", "?"),
            })
        st.dataframe(table_rows, use_container_width=True)

    # ── Benchmark warning ────────────────────────────────────────────────────
    if (training or {}).get("benchmark_warning"):
        st.warning("⚠️ Best model does NOT beat all benchmarks by ≥ 0.02 R². "
                   "Consider running more candidates.")
    else:
        st.success("✅ Best model beats benchmarks by a meaningful margin.")

    # ── Skipped models ───────────────────────────────────────────────────────
    skipped = (training or {}).get("skipped_models", [])
    if skipped:
        with st.expander(f"Skipped models ({len(skipped)})", expanded=False):
            for s in skipped:
                if isinstance(s, dict):
                    st.write(f"- **{s.get('name', '?')}**: {s.get('reason', '?')}")
                else:
                    st.write(f"- **{s}**")

    # ── Interactive Forecast Comparison ──────────────────────────────────────
    _render_forecast_comparison(output_dir, visible_models=visible, best_name=best_name or "")


# ─────────────────────────────────────────────────────────────────────────────
# Forecast Comparison Plot (shared helper)
# ─────────────────────────────────────────────────────────────────────────────

def _render_forecast_comparison(
    output_dir: Path,
    visible_models: list[str] | None = None,
    best_name: str = "",
) -> None:
    """
    Interactive holdout forecast overlay: actual vs every model's predictions.
    Reads forecast_comparison.npz written by step 13.
    Falls back to loading each candidate-*.joblib + holdout.npz if npz is absent.
    """
    st.subheader("📊 Interactive Holdout Forecast Comparison")

    npz_path = output_dir / "forecast_comparison.npz"
    holdout_path = output_dir / "holdout.npz"

    preds: dict[str, np.ndarray] = {}
    y_test: np.ndarray | None = None

    # Primary path: pre-computed forecast_comparison.npz from step 13
    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=False)
            y_test = data["y_test"].astype(float)
            for key in data.files:
                if key != "y_test":
                    preds[key] = data[key].astype(float)
        except Exception as e:
            st.warning(f"Could not load forecast_comparison.npz: {e}")

    # Fallback: load each candidate-*.joblib and run .predict on holdout X_test
    if y_test is None and holdout_path.exists():
        try:
            hdata = np.load(holdout_path, allow_pickle=False)
            X_test = hdata["X_test"]
            y_test = hdata["y_test"].astype(float)
            for jlib in sorted(output_dir.glob("candidate-*.joblib")):
                name = jlib.stem.replace("candidate-", "")
                try:
                    mdl = joblib.load(jlib)
                    y_pred = mdl.predict(X_test)
                    preds[name] = np.asarray(y_pred, dtype=float)
                except Exception:
                    pass
        except Exception as e:
            st.warning(f"Could not load holdout.npz for fallback: {e}")

    if y_test is None or not preds:
        st.info("Forecast comparison data not yet available. "
                "Run step 13 to generate forecast_comparison.npz.")
        return

    # Filter to visible models if filter is active
    if visible_models:
        preds = {k: v for k, v in preds.items() if k in visible_models}

    if not preds:
        st.warning("No model predictions available for the selected filter.")
        return

    # Model selector for the plot (independent of the metric filter above)
    all_model_names = sorted(preds.keys())
    
    # Identify best benchmark (if any benchmarks exist)
    benchmarks = [n for n in all_model_names if n in _BENCHMARK_NAMES]
    best_benchmark = None
    if benchmarks:
        # Get evaluation data to find best benchmark by R²
        eval_data = _read_json(output_dir / "step-14-evaluation.json") or {}
        best_r2 = -999
        for b in benchmarks:
            for cand in eval_data.get("candidates", []):
                if cand.get("model_name") == b and (cand.get("r2") or -999) > best_r2:
                    best_r2 = cand.get("r2")
                    best_benchmark = b
    
    # Default selection: best model + best benchmark if available
    default_sel = [best_name] if best_name else []
    if best_benchmark and best_benchmark not in default_sel:
        default_sel.append(best_benchmark)
    if not default_sel:
        default_sel = all_model_names  # fallback to all if none selected
    
    selected_for_plot = st.multiselect(
        "Models to overlay on forecast plot:",
        options=all_model_names,
        default=default_sel,
        key="forecast_comparison_select",
    )

    # Zoom / index range
    n = len(y_test)
    c1, c2 = st.columns(2)
    zoom_start = c1.number_input("Plot from index", min_value=0, max_value=n - 1,
                                  value=0, step=max(1, n // 20),
                                  key="fc_zoom_start")
    zoom_end = c2.number_input("Plot to index", min_value=1, max_value=n,
                                value=n, step=max(1, n // 20),
                                key="fc_zoom_end")
    zoom_start = int(zoom_start)
    zoom_end = int(zoom_end)
    if zoom_end <= zoom_start:
        zoom_end = min(zoom_start + 100, n)

    x_idx = np.arange(zoom_start, zoom_end)
    y_actual = y_test[zoom_start:zoom_end]

    fig = go.Figure()

    # Actual (always shown, thick black)
    fig.add_trace(go.Scatter(
        x=x_idx, y=y_actual,
        mode="lines", name="Actual",
        line=dict(color="black", width=2.5),
    ))

    # Colour palette for models
    _PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]
    _DASH_STYLES = ["solid", "dot", "dash", "longdash", "dashdot"]

    for i, name in enumerate(selected_for_plot):
        if name not in preds:
            continue
        y_model = preds[name][zoom_start:zoom_end]
        is_best = name == best_name
        is_best_benchmark = name == best_benchmark
        is_benchmark = name in _BENCHMARK_NAMES
        
        # Styling: best model (green, solid), best benchmark (orange, dashed), others
        if is_best:
            color = "#00b450"
            width = 2.5
            dash = "solid"
            label = f"★ {name}"
        elif is_best_benchmark:
            color = "#ff7f0e"
            width = 2.5
            dash = "dash"
            label = f"🏆 {name}"
        elif is_benchmark:
            color = _BENCHMARK_COLOR
            width = 1.5
            dash = _DASH_STYLES[(i + 1) % len(_DASH_STYLES)]
            label = name
        else:
            color = _PALETTE[i % len(_PALETTE)]
            width = 1.5
            dash = _DASH_STYLES[(i + 1) % len(_DASH_STYLES)]
            label = name

        fig.add_trace(go.Scatter(
            x=x_idx, y=y_model,
            mode="lines", name=label,
            line=dict(color=color, width=width, dash=dash),
        ))

    fig.update_layout(
        title=f"Holdout Forecast Comparison (indices {zoom_start}–{zoom_end})",
        xaxis_title="Holdout Index",
        yaxis_title="Target Value",
        template="plotly_white",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residual overlay (absolute error per model)
    if st.checkbox("Show absolute error per model", key="fc_show_error"):
        fig_err = go.Figure()
        for i, name in enumerate(selected_for_plot):
            if name not in preds:
                continue
            abs_err = np.abs(y_actual - preds[name][zoom_start:zoom_end])
            color = _PALETTE[i % len(_PALETTE)]
            fig_err.add_trace(go.Scatter(
                x=x_idx, y=abs_err,
                mode="lines", name=name,
                line=dict(color=color, width=1.2),
            ))
        fig_err.update_layout(
            title="Absolute Error per Model",
            xaxis_title="Holdout Index", yaxis_title="|Error|",
            template="plotly_white", height=320, hovermode="x unified",
        )
        st.plotly_chart(fig_err, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Best Model
# ─────────────────────────────────────────────────────────────────────────────

def _render_best_model_tab(output_dir: Path) -> None:
    selection = _read_json(output_dir / "step-15-selection.json")
    evaluation = _read_json(output_dir / "step-14-evaluation.json")
    feat_info = _read_json(output_dir / "step-12-features.json")

    if not selection:
        st.info("Model selection data not yet available.")
        return

    best_name = selection.get("selected_model")
    quality_flag = selection.get("quality_flag", "unknown")

    if not best_name:
        st.error(f"No viable candidate selected. Quality flag: {quality_flag}")
        if selection.get("rationale"):
            st.write(selection["rationale"])
        return

    st.success(f"**Selected model: `{best_name}`**  |  Quality: `{quality_flag}`")

    rationale = selection.get("rationale")
    if rationale:
        st.info(rationale)

    # ── Key metrics ──────────────────────────────────────────────────────────
    st.subheader("📊 Performance Metrics")
    best_eval = None
    if evaluation:
        candidates = evaluation.get("candidates", {})
        if isinstance(candidates, dict):
            best_eval = candidates.get(best_name)
        else:
            best_eval = next(
                (c for c in candidates
                 if c.get("model_name") == best_name),
                None,
            )

    if best_eval:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Holdout R²", f"{best_eval.get('r2', 0):.4f}")
        c2.metric("RMSE", f"{best_eval.get('rmse', 0):.4f}")
        c3.metric("MAE", f"{best_eval.get('mae', 0):.4f}")
        cv_mean = best_eval.get("cv_mean_r2")
        cv_std = best_eval.get("cv_std_r2")
        c4.metric("CV R²", f"{cv_mean:.4f}" if cv_mean is not None else "N/A",
                  delta=f"±{cv_std:.4f}" if cv_std else None, delta_color="off")
        mape = best_eval.get("mape")
        c5.metric("MAPE", f"{mape:.2f}%" if mape else "N/A")

        target_stats = evaluation.get("target_stats", {})
        if target_stats:
            st.caption(
                f"Target stats — mean: {target_stats.get('mean', '?'):.2f}  "
                f"std: {target_stats.get('std', '?'):.2f}  "
                f"min: {target_stats.get('min', '?')}  "
                f"max: {target_stats.get('max', '?')}"
            )

    # ── Residual analysis from holdout.npz ──────────────────────────────────
    st.subheader("🔬 Residual Analysis")
    holdout_path = output_dir / "holdout.npz"
    model_path = output_dir / "model.joblib"

    if holdout_path.exists() and model_path.exists():
        try:
            data = np.load(holdout_path)
            X_test = data.get("X_test")
            y_test = data.get("y_test")
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)

            residuals = y_test - y_pred
            c1, c2 = st.columns(2)

            with c1:
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=y_test, y=y_pred, mode="markers",
                    marker=dict(size=5, color="rgba(26,118,255,0.5)"),
                    name="Predictions",
                ))
                min_v = float(min(y_test.min(), y_pred.min()))
                max_v = float(max(y_test.max(), y_pred.max()))
                fig_scatter.add_trace(go.Scatter(
                    x=[min_v, max_v], y=[min_v, max_v],
                    mode="lines", name="Perfect", line=dict(color="red", dash="dash"),
                ))
                fig_scatter.update_layout(title="Actual vs Predicted",
                                           xaxis_title="Actual", yaxis_title="Predicted",
                                           template="plotly_white", height=360)
                st.plotly_chart(fig_scatter, use_container_width=True)

            with c2:
                fig_resid = go.Figure()
                fig_resid.add_trace(go.Scatter(
                    x=y_pred, y=residuals, mode="markers",
                    marker=dict(size=5, color=residuals,
                                colorscale="RdBu", showscale=True),
                ))
                fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                fig_resid.update_layout(title="Residuals vs Predicted",
                                         xaxis_title="Predicted", yaxis_title="Residuals",
                                         template="plotly_white", height=360)
                st.plotly_chart(fig_resid, use_container_width=True)

            # Residual distribution
            fig_hist = go.Figure(go.Histogram(x=residuals, nbinsx=40,
                                               marker_color="rgba(44,160,44,0.7)"))
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
            fig_hist.update_layout(title="Residual Distribution",
                                    xaxis_title="Residual", template="plotly_white", height=280)
            st.plotly_chart(fig_hist, use_container_width=True)

            # Time-ordered forecast plot
            fig_ts = go.Figure()
            x_idx = np.arange(len(y_test))
            fig_ts.add_trace(go.Scatter(x=x_idx, y=y_test, mode="lines",
                                         name="Actual", line=dict(color="blue", width=1.5)))
            fig_ts.add_trace(go.Scatter(x=x_idx, y=y_pred, mode="lines",
                                         name="Predicted",
                                         line=dict(color="red", width=1.5, dash="dot")))
            fig_ts.update_layout(title="Holdout Forecast (chronological)",
                                  xaxis_title="Holdout Index", yaxis_title="Value",
                                  template="plotly_white", height=320, hovermode="x unified")
            st.plotly_chart(fig_ts, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not generate residual plots: {e}")

    # ── SHAP ────────────────────────────────────────────────────────────────
    st.subheader("🎯 SHAP Feature Importance")
    _render_shap_section(output_dir, feat_info)

    # ── Candidate Analysis ───────────────────────────────────────────────────
    cand_analysis = selection.get("candidate_analysis")
    if cand_analysis:
        st.subheader("🔍 Candidate Analysis")
        if isinstance(cand_analysis, dict):
            for model_name, analysis in cand_analysis.items():
                with st.expander(f"{model_name}"):
                    st.write(analysis)
        elif isinstance(cand_analysis, str):
            st.write(cand_analysis)


def _render_shap_section(output_dir: Path, feat_info: dict | None) -> None:
    shap_path = output_dir / "shap_values.npz"
    eval_data = _read_json(output_dir / "step-14-evaluation.json") or {}
    shap_meta = eval_data.get("shap_artifacts", {})

    status = shap_meta.get("status", "unknown")

    if status == "skipped":
        st.info(f"SHAP not computed: {shap_meta.get('shap_skipped_reason', 'model type not supported')}")
        return
    if status == "failed":
        st.warning(f"SHAP computation failed: {shap_meta.get('shap_error', 'unknown error')}")
        return
    if not shap_path.exists():
        st.info("SHAP values not yet available.")
        return

    try:
        shap_data = np.load(shap_path, allow_pickle=True)
        shap_values = shap_data["shap_values"].astype(float)       # (n, f)
        base_values = shap_data.get("base_values")
        expected_val = shap_data.get("expected_value")
        X_sample = shap_data.get("X_test_sample")

        # Feature names: prefer from npz, fallback to step-12 JSON
        if "feature_names" in shap_data:
            feature_names = list(shap_data["feature_names"].astype(str))
        elif feat_info:
            feature_names = feat_info.get("feature_names", [])
            if not feature_names:
                feature_names = [f"f{i}" for i in range(shap_values.shape[1])]
        else:
            feature_names = [f"f{i}" for i in range(shap_values.shape[1])]

        n_features = shap_values.shape[1]
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_k = min(20, n_features)
        top_idx = np.argsort(mean_abs_shap)[::-1][:top_k]
        top_names = [feature_names[i] if i < len(feature_names) else f"f{i}"
                     for i in top_idx]
        top_vals = mean_abs_shap[top_idx]

        n_samples = shap_values.shape[0]
        explainer_type = shap_meta.get("explainer_type", "unknown")
        c1, c2, c3 = st.columns(3)
        c1.metric("Explainer", explainer_type)
        c2.metric("Samples", n_samples)
        if expected_val is not None:
            ev = float(expected_val) if not hasattr(expected_val, "__iter__") else float(expected_val.flat[0])
            c3.metric("Expected value (baseline)", f"{ev:.4f}")

        # SHAP bar chart (mean |SHAP|)
        fig_shap = go.Figure(go.Bar(
            x=top_vals[::-1], y=top_names[::-1],
            orientation="h",
            marker_color="steelblue",
        ))
        fig_shap.update_layout(
            title=f"Top {top_k} Features — Mean |SHAP| Value",
            xaxis_title="Mean |SHAP|", template="plotly_white",
            height=max(300, top_k * 20),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # SHAP beeswarm approximation: scatter of SHAP values for top 10 features
        top10_idx = top_idx[:10]
        top10_names = [feature_names[i] if i < len(feature_names) else f"f{i}"
                       for i in top10_idx]

        if X_sample is not None and X_sample.shape[1] == n_features:
            st.markdown("**SHAP Value Distribution (top 10 features)**")
            with st.expander("Beeswarm-style SHAP plot", expanded=True):
                fig_bee = go.Figure()
                for rank, (feat_idx, feat_name) in enumerate(zip(top10_idx, top10_names)):
                    shap_col = shap_values[:, feat_idx]
                    feat_col = X_sample[:, feat_idx]
                    # Normalise feature value for color (0→1)
                    f_min, f_max = feat_col.min(), feat_col.max()
                    feat_norm = (feat_col - f_min) / (f_max - f_min + 1e-9)
                    fig_bee.add_trace(go.Scatter(
                        x=shap_col,
                        y=[feat_name] * len(shap_col),
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=feat_norm,
                            colorscale="RdBu",
                            showscale=(rank == 0),
                            colorbar=dict(title="Feature value\n(low→high)", len=0.5)
                            if rank == 0 else None,
                            opacity=0.6,
                        ),
                        showlegend=False,
                    ))
                fig_bee.add_vline(x=0, line_dash="dash", line_color="black")
                fig_bee.update_layout(
                    title="SHAP Values by Feature (color = feature magnitude)",
                    xaxis_title="SHAP value", template="plotly_white",
                    height=max(350, len(top10_names) * 35),
                )
                st.plotly_chart(fig_bee, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not render SHAP plots: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Report
# ─────────────────────────────────────────────────────────────────────────────

def _render_report_tab(output_dir: Path) -> None:
    report_path = output_dir / "step-16-report.md"
    model_path = output_dir / "model.joblib"

    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8", errors="replace")
        c1, c2 = st.columns([5, 1])
        with c2:
            st.download_button("⬇️ Download Report", report_text,
                               file_name="forecast_report.md", mime="text/markdown")
        if model_path.exists():
            with open(model_path, "rb") as f:
                model_bytes = f.read()
            c2.download_button("⬇️ Download Model", model_bytes,
                               file_name="model.joblib",
                               mime="application/octet-stream")
        with c1:
            st.markdown(report_text)
    else:
        st.info("Report not yet generated.")

    # Artifact index
    st.subheader("📁 Run Artifacts")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            st.caption(f"`{p.name}` — {size_kb:.1f} KB")


def _render_audit_tab(output_dir: Path) -> None:
    """Render Step 17 Critical Self-Audit results."""
    st.subheader("🔐 Critical Self-Audit Results")
    
    audit_results = _parse_audit_results(output_dir)
    if not audit_results:
        st.info("Audit results not yet available (step 17 pending).")
        return
    
    # Overall result banner
    overall = audit_results.get("overall_audit_result", "unknown")
    if overall == "pass":
        st.success(f"✅ **Audit Passed** | Confidence: {audit_results.get('audit_confidence', 'N/A')}")
    elif overall == "fail":
        st.error(f"❌ **Audit Failed** | Remediation Required")
    else:
        st.info(f"⚠️ **Audit Status:** {overall}")
    
    # Audit confidence if available
    if "audit_confidence" in audit_results:
        st.metric("Audit Confidence", f"{audit_results['audit_confidence']:.1%}")
    
    st.markdown("---")
    
    # Check results
    st.subheader("📊 Check Results")
    checks = audit_results.get("checks", {})
    
    if checks:
        # Create columns for check summary
        check_stats = {"pass": 0, "marginal": 0, "fail": 0}
        for check_result in checks.values():
            status = check_result.get("status", "unknown")
            if status in check_stats:
                check_stats[status] += 1
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Checks", len(checks))
        col2.metric("✅ Pass", check_stats["pass"])
        col3.metric("🟡 Marginal", check_stats["marginal"])
        col4.metric("🔴 Fail", check_stats["fail"])
        
        st.markdown("---")
        
        # Detailed check results
        for check_name, check_result in checks.items():
            status = check_result.get("status", "unknown")
            
            if status == "pass":
                emoji = "✅"
                color = "green"
            elif status == "marginal":
                emoji = "🟡"
                color = "orange"
            else:
                emoji = "🔴"
                color = "red"
            
            findings = check_result.get("findings", [])
            severity = check_result.get("severity", "low")
            
            with st.expander(f"{emoji} **{check_name.replace('_', ' ').title()}** [{status.upper()}]"):
                # Severity badge
                if severity == "high":
                    st.error(f"**Severity:** {severity.upper()}")
                elif severity == "medium":
                    st.warning(f"**Severity:** {severity.upper()}")
                else:
                    st.info(f"**Severity:** {severity.upper()}")
                
                # Findings
                if findings:
                    st.write("**Findings:**")
                    for finding in findings:
                        st.caption(f"• {finding}")
                
                # Full check details
                with st.expander("📋 Full Details"):
                    st.json(check_result)
    else:
        st.info("No check results available.")
    
    st.markdown("---")
    
    # Remediation actions
    if overall == "fail":
        st.subheader("🔧 Remediation Actions Required")
        remediation_actions = audit_results.get("remediation_actions", [])
        
        if remediation_actions:
            auto_count = sum(1 for a in remediation_actions if a.get("type") == "[AUTO]")
            manual_count = len(remediation_actions) - auto_count
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Actions", len(remediation_actions))
            col2.metric("🟢 Auto", auto_count)
            col3.metric("🔴 Manual", manual_count)

            # ── Remediation trigger button ────────────────────────────────────
            st.markdown("---")
            if auto_count > 0:
                orchestrator_exists = (output_dir / "code" / "orchestrator.py").exists()
                if orchestrator_exists:
                    st.info(
                        f"**{auto_count} AUTO action(s) detected.** "
                        "The orchestrator will re-run the affected steps automatically with corrected parameters."
                    )
                    if st.button("🔄 Trigger Auto-Remediation & Restart Pipeline",
                                 type="primary", use_container_width=True,
                                 key="btn_auto_remediate"):
                        with st.spinner("Deleting affected artifacts and restarting pipeline…"):
                            launched = _trigger_auto_remediation(output_dir)
                        if launched:
                            st.success(
                                "✅ Remediation pipeline started! "
                                "Switch to the Live Progress tab or refresh to monitor."
                            )
                        # Page will reload on next interaction and show new progress
                else:
                    st.warning(
                        "orchestrator.py not found in this run's code/ directory. "
                        "Cannot trigger auto-remediation from the UI. "
                        "Re-run the pipeline via the agent to generate the orchestrator."
                    )
            if manual_count > 0:
                st.warning(
                    f"**{manual_count} MANUAL action(s) require human review.** "
                    "See details below. After addressing them, re-run the pipeline with the corrected parameters."
                )

            st.markdown("---")
            
            for i, action in enumerate(remediation_actions, 1):
                action_id = action.get("action_id", "unknown")
                action_type = action.get("type", "UNKNOWN")
                severity = action.get("severity", "medium")
                description = action.get("description", _REMEDIATION_DESCRIPTIONS.get(action_id, ""))
                
                # Action badge
                if action_type == "[AUTO]":
                    emoji = "🟢"
                else:
                    emoji = "🔴"
                
                with st.expander(f"{emoji} **{i}. {action_id}** ({action_type})"):
                    st.write(description)
                    st.caption(f"**Severity:** {severity.upper()}")
                    
                    # Affected steps
                    affected = _REMEDIATION_STEPS_MAP.get(action_id, [])
                    if affected:
                        st.write(f"**Affected Steps:** {', '.join(map(str, affected))}")
                    
                    # Full action details
                    with st.expander("📋 Full Details"):
                        st.json(action)
        else:
            st.info("No remediation actions required.")

        # Show remediation_required.json if orchestrator wrote it
        req_file = output_dir / "remediation_required.json"
        if req_file.exists():
            st.markdown("---")
            st.subheader("📋 Remediation Required (written by orchestrator)")
            req_data = _read_json(req_file)
            if req_data:
                iterations = req_data.get("remediation_iterations_attempted", 0)
                final_result = req_data.get("final_audit_result", "unknown")
                st.error(
                    f"Orchestrator exhausted {iterations} remediation attempt(s). "
                    f"Final audit result: **{final_result.upper()}**. "
                    "Manual intervention is required."
                )
                pending = req_data.get("pending_manual_actions", [])
                for action in pending:
                    with st.expander(f"🔴 {action.get('action_id', 'unknown')}"):
                        st.write(action.get("description", ""))
                        st.json(action)
                with st.expander("📄 Full remediation_required.json"):
                    st.json(req_data)
    
    st.markdown("---")
    
    # Critical findings
    critical_findings = audit_results.get("critical_findings", [])
    if critical_findings:
        st.subheader("⚠️ Critical Findings")
        for finding in critical_findings:
            check = finding.get("check", "unknown")
            desc = finding.get("description", "")
            severity = finding.get("severity", "high")
            
            st.error(f"**{check}** ({severity.upper()}): {desc}")


def _render_judge_tab(output_dir: Path) -> None:
    """Render Step 18 LLM-as-a-Judge results."""
    judge_json_path = output_dir / "step-18-judge.json"
    judge_md_path = output_dir / "step-18-judge.md"

    if not judge_json_path.exists() and not judge_md_path.exists():
        st.info("Judge report not yet available (step 18 pending).")
        return

    judge = _read_json(judge_json_path) if judge_json_path.exists() else {}

    if judge:
        st.markdown(
            _html_block("""
            <style>
            .judge-grid {display:grid; gap:12px; grid-template-columns:repeat(12,minmax(0,1fr));}
            .judge-card {
                border:1px solid rgba(148,163,184,.22);
                background:linear-gradient(180deg,rgba(30,41,59,.64),rgba(15,23,42,.72));
                border-radius:8px;
                padding:18px;
                min-height:100%;
                box-shadow:0 8px 26px rgba(0,0,0,.18);
            }
            .judge-span-3 {grid-column:span 3;}
            .judge-span-4 {grid-column:span 4;}
            .judge-span-5 {grid-column:span 5;}
            .judge-span-6 {grid-column:span 6;}
            .judge-span-7 {grid-column:span 7;}
            .judge-span-12 {grid-column:span 12;}
            .judge-title {font-size:18px;font-weight:700;margin:0 0 14px;color:#e5e7eb;}
            .judge-kicker {font-size:12px;text-transform:uppercase;color:#94a3b8;margin-bottom:4px;}
            .judge-status {font-size:25px;font-weight:800;color:#4ade80;margin:0 0 8px;}
            .judge-status.warn {color:#facc15;}
            .judge-status.bad {color:#f87171;}
            .judge-body {color:#e5e7eb;line-height:1.55;font-size:14px;}
            .judge-muted {color:#9ca3af;font-size:13px;line-height:1.45;}
            .judge-pill {display:inline-block;border-radius:5px;padding:3px 9px;font-weight:700;font-size:13px;}
            .judge-high {background:rgba(34,197,94,.18);color:#4ade80;}
            .judge-medium,.judge-unclear {background:rgba(250,204,21,.18);color:#facc15;}
            .judge-low {background:rgba(248,113,113,.18);color:#f87171;}
            .judge-metric-value {font-size:26px;font-weight:800;color:#60a5fa;margin:2px 0 8px;}
            .judge-list {margin:8px 0 0 18px;padding:0;color:#e5e7eb;font-size:14px;line-height:1.55;}
            .judge-source-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;}
            .judge-source {border:1px solid rgba(148,163,184,.16);border-radius:6px;padding:8px;color:#cbd5e1;background:rgba(15,23,42,.38);}
            @media (max-width: 900px) {
                .judge-grid {grid-template-columns:1fr;}
                .judge-span-3,.judge-span-4,.judge-span-5,.judge-span-6,.judge-span-7,.judge-span-12 {grid-column:span 1;}
                .judge-source-grid {grid-template-columns:1fr;}
            }
            </style>
            """),
            unsafe_allow_html=True,
        )

        valid_statuses = {
            "mvp_discussion_ready",
            "mvp_discussion_ready_with_caveats",
            "needs_validation_before_mvp_discussion",
            "not_mvp_ready",
            "no_reliable_forecast_use_case_supported",
        }
        status = str(judge.get("status", "unknown"))
        status_label = str(judge.get("status_label") or status).replace("_", " ").title()
        status_class = "bad" if status in {"not_mvp_ready", "no_reliable_forecast_use_case_supported"} else (
            "warn" if status in {"mvp_discussion_ready_with_caveats", "needs_validation_before_mvp_discussion"} else ""
        )
        recommendation = judge.get("final_recommendation", {})
        if not isinstance(recommendation, dict):
            recommendation = {}
        use_case = judge.get("use_case", {})
        if not isinstance(use_case, dict):
            use_case = {}
        ratings = judge.get("ratings", {})
        if not isinstance(ratings, dict):
            ratings = {}

        if status not in valid_statuses:
            st.error(
                "Final recommendation uses an invalid Step 18 status. "
                f"Received: `{status}`"
            )

        def _rating_item(name: str) -> dict:
            item = ratings.get(name, {})
            return item if isinstance(item, dict) else {"rating": item}

        def _pill(value: object) -> str:
            rating = str(value or "unclear").lower()
            css = rating if rating in {"high", "medium", "low", "unclear"} else "unclear"
            return f'<span class="judge-pill judge-{css}">{_html_escape(str(value).title() if value else "Unclear")}</span>'

        summary = recommendation.get("summary") or judge.get("status_reason", "")
        strongest = recommendation.get("strongest_supporting_reason", "")
        caveat = recommendation.get("main_caveat", "")
        use_case_type = str(use_case.get("type", "unclear")).replace("_", " ").title()

        st.markdown(
            _html_block(f"""
            <div class="judge-grid">
              <div class="judge-card judge-span-6">
                <div class="judge-title">Final Recommendation</div>
                <div class="judge-status {status_class}">{_html_escape(status_label)}</div>
                <div class="judge-body">{_html_escape(summary)}</div>
                <div class="judge-grid" style="margin-top:14px;">
                  <div class="judge-card judge-span-6"><div class="judge-kicker">Strongest reason</div><div class="judge-muted">{_html_escape(strongest)}</div></div>
                  <div class="judge-card judge-span-6"><div class="judge-kicker">Main caveat</div><div class="judge-muted">{_html_escape(caveat)}</div></div>
                </div>
              </div>
              <div class="judge-card judge-span-6">
                <div class="judge-title">Use Case</div>
                <div class="judge-body"><strong>{_html_escape(use_case.get("title", "Use case unclear"))}</strong></div>
                <div class="judge-muted" style="margin-top:8px;">{_html_escape(use_case.get("description", ""))}</div>
                <div class="judge-grid" style="margin-top:14px;">
                  <div class="judge-card judge-span-4"><div class="judge-kicker">Type</div><div class="judge-body">{_html_escape(use_case_type)}</div></div>
                  <div class="judge-card judge-span-5"><div class="judge-kicker">Decision context</div><div class="judge-muted">{_html_escape(use_case.get("decision_context", ""))}</div></div>
                  <div class="judge-card judge-span-3"><div class="judge-kicker">Evidence strength</div>{_pill(use_case.get("evidence_strength"))}</div>
                </div>
              </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        if ratings:
            cards = []
            for key, label in [
                ("forecastability", "Forecastability"),
                ("use_case_potential", "Use Case Potential"),
                ("business_potential", "Business Potential"),
                ("business_value_evidence", "Business Value Evidence"),
            ]:
                item = _rating_item(key)
                rating = _metric_display_value(item.get("rating"))
                headline = str(item.get("headline") or label).strip()
                explanation = str(item.get("explanation") or "").strip()
                cards.append(
                    _html_block(f"""
                    <div class="judge-card judge-span-3">
                      <div style="display:flex;justify-content:space-between;gap:10px;align-items:start;">
                        <div class="judge-title" style="font-size:16px;margin-bottom:8px;">{_html_escape(label)}</div>
                        {_pill(rating)}
                      </div>
                      <div class="judge-body"><strong>{_html_escape(headline)}</strong></div>
                      <div class="judge-muted" style="margin-top:8px;">{_html_escape(explanation)}</div>
                    </div>
                    """)
                )
            st.markdown(
                _html_block(f"""
                <div class="judge-title" style="margin-top:12px;">Assessment Scores</div>
                <div class="judge-grid">{"".join(cards)}</div>
                """),
                unsafe_allow_html=True,
            )

        metric_meaning = judge.get("metric_meaning", {})
        if isinstance(metric_meaning, dict) and metric_meaning:
            metric_cards = []
            for key, label in [
                ("r2", "R²"),
                ("rmse", "RMSE"),
                ("mae", "MAE"),
                ("baseline", "Baseline"),
                ("target_scale", "Target Scale"),
            ]:
                item = metric_meaning.get(key)
                if not item:
                    continue
                if isinstance(item, dict):
                    value = item.get("value", item.get("actual_value", item.get("metric_value")))
                    unit = item.get("unit")
                    if value is None and "available" in item:
                        value_text = "Available" if item.get("available") else "Not available"
                    elif value is not None:
                        value_text = str(value)
                        if unit:
                            value_text = f"{value_text} {unit}"
                    else:
                        value_text = ""
                    meaning = item.get("meaning", "")
                    implication = item.get("relation_to_use_case") or item.get("explanation") or ""
                    metric_cards.append(
                        _html_block(f"""
                        <div class="judge-card judge-span-3">
                          <div class="judge-kicker">{_html_escape(label)}</div>
                          <div class="judge-metric-value">{_html_escape(value_text)}</div>
                          <div class="judge-body">{_html_escape(meaning)}</div>
                          {f'<div class="judge-muted" style="margin-top:8px;">{_html_escape(implication)}</div>' if implication else ''}
                        </div>
                        """)
                    )
                else:
                    metric_cards.append(
                        f'<div class="judge-card judge-span-3"><div class="judge-kicker">{_html_escape(label)}</div><div class="judge-body">{_html_escape(item)}</div></div>'
                    )
            st.markdown(
                _html_block(f"""
                <div class="judge-title" style="margin-top:12px;">Metric Meaning for This Use Case</div>
                <div class="judge-grid">{"".join(metric_cards)}</div>
                """),
                unsafe_allow_html=True,
            )

        business = judge.get("business_potential_and_evidence", {})
        if isinstance(business, dict) and business:
            points = business.get("supported_discussion_points", [])
            limits = business.get("evidence_limits", [])
            points_html = "".join(f"<li>{_html_escape(point)}</li>" for point in points) or "<li>No supported discussion points documented.</li>"
            limits_html = "".join(f"<li>{_html_escape(limit)}</li>" for limit in limits) or "<li>No evidence limits documented.</li>"
            st.markdown(
                _html_block(f"""
                <div class="judge-grid" style="margin-top:12px;">
                  <div class="judge-card judge-span-7">
                    <div class="judge-title">Business Potential and Evidence</div>
                    <div class="judge-kicker">Supported discussion points</div>
                    <ul class="judge-list">{points_html}</ul>
                  </div>
                  <div class="judge-card judge-span-5">
                    <div class="judge-title">Evidence limits</div>
                    <ul class="judge-list">{limits_html}</ul>
                  </div>
                </div>
                """),
                unsafe_allow_html=True,
            )

        risks = judge.get("risks_and_caveats", [])
        sources = judge.get("sources", [])

        if risks:
            with st.expander("Risks and caveats", expanded=False):
                if risks:
                    for risk in risks:
                        st.markdown(f"- {risk}")
                else:
                    st.caption("No risks documented.")

        if sources:
            sources_html = "".join(f'<div class="judge-source">{_html_escape(source)}</div>' for source in sources)
            st.markdown(
                _html_block(f"""
                <div class="judge-card" style="margin-top:12px;">
                  <div class="judge-title">Sources</div>
                  <div class="judge-source-grid">{sources_html}</div>
                </div>
                """),
                unsafe_allow_html=True,
            )

    if judge_md_path.exists():
        judge_text = judge_md_path.read_text(encoding="utf-8", errors="replace")
        st.markdown("---")
        st.download_button(
            "⬇️ Download Judge Report",
            judge_text,
            file_name="step-18-judge.md",
            mime="text/markdown",
        )
        with st.expander("Rendered Markdown Report"):
            st.markdown(judge_text)

    if judge_json_path.exists():
        with st.expander("📋 Judge JSON"):
            st.json(judge or _read_json(judge_json_path))


# ─────────────────────────────────────────────────────────────────────────────
# Launch mode handlers
# ─────────────────────────────────────────────────────────────────────────────

def _handle_cli_mode(prompt: str, output_dir: Path, model: str) -> None:
    """Copilot CLI mode: launch the agent subprocess and stream live progress."""
    st.subheader("⏱️ Pipeline Running (CLI)")
    started_at = time.monotonic()
    process = _start_pipeline_process(prompt, ROOT_DIR, model=model)
    status_ph = st.empty()

    while process.poll() is None:
        with status_ph.container():
            _render_live_status(output_dir, started_at)
        time.sleep(1.0)
    with status_ph.container():
        _render_live_status(output_dir, started_at)

    stdout, stderr = process.communicate()
    with st.expander("📝 Execution Logs"):
        st.code((stdout or "").strip() or "<no stdout>", language="text")
        if stderr:
            st.code(stderr.strip(), language="bash")

    if process.returncode != 0:
        st.error(f"❌ Pipeline failed (exit {process.returncode})")
    else:
        st.success("✅ Pipeline completed successfully!")
        # Save metadata when pipeline completes successfully
        elapsed = time.monotonic() - started_at
        _save_metadata(output_dir, model, elapsed)


def _handle_rerun(output_dir: Path, target_column: str, csv_path_override: str | None) -> None:
    """
    Re-run mode: execute an already-generated orchestrator.py directly via Python.
    No Copilot CLI or agent needed.
    """
    st.subheader("🔄 Re-running with existing scripts")
    orchestrator = output_dir / "code" / "orchestrator.py"
    if not orchestrator.exists():
        st.error(
            f"`orchestrator.py` not found at `{orchestrator}`.\n\n"
            "Generate the pipeline scripts first by running the agent in VS Code Copilot Chat."
        )
        return

    progress_data = _read_json(output_dir / "progress.json") or {}
    csv_path = csv_path_override or progress_data.get("csv_path", "")
    if not csv_path:
        st.error("No CSV path found. Supply a CSV path override in the sidebar.")
        return

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    st.caption(f"Script: `{orchestrator}`")
    st.caption(f"CSV: `{csv_path}`  |  Target: `{target_column}`  |  Run ID: `{run_id}`")

    try:
        process = subprocess.Popen(
            [
                "python", str(orchestrator),
                "--csv-path", csv_path,
                "--target-column", target_column,
                "--output-dir", str(output_dir),
                "--run-id", run_id,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except Exception as e:
        st.error(f"Failed to launch orchestrator: {e}")
        return

    started_at = time.monotonic()
    status_ph = st.empty()

    while process.poll() is None:
        with status_ph.container():
            _render_live_status(output_dir, started_at)
        time.sleep(1.0)
    with status_ph.container():
        _render_live_status(output_dir, started_at)

    stdout, stderr = process.communicate()
    with st.expander("📝 Execution Logs"):
        st.code((stdout or "").strip() or "<no stdout>", language="text")
        if stderr:
            st.code(stderr.strip(), language="bash")

    if process.returncode != 0:
        st.error(f"❌ Re-run failed (exit {process.returncode})")
    else:
        st.success("✅ Re-run completed successfully!")
        # Save metadata when pipeline completes successfully
        elapsed = time.monotonic() - started_at
        # Use 'orchestrator-rerun' as model name for re-runs
        _save_metadata(output_dir, "orchestrator-rerun", elapsed)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Time Series Forecasting Pipeline",
                       layout="wide", initial_sidebar_state="expanded")
    st.title("⏱️ Time Series Forecasting Pipeline")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")

        # Existing run browser
        existing = _existing_runs()
        run_options = ["— New run —"] + [r.name for r in existing]
        chosen_run = st.selectbox("📂 Browse existing run", run_options)

        st.markdown("---")
        st.subheader("▶️ Start New Run")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        output_dir_input = st.text_input(
            "Output directory",
            value=str(DEFAULT_RUNS_DIR / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")),
        )
        selected_model = st.selectbox(
            "Copilot model",
            options=["claude-haiku-4.5", "claude-sonnet-4.6", "gpt-5.4-mini", ],
            index=0,
        )

        target_column: str | None = None
        if uploaded is not None:
            _raw = uploaded.getvalue()
            dataframe = _read_uploaded_dataframe(_raw)
            recommended_col = _recommend_target_column(dataframe)
            st.info(f"💡 Empfohlene Zielspalte: **{recommended_col}**")
            default_idx = list(dataframe.columns).index(recommended_col) if recommended_col in dataframe.columns else 0
            target_column = st.selectbox(
                "🎯 Zielspalte auswählen",
                options=dataframe.columns,
                index=default_idx,
            )
            st.caption(f"{dataframe.shape[0]} rows × {dataframe.shape[1]} cols")

        submitted = st.button("▶️ Run Pipeline (CLI)", type="primary",
                              use_container_width=True,
                              disabled=(uploaded is None))

        # Re-run with existing scripts (no agent needed)
        active_dir_for_rerun: Path | None = None
        if chosen_run != "— New run —":
            candidate = DEFAULT_RUNS_DIR / chosen_run
            orchestrator_path = candidate / "code" / "orchestrator.py"
            if orchestrator_path.exists():
                st.markdown("---")
                rerun_target = st.text_input(
                    "Re-run target column",
                    value=(_read_json(candidate / "progress.json") or {}).get(
                        "target_column", ""),
                )
                rerun_csv = st.text_input("Re-run CSV path (optional override)", value="")
                if st.button("🔄 Re-run existing scripts", use_container_width=True):
                    active_dir_for_rerun = candidate
                    st.session_state["rerun_target"] = rerun_target
                    st.session_state["rerun_csv"] = rerun_csv or None

    # ── Determine active output_dir ───────────────────────────────────────────
    active_dir: Path | None = None
    if chosen_run != "— New run —":
        active_dir = DEFAULT_RUNS_DIR / chosen_run

    # ── Re-run with existing scripts ─────────────────────────────────────────
    if active_dir_for_rerun is not None:
        _handle_rerun(active_dir_for_rerun,
                      st.session_state.get("rerun_target", ""),
                      st.session_state.get("rerun_csv"))
        active_dir = active_dir_for_rerun

    # ── Handle new run ────────────────────────────────────────────────────────
    elif submitted:
        if uploaded is None:
            st.error("Please upload a CSV file.")
            return
        if not target_column:
            st.error("Could not determine target column.")
            return

        csv_path = _save_uploaded_csv(uploaded, DEFAULT_UPLOAD_DIR)
        active_dir = Path(output_dir_input).expanduser()
        active_dir.mkdir(parents=True, exist_ok=True)
        normalized_target = _normalize_column_name(target_column)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        prompt = _render_single_agent_prompt(
            csv_path=csv_path, target_column=normalized_target,
            output_dir=active_dir, copilot_model=selected_model,
        )

        _handle_cli_mode(prompt, active_dir, selected_model)

    # ── Show results in tabs ──────────────────────────────────────────────────
    if active_dir is None:
        st.markdown("""
## Getting Started

1. **Upload** a CSV with time-series data and select the target column, OR
2. **Browse** an existing run from the sidebar dropdown.

The pipeline runs 9 steps:
- **10** CSV cleansing & outlier detection
- **11** Deep time-series EDA (ADF, KPSS, Hurst, ACF/PACF, MI, STL)
- **12** Adaptive feature engineering (lags, Fourier, PCA factors)
- **13** Multi-tier model training (classical TS, FAAR, ML hybrids)
- **14** Evaluation + SHAP computation
- **15** Model selection with weighted ranking
- **16** Full report generation
- **17** Critical self-audit & remediation (validates results, triggers re-runs if needed)
- **18** LLM-as-a-Judge customer-facing judgement
        """)
        return

    # Check if there is any data to show
    has_data = any((active_dir / f).exists() for f in [
        "step-10-cleanse.json", "step-11-exploration.json",
        "step-13-training.json", "step-16-report.md"
    ])
    if not has_data:
        st.info(f"No pipeline output found in `{active_dir.name}` yet.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔍 EDA",
        "⚙️ Features",
        "🏋️ Model Comparison",
        "🏆 Best Model",
        "📄 Report",
        "🔐 Audit",
        "⚖️ Judge",
    ])

    with tab1:
        _render_eda_tab(active_dir)

    with tab2:
        _render_features_tab(active_dir)

    with tab3:
        _render_model_comparison_tab(active_dir)

    with tab4:
        _render_best_model_tab(active_dir)

    with tab5:
        _render_report_tab(active_dir)
    
    with tab6:
        _render_audit_tab(active_dir)

    with tab7:
        _render_judge_tab(active_dir)


if __name__ == "__main__":
    main()
