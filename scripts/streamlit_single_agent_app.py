"""
Streamlit UI for Single Agent Pipeline — Professional data scientist dashboard.

Five views:
  Tab 1 — EDA       : stationarity, Hurst, ACF/PACF, MI, outliers, seasonality
  Tab 2 — Models    : filterable comparison of all trained candidates
  Tab 3 — Best Model: SHAP, residuals, detailed metrics
  Tab 4 — Report    : full step-16-report.md
  Tab 5 — Audit     : critical self-audit results, remediation actions
"""

from __future__ import annotations

import json
import re
import subprocess
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
        "Follow exactly the contract in "
        "`@.github/agents/Single Agent Pipeline.agent.md`."
    )


def _save_uploaded_csv(uploaded_file, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / uploaded_file.name
    destination_path.write_bytes(uploaded_file.getvalue())
    return destination_path


def _start_pipeline_process(prompt: str, working_dir: Path,
                              model: str = "claude-haiku-4.5") -> subprocess.Popen[str]:
    command = ["copilot", "--allow-all-tools", "--allow-all-paths",
               "--allow-all-urls", "--no-ask-user", "--model", model]
    if "gpt" in model.lower():
        command.extend(["--reasoning-effort", "low"])
    command.extend(["-s", "-p", prompt])
    return subprocess.Popen(command, cwd=working_dir,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _copilot_cli_available() -> bool:
    """Return True when the `copilot` CLI binary is on PATH."""
    try:
        result = subprocess.run(
            ["copilot", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


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
    
    # Parse audit results if available (step 17)
    audit_results = _parse_audit_results(output_dir)
    remediation_triggered = False
    restart_step = None

    if progress:
        completed = progress.get("completed_steps", [])
        completed_set: set[str] = set()
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
# Tab 2 — Model Comparison
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
    


# ─────────────────────────────────────────────────────────────────────────────
# Launch mode handlers
# ─────────────────────────────────────────────────────────────────────────────

def _handle_vscode_mode(prompt: str, output_dir: Path, llm_model: str) -> None:
    """
    VS Code Chat mode: display the agent prompt so the user can paste it into
    VS Code Copilot Chat, then poll the output directory for live progress.
    """
    st.subheader("📋 Paste this into VS Code Copilot Chat")
    st.markdown(
        "1. Open **GitHub Copilot Chat** in VS Code (`Ctrl+Alt+I`)\n"
        "2. Select the **`Single Agent Pipeline`** agent from the agent dropdown\n"
        "   (or prefix your message with `@Single Agent Pipeline`)\n"
        "3. Copy the prompt below and send it\n"
        "4. This page will automatically update as the agent writes output files"
    )
    st.code(prompt, language="text")
    st.info(
        f"The agent will write its output to:\n`{output_dir}`\n\n"
        "This dashboard will refresh every 3 seconds once output files appear."
    )

    # Poll for progress while the user runs the agent
    st.markdown("---")
    st.subheader("⏱️ Live Progress Monitor")
    started_at = time.monotonic()
    monitor_ph = st.empty()

    # Run up to ~6 hours of polling (7200 iterations × 3 s)
    # In practice the user stops monitoring by navigating away or the run completes.
    for _ in range(7200):
        with monitor_ph.container():
            progress = _render_live_status(output_dir, started_at)
            if progress and progress.get("status") in ("completed", "failed", "error"):
                break
        time.sleep(3.0)
    
    # Save metadata when pipeline completes
    elapsed = time.monotonic() - started_at
    _save_metadata(output_dir, llm_model, elapsed)


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

    cli_available = _copilot_cli_available()

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

        # Mode selector — only show CLI option when available
        if cli_available:
            launch_mode = st.radio(
                "Launch mode",
                options=["VS Code Chat", "Copilot CLI"],
                index=0,
                help=("VS Code Chat: generates the prompt for you to paste into "
                      "VS Code Copilot Chat, then monitors progress.\n"
                      "Copilot CLI: runs the agent automatically via the CLI."),
            )
        else:
            launch_mode = "VS Code Chat"
            st.info("Copilot CLI not found — using VS Code Chat mode.")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        output_dir_input = st.text_input(
            "Output directory",
            value=str(DEFAULT_RUNS_DIR / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")),
        )
        selected_model = st.selectbox(
            "Copilot model",
            options=["claude-haiku-4.5", "gpt-5-mini"],
            index=0,
        )

        target_column: str | None = None
        if uploaded is not None:
            _raw = uploaded.getvalue()
            _first_line = _raw.split(b"\n", 1)[0].decode("utf-8", errors="replace")
            _sep = ";" if _first_line.count(";") > _first_line.count(",") else ","
            dataframe = pl.read_csv(BytesIO(_raw), separator=_sep, try_parse_dates=True, truncate_ragged_lines=True)
            target_column = st.selectbox("🎯 Target column", options=dataframe.columns)
            st.caption(f"{dataframe.shape[0]} rows × {dataframe.shape[1]} cols")

        if launch_mode == "VS Code Chat":
            submitted = st.button("📋 Generate Prompt & Monitor", type="primary",
                                  use_container_width=True,
                                  disabled=(uploaded is None))
        else:
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

        if launch_mode == "VS Code Chat":
            _handle_vscode_mode(prompt, active_dir, selected_model)
        else:
            _handle_cli_mode(prompt, active_dir, selected_model)

    # ── Show results in tabs ──────────────────────────────────────────────────
    if active_dir is None:
        st.markdown("""
## Getting Started

1. **Upload** a CSV with time-series data and select the target column, OR
2. **Browse** an existing run from the sidebar dropdown.

The pipeline runs 8 steps:
- **10** CSV cleansing & outlier detection
- **11** Deep time-series EDA (ADF, KPSS, Hurst, ACF/PACF, MI, STL)
- **12** Adaptive feature engineering (lags, Fourier, PCA factors)
- **13** Multi-tier model training (classical TS, FAAR, ML hybrids)
- **14** Evaluation + SHAP computation
- **15** Model selection with weighted ranking
- **16** Full report generation
- **17** Critical self-audit & remediation (validates results, triggers re-runs if needed)
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 EDA",
        "🏋️ Model Comparison",
        "🏆 Best Model",
        "📄 Report",
        "🔐 Audit",
    ])

    with tab1:
        _render_eda_tab(active_dir)

    with tab2:
        _render_model_comparison_tab(active_dir)

    with tab3:
        _render_best_model_tab(active_dir)

    with tab4:
        _render_report_tab(active_dir)
    
    with tab5:
        _render_audit_tab(active_dir)


if __name__ == "__main__":
    main()
