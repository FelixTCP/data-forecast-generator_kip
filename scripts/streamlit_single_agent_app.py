"""
Streamlit UI for Single Agent Pipeline - Professional data scientist dashboard.
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
]


def _normalize_column_name(value: str) -> str:
    return re.sub(r"_+", "_", value.strip().lower().replace(" ", "_")).strip("_")


def _render_single_agent_prompt(
    csv_path: Path,
    target_column: str,
    output_dir: Path,
    copilot_model: str,
) -> str:
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


def _run_pipeline(
    prompt: str, working_dir: Path, model: str = "claude-haiku-4.5"
) -> subprocess.CompletedProcess[str]:
    command = [
        "copilot",
        "--allow-all-tools",
        "--allow-all-paths",
        "--allow-all-urls",
        "--no-ask-user",
        "--model",
        model,
    ]
    if "gpt" in model.lower():
        command.extend(["--reasoning-effort", "low"])
    command.extend(["-s", "-p", prompt])
    return subprocess.run(
        command,
        cwd=working_dir,
        text=True,
        capture_output=True,
        check=False,
    )


def _start_pipeline_process(
    prompt: str, working_dir: Path, model: str = "claude-haiku-4.5"
) -> subprocess.Popen[str]:
    command = [
        "copilot",
        "--allow-all-tools",
        "--allow-all-paths",
        "--allow-all-urls",
        "--no-ask-user",
        "--model",
        model,
    ]
    if "gpt" in model.lower():
        command.extend(["--reasoning-effort", "low"])
    command.extend(["-s", "-p", prompt])
    return subprocess.Popen(
        command,
        cwd=working_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _read_progress(output_dir: Path) -> dict | None:
    progress_path = output_dir / "progress.json"
    if not progress_path.exists():
        return None
    return json.loads(progress_path.read_text(encoding="utf-8"))


def _format_step_label(step: str | None) -> str:
    if not step:
        return "waiting to start"
    return step.replace("-", " ").title()


def _format_elapsed(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _render_live_status(output_dir: Path, started_at: float) -> dict | None:
    progress = _read_progress(output_dir)
    elapsed = _format_elapsed(time.monotonic() - started_at)

    completed_count = 0
    current_step = None
    status = "running"
    errors: list[str] = []

    if progress:
        completed = progress.get("completed_steps", [])
        completed_count = len(completed) if isinstance(completed, list) else 0
        current_step = progress.get("current_step")
        status = str(progress.get("status", "running"))
        raw_errors = progress.get("errors", [])
        if isinstance(raw_errors, list):
            errors = [str(item) for item in raw_errors]

    fraction = min(1.0, completed_count / len(PIPELINE_STEPS))
    st.progress(
        fraction, text=f"Completed {completed_count}/{len(PIPELINE_STEPS)} steps"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", status.upper())
    c2.metric("Step", _format_step_label(current_step))
    c3.metric("Elapsed", elapsed)
    current_model = None
    if progress:
        current_model = progress.get("current_model") or progress.get("current_substep")
    c4.metric("Model", current_model or "-")

    if progress:
        st.caption("📊 Live progress snapshot")
        with st.expander("Show raw progress.json"):
            st.json(progress)

        completed_models = progress.get("completed_models", [])
        if completed_models:
            st.write(f"✓ Completed models: {', '.join(completed_models)}")

        model_progress = progress.get("model_progress")
        if isinstance(model_progress, (int, float)):
            try:
                mp = float(model_progress)
                st.progress(min(1.0, max(0.0, mp)), text="Model training progress")
            except Exception:
                pass

    if errors:
        st.error("\n".join(errors))

    return progress


def _load_evaluation_metrics(output_dir: Path) -> dict | None:
    """Load model evaluation results."""
    eval_path = output_dir / "step-14-evaluation.json"
    if not eval_path.exists():
        return None
    return json.loads(eval_path.read_text(encoding="utf-8"))


def _load_training_results(output_dir: Path) -> dict | None:
    """Load model training results."""
    train_path = output_dir / "step-13-training.json"
    if not train_path.exists():
        return None
    return json.loads(train_path.read_text(encoding="utf-8"))


def _load_selection_results(output_dir: Path) -> dict | None:
    """Load model selection results."""
    select_path = output_dir / "step-15-selection.json"
    if not select_path.exists():
        return None
    return json.loads(select_path.read_text(encoding="utf-8"))


def _load_features_data(output_dir: Path) -> pl.DataFrame | None:
    """Load features data."""
    features_path = output_dir / "features.parquet"
    if not features_path.exists():
        return None
    return pl.read_parquet(features_path)


def _load_features_info(output_dir: Path) -> dict | None:
    """Load features info from step-12."""
    features_path = output_dir / "step-12-features.json"
    if not features_path.exists():
        return None
    return json.loads(features_path.read_text(encoding="utf-8"))


def _load_holdout(output_dir: Path) -> tuple | None:
    """Load holdout test data."""
    holdout_path = output_dir / "holdout.npz"
    if not holdout_path.exists():
        return None
    data = np.load(holdout_path)
    return data.get("X_test"), data.get("y_test")


def _render_features_overview(output_dir: Path) -> None:
    """Render feature extraction overview with importance."""
    st.subheader("🔧 Feature Engineering Overview")

    features_info = _load_features_info(output_dir)
    if not features_info:
        st.warning("Feature information not available.")
        return

    # Feature list
    features_list = features_info.get("features", [])
    n_features = len(features_list)

    c1, c2, c3 = st.columns(3)
    c1.metric("📊 Total Features", n_features)
    c2.metric("⏰ Time Features", sum(1 for f in features_list if "time" in f.lower() or "date" in f.lower() or "year" in f.lower() or "month" in f.lower()))
    c3.metric("📈 Lag Features", sum(1 for f in features_list if "lag" in f.lower()))

    # Feature creation reasons
    with st.expander("📝 Feature Creation Details", expanded=False):
        feature_creation = features_info.get("feature_creation", {})
        if feature_creation:
            for feat_name, reason in feature_creation.items():
                st.write(f"**{feat_name}**: {reason}")
        else:
            # Display feature list grouped by type
            time_features = [f for f in features_list if any(x in f.lower() for x in ["time", "date", "year", "month", "day", "hour", "dow"])]
            lag_features = [f for f in features_list if "lag" in f.lower()]
            rolling_features = [f for f in features_list if "rolling" in f.lower() or "mean" in f.lower() or "std" in f.lower()]
            other_features = [f for f in features_list if f not in time_features and f not in lag_features and f not in rolling_features]

            if time_features:
                st.write(f"**⏰ Time Features ({len(time_features)}):**")
                st.write(", ".join(time_features[:10]) + ("..." if len(time_features) > 10 else ""))

            if lag_features:
                st.write(f"**📊 Lag Features ({len(lag_features)}):**")
                st.write(", ".join(lag_features[:10]) + ("..." if len(lag_features) > 10 else ""))

            if rolling_features:
                st.write(f"**📈 Rolling Features ({len(rolling_features)}):**")
                st.write(", ".join(rolling_features[:10]) + ("..." if len(rolling_features) > 10 else ""))

            if other_features:
                st.write(f"**🔸 Other Features ({len(other_features)}):**")
                st.write(", ".join(other_features[:10]) + ("..." if len(other_features) > 10 else ""))

    # Feature importance from model
    model_path = output_dir / "model.joblib"
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            if hasattr(model, "feature_importances_"):
                st.write("### 🎯 Feature Importances (from trained model)")
                importances = model.feature_importances_
                sorted_indices = np.argsort(importances)[-15:]  # Top 15
                sorted_names = [features_list[i] if i < len(features_list) else f"Feature_{i}" for i in sorted_indices]
                sorted_importances = importances[sorted_indices]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=sorted_names,
                    x=sorted_importances,
                    orientation="h",
                    marker_color="rgba(26, 118, 255, 0.8)"
                ))
                fig.update_layout(
                    title="Top 15 Feature Importances",
                    xaxis_title="Importance Score",
                    template="plotly_white",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            elif hasattr(model, "coef_"):
                st.write("### 🎯 Feature Coefficients (Linear Model)")
                coef = model.coef_.flatten()
                sorted_indices = np.argsort(np.abs(coef))[-15:]  # Top 15
                sorted_names = [features_list[i] if i < len(features_list) else f"Feature_{i}" for i in sorted_indices]
                sorted_coefs = coef[sorted_indices]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=sorted_names,
                    x=sorted_coefs,
                    orientation="h",
                    marker_color=["red" if x < 0 else "blue" for x in sorted_coefs]
                ))
                fig.update_layout(
                    title="Top 15 Feature Coefficients (Red=Negative, Blue=Positive)",
                    xaxis_title="Coefficient",
                    template="plotly_white",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not visualize feature importances: {e}")

    # Excluded features
    excluded = features_info.get("features_excluded", [])
    if excluded:
        with st.expander("⚠️ Excluded Features", expanded=False):
            for feat in excluded[:20]:
                if isinstance(feat, dict):
                    st.write(f"- **{feat.get('name', 'unknown')}**: {feat.get('reason', 'unknown reason')}")
                else:
                    st.write(f"- {feat}")


def _render_metrics_dashboard(evaluation: dict) -> None:
    """Render a comprehensive metrics dashboard."""
    st.subheader("📈 Model Performance Dashboard")

    if not evaluation:
        st.warning("No evaluation metrics available.")
        return

    # Key metrics cards
    candidates = evaluation.get("candidates", [])
    if candidates:
        best_candidate = max(candidates, key=lambda x: x.get("r2", -float("inf")))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Best R²",
            f"{best_candidate.get('r2', 0):.4f}",
            help="Coefficient of determination",
        )
        c2.metric("RMSE", f"{best_candidate.get('rmse', 0):.4f}", help="Root Mean Squared Error")
        c3.metric("MAE", f"{best_candidate.get('mae', 0):.4f}", help="Mean Absolute Error")
        c4.metric(
            "Quality", evaluation.get("quality_assessment", "—"), help="Overall quality assessment"
        )

    # Candidate comparison
    if len(candidates) > 1:
        st.write("### Candidate Model Comparison")
        df_candidates = (
            pl.DataFrame(candidates)
            .select(["model_name", "r2", "rmse", "mae"])
            .to_pandas()
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df_candidates["model_name"],
                y=df_candidates["r2"],
                name="R²",
                marker_color="rgb(26, 118, 255)",
            )
        )
        fig.update_layout(
            title="Model R² Comparison",
            xaxis_title="Model",
            yaxis_title="R² Score",
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Error metrics comparison
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=df_candidates["model_name"],
                y=df_candidates["rmse"],
                name="RMSE",
                marker_color="rgb(255, 127, 14)",
            )
        )
        fig2.add_trace(
            go.Bar(
                x=df_candidates["model_name"],
                y=df_candidates["mae"],
                name="MAE",
                marker_color="rgb(44, 160, 44)",
            )
        )
        fig2.update_layout(
            title="Error Metrics Comparison",
            xaxis_title="Model",
            yaxis_title="Error",
            barmode="group",
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig2, use_container_width=True)


def _generate_inference_plots(output_dir: Path, features: pl.DataFrame | None) -> None:
    """Generate and display inference visualization plots."""
    st.subheader("🔮 Model Inference & Predictions")

    # Load model and test data
    model_path = output_dir / "model.joblib"
    if not model_path.exists():
        st.warning("Model artifact not found.")
        return

    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    holdout_data = _load_holdout(output_dir)
    if holdout_data is None or holdout_data[0] is None:
        st.warning("Holdout test data not found.")
        return

    X_test, y_test = holdout_data

    # Make predictions
    try:
        if hasattr(X_test, "to_pandas"):
            X_test_pd = X_test.to_pandas()
        else:
            X_test_pd = X_test if isinstance(X_test, np.ndarray) else np.array(X_test)

        y_pred = model.predict(X_test_pd)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    # Actual vs Predicted scatter plot
    c1, c2 = st.columns(2)

    with c1:
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                mode="markers",
                marker=dict(size=6, color="rgba(26, 118, 255, 0.6)"),
                name="Predictions",
            )
        )
        # Add perfect prediction line
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig1.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="red", dash="dash"),
            )
        )
        fig1.update_layout(
            title="Actual vs Predicted Values",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            template="plotly_white",
            hovermode="closest",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        # Residuals plot
        residuals = y_test - y_pred
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                marker=dict(size=6, color=residuals, colorscale="RdBu", showscale=True),
                name="Residuals",
            )
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        fig2.update_layout(
            title="Residual Plot",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals",
            template="plotly_white",
            hovermode="closest",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Residuals distribution
    fig3 = go.Figure()
    fig3.add_trace(
        go.Histogram(
            x=residuals, nbinsx=30, marker_color="rgba(44, 160, 44, 0.7)", name="Residuals"
        )
    )
    fig3.update_layout(
        title="Residuals Distribution",
        xaxis_title="Residual Value",
        yaxis_title="Frequency",
        template="plotly_white",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Prediction error distribution
    errors = np.abs(y_test - y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("Test MAE", f"{mae:.4f}")
    c2.metric("Test RMSE", f"{rmse:.4f}")
    c3.metric("Test R²", f"{r2:.4f}")

    # Time series like plot (ordered predictions)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=y_test, mode="lines+markers", name="Actual", line=dict(color="blue", width=2)))
    fig4.add_trace(
        go.Scatter(
            y=y_pred,
            mode="lines+markers",
            name="Predicted",
            line=dict(color="red", width=2, dash="dot"),
        )
    )
    fig4.update_layout(
        title="Time Series Forecast (Test Set)",
        xaxis_title="Time Index",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig4, use_container_width=True)


def main() -> None:
    # Configure page
    st.set_page_config(
        page_title="Time Series Forecasting Pipeline",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title("Time Series Forecasting Pipeline")
    st.markdown(
        """
    Professional data scientist dashboard for automated regression forecasting.
    Upload your CSV, configure parameters, and let the agent discover the best model.
    """
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")

        uploaded = st.file_uploader("📁 Upload CSV", type=["csv"])
        output_dir_input = st.text_input(
            "📂 Output Directory",
            value=str(DEFAULT_RUNS_DIR / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")),
        )

        selected_model = st.selectbox(
            "🤖 Copilot Model",
            options=["claude-haiku-4.5", "gpt-5-mini"],
            index=0,
            help="Fast & quality (Haiku 4.5) vs Budget (GPT-5-mini)",
        )

        target_column: str | None = None
        if uploaded is not None:
            uploaded_bytes = uploaded.getvalue()
            dataframe = pl.read_csv(BytesIO(uploaded_bytes), try_parse_dates=True)
            target_column = st.selectbox("🎯 Target Column", options=dataframe.columns, index=0)
            st.write(f"**Dataset shape:** {dataframe.shape[0]} rows × {dataframe.shape[1]} cols")
        else:
            st.info("Upload a CSV to get started.")

        submitted = st.button("▶️ Run Pipeline", type="primary", use_container_width=True)

    if not submitted:
        # Show welcome message
        st.markdown(
            """
        ## Getting Started

        1. **Upload** a CSV file with time series data
        2. **Select** the target column to predict
        3. **Choose** a Copilot model
        4. **Click** "Run Pipeline"

        The pipeline will:
        - Clean and explore your data
        - Engineer features automatically
        - Train multiple model types
        - Evaluate and select the best model
        - Generate inference visualizations

        """
        )
        return

    if uploaded is None:
        st.error("Please upload a CSV file.")
        return

    if not target_column:
        st.error("Could not determine target column.")
        return

    csv_path = _save_uploaded_csv(uploaded, DEFAULT_UPLOAD_DIR)
    output_dir = Path(output_dir_input).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_target = _normalize_column_name(target_column)
    prompt = _render_single_agent_prompt(
        csv_path=csv_path,
        target_column=normalized_target,
        output_dir=output_dir,
        copilot_model=selected_model,
    )

    # Main content area
    st.markdown("---")
    st.subheader("⏱️ Pipeline Execution")
    started_at = time.monotonic()
    process = _start_pipeline_process(prompt, ROOT_DIR, model=selected_model)
    status_placeholder = st.empty()

    while process.poll() is None:
        with status_placeholder.container():
            _render_live_status(output_dir, started_at)
        time.sleep(1.0)

    with status_placeholder.container():
        _render_live_status(output_dir, started_at)

    stdout, stderr = process.communicate()
    result = subprocess.CompletedProcess(
        args=process.args,
        returncode=process.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
    )

    # Execution log (in expander)
    with st.expander("📝 Execution Logs"):
        st.code((result.stdout or "").strip() or "<no stdout>", language="text")
        if result.stderr:
            st.code(result.stderr.strip(), language="bash")

    if result.returncode != 0:
        st.error(f"❌ Pipeline failed with exit code {result.returncode}")
        return

    st.success("✅ Pipeline completed successfully!")

    # Display results
    st.markdown("---")

    # Load and display metrics
    evaluation = _load_evaluation_metrics(output_dir)
    training = _load_training_results(output_dir)
    selection = _load_selection_results(output_dir)

    if evaluation:
        _render_metrics_dashboard(evaluation)

    # Load and display features overview
    _render_features_overview(output_dir)

    # Load features for context
    features = _load_features_data(output_dir)

    # Generate inference visualizations
    _generate_inference_plots(output_dir, features)

    # Display report
    st.markdown("---")
    st.subheader("📄 Final Report")
    report_path = output_dir / "step-16-report.md"
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info("Report not yet generated.")

    # Summary stats
    st.markdown("---")
    st.subheader("📊 Run Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Output Directory", str(output_dir)[:40] + "...")
    c2.metric("Model Used", selected_model)
    if selection:
        c3.metric("Selected Model", selection.get("selected_model", "—"))


if __name__ == "__main__":
    main()
