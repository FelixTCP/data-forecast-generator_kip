"""
Inference application with XAI explainability - show WHY predictions are made.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "output"


@st.cache_data
def _list_run_directories() -> list[str]:
    """List all available run directories."""
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(
        [d.name for d in ARTIFACTS_DIR.iterdir() if d.is_dir() and (d / "model.joblib").exists()],
        reverse=True
    )


def _load_run_artifacts(run_dir: Path) -> dict | None:
    """Load all artifacts from a run directory."""
    artifacts = {}
    try:
        if (run_dir / "model.joblib").exists():
            artifacts["model"] = joblib.load(run_dir / "model.joblib")
        if (run_dir / "features.parquet").exists():
            import polars as pl
            artifacts["features_df"] = pl.read_parquet(run_dir / "features.parquet")
        if (run_dir / "cleaned.parquet").exists():
            import polars as pl
            artifacts["cleaned_df"] = pl.read_parquet(run_dir / "cleaned.parquet")
        if (run_dir / "holdout.npz").exists():
            data = np.load(run_dir / "holdout.npz")
            artifacts["X_test"] = data.get("X_test")
            artifacts["y_test"] = data.get("y_test")
        if (run_dir / "step-12-features.json").exists():
            artifacts["features_info"] = json.loads(
                (run_dir / "step-12-features.json").read_text(encoding="utf-8")
            )
        if (run_dir / "step-10-cleanse.json").exists():
            artifacts["cleanse_info"] = json.loads(
                (run_dir / "step-10-cleanse.json").read_text(encoding="utf-8")
            )
        return artifacts if artifacts else None
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        return None


def _forecast_k_steps(
    model, X_last: np.ndarray, k: int, n_features: int
) -> np.ndarray:
    """Forecast k steps ahead using the trained model."""
    forecasts = []
    X_current = X_last.copy() if isinstance(X_last, np.ndarray) else np.array(X_last)

    for step in range(k):
        if X_current.ndim == 1:
            X_current = X_current.reshape(1, -1)

        y_pred = model.predict(X_current)
        if hasattr(y_pred, "__len__"):
            y_pred = y_pred[0] if len(y_pred) > 0 else y_pred

        forecasts.append(float(y_pred))

        if X_current.shape[1] > 1:
            X_current = np.roll(X_current, -1, axis=1)
            X_current[0, -1] = y_pred
        else:
            X_current[0, 0] = y_pred

    return np.array(forecasts)


def _render_forecast_plot(historical_y: np.ndarray, forecasts: np.ndarray) -> None:
    """Render forecast visualization."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=historical_y,
            mode="lines+markers",
            name="Historical Data",
            line=dict(color="blue", width=2),
            marker=dict(size=4),
        )
    )

    forecast_x = np.arange(len(historical_y), len(historical_y) + len(forecasts))
    fig.add_trace(
        go.Scatter(
            x=forecast_x,
            y=forecasts,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="red", width=2, dash="dot"),
            marker=dict(size=6),
        )
    )

    fig.add_vline(x=len(historical_y) - 0.5, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Historical Data & Forecast",
        xaxis_title="Time Index",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_shap_explanations(model, X_test: np.ndarray, features_info: dict) -> None:
    """Render SHAP-based explanations."""
    if not HAS_SHAP:
        st.warning("SHAP not installed. Install with: pip install shap")
        return

    st.subheader("🔍 SHAP Explainability Analysis")

    # Get feature names
    feature_names = features_info.get("features", [f"Feature_{i}" for i in range(X_test.shape[1])])

    try:
        # Sample data for SHAP (use subset for performance)
        sample_size = min(100, X_test.shape[0])
        X_sample = X_test[:sample_size] if X_test.shape[0] > sample_size else X_test

        # Create SHAP explainer
        with st.spinner("Computing SHAP values... (this may take a moment)"):
            if hasattr(model, "predict"):
                explainer = shap.KernelExplainer(model.predict, X_sample[:min(20, len(X_sample))])
                shap_values = explainer.shap_values(X_sample)
            else:
                st.warning("Model does not support SHAP analysis")
                return

        # SHAP feature importance (mean absolute SHAP)
        if isinstance(shap_values, np.ndarray):
            mean_shap = np.abs(shap_values).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)

        # Plot feature importance
        st.write("### 📊 Feature Importance (by mean |SHAP value|)")
        top_indices = np.argsort(mean_shap)[-15:]
        top_names = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" for i in top_indices]
        top_values = mean_shap[top_indices]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=top_names,
                x=top_values,
                orientation="h",
                marker_color="rgba(26, 118, 255, 0.8)",
            )
        )
        fig.update_layout(
            title="Top 15 Features by SHAP Importance",
            xaxis_title="Mean |SHAP value|",
            template="plotly_white",
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.write("### 📈 SHAP Impact Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean |SHAP|", f"{mean_shap.mean():.6f}")
        c2.metric("Max SHAP", f"{mean_shap.max():.6f}")
        c3.metric("Most Important", top_names[-1] if len(top_names) > 0 else "—")

        # Detailed feature contributions for a single prediction
        st.write("### 🎯 Feature Contributions to a Single Prediction")
        sample_idx = st.slider("Select sample for detailed explanation", 0, len(X_sample) - 1, 0)

        sample_shap = shap_values[sample_idx]
        sample_features = X_sample[sample_idx]

        # Create contribution dataframe
        contrib_df = pl.DataFrame({
            "Feature": feature_names[: len(sample_shap)],
            "Value": [float(v) for v in sample_features[: len(sample_shap)]],
            "SHAP": [float(v) for v in sample_shap],
            "|SHAP|": [float(v) for v in np.abs(sample_shap)],
        }).sort("|SHAP|", descending=True).head(15)

        fig2 = go.Figure()
        colors = ["red" if x < 0 else "blue" for x in contrib_df["SHAP"].to_list()]
        fig2.add_trace(
            go.Bar(
                y=contrib_df["Feature"].to_list(),
                x=contrib_df["SHAP"].to_list(),
                orientation="h",
                marker_color=colors,
                text=[f"{x:.4f}" for x in contrib_df["SHAP"].to_list()],
                textposition="outside",
            )
        )
        fig2.update_layout(
            title=f"Feature Contributions to Prediction {sample_idx} (Red=Negative, Blue=Positive)",
            xaxis_title="SHAP Value",
            template="plotly_white",
            height=500,
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Display as table
        st.write("**Detailed Contributions:**")
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"SHAP analysis failed: {e}")


def _render_feature_analysis(features_info: dict, X_test: np.ndarray) -> None:
    """Render feature statistics and analysis."""
    st.subheader("📊 Feature Statistics")

    feature_names = features_info.get("features", [f"Feature_{i}" for i in range(X_test.shape[1])])

    # Feature summary stats
    stats_data = []
    for i, name in enumerate(feature_names[: X_test.shape[1]]):
        col_data = X_test[:, i]
        stats_data.append({
            "Feature": name,
            "Mean": float(np.mean(col_data)),
            "Std": float(np.std(col_data)),
            "Min": float(np.min(col_data)),
            "Max": float(np.max(col_data)),
        })

    stats_df = pl.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Feature distributions
    st.write("### Feature Distribution Heatmap")
    fig = go.Figure(data=go.Heatmap(z=X_test.T, colorscale="Viridis"))
    fig.update_layout(
        title="Feature Values Heatmap (Samples × Features)",
        xaxis_title="Sample Index",
        yaxis_title="Feature",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Inference with Explainability",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔮 Inference with Explainability")
    st.markdown("""
    Load a trained model and make transparent predictions with XAI analysis.
    Understand WHY your model makes specific predictions using SHAP values.
    """)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")

        runs = _list_run_directories()
        if not runs:
            st.error("No completed runs found in output/")
            return

        selected_run = st.selectbox("📁 Select Run", options=runs, help="Choose a completed pipeline run")

        run_path = ARTIFACTS_DIR / selected_run
        st.write(f"**Run ID:** {selected_run}")

        with st.spinner("Loading artifacts..."):
            artifacts = _load_run_artifacts(run_path)

        if not artifacts or "model" not in artifacts:
            st.error("Failed to load model artifacts.")
            return

        st.success("✅ Model loaded")

        st.markdown("---")
        k = st.slider(
            "🔢 Forecast K Steps",
            min_value=1,
            max_value=100,
            value=10,
        )

        run_inference = st.button("▶️ Generate Forecast & Explain", type="primary", use_container_width=True)

    if not run_inference:
        st.markdown("""
        ## How It Works

        1. **Select a run** with a trained model
        2. **Set forecast length** (k steps ahead)
        3. **Click "Generate Forecast & Explain"**
        4. **View SHAP explanations** showing which features drove the predictions

        ### What is SHAP?
        SHAP (SHapley Additive exPlanations) values show the contribution of each feature to a model's prediction.
        - **Positive** values push predictions higher
        - **Negative** values push predictions lower
        - Feature importance is computed as the mean |SHAP value|
        """)
        return

    X_test = artifacts.get("X_test")
    y_test = artifacts.get("y_test")
    features_info = artifacts.get("features_info", {})
    cleanse_info = artifacts.get("cleanse_info", {})
    model = artifacts["model"]

    if X_test is None:
        st.error("No test data available.")
        return

    historical_y = y_test.flatten() if hasattr(y_test, "flatten") else np.array(y_test)
    n_features = X_test.shape[1] if hasattr(X_test, "shape") else len(X_test[0])
    X_last = X_test[-1]

    try:
        forecasts = _forecast_k_steps(model, X_last, k, n_features)
    except Exception as e:
        st.error(f"Forecast generation failed: {e}")
        return

    st.success("✅ Forecast generated!")

    st.markdown("---")

    # Forecast visualization
    _render_forecast_plot(historical_y, forecasts)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast Length", k)
    c2.metric("Forecast Mean", f"{np.mean(forecasts):.4f}")
    c3.metric("Forecast Range", f"{np.max(forecasts) - np.min(forecasts):.4f}")
    c4.metric("vs Historical Mean", f"{np.mean(forecasts) - np.mean(historical_y):+.4f}")

    # SHAP explanations
    st.markdown("---")
    _render_shap_explanations(model, X_test, features_info)

    # Feature analysis
    st.markdown("---")
    _render_feature_analysis(features_info, X_test)

    # Export
    st.markdown("---")
    st.subheader("💾 Export Results")

    forecast_export = pl.DataFrame({
        "step": np.arange(1, k + 1).tolist(),
        "forecasted_value": forecasts.tolist(),
    })

    csv = forecast_export.write_csv()
    st.download_button(
        label="📥 Download Forecast (CSV)",
        data=csv,
        file_name=f"forecast_{selected_run}_{k}steps.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
