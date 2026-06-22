"""
Standalone Inference App — autoregressive k-step forecast from pipeline artifacts.

Requires per-run artifacts:
  • model.joblib            — fitted best model
  • features_future.parquet — pre-scaled future feature rows (written by step 12)
  • holdout.npz             — X_test / y_test for residual bootstrap CI
  • step-12-features.json   — feature names, future_inference metadata
  • scaler.joblib           — StandardScaler (optional but needed for AR update)

If features_future.parquet is absent → inference is blocked with a clear message.

Coherence strategy
------------------
Step-12 fills ALL k future rows with identical lag values (last known observation).
That produces a flat, unrealistic forecast.  This app corrects it via an
**autoregressive update loop**:

  1. Predict y_hat for step s  (model operates in scaled feature space)
  2. Convert y_hat to scaled lag space using the saved StandardScaler
  3. Overwrite lag_1 … lag_N in row s+1 with the freshly scaled predictions
  4. Propagate: lag_2[s+1] ← lag_1[s],  lag_3[s+1] ← lag_2[s], …
  5. Incrementally update rolling-mean features  (rolling_std / min / max are held)

Bootstrap CI inflates by √(step) so uncertainty widens naturally with horizon.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Constants / paths
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT_DIR / "output"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _eligible_runs() -> list[Path]:
    """Return run directories that have all required inference artifacts."""
    if not RUNS_DIR.exists():
        return []
    runs = []
    for p in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        if (p / "features_future.parquet").exists() and (p / "model.joblib").exists():
            runs.append(p)
    return runs


def _run_label(p: Path) -> str:
    meta = _read_json(p / "meta_data.json") or {}
    prog = _read_json(p / "progress.json") or {}
    target = prog.get("target_column", "?")
    model  = meta.get("llm_model_name", "")
    ts     = p.name  # ISO timestamp directory name
    label  = f"{ts}  •  target={target}"
    if model:
        label += f"  •  {model}"
    return label


# ─────────────────────────────────────────────────────────────────────────────
# Autoregressive inference engine
# ─────────────────────────────────────────────────────────────────────────────

_LAG_PAT   = re.compile(r"^(.+)_lag_(\d+)$")
_RMEAN_PAT = re.compile(r"^(.+)_roll_mean_(\d+)$")


def _build_ar_predictions(
    X_future_raw: np.ndarray,          # (k_available, n_features) — from parquet (lag cols may be unscaled!)
    feature_names: list[str],
    model,
    scaler,
    y_hist: np.ndarray,                # unscaled historical target values (for roll seeds)
    X_hist_last: np.ndarray,           # last row(s) of X_test — properly scaled, shape (m, n_features)
    k: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Run the autoregressive forecast loop.

    The step-12 pipeline writes future lag columns as raw (unscaled) placeholder
    values.  This function re-seeds all lag columns from the properly-scaled
    holdout tail before the first prediction, then propagates each new prediction
    back into the next row's lag slots.

    Returns
    -------
    y_preds : (k,)   unscaled predictions
    warnings : list of human-readable warnings encountered
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    warnings_out: list[str] = []

    # Scaler lookup helpers — operate on the feature-space index
    scaler_mean  = np.array(scaler.mean_)  if scaler is not None else None
    scaler_scale = np.array(scaler.scale_) if scaler is not None else None

    def _to_scaled(val: float, col_idx: int) -> float:
        if scaler_mean is None or col_idx >= len(scaler_mean):
            return val
        std = scaler_scale[col_idx]
        return (val - scaler_mean[col_idx]) / (std if std > 0 else 1.0)

    def _to_unscaled(scaled_val: float, col_idx: int) -> float:
        if scaler_mean is None or col_idx >= len(scaler_mean):
            return scaled_val
        return scaled_val * scaler_scale[col_idx] + scaler_mean[col_idx]

    # ── Identify lag / rolling-mean columns ──────────────────────────────────
    lag_cols: dict[str, list[tuple[int, int]]] = {}   # base → [(lag_n, col_idx)]
    for fname, idx in name_to_idx.items():
        m = _LAG_PAT.match(fname)
        if m:
            base, lag_n = m.group(1), int(m.group(2))
            lag_cols.setdefault(base, []).append((lag_n, idx))
    for base in lag_cols:
        lag_cols[base].sort()

    roll_cols: dict[str, list[tuple[int, int]]] = {}  # base → [(window, col_idx)]
    for fname, idx in name_to_idx.items():
        m = _RMEAN_PAT.match(fname)
        if m:
            base, window = m.group(1), int(m.group(2))
            roll_cols.setdefault(base, []).append((window, idx))

    # ── Work matrix: take future rows but keep non-lag features as-is ────────
    X_work = np.array(X_future_raw[:k], dtype=float)
    if len(X_work) < k:
        pad = np.tile(X_work[-1], (k - len(X_work), 1))
        X_work = np.vstack([X_work, pad])

    # ── Re-seed lag features in row 0 from holdout tail (properly scaled) ────
    # The parquet stores lag columns as unscaled raw values; the model expects
    # StandardScaled inputs.  Pull the correct scaled values from X_hist_last.
    #
    # For a base with lags [1, 2, 3]:
    #   lag_1[future_0] = scaled(y_test[-1])
    #   lag_2[future_0] = X_hist_last[-1, lag_1_idx]   (= scaled y_test[-2])
    #   lag_3[future_0] = X_hist_last[-1, lag_2_idx]   (= scaled y_test[-3])
    if X_hist_last is not None and len(X_hist_last) > 0:
        last_hist_row = X_hist_last[-1]
        for base, lags_sorted in lag_cols.items():
            # lags_sorted: [(1,idx1), (2,idx2), (3,idx3), …]
            for i, (lag_n, col_idx) in enumerate(lags_sorted):
                if i == 0:
                    # lag_1: seed with the very last y value
                    y_last = float(y_hist[-1])
                    X_work[0, col_idx] = _to_scaled(y_last, col_idx)
                else:
                    # lag_N: copy lag_(N-1) from last holdout row
                    _, prev_col_idx = lags_sorted[i - 1]
                    X_work[0, col_idx] = last_hist_row[prev_col_idx]

    # ── AR loop ───────────────────────────────────────────────────────────────
    y_preds_unscaled: list[float] = []

    for s in range(k):
        row = X_work[s].copy()

        y_s = float(model.predict(row.reshape(1, -1)).flatten()[0])
        y_preds_unscaled.append(y_s)

        if s + 1 >= k:
            break

        next_row = X_work[s + 1].copy()

        # Shift lag slots: lag_N+1[s+1] ← lag_N[s], lag_1[s+1] ← scaled(y_s)
        for base, lags_sorted in lag_cols.items():
            for i in range(len(lags_sorted) - 1, 0, -1):
                _, idx_dst = lags_sorted[i]
                _, idx_src = lags_sorted[i - 1]
                next_row[idx_dst] = row[idx_src]
            lag_1_n, lag_1_idx = lags_sorted[0]
            if lag_1_n == 1:
                next_row[lag_1_idx] = _to_scaled(y_s, lag_1_idx)

        # Incremental rolling-mean update: µ_{t+1} = µ_t + (y_enter − y_leave) / W
        for base, rolls in roll_cols.items():
            for (window, col_idx) in rolls:
                rm_unscaled = _to_unscaled(row[col_idx], col_idx)
                leave_idx = -(window - s)
                y_leave = float(y_hist[leave_idx]) if s < window and len(y_hist) >= window else y_s
                new_rm = rm_unscaled + (y_s - y_leave) / window
                next_row[col_idx] = _to_scaled(new_rm, col_idx)

        X_work[s + 1] = next_row

    return np.array(y_preds_unscaled), X_work[:k], warnings_out


# ─────────────────────────────────────────────────────────────────────────────
# SHAP helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_shap(
    model,
    X_future_ar: np.ndarray,   # (k, n_features) — AR-corrected feature rows
    X_background: np.ndarray,  # X_test, used as background for linear explainer
) -> np.ndarray | None:
    """
    Compute SHAP values for k forecast rows.

    Returns ndarray of shape (k, n_features) or None on failure.
    Strategy (tries each in order, uses first that succeeds):
      1. sklearn Pipeline  → unwrap to (preprocessing, final_estimator),
         then apply TreeExplainer or LinearExplainer on the transformed data
      2. TreeExplainer     — exact, fast for tree ensembles
      3. LinearExplainer   — exact for linear models
      4. PermutationExplainer — slow fallback, model-agnostic
    """
    try:
        import shap  # noqa: PLC0415
        from sklearn.pipeline import Pipeline  # noqa: PLC0415
    except ImportError:
        return None

    # ── Unwrap sklearn Pipeline ───────────────────────────────────────────────
    # If the model is a Pipeline, split into preprocessing steps + final estimator.
    # SHAP explainers work on the final estimator operating on transformed data.
    inner_model = model
    X_explain   = X_future_ar
    X_bg        = X_background
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        preprocessor = model[:-1]   # all steps except the last
        inner_model  = model[-1]
        try:
            X_explain = preprocessor.transform(X_future_ar)
            X_bg      = preprocessor.transform(X_background)
        except Exception:
            # If transform fails, fall back to raw data with full pipeline
            inner_model = model
            X_explain   = X_future_ar
            X_bg        = X_background

    # ── Try TreeExplainer ─────────────────────────────────────────────────────
    try:
        explainer = shap.TreeExplainer(inner_model)
        sv = explainer.shap_values(X_explain)
        if isinstance(sv, list):
            sv = sv[0]
        return np.array(sv)
    except Exception:
        pass

    # ── Try LinearExplainer ───────────────────────────────────────────────────
    try:
        bg = shap.maskers.Independent(X_bg, max_samples=min(200, len(X_bg)))
        explainer = shap.LinearExplainer(inner_model, bg)
        sv = explainer.shap_values(X_explain)
        return np.array(sv)
    except Exception:
        pass

    # ── Fallback: PermutationExplainer (model-agnostic, slower) ──────────────
    try:
        bg = shap.maskers.Independent(X_bg, max_samples=100)
        explainer = shap.PermutationExplainer(model.predict, bg)
        sv = explainer(X_future_ar).values
        return np.array(sv)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI helper
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_ci(
    y_future: np.ndarray,
    residuals: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build step-wise bootstrap CI that widens naturally with the forecast horizon.

    Uncertainty inflation: σ·√(step) so that confidence intervals fan out.
    """
    k = len(y_future)
    rng = np.random.default_rng(seed)
    # Resample residuals once per bootstrap draw, then scale by √(step)
    boot_samples = rng.choice(residuals, size=(n_boot, k), replace=True)
    # Horizon scaling: step s gets inflation √((s+1))
    horizon_scale = np.sqrt(np.arange(1, k + 1))
    boot_samples = boot_samples * horizon_scale[None, :]
    boot_preds = y_future[None, :] + boot_samples
    lower = np.percentile(boot_preds, 100 * (alpha / 2), axis=0)
    upper = np.percentile(boot_preds, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Forecast Inference",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🔮 Forecast Inference")
    st.caption(
        "Autoregressive k-step ahead forecast — powered by pipeline artifacts from the "
        "Single Agent Pipeline."
    )

    # ── Sidebar: run selection ────────────────────────────────────────────────
    with st.sidebar:
        st.header("Run")
        eligible = _eligible_runs()
        if not eligible:
            st.error(
                "No eligible runs found.\n\n"
                "A run must contain **features_future.parquet** and **model.joblib**.\n"
                "Re-run the pipeline to generate these artifacts."
            )
            st.stop()

        labels = {_run_label(p): p for p in eligible}
        chosen_label = st.selectbox("Select run", list(labels.keys()))
        output_dir: Path = labels[chosen_label]

        # ── Compact run info ribbon ───────────────────────────────────────────
        prog = _read_json(output_dir / "progress.json") or {}
        target_col: str = prog.get("target_column", "unknown")

        feat_info = _read_json(output_dir / "step-12-features.json") or {}
        future_info = feat_info.get("future_inference", {}) or {}
        k_available = int(future_info.get("k_future", 10))
        last_known  = str(future_info.get("last_known_date", "—"))
        time_step   = str(future_info.get("time_step", "—"))
        n_features  = len(feat_info.get("features", []))

        meta = _read_json(output_dir / "meta_data.json") or {}
        model_name = meta.get("llm_model_name", "—")

        st.markdown(
            f"""
            <style>
            .run-info {{font-size:12px; line-height:1.7; color: var(--text-color);}}
            .run-info b {{font-weight:600;}}
            .run-info .val {{
                font-family: monospace;
                background: rgba(128,128,128,.1);
                border-radius: 3px;
                padding: 1px 5px;
            }}
            </style>
            <div class="run-info">
              <b>Run</b> <span class="val">{output_dir.name}</span><br>
              <b>Target</b> <span class="val">{target_col}</span><br>
              <b>Model</b> <span class="val">{model_name}</span><br>
              <b>Last date</b> <span class="val">{last_known[:10] if len(last_known) > 10 else last_known}</span>
              &nbsp;&nbsp;<b>Step</b> <span class="val">{time_step}</span><br>
              <b>Horizon</b> <span class="val">{k_available}</span>
              &nbsp;&nbsp;<b>Features</b> <span class="val">{n_features}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        k = st.slider(
            "Steps to forecast (k)",
            min_value=1,
            max_value=k_available,
            value=min(k_available, 10),
        )
        n_history = st.slider(
            "Historical context points",
            min_value=10,
            max_value=200,
            value=60,
            help="How many holdout observations to show before the forecast.",
        )
        ci_level = st.select_slider(
            "Confidence interval",
            options=[80, 90, 95, 99],
            value=95,
            help="Bootstrap CI width — wider = more conservative.",
        )
        run_btn = st.button("▶️  Generate Forecast", type="primary", use_container_width=True)

    # ── Guard: features_future.parquet required ───────────────────────────────
    future_path  = output_dir / "features_future.parquet"
    model_path   = output_dir / "model.joblib"
    holdout_path = output_dir / "holdout.npz"

    if not future_path.exists():
        st.error(
            "**`features_future.parquet` not found in this run.**\n\n"
            "Inference requires pre-computed future feature rows written by step 12 "
            "of the pipeline.  Select a more recent run or re-run the pipeline."
        )
        st.stop()

    missing = [p.name for p in [model_path, holdout_path] if not p.exists()]
    if missing:
        st.warning(f"Missing artifacts: {', '.join(missing)} — inference not possible.")
        st.stop()

    # ── Session-state cache key — invalidated when run / k / ci_level change ──
    cache_key = f"forecast__{output_dir.name}__{k}__{ci_level}"

    if run_btn:
        # Clear stale cache so we recompute fresh on explicit button press
        st.session_state.pop(cache_key, None)

    if cache_key not in st.session_state:
        if not run_btn:
            # Nothing cached yet and button not pressed → show idle card
            st.info(
                f"Ready to forecast **{k}** step(s) ahead for target `{target_col}`.  "
                f"Click **▶️ Generate Forecast** in the sidebar."
            )
            _render_artifact_summary(output_dir, feat_info, future_info, target_col)
            return

        # ── Load artifacts ────────────────────────────────────────────────────
        with st.spinner("Loading model and features…"):
            model      = joblib.load(model_path)
            holdout    = np.load(holdout_path)
            X_test     = holdout.get("X_test")
            y_test     = holdout.get("y_test")

            scaler_path = output_dir / "scaler.joblib"
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None

            future_df = pl.read_parquet(future_path)
            if "is_future" in future_df.columns:
                future_df = future_df.drop("is_future")

        if X_test is None or y_test is None:
            st.error("`holdout.npz` is missing X_test or y_test — cannot compute CI.")
            st.stop()

        feature_names: list[str] = feat_info.get("features", [])
        available_cols = [c for c in feature_names if c in future_df.columns]
        missing_feats  = [c for c in feature_names if c not in future_df.columns]
        if missing_feats:
            st.warning(
                f"{len(missing_feats)} feature(s) from step-12 not found in "
                f"features_future.parquet: {missing_feats[:5]}…"
            )

        X_future_raw = future_df.select(available_cols).to_numpy().astype(float)
        y_hist = y_test.flatten()

        # ── AR inference ─────────────────────────────────────────────────────
        with st.spinner(f"Running autoregressive inference for {k} step(s)…"):
            y_future, X_future_ar, ar_warnings = _build_ar_predictions(
                X_future_raw   = X_future_raw,
                feature_names  = available_cols,
                model          = model,
                scaler         = scaler,
                y_hist         = y_hist,
                X_hist_last    = X_test,
                k              = k,
            )

        # ── Bootstrap CI ──────────────────────────────────────────────────────
        y_pred_test = model.predict(X_test).flatten()
        residuals   = (y_hist - y_pred_test).astype(float)
        alpha       = 1 - ci_level / 100
        lower_ci, upper_ci = _bootstrap_ci(y_future, residuals, alpha=alpha)

        # ── Store everything in session state ─────────────────────────────────
        st.session_state[cache_key] = dict(
            y_future      = y_future,
            X_future_ar   = X_future_ar,
            lower_ci      = lower_ci,
            upper_ci      = upper_ci,
            residuals     = residuals,
            y_hist        = y_hist,
            available_cols= available_cols,
            ar_warnings   = ar_warnings,
            model         = model,
            X_test        = X_test,
        )

    # ── Retrieve from cache ───────────────────────────────────────────────────
    c = st.session_state[cache_key]
    y_future       = c["y_future"]
    X_future_ar    = c["X_future_ar"]
    lower_ci       = c["lower_ci"]
    upper_ci       = c["upper_ci"]
    residuals      = c["residuals"]
    y_hist         = c["y_hist"]
    available_cols = c["available_cols"]
    model          = c["model"]
    X_test         = c["X_test"]

    for w in c["ar_warnings"]:
        st.warning(w)

    # ── Plot ──────────────────────────────────────────────────────────────────
    hist_show = y_hist[-n_history:]
    hist_x    = np.arange(len(hist_show))
    fore_x    = np.arange(len(hist_show), len(hist_show) + k)

    fig = go.Figure()

    # Historical holdout
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_show,
        mode="lines",
        name="Holdout (actual)",
        line=dict(color="#1f77b4", width=2),
    ))

    # CI band
    fig.add_trace(go.Scatter(
        x=np.concatenate([fore_x, fore_x[::-1]]),
        y=np.concatenate([upper_ci, lower_ci[::-1]]),
        fill="toself",
        fillcolor="rgba(214,39,40,0.13)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=True,
        name=f"{ci_level}% CI",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fore_x, y=y_future,
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#d62728", width=2.5, dash="dot"),
        marker=dict(size=7, symbol="circle"),
    ))

    # Vertical separator
    fig.add_vline(
        x=float(len(hist_show)) - 0.5,
        line_dash="dash",
        line_color="#888",
        annotation_text="forecast start",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Autoregressive {k}-step forecast — target: {target_col}  "
              f"({ci_level}% bootstrap CI, σ·√step inflation)",
        xaxis_title="Time index (holdout)",
        yaxis_title=target_col,
        template="plotly_white",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    holdout_mean = float(np.mean(hist_show))
    fore_mean    = float(np.mean(y_future))
    ci_width_avg = float(np.mean(upper_ci - lower_ci))
    delta_pct    = (fore_mean - holdout_mean) / abs(holdout_mean) * 100 if holdout_mean != 0 else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Forecast steps", k)
    c2.metric("Forecast mean",   f"{fore_mean:.4f}")
    c3.metric("Δ vs holdout mean", f"{fore_mean - holdout_mean:+.4f}",
              delta=f"{delta_pct:+.1f}%", delta_color="off")
    c4.metric(f"Mean CI width ({ci_level}%)", f"{ci_width_avg:.4f}")
    c5.metric("Holdout RMSE",
              f"{float(np.sqrt(np.mean(residuals**2))):.4f}")

    # ── Step-by-step table ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Forecast table")
    out_df = pl.DataFrame({
        "step":                  list(range(1, k + 1)),
        "forecasted_value":      y_future.tolist(),
        f"lower_{ci_level}ci":   lower_ci.tolist(),
        f"upper_{ci_level}ci":   upper_ci.tolist(),
        "ci_width":              (upper_ci - lower_ci).tolist(),
    })
    st.dataframe(out_df, use_container_width=True, hide_index=True)

    csv_bytes = out_df.write_csv()
    st.download_button(
        label="📥 Download forecast CSV",
        data=csv_bytes,
        file_name=f"forecast_{output_dir.name}_k{k}.csv",
        mime="text/csv",
    )

    # ── SHAP feature attribution ─────────────────────────────────────────────
    with st.expander("🎯 SHAP feature attribution per forecast step", expanded=False):
        shap_cache_key = cache_key + "__shap"
        if shap_cache_key not in st.session_state:
            with st.spinner("Computing SHAP values…"):
                st.session_state[shap_cache_key] = _compute_shap(model, X_future_ar, X_test)
        shap_vals = st.session_state[shap_cache_key]

        if shap_vals is None:
            st.warning("SHAP computation failed for this model type.")
        else:
            # ── Heatmap: features × steps ────────────────────────────────────
            st.markdown(
                "Each cell shows the SHAP value (additive contribution to the prediction) "
                "for that feature at that forecast step. "
                "Red = pushes prediction up · Blue = pushes prediction down."
            )
            # Sort features by mean |SHAP| across all steps (most important on top)
            mean_abs = np.abs(shap_vals).mean(axis=0)
            order = np.argsort(mean_abs)[::-1]
            top_n = min(20, len(available_cols))
            idx_top = order[:top_n]
            feat_labels = [available_cols[i] for i in idx_top]
            z = shap_vals[:, idx_top].T   # (top_n, k)

            # symmetric color scale
            abs_max = float(np.abs(z).max()) or 1.0

            fig_heat = go.Figure(go.Heatmap(
                z=z,
                x=[f"Step {s+1}" for s in range(k)],
                y=feat_labels,
                colorscale="RdBu_r",
                zmid=0,
                zmin=-abs_max,
                zmax=abs_max,
                text=[[f"{v:+.3f}" for v in row] for row in z],
                texttemplate="%{text}",
                hovertemplate="Feature: %{y}<br>%{x}<br>SHAP: %{z:+.4f}<extra></extra>",
            ))
            fig_heat.update_layout(
                title=f"SHAP values — top {top_n} features by mean |SHAP| across {k} steps",
                height=max(340, top_n * 26 + 100),
                template="plotly_white",
                xaxis_title="Forecast step",
                yaxis_title="Feature",
                margin=dict(l=200, r=20, t=60, b=40),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # ── Per-step detail ───────────────────────────────────────────────
            st.markdown("---")
            st.markdown("**Per-step feature attribution**")
            step_sel = st.selectbox(
                "Select forecast step",
                options=list(range(1, k + 1)),
                format_func=lambda s: f"Step {s}  (ŷ = {y_future[s-1]:.4f})",
                key="shap_step_sel",
            )
            sv_step = shap_vals[step_sel - 1]  # (n_features,)
            # Sort by SHAP value for this step
            sort_idx = np.argsort(sv_step)
            feat_sorted  = [available_cols[i] for i in sort_idx]
            shap_sorted  = sv_step[sort_idx]
            colors = ["#d62728" if v >= 0 else "#1f77b4" for v in shap_sorted]

            fig_bar = go.Figure(go.Bar(
                x=shap_sorted,
                y=feat_sorted,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.4f}" for v in shap_sorted],
                textposition="outside",
                hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
            ))
            fig_bar.update_layout(
                title=f"SHAP attribution — Step {step_sel}  (ŷ = {y_future[step_sel-1]:.4f})",
                xaxis_title="SHAP value",
                template="plotly_white",
                height=max(400, len(available_cols) * 20 + 80),
                margin=dict(l=200, r=80, t=60, b=40),
            )
            fig_bar.add_vline(x=0, line_color="#555", line_width=1)
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Residual diagnostics ──────────────────────────────────────────────────
    with st.expander("🔬 Holdout residual diagnostics"):
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(
            x=residuals, nbinsx=40,
            marker_color="#1f77b4", opacity=0.75,
            name="Residuals",
        ))
        fig_res.update_layout(
            title="Holdout residual distribution",
            xaxis_title="Residual (actual − predicted)",
            yaxis_title="Count",
            template="plotly_white",
            height=300,
        )
        st.plotly_chart(fig_res, use_container_width=True)
        r2_col, rmse_col, mae_col = st.columns(3)
        r2_col.metric("R² (holdout)",   f"{1 - np.var(residuals) / np.var(y_hist):.4f}")
        rmse_col.metric("RMSE",         f"{float(np.sqrt(np.mean(residuals**2))):.4f}")
        mae_col.metric("MAE",           f"{float(np.mean(np.abs(residuals))):.4f}")


def _render_artifact_summary(
    output_dir: Path,
    feat_info: dict,
    future_info: dict,
    target_col: str,
) -> None:
    """Show a quick artifact summary card while the app is idle."""
    st.markdown("### Pipeline artifacts")
    checks = {
        "model.joblib":              (output_dir / "model.joblib").exists(),
        "features_future.parquet":   (output_dir / "features_future.parquet").exists(),
        "holdout.npz":               (output_dir / "holdout.npz").exists(),
        "scaler.joblib":             (output_dir / "scaler.joblib").exists(),
        "step-12-features.json":     (output_dir / "step-12-features.json").exists(),
    }
    for name, ok in checks.items():
        icon = "✅" if ok else "❌"
        st.write(f"{icon} `{name}`")

    st.markdown("### Feature summary")
    features = feat_info.get("features", [])
    if features:
        st.write(f"**{len(features)} feature(s):** {', '.join(features[:8])}" +
                 (f" … +{len(features)-8} more" if len(features) > 8 else ""))

    placeholder_lags = future_info.get("placeholder_lags", [])
    if placeholder_lags:
        st.info(
            f"Step 12 wrote {len(placeholder_lags)} lag(s) as placeholder "
            f"(identical to last known value). "
            "This app updates them autoregressively at inference time for coherent forecasts."
        )


if __name__ == "__main__":
    main()
