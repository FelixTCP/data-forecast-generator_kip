"""Step 13 — Model Training."""
import argparse
import json
import os
import sys
import warnings
import numpy as np
import polars as pl
import joblib
from datetime import datetime, timezone
from tqdm import tqdm

warnings.filterwarnings("ignore")


def _score_benchmark(y_true, y_pred):
    """Compute R², RMSE, MAE for a benchmark."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Handle NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    r2 = float(r2_score(y_true[mask], y_pred[mask]))
    rmse = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
    mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    return {"r2": r2, "rmse": rmse, "mae": mae}


def update_progress(output_dir, step, status, extra=None):
    progress_path = os.path.join(output_dir, "progress.json")
    with open(progress_path) as f:
        progress = json.load(f)
    progress["status"] = status
    progress["current_step"] = step
    if extra:
        progress.update(extra)
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def make_ser(obj):
    if isinstance(obj, dict):
        return {k: make_ser(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_ser(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        return str(obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--force-expansion-models", default="false")
    parser.add_argument("--regularization", default="")
    parser.add_argument("--extra-models", default="")
    parser.add_argument("--split-mode", default="time_series")
    args = parser.parse_args()

    output_dir = args.output_dir
    run_id = args.run_id
    force_expansion = args.force_expansion_models.lower() == "true"
    extra_models = [m.strip() for m in args.extra_models.split(",") if m.strip()]

    update_progress(output_dir, "13-model-training", "running")

    # Load inputs
    step11 = json.load(open(os.path.join(output_dir, "step-11-exploration.json")))
    step12 = json.load(open(os.path.join(output_dir, "step-12-features.json")))

    target_col = step12["target_column"]
    feature_cols = step12["features"]
    split_mode = step12["split_strategy"]["resolved_mode"]
    primary_period = step11.get("ts_diagnostics", {}).get("primary_seasonal_period")
    acf_pacf = step11.get("acf_pacf_orders", {})

    # Load features parquet
    df_pl = pl.read_parquet(os.path.join(output_dir, "features.parquet"))
    import pandas as pd
    df = df_pl.to_pandas()

    # Ensure feature cols exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        print("ERROR: No feature columns found in parquet", file=sys.stderr)
        sys.exit(1)

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    N = len(y)
    if N < 20:
        print(f"ERROR: Too few rows: {N}", file=sys.stderr)
        sys.exit(1)

    # Pre-training leakage gate
    from scipy.stats import pearsonr
    for i, col in enumerate(feature_cols):
        x_col = X[:, i]
        mask = ~(np.isnan(x_col) | np.isnan(y))
        if mask.sum() < 10:
            continue
        try:
            r, _ = pearsonr(x_col[mask], y[mask])
            if abs(r) > 0.98:
                # Check if it's a legitimate lag feature
                if f"{target_col}_lag" not in col and f"{target_col}_roll" not in col and f"{target_col}_diff" not in col:
                    print(f"ERROR: Leaked feature '{col}' (|r|={abs(r):.4f})", file=sys.stderr)
                    sys.exit(1)
        except Exception:
            pass

    # Time-series split: chronological holdout (last 20%)
    holdout_size = max(int(N * 0.20), 30)
    train_end = N - holdout_size
    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[train_end:], y[train_end:]

    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

    # Save holdout arrays
    holdout_path = os.path.join(output_dir, "holdout.npz")
    np.savez(holdout_path, X_test=X_test, y_test=y_test,
             feature_names=np.array(feature_cols, dtype=str))

    # ─── MANDATORY BENCHMARKS ────────────────────────────────────────────────
    benchmarks = {}
    benchmark_predictions = {}

    # 1. naive_persistence
    y_pred_naive = np.concatenate([[y_train[-1]], y_test[:-1]])
    benchmarks["naive_persistence"] = _score_benchmark(y_test, y_pred_naive)
    benchmark_predictions["naive_persistence"] = y_pred_naive

    # 2. seasonal_naive
    m = primary_period or 1
    y_hist = np.concatenate([y_train, y_test])
    try:
        y_pred_seasonal = np.array([y_hist[i - m] for i in range(len(y_train), len(y_hist))])
    except Exception:
        y_pred_seasonal = y_pred_naive.copy()
    benchmarks["seasonal_naive"] = _score_benchmark(y_test, y_pred_seasonal)
    benchmarks["seasonal_naive"]["seasonal_period"] = m
    benchmark_predictions["seasonal_naive"] = y_pred_seasonal

    # 3. auto_arima_benchmark
    print("Running auto_arima benchmark...")
    try:
        import pmdarima
        bm_model = pmdarima.auto_arima(y_train, seasonal=True, m=min(m, 12),
                                        max_p=3, max_q=3, suppress_warnings=True,
                                        error_action="ignore", stepwise=True)
        y_pred_arima = bm_model.predict(n_periods=len(y_test))
    except Exception as e:
        print(f"auto_arima failed ({e}), using ARIMA(1,1,1)")
        try:
            from statsmodels.tsa.arima.model import ARIMA as _ARIMA
            bm_model_arima = _ARIMA(y_train, order=(1, 1, 1)).fit()
            y_pred_arima = bm_model_arima.forecast(steps=len(y_test))
        except Exception as e2:
            print(f"ARIMA(1,1,1) failed ({e2}), using persistence")
            y_pred_arima = y_pred_naive.copy()
    benchmarks["auto_arima_benchmark"] = _score_benchmark(y_test, np.asarray(y_pred_arima, dtype=float))
    benchmark_predictions["auto_arima_benchmark"] = np.asarray(y_pred_arima, dtype=float)

    # 4. ar1_benchmark
    print("Running AR(1) benchmark...")
    try:
        from statsmodels.tsa.ar_model import AutoReg as _AutoReg
        ar1_model = _AutoReg(y_train, lags=1, old_names=False).fit()
        start_idx = len(y_train)
        end_idx = len(y_train) + len(y_test) - 1
        y_pred_ar1 = np.asarray(ar1_model.predict(start=start_idx, end=end_idx), dtype=float)
    except Exception as e:
        print(f"AR(1) failed ({e}), using persistence")
        y_pred_ar1 = y_pred_naive.copy()
    benchmarks["ar1_benchmark"] = _score_benchmark(y_test, y_pred_ar1)
    benchmarks["ar1_benchmark"]["lags"] = 1
    benchmark_predictions["ar1_benchmark"] = y_pred_ar1

    print(f"Benchmarks: naive={benchmarks['naive_persistence']['r2']:.4f}, "
          f"seasonal={benchmarks['seasonal_naive']['r2']:.4f}, "
          f"arima={benchmarks['auto_arima_benchmark']['r2']:.4f}, "
          f"ar1={benchmarks['ar1_benchmark']['r2']:.4f}")

    # ─── ML CANDIDATES ───────────────────────────────────────────────────────
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
    from sklearn.svm import SVR

    tscv = TimeSeriesSplit(n_splits=5)

    def build_pipeline(estimator):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    def train_candidate(name, estimator, X_tr, y_tr, X_te, y_te, tscv_obj):
        print(f"Training {name}...")
        pipe = build_pipeline(estimator)
        try:
            cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=tscv_obj, scoring="r2", n_jobs=1)
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
        except Exception as e:
            print(f"  CV failed for {name}: {e}")
            cv_mean, cv_std = None, None

        try:
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            r2 = float(r2_score(y_te, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
            mae = float(mean_absolute_error(y_te, y_pred))
            residuals = y_te - y_pred
            return {
                "model_name": name,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "cv_mean_r2": cv_mean,
                "cv_std_r2": cv_std,
                "residual_mean": float(np.mean(residuals)),
                "residual_max_abs": float(np.max(np.abs(residuals))),
                "model_worse_than_mean_baseline": r2 < 0,
                "fitted_pipeline": pipe,
                "predictions": y_pred.tolist(),
            }
        except Exception as e:
            print(f"  Training failed for {name}: {e}")
            return {
                "model_name": name,
                "r2": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "cv_mean_r2": None,
                "cv_std_r2": None,
                "error": str(e),
                "model_worse_than_mean_baseline": True,
                "fitted_pipeline": None,
                "predictions": [],
            }

    candidates_def = [
        ("ridge", Ridge(alpha=10.0, random_state=None)),
        ("random_forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)),
        ("gradient_boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]

    # Add XGBoost if available
    try:
        import xgboost as xgb
        candidates_def.append(("xgboost", xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)))
    except ImportError:
        print("XGBoost not available, skipping")

    # Force expansion models if requested
    if force_expansion or extra_models:
        candidates_def.extend([
            ("elasticnet", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000)),
            ("histgradientboosting", HistGradientBoostingRegressor(random_state=42)),
            ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.1)),
        ])

    results = []
    fitted_models = {}

    for name, estimator in tqdm(candidates_def, desc="Training candidates"):
        result = train_candidate(name, estimator, X_train, y_train, X_test, y_test, tscv)
        pipe = result.pop("fitted_pipeline", None)
        results.append(result)
        if pipe is not None and not np.isnan(result["r2"]):
            fitted_models[name] = pipe
            candidate_path = os.path.join(output_dir, f"candidate-{name}.joblib")
            joblib.dump(pipe, candidate_path)
            result["joblib_path"] = candidate_path

    # Select best candidate
    valid_results = [r for r in results if r.get("r2") is not None and not (isinstance(r["r2"], float) and np.isnan(r["r2"])) and r["r2"] >= 0]
    if not valid_results:
        # Fallback: use best by r2 even if negative
        valid_results = [r for r in results if r.get("r2") is not None and not (isinstance(r["r2"], float) and np.isnan(r["r2"]))]

    if not valid_results:
        print("ERROR: All candidates failed", file=sys.stderr)
        sys.exit(1)

    best = max(valid_results, key=lambda x: x["r2"] if x["r2"] is not None else -999)
    best_name = best["model_name"]
    print(f"Best candidate: {best_name} (R²={best['r2']:.4f})")

    # Save best model
    if best_name in fitted_models:
        model_path = os.path.join(output_dir, "model.joblib")
        joblib.dump(fitted_models[best_name], model_path)
    else:
        print(f"WARNING: Best model {best_name} has no fitted pipeline", file=sys.stderr)
        # Create a simple model as fallback
        from sklearn.linear_model import Ridge
        fallback_pipe = build_pipeline(Ridge(alpha=10.0))
        fallback_pipe.fit(X_train, y_train)
        model_path = os.path.join(output_dir, "model.joblib")
        joblib.dump(fallback_pipe, model_path)

    # Benchmark warning
    best_r2 = best["r2"] if best["r2"] is not None else -999
    arima_r2 = benchmarks["auto_arima_benchmark"]["r2"]
    ar1_r2 = benchmarks["ar1_benchmark"]["r2"]
    benchmark_warning = (
        (not np.isnan(arima_r2) and best_r2 < arima_r2 + 0.02) or
        (not np.isnan(ar1_r2) and best_r2 < ar1_r2 + 0.02)
    )

    # Build output JSON
    result_json = {
        "step": "13-model-training",
        "run_id": run_id,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_cols": feature_cols,
        "split_mode": split_mode,
        "primary_seasonal_period": primary_period,
        "best_model": best_name,
        "best_r2": best_r2,
        "benchmark_warning": benchmark_warning,
        "benchmarks": benchmarks,
        "candidates": [
            {k: v for k, v in r.items() if k not in ("predictions",)}
            for r in results
        ],
        "artifacts": {
            "model_joblib": model_path,
            "holdout_npz": holdout_path,
        },
        "context": {
            "target_column": target_col,
            "feature_cols": feature_cols,
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
        }
    }

    out_json = os.path.join(output_dir, "step-13-training.json")
    with open(out_json, "w") as f:
        json.dump(make_ser(result_json), f, indent=2)

    # Update progress
    with open(os.path.join(output_dir, "progress.json")) as f:
        progress = json.load(f)
    progress["status"] = "running"
    progress["current_step"] = "14-model-evaluation"
    if "13-model-training" not in progress.get("completed_steps", []):
        progress["completed_steps"].append("13-model-training")
    with open(os.path.join(output_dir, "progress.json"), "w") as f:
        json.dump(progress, f, indent=2)

    print(f"Step 13 complete. Best: {best_name} R²={best_r2:.4f}")
    print(f"Model saved: {model_path}")
    print(f"Holdout saved: {holdout_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
