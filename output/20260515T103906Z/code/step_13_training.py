"""Step 13 — Model Training (Time-Series Focused).

Runnable:
    python step_13_training.py --output-dir <dir> --run-id <id>
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure CODE_DIR is importable
CODE_DIR = Path(__file__).parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ts_helpers import TimeSeriesPredictor


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_benchmark(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite_mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if finite_mask.sum() == 0:
        return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    yt = y_true[finite_mask]
    yp = y_pred[finite_mask]
    r2 = float(r2_score(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    return {"r2": r2, "rmse": rmse, "mae": mae}


def _ts_backtest_r2(model_factory, y: np.ndarray, n_splits: int = 3) -> list[float]:
    """Simple expanding-window backtest for TS models."""
    n = len(y)
    fold_size = max(n // (n_splits + 1), 30)
    scores = []
    for fold in range(n_splits):
        train_end = fold_size * (fold + 1)
        test_end = min(train_end + fold_size, n)
        if test_end > n or train_end < 30:
            continue
        y_tr = y[:train_end]
        y_te = y[train_end:test_end]
        try:
            m = model_factory(y_tr)
            preds = m.forecast(steps=len(y_te))
            preds = np.asarray(preds, dtype=float)
            if len(preds) == len(y_te) and np.all(np.isfinite(preds)):
                scores.append(float(r2_score(y_te, preds)))
        except Exception:
            pass
    return scores


def _cv_sklearn(estimator, X, y, n_splits=3) -> tuple[float, float]:
    """TimeSeriesSplit cross-validation, returns (mean_r2, std_r2)."""
    from sklearn.base import clone as _clone
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        try:
            est = _clone(estimator)
            est.fit(X_tr, y_tr)
            y_pred_val = est.predict(X_val)
            scores.append(float(r2_score(y_val, y_pred_val)))
        except Exception:
            scores.append(float("nan"))
    valid = [s for s in scores if np.isfinite(s)]
    if not valid:
        return float("nan"), float("nan")
    return float(np.mean(valid)), float(np.std(valid))


def update_progress(progress_path: Path, updates: dict):
    if progress_path.exists():
        with open(progress_path) as f:
            p = json.load(f)
    else:
        p = {}
    p.update(updates)
    with open(progress_path, "w") as f:
        json.dump(p, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Step 13: Model Training")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    progress_path = output_dir / "progress.json"

    update_progress(progress_path, {
        "status": "running",
        "current_step": "13-model-training",
        "current_model": None,
        "completed_models": [],
        "model_progress": 0.0,
        "model_history": [],
    })

    try:
        # ── Load leakage audit ────────────────────────────────────────────────
        leakage_path = output_dir / "leakage_audit.json"
        if leakage_path.exists():
            with open(leakage_path) as f:
                leakage_audit = json.load(f)
            if leakage_audit.get("status") == "fail":
                raise RuntimeError(
                    f"Leakage audit failed — halting training. "
                    f"Violations: {leakage_audit.get('pearson_violations', [])}"
                )
        else:
            print("WARNING: No leakage_audit.json found — proceeding without leakage check")

        # ── Load context ──────────────────────────────────────────────────────
        with open(output_dir / "step-10-cleanse.json") as f:
            ctx10 = json.load(f)
        with open(output_dir / "step-11-exploration.json") as f:
            ctx11 = json.load(f)
        with open(output_dir / "step-12-features.json") as f:
            ctx12 = json.load(f)

        target_col = ctx10["target_column_normalized"]
        time_col = ctx10["time_column"]
        feature_list = ctx12["features"]
        holdout_start = ctx12["split_strategy"]["holdout_start_index"]
        holdout_size = ctx12["split_strategy"]["holdout_size"]

        ts = ctx11["ts_diagnostics"]
        primary_period = ts.get("primary_seasonal_period") or 1
        stationarity = ts.get("stationarity_conclusion", "stationary")
        model_class_recs = [m["model_class"] for m in ctx11.get("model_class_recommendations", [])]
        pca_info = ctx12.get("pca_info", {})

        print(f"Target: {target_col}, Time: {time_col}")
        print(f"Features: {len(feature_list)}, Holdout start: {holdout_start}")

        # ── Load feature matrix ────────────────────────────────────────────────
        df = pl.read_parquet(output_dir / "features.parquet")
        print(f"Loaded features.parquet: {df.shape}")

        # Extract arrays
        y_all = df[target_col].cast(pl.Float64).to_numpy()
        valid_features = [f for f in feature_list if f in df.columns]

        X_all = np.column_stack([df[f].cast(pl.Float64).to_numpy() for f in valid_features])

        # Fill NaN
        for j in range(X_all.shape[1]):
            nan_mask = np.isnan(X_all[:, j])
            if nan_mask.any():
                col_mean = float(np.nanmean(X_all[:, j]))
                X_all[nan_mask, j] = col_mean
        nan_y = np.isnan(y_all)
        if nan_y.any():
            y_all[nan_y] = float(np.nanmean(y_all[~nan_y]))

        # Split
        X_train = X_all[:holdout_start]
        X_test = X_all[holdout_start:]
        y_train = y_all[:holdout_start]
        y_test = y_all[holdout_start:]

        print(f"Train: {len(y_train)}, Test: {len(y_test)}")

        # Pre-training leakage gate
        for j, fname in enumerate(valid_features):
            corr = float(np.corrcoef(X_train[:, j], y_train)[0, 1])
            if abs(corr) >= 0.98:
                raise RuntimeError(
                    f"Leaked feature detected at training time: {fname} |r|={abs(corr):.4f}"
                )

        # Save holdout data
        np.savez(
            output_dir / "holdout.npz",
            X_test=X_test,
            y_test=y_test,
            feature_names=np.array(valid_features),
        )
        print(f"Saved holdout.npz: X_test={X_test.shape}, y_test={y_test.shape}")

        # ── STEP A: Mandatory benchmarks ──────────────────────────────────────
        print("\n=== MANDATORY BENCHMARKS ===")
        benchmarks = {}
        benchmark_predictions = {}

        update_progress(progress_path, {"current_model": "naive_persistence"})

        # 1. naive_persistence
        y_pred_naive = np.concatenate([[y_train[-1]], y_test[:-1]])
        benchmarks["naive_persistence"] = _score_benchmark(y_test, y_pred_naive)
        benchmark_predictions["naive_persistence"] = y_pred_naive
        print(f"naive_persistence: {benchmarks['naive_persistence']}")

        update_progress(progress_path, {"current_model": "seasonal_naive"})

        # 2. seasonal_naive
        m = primary_period or 1
        y_hist = np.concatenate([y_train, y_test])
        y_pred_seasonal = np.array([y_hist[i - m] for i in range(len(y_train), len(y_hist))])
        benchmarks["seasonal_naive"] = _score_benchmark(y_test, y_pred_seasonal)
        benchmarks["seasonal_naive"]["seasonal_period"] = m
        benchmark_predictions["seasonal_naive"] = y_pred_seasonal
        print(f"seasonal_naive (m={m}): {benchmarks['seasonal_naive']}")

        update_progress(progress_path, {"current_model": "auto_arima_benchmark"})

        # 3. auto_arima_benchmark
        try:
            import pmdarima
            print("Fitting auto_arima (stepwise=True, max_p=5, max_q=2)...")
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bm_model = pmdarima.auto_arima(
                    y_train, seasonal=True, m=min(m, 52),
                    max_p=5, max_q=2, max_P=2, max_Q=2,
                    d=0 if stationarity == "stationary" else None,
                    D=0, stepwise=True, suppress_warnings=True,
                    error_action="ignore", n_jobs=1,
                )
            t1 = time.time()
            print(f"auto_arima fitted in {t1-t0:.1f}s")
            if hasattr(bm_model, "predict"):
                y_pred_arima = bm_model.predict(n_periods=len(y_test))
            else:
                y_pred_arima = bm_model.forecast(steps=len(y_test))
        except Exception as e:
            print(f"pmdarima failed ({e}), falling back to ARIMA(1,1,1)")
            from statsmodels.tsa.arima.model import ARIMA as _ARIMA
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bm_model = _ARIMA(y_train, order=(1, 1, 1)).fit()
            y_pred_arima = np.asarray(bm_model.forecast(steps=len(y_test)), dtype=float)

        benchmarks["auto_arima_benchmark"] = _score_benchmark(y_test, y_pred_arima)
        benchmark_predictions["auto_arima_benchmark"] = np.asarray(y_pred_arima, dtype=float)
        print(f"auto_arima_benchmark: {benchmarks['auto_arima_benchmark']}")

        update_progress(progress_path, {"current_model": "ar1_benchmark"})

        # 4. ar1_benchmark
        from statsmodels.tsa.ar_model import AutoReg as _AutoReg
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar1_model = _AutoReg(y_train, lags=1, old_names=False).fit()
            start_idx = len(y_train)
            end_idx = len(y_train) + len(y_test) - 1
            y_pred_ar1 = np.asarray(ar1_model.predict(start=start_idx, end=end_idx), dtype=float)
        except Exception as e:
            print(f"AR(1) failed ({e}), using persistence fallback")
            y_pred_ar1 = np.concatenate([[y_train[-1]], y_test[:-1]])
        benchmarks["ar1_benchmark"] = _score_benchmark(y_test, y_pred_ar1)
        benchmarks["ar1_benchmark"]["lags"] = 1
        benchmark_predictions["ar1_benchmark"] = np.asarray(y_pred_ar1, dtype=float)
        print(f"ar1_benchmark: {benchmarks['ar1_benchmark']}")

        # Record benchmark predictions in holdout.npz
        np.savez(
            output_dir / "holdout.npz",
            X_test=X_test,
            y_test=y_test,
            feature_names=np.array(valid_features),
            y_pred_naive=y_pred_naive,
            y_pred_seasonal=y_pred_seasonal,
            y_pred_arima=np.asarray(y_pred_arima, dtype=float),
            y_pred_ar1=np.asarray(y_pred_ar1, dtype=float),
        )

        # ── STEP B: Train candidate models ────────────────────────────────────
        print("\n=== CANDIDATE MODEL TRAINING ===")
        model_history = []
        completed_models = []
        candidate_results = {}

        def _record_candidate(name, model, cv_mean, cv_std, train_r2, fit_time, notes="", preds=None):
            if model is not None:
                cand_path = str(output_dir / f"candidate-{name}.joblib")
                joblib.dump(model, cand_path)
            entry = {
                "model_name": name,
                "status": "completed",
                "cv_mean_r2": cv_mean,
                "cv_std_r2": cv_std,
                "train_r2": train_r2,
                "fit_time_sec": fit_time,
                "notes": notes,
            }
            model_history.append(entry)
            candidate_results[name] = {"cv_mean": cv_mean, "cv_std": cv_std, "preds": preds}
            completed_models.append(name)
            update_progress(progress_path, {
                "current_model": name,
                "completed_models": completed_models,
                "model_progress": len(completed_models) / 6.0,
                "model_history": model_history,
            })
            print(f"  → {name}: cv_r2={cv_mean:.4f}±{cv_std:.4f}, fit={fit_time:.1f}s")

        # ── Ridge ─────────────────────────────────────────────────────────────
        update_progress(progress_path, {"current_model": "ridge"})
        try:
            t0 = time.time()
            ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=42))])
            cv_mean, cv_std = _cv_sklearn(ridge, X_train, y_train, n_splits=3)
            ridge.fit(X_train, y_train)
            train_preds = ridge.predict(X_train)
            holdout_preds_ridge = ridge.predict(X_test)
            train_r2 = float(r2_score(y_train, train_preds))
            t1 = time.time()
            _record_candidate("ridge", ridge, cv_mean, cv_std, train_r2, t1-t0,
                             preds=holdout_preds_ridge)
        except Exception as e:
            print(f"Ridge failed: {e}")
            model_history.append({"model_name": "ridge", "status": "failed", "error": str(e)})

        # ── ElasticNet ────────────────────────────────────────────────────────
        update_progress(progress_path, {"current_model": "elasticnet"})
        try:
            t0 = time.time()
            enet = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=42))])
            cv_mean, cv_std = _cv_sklearn(enet, X_train, y_train, n_splits=3)
            enet.fit(X_train, y_train)
            holdout_preds_enet = enet.predict(X_test)
            train_r2 = float(r2_score(y_train, enet.predict(X_train)))
            t1 = time.time()
            _record_candidate("elasticnet", enet, cv_mean, cv_std, train_r2, t1-t0,
                             preds=holdout_preds_enet)
        except Exception as e:
            print(f"ElasticNet failed: {e}")
            model_history.append({"model_name": "elasticnet", "status": "failed", "error": str(e)})

        # ── Random Forest ─────────────────────────────────────────────────────
        update_progress(progress_path, {"current_model": "random_forest"})
        try:
            t0 = time.time()
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5,
                                       random_state=42, n_jobs=-1)
            cv_mean, cv_std = _cv_sklearn(rf, X_train, y_train, n_splits=3)
            rf.fit(X_train, y_train)
            holdout_preds_rf = rf.predict(X_test)
            train_r2 = float(r2_score(y_train, rf.predict(X_train)))
            t1 = time.time()
            _record_candidate("random_forest", rf, cv_mean, cv_std, train_r2, t1-t0,
                             preds=holdout_preds_rf)
        except Exception as e:
            print(f"RandomForest failed: {e}")
            model_history.append({"model_name": "random_forest", "status": "failed", "error": str(e)})

        # ── HistGradientBoosting ───────────────────────────────────────────────
        update_progress(progress_path, {"current_model": "hist_gbm"})
        try:
            t0 = time.time()
            hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                max_depth=5, random_state=42)
            cv_mean, cv_std = _cv_sklearn(hgb, X_train, y_train, n_splits=3)
            hgb.fit(X_train, y_train)
            holdout_preds_hgb = hgb.predict(X_test)
            train_r2 = float(r2_score(y_train, hgb.predict(X_train)))
            t1 = time.time()
            _record_candidate("hist_gbm", hgb, cv_mean, cv_std, train_r2, t1-t0,
                             preds=holdout_preds_hgb)
        except Exception as e:
            print(f"HistGBM failed: {e}")
            model_history.append({"model_name": "hist_gbm", "status": "failed", "error": str(e)})

        # ── HoltWinters (ExponentialSmoothing) ────────────────────────────────
        if "HoltWinters-ETS" in model_class_recs or "HoltWinters" in model_class_recs:
            update_progress(progress_path, {"current_model": "holt_winters"})
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                t0 = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hw = ExponentialSmoothing(
                        y_train,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=min(primary_period, len(y_train) // 3),
                        initialization_method="estimated",
                    ).fit(optimized=True, use_brute=False)
                hw_preds = np.asarray(hw.forecast(steps=len(y_test)), dtype=float)

                # Backtest for CV-like score
                bt_scores = _ts_backtest_r2(
                    lambda y: ExponentialSmoothing(
                        y, trend="add", seasonal="add",
                        seasonal_periods=min(primary_period, len(y) // 3),
                        initialization_method="estimated",
                    ).fit(optimized=True, use_brute=False),
                    y_train,
                    n_splits=2,
                )
                cv_mean_hw = float(np.mean(bt_scores)) if bt_scores else float("nan")
                cv_std_hw = float(np.std(bt_scores)) if bt_scores else float("nan")

                # Wrap in TimeSeriesPredictor for uniform predict interface
                hw_wrapper = TimeSeriesPredictor(hw, hw_preds, "HoltWinters", bt_scores)
                train_r2_hw = float(r2_score(y_train, hw.fittedvalues))
                t1 = time.time()
                _record_candidate("holt_winters", hw_wrapper, cv_mean_hw, cv_std_hw,
                                 train_r2_hw, t1-t0,
                                 notes=f"seasonal_periods={min(primary_period, len(y_train)//3)}",
                                 preds=hw_preds)
            except Exception as e:
                import traceback
                print(f"HoltWinters failed: {e}\n{traceback.format_exc()}")
                model_history.append({"model_name": "holt_winters", "status": "failed", "error": str(e)})

        # ── FAAR-ARIMA (Factor-Augmented ARIMA) ───────────────────────────────
        if "FAAR-ARIMA" in model_class_recs and pca_info:
            update_progress(progress_path, {"current_model": "faar_arima"})
            try:
                pca_cols = [f"pca_factor_{i+1}" for i in range(pca_info.get("pca_n_components", 0))]
                pca_available = [c for c in pca_cols if c in df.columns]
                if pca_available:
                    X_exog_all = np.column_stack([df[c].cast(pl.Float64).to_numpy() for c in pca_available])
                    X_exog_train = X_exog_all[:holdout_start]
                    X_exog_test = X_exog_all[holdout_start:]

                    import pmdarima
                    t0 = time.time()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        faar = pmdarima.auto_arima(
                            y_train,
                            exogenous=X_exog_train,
                            seasonal=False,
                            max_p=3, max_q=2,
                            d=0 if stationarity == "stationary" else None,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action="ignore",
                        )
                    faar_preds = np.asarray(faar.predict(n_periods=len(y_test), exogenous=X_exog_test), dtype=float)

                    # Backtest
                    n_bt = 3
                    bt_scores_faar = []
                    fold_sz = len(y_train) // (n_bt + 1)
                    for fold in range(n_bt):
                        te = fold_sz * (fold + 1)
                        te_end = min(te + fold_sz, len(y_train))
                        if te_end > len(y_train) or te < 30:
                            continue
                        y_bt_tr = y_train[:te]
                        y_bt_te = y_train[te:te_end]
                        X_bt_tr = X_exog_train[:te]
                        X_bt_te = X_exog_train[te:te_end]
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                bt_model = pmdarima.ARIMA(
                                    order=faar.order, seasonal_order=faar.seasonal_order,
                                ).fit(y_bt_tr, X=X_bt_tr)
                            bt_preds = bt_model.predict(n_periods=len(y_bt_te), exogenous=X_bt_te)
                            bt_scores_faar.append(float(r2_score(y_bt_te, bt_preds)))
                        except Exception:
                            pass

                    cv_mean_faar = float(np.mean(bt_scores_faar)) if bt_scores_faar else float("nan")
                    cv_std_faar = float(np.std(bt_scores_faar)) if bt_scores_faar else float("nan")

                    faar_wrapper = TimeSeriesPredictor(faar, faar_preds, "FAAR-ARIMA", bt_scores_faar)
                    t1 = time.time()

                    train_preds_faar = np.asarray(faar.predict_in_sample(exogenous=X_exog_train), dtype=float)
                    train_r2_faar = float(r2_score(y_train, train_preds_faar))

                    _record_candidate("faar_arima", faar_wrapper, cv_mean_faar, cv_std_faar,
                                     train_r2_faar, t1-t0,
                                     notes=f"order={faar.order}, n_exog={len(pca_available)}",
                                     preds=faar_preds)
                else:
                    print("FAAR-ARIMA: No PCA columns found in features.parquet, skipping")
            except Exception as e:
                import traceback
                print(f"FAAR-ARIMA failed: {e}\n{traceback.format_exc()}")
                model_history.append({"model_name": "faar_arima", "status": "failed", "error": str(e)})

        # ── XGBoost (if available) ────────────────────────────────────────────
        try:
            import xgboost as xgb
            if "XGBoost" in model_class_recs:
                update_progress(progress_path, {"current_model": "xgboost"})
                try:
                    t0 = time.time()
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=300, learning_rate=0.05, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, n_jobs=-1, verbosity=0,
                    )
                    cv_mean, cv_std = _cv_sklearn(xgb_model, X_train, y_train, n_splits=3)
                    xgb_model.fit(X_train, y_train)
                    holdout_preds_xgb = xgb_model.predict(X_test)
                    train_r2 = float(r2_score(y_train, xgb_model.predict(X_train)))
                    t1 = time.time()
                    _record_candidate("xgboost", xgb_model, cv_mean, cv_std, train_r2, t1-t0,
                                     preds=holdout_preds_xgb)
                except Exception as e:
                    print(f"XGBoost failed: {e}")
                    model_history.append({"model_name": "xgboost", "status": "failed", "error": str(e)})
        except ImportError:
            print("XGBoost not installed, skipping")

        # ── Select best model (by CV R²) ──────────────────────────────────────
        # Prefer finite CV R² candidates; fall back to holdout R² if all CV are nan
        finite_cv = {
            name: info for name, info in candidate_results.items()
            if info["cv_mean"] is not None and np.isfinite(info["cv_mean"])
        }
        if finite_cv:
            best_name = max(finite_cv, key=lambda n: finite_cv[n]["cv_mean"])
        elif candidate_results:
            # Fall back: choose by holdout R²
            def _holdout_r2(name):
                preds = candidate_results[name].get("preds")
                if preds is not None:
                    p = np.asarray(preds, dtype=float)
                    finite = np.isfinite(p) & np.isfinite(y_test)
                    if finite.sum() > 0:
                        return float(r2_score(y_test[finite], p[finite]))
                return float("-inf")
            best_name = max(candidate_results, key=_holdout_r2)
        valid_candidates = candidate_results  # keep for downstream compat
        if valid_candidates:
            best_path = output_dir / f"candidate-{best_name}.joblib"
            if best_path.exists():
                best_model = joblib.load(best_path)
                joblib.dump(best_model, output_dir / "model.joblib")
                print(f"\nBest model (by CV R²): {best_name} → saved as model.joblib")
            else:
                # Use first available
                for name in candidate_results:
                    cpath = output_dir / f"candidate-{name}.joblib"
                    if cpath.exists():
                        joblib.dump(joblib.load(cpath), output_dir / "model.joblib")
                        best_name = name
                        print(f"Best model fallback: {name}")
                        break
        else:
            print("WARNING: No valid candidates — using persistence as fallback model")
            best_name = "naive_persistence"

        # Benchmark warning
        best_holdout_r2 = float("nan")
        if best_name in candidate_results and candidate_results[best_name]["preds"] is not None:
            best_preds = candidate_results[best_name]["preds"]
            best_holdout_r2 = float(r2_score(y_test, np.asarray(best_preds, dtype=float)))

        auto_arima_r2 = benchmarks["auto_arima_benchmark"].get("r2", float("nan"))
        ar1_r2 = benchmarks["ar1_benchmark"].get("r2", float("nan"))
        benchmark_warning = (
            not np.isnan(best_holdout_r2)
            and not np.isnan(auto_arima_r2)
            and not np.isnan(ar1_r2)
            and (best_holdout_r2 - auto_arima_r2 < 0.02 or best_holdout_r2 - ar1_r2 < 0.02)
        )

        # ── Build step output JSON ─────────────────────────────────────────────
        # Collect holdout predictions for all candidates
        holdout_preds_all = {}
        for name, info in candidate_results.items():
            if info.get("preds") is not None:
                holdout_preds_all[name] = [float(v) for v in info["preds"]]

        step_output = {
            "step": "13-model-training",
            "run_id": args.run_id,
            "best_model": best_name,
            "best_model_holdout_r2": float(best_holdout_r2) if np.isfinite(best_holdout_r2) else None,
            "benchmark_warning": benchmark_warning,
            "benchmarks": benchmarks,
            "model_history": model_history,
            "holdout_predictions": holdout_preds_all,
            "candidate_model_files": [
                str(output_dir / f"candidate-{name}.joblib")
                for name in candidate_results
                if (output_dir / f"candidate-{name}.joblib").exists()
            ],
            "artifacts": {
                "model_joblib": str(output_dir / "model.joblib"),
                "holdout_npz": str(output_dir / "holdout.npz"),
            },
            "features_used": valid_features,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "primary_seasonal_period": int(primary_period),
        }

        out_path = output_dir / "step-13-training.json"
        with open(out_path, "w") as f:
            json.dump(step_output, f, indent=2)
        print(f"Written: {out_path}")

        # Update progress
        with open(progress_path) as f:
            p = json.load(f)
        if "13-model-training" not in p.get("completed_steps", []):
            p.setdefault("completed_steps", []).append("13-model-training")
        p["status"] = "running"
        p["current_step"] = "13-model-training"
        p["model_history"] = model_history
        with open(progress_path, "w") as f:
            json.dump(p, f, indent=2)

        print("\nStep 13 complete.")
        sys.exit(0)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR in step 13: {e}\n{tb}", file=sys.stderr)
        if progress_path.exists():
            with open(progress_path) as f:
                p = json.load(f)
            p["status"] = "error"
            p.setdefault("errors", []).append({"step": "13-model-training", "error": str(e), "traceback": tb})
            with open(progress_path, "w") as f:
                json.dump(p, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
