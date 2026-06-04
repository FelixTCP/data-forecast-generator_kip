#!/usr/bin/env python
"""
STEP 13 — Model Training

Train candidate models (Ridge, RandomForest, GradientBoosting, XGBoost).
Split chronologically if time-series, else randomly.
Save best model as model.joblib and holdout arrays as holdout.npz.

Exit code: 0=success, non-zero=failure
"""

import sys
import json
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_step12(output_dir: Path) -> dict:
    return json.loads((output_dir / "step-12-features.json").read_text())

def load_step11(output_dir: Path) -> dict:
    return json.loads((output_dir / "step-11-exploration.json").read_text())

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R², RMSE, MAE."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "residual_mean": float(np.mean(y_true - y_pred)),
        "residual_max_abs": float(np.max(np.abs(y_true - y_pred)))
    }

def compute_cv_score(model, X_train: np.ndarray, y_train: np.ndarray, cv_folds: int = 5) -> dict:
    """Compute cross-validation R² score."""
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    
    cv = TimeSeriesSplit(n_splits=cv_folds)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    
    return {
        "cv_mean": float(np.mean(scores)),
        "cv_std": float(np.std(scores)),
        "cv_scores": [float(s) for s in scores]
    }

def train_ridge(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """Train Ridge regression."""
    from sklearn.linear_model import Ridge
    
    logger.info("Training Ridge...")
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = compute_metrics(y_train, y_pred_train)
    cv_score = compute_cv_score(model, X_train, y_train)
    
    return model, y_pred_test, train_metrics, cv_score

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """Train Random Forest."""
    from sklearn.ensemble import RandomForestRegressor
    
    logger.info("Training Random Forest...")
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = compute_metrics(y_train, y_pred_train)
    cv_score = compute_cv_score(model, X_train, y_train)
    
    return model, y_pred_test, train_metrics, cv_score

def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """Train Gradient Boosting."""
    from sklearn.ensemble import GradientBoostingRegressor
    
    logger.info("Training Gradient Boosting...")
    model = GradientBoostingRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = compute_metrics(y_train, y_pred_train)
    cv_score = compute_cv_score(model, X_train, y_train)
    
    return model, y_pred_test, train_metrics, cv_score

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """Train XGBoost if available."""
    try:
        import xgboost as xgb
        
        logger.info("Training XGBoost...")
        model = xgb.XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_metrics = compute_metrics(y_train, y_pred_train)
        cv_score = compute_cv_score(model, X_train, y_train)
        
        return model, y_pred_test, train_metrics, cv_score
    except ImportError:
        logger.warning("XGBoost not available, skipping")
        return None, None, None, None

def compute_benchmarks(y_train: np.ndarray, y_test: np.ndarray, primary_seasonal_period: int = None) -> dict:
    """Compute benchmark predictions."""
    benchmarks = {}
    
    # 1. Naive persistence
    y_pred_naive = np.concatenate([[y_train[-1]], y_test[:-1]])
    benchmarks["naive_persistence"] = compute_metrics(y_test, y_pred_naive)
    
    # 2. Seasonal naive
    if primary_seasonal_period is None or primary_seasonal_period < 1:
        primary_seasonal_period = 1
    
    y_full = np.concatenate([y_train, y_test])
    try:
        y_pred_seasonal = np.array([y_full[max(0, i - primary_seasonal_period)] for i in range(len(y_train), len(y_full))])
        benchmarks["seasonal_naive"] = compute_metrics(y_test, y_pred_seasonal)
        benchmarks["seasonal_naive"]["seasonal_period"] = primary_seasonal_period
    except:
        benchmarks["seasonal_naive"] = benchmarks["naive_persistence"].copy()
        benchmarks["seasonal_naive"]["seasonal_period"] = 1
    
    # 3. AR(1) from statsmodels
    try:
        from statsmodels.tsa.ar_model import AutoReg
        ar1 = AutoReg(y_train, lags=1, old_names=False).fit()
        start_idx = len(y_train)
        end_idx = len(y_train) + len(y_test) - 1
        y_pred_ar1 = np.asarray(ar1.predict(start=start_idx, end=end_idx), dtype=float)
        benchmarks["ar1_benchmark"] = compute_metrics(y_test, y_pred_ar1)
        benchmarks["ar1_benchmark"]["lags"] = 1
    except Exception as e:
        logger.warning(f"AR(1) failed: {e}")
        y_pred_ar1 = np.concatenate([[y_train[-1]], y_test[:-1]])
        benchmarks["ar1_benchmark"] = compute_metrics(y_test, y_pred_ar1)
        benchmarks["ar1_benchmark"]["lags"] = 1
    
    return benchmarks

def main():
    parser = argparse.ArgumentParser(description="STEP 13 — Model Training")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--run-id", required=True, help="Run ID")
    
    args = parser.parse_args()
    
    try:
        output_dir = Path(args.output_dir)
        
        # Load metadata
        step12 = load_step12(output_dir)
        step11 = load_step11(output_dir)
        
        target_col = step12["target_column"]
        split_mode = step12["split_strategy"]["resolved_mode"]
        time_col = step12["split_strategy"].get("time_column")
        
        primary_seasonal_period = step11.get("ts_diagnostics", {}).get("primary_seasonal_period")
        
        logger.info(f"Loading features from features.parquet...")
        
        # Load features
        import polars as pl
        df = pl.read_parquet(output_dir / "features.parquet")
        df_pd = df.to_pandas()
        
        logger.info(f"Loaded {df_pd.shape[0]} rows x {df_pd.shape[1]} columns")
        
        # Split into X and y
        X = df_pd.drop(columns=[target_col])
        y = df_pd[target_col].values
        
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Split train/test
        if split_mode == "time_series":
            # Chronological split at 80%
            split_idx = int(0.8 * len(X))
            X_train = X.iloc[:split_idx].values
            y_train = y[:split_idx]
            X_test = X.iloc[split_idx:].values
            y_test = y[split_idx:]
            logger.info(f"Time-series split: train={len(X_train)}, test={len(X_test)}")
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )
            logger.info(f"Random split: train={len(X_train)}, test={len(X_test)}")
        
        # Compute benchmarks
        logger.info("Computing benchmarks...")
        benchmarks = compute_benchmarks(y_train, y_test, primary_seasonal_period)
        
        # Train models
        candidates = {}
        
        # Ridge
        try:
            model, y_pred, train_metrics, cv_score = train_ridge(X_train, y_train, X_test)
            test_metrics = compute_metrics(y_test, y_pred)
            candidates["ridge"] = {
                "model": model,
                "y_pred": y_pred,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "cv_score": cv_score
            }
            joblib.dump(model, output_dir / "candidate-ridge.joblib")
        except Exception as e:
            logger.error(f"Ridge training failed: {e}")
        
        # Random Forest
        try:
            model, y_pred, train_metrics, cv_score = train_random_forest(X_train, y_train, X_test)
            test_metrics = compute_metrics(y_test, y_pred)
            candidates["random_forest"] = {
                "model": model,
                "y_pred": y_pred,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "cv_score": cv_score
            }
            joblib.dump(model, output_dir / "candidate-random_forest.joblib")
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
        
        # Gradient Boosting
        try:
            model, y_pred, train_metrics, cv_score = train_gradient_boosting(X_train, y_train, X_test)
            test_metrics = compute_metrics(y_test, y_pred)
            candidates["gradient_boosting"] = {
                "model": model,
                "y_pred": y_pred,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "cv_score": cv_score
            }
            joblib.dump(model, output_dir / "candidate-gradient_boosting.joblib")
        except Exception as e:
            logger.error(f"Gradient Boosting training failed: {e}")
        
        # XGBoost (optional)
        try:
            model, y_pred, train_metrics, cv_score = train_xgboost(X_train, y_train, X_test)
            if model is not None:
                test_metrics = compute_metrics(y_test, y_pred)
                candidates["xgboost"] = {
                    "model": model,
                    "y_pred": y_pred,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "cv_score": cv_score
                }
                joblib.dump(model, output_dir / "candidate-xgboost.joblib")
        except Exception as e:
            logger.warning(f"XGBoost training failed: {e}")
        
        if not candidates:
            raise RuntimeError("No models trained successfully")
        
        logger.info(f"Successfully trained {len(candidates)} models")
        
        # Find best model by test R²
        best_name = max(candidates.keys(), key=lambda k: candidates[k]["test_metrics"]["r2"])
        best_model = candidates[best_name]["model"]
        
        logger.info(f"Best model: {best_name} (R²={candidates[best_name]['test_metrics']['r2']:.4f})")
        
        # Save best model as model.joblib
        joblib.dump(best_model, output_dir / "model.joblib")
        
        # Save holdout arrays
        np.savez(output_dir / "holdout.npz", X_test=X_test, y_test=y_test)
        
        # Build output JSON
        candidates_json = []
        for name, data in candidates.items():
            candidates_json.append({
                "name": name,
                "train_r2": data["train_metrics"]["r2"],
                "train_rmse": data["train_metrics"]["rmse"],
                "train_mae": data["train_metrics"]["mae"],
                "r2": data["test_metrics"]["r2"],
                "rmse": data["test_metrics"]["rmse"],
                "mae": data["test_metrics"]["mae"],
                "cv_mean": data["cv_score"]["cv_mean"],
                "cv_std": data["cv_score"]["cv_std"],
                "model_worse_than_mean_baseline": data["test_metrics"]["r2"] < 0
            })
        
        benchmarks_json = []
        for name, metrics in benchmarks.items():
            benchmarks_json.append({
                "name": name,
                "r2": metrics.get("r2", None),
                "rmse": metrics.get("rmse", None),
                "mae": metrics.get("mae", None),
                **{k: v for k, v in metrics.items() if k not in ["r2", "rmse", "mae"]}
            })
        
        output = {
            "step": "13-model-training",
            "run_id": args.run_id,
            "split_mode": split_mode,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_count": X_train.shape[1],
            "model": {
                "name": best_name,
                "r2": candidates[best_name]["test_metrics"]["r2"],
                "rmse": candidates[best_name]["test_metrics"]["rmse"],
                "mae": candidates[best_name]["test_metrics"]["mae"],
                "joblib": str(output_dir / "model.joblib")
            },
            "candidates": candidates_json,
            "benchmarks": benchmarks_json,
            "artifacts": {
                "model_joblib": str(output_dir / "model.joblib"),
                "holdout_npz": str(output_dir / "holdout.npz"),
                "candidates": {
                    name: str(output_dir / f"candidate-{name}.joblib")
                    for name in candidates.keys()
                }
            }
        }
        
        # Write output JSON
        (output_dir / "step-13-training.json").write_text(json.dumps(output, indent=2))
        logger.info("Wrote step-13-training.json")
        
        # Update progress
        progress_file = output_dir / "progress.json"
        progress = json.loads(progress_file.read_text())
        progress["status"] = "completed"
        progress["completed_steps"].append("13-model-training")
        progress["current_step"] = "14-model-evaluation"
        progress_file.write_text(json.dumps(progress, indent=2))
        
        logger.info("STEP 13 completed successfully")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"STEP 13 failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
