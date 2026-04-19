#!/usr/bin/env python3
"""Step 14: evaluate candidates, run leakage stress tests, trigger expansion if needed."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

STEP_NAME = "14-model-evaluation"
STEP12_JSON = "step-12-features.json"
STEP13_JSON = "step-13-training.json"
STEP14_JSON = "step-14-evaluation.json"
PROGRESS_FILE = "progress.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _update_progress(
    output_dir: Path,
    *,
    run_id: str,
    csv_path: str,
    target_column: str,
    status: str,
    current_step: str,
    error: str | None = None,
    mark_completed: bool = False,
) -> None:
    path = output_dir / PROGRESS_FILE
    progress = _load_json(path)
    if not progress:
        progress = {
            "run_id": run_id,
            "csv_path": csv_path,
            "target_column": target_column,
            "status": "running",
            "current_step": current_step,
            "completed_steps": [],
            "errors": [],
        }

    progress["run_id"] = run_id
    progress["csv_path"] = csv_path
    progress["target_column"] = target_column
    progress["status"] = status
    progress["current_step"] = current_step

    completed = progress.setdefault("completed_steps", [])
    if mark_completed and current_step not in completed:
        completed.append(current_step)

    if error:
        progress.setdefault("errors", []).append({"step": current_step, "error": error})

    _write_json(path, progress)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "residual_mean": float(np.mean(residuals)),
        "residual_max_abs": float(np.max(np.abs(residuals))),
    }


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if np.any(y_true == 0):
        return None
    return float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100.0


def _split_for_probe(
    X: np.ndarray,
    y: np.ndarray,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if split_mode == "time_series":
        split_idx = max(int(len(X) * 0.8), 1)
        split_idx = min(split_idx, len(X) - 1)
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )
    return X_train, X_test, y_train, y_test


def _run_leakage_probe(output_dir: Path, split_mode: str) -> dict[str, Any]:
    step12 = _load_json(output_dir / STEP12_JSON)
    if not step12:
        return {"triggered": True, "status": "fail", "details": ["Missing step-12-features.json"]}

    features_path = Path(step12.get("artifacts", {}).get("features_parquet", ""))
    if not features_path.exists():
        return {"triggered": True, "status": "fail", "details": ["Missing features.parquet"]}

    context = step12.get("context", {})
    target = context.get("target_column")
    features = step12.get("features", [])
    if not target or target not in pl.read_parquet(features_path, n_rows=1).columns:
        return {"triggered": True, "status": "fail", "details": ["Target unavailable for leakage probe"]}

    df = pl.read_parquet(features_path)
    if target not in df.columns:
        return {"triggered": True, "status": "fail", "details": ["Target missing in feature matrix"]}

    target_derived = [f for f in features if f.startswith("target_") and f in df.columns]
    rolling_target = [f for f in target_derived if "roll" in f]

    details: list[dict[str, Any]] = []
    y = df[target].cast(pl.Float64).to_numpy()

    def evaluate_subset(subset: list[str], label: str) -> None:
        if len(subset) < 1:
            details.append({"probe": label, "status": "skip", "reason": "no_features"})
            return
        X = df.select(subset).to_numpy()
        X_train, X_test, y_train, y_test = _split_for_probe(X, y, split_mode)
        model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        met = _metrics(y_test, pred)
        details.append({"probe": label, "status": "ok", **met})

    evaluate_subset([f for f in features if f not in target_derived], "remove_target_derived_features")
    evaluate_subset([f for f in features if f not in rolling_target], "remove_target_rolling_features")

    if target_derived:
        X_td = df.select(target_derived).to_numpy()
        X_train, X_test, y_train, y_test = _split_for_probe(X_td, y, split_mode)
        probe = LinearRegression()
        probe.fit(X_train, y_train)
        pred = probe.predict(X_test)
        r2 = float(r2_score(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        lin_record = {
            "probe": "linear_reconstruction_target_derived_only",
            "status": "ok",
            "r2": r2,
            "rmse": rmse,
            "feature_count": len(target_derived),
        }
        details.append(lin_record)
        if r2 > 0.995 or rmse < 1e-6:
            return {
                "triggered": True,
                "status": "fail",
                "details": details,
                "reason": "linear probe shows near-perfect reconstruction",
            }

    return {
        "triggered": True,
        "status": "pass",
        "details": details,
    }


def _run_expansion_candidates(
    output_dir: Path,
    split_mode: str,
) -> list[dict[str, Any]]:
    step12 = _load_json(output_dir / STEP12_JSON)
    features_path = Path(step12["artifacts"]["features_parquet"])
    context = step12.get("context", {})
    target = context.get("target_column")
    features = step12.get("features", [])

    df = pl.read_parquet(features_path)
    X = df.select(features).to_numpy()
    y = df[target].cast(pl.Float64).to_numpy()

    X_train, X_test, y_train, y_test = _split_for_probe(X, y, split_mode)
    cv = TimeSeriesSplit(n_splits=5) if split_mode == "time_series" else KFold(n_splits=5, shuffle=True, random_state=42)

    candidate_defs: list[tuple[str, Any]] = [
        (
            "elastic_net",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42, max_iter=5000)),
                ]
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(random_state=42),
        ),
        (
            "svr_rbf",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")),
                ]
            ),
        ),
    ]

    records: list[dict[str, Any]] = []
    for name, estimator in candidate_defs:
        try:
            cv_scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_test)
            met = _metrics(y_test, pred)
            mape = _mape(y_test, pred)
            path = (output_dir / f"candidate-{name}.joblib").resolve()
            joblib.dump(estimator, path)
            rec: dict[str, Any] = {
                "model_name": name,
                "status": "ok",
                "cv_mean_r2": float(np.mean(cv_scores)),
                "cv_std_r2": float(np.std(cv_scores)),
                **met,
                "artifact": str(path),
            }
            if mape is not None:
                rec["mape"] = mape
            records.append(rec)
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "model_name": name,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return records


def _quality_from_r2(r2: float) -> str:
    if r2 >= 0.50:
        return "acceptable"
    if r2 >= 0.25:
        return "marginal"
    return "subpar"


def _diagnose_subpar(candidates: list[dict[str, Any]], target_stats: dict[str, float]) -> str:
    notes: list[str] = []
    ok = [c for c in candidates if c.get("status", "ok") == "ok"]
    if not ok:
        return "No successful candidates were available for diagnostics."

    best = max(ok, key=lambda c: c["r2"])
    best_cv = float(best.get("cv_mean_r2", 0.0))
    best_r2 = float(best.get("r2", 0.0))
    best_rmse = float(best.get("rmse", 0.0))

    if best_rmse > target_stats["std"]:
        notes.append("RMSE exceeds target standard deviation, suggesting weak learning signal.")
    if best_cv < 0.10:
        notes.append("Cross-validation R2 is near zero, indicating uninformative features.")
    if best_cv - best_r2 > 0.20:
        notes.append("Holdout performance is materially below CV, indicating potential overfitting or split mismatch.")

    mean = target_stats["mean"]
    std = max(target_stats["std"], 1e-9)
    # Simple skew proxy without scipy dependency.
    skew_proxy = abs((mean - target_stats.get("median", mean)) / std)
    if skew_proxy > 1.0:
        notes.append("Target appears skewed; consider log-transforming the target.")

    if not notes:
        notes.append("Subpar performance likely due to weak feature-target relationship under current split strategy.")

    return " ".join(notes)


def run_step(output_dir: Path, run_id: str) -> tuple[dict[str, Any], bool, str | None]:
    step13 = _load_json(output_dir / STEP13_JSON)
    if not step13:
        raise FileNotFoundError("Missing step-13-training.json.")

    holdout_path = Path(step13["artifacts"]["holdout_npz"])
    if not holdout_path.exists():
        raise FileNotFoundError(f"Missing holdout npz at {holdout_path}")

    holdout = np.load(holdout_path, allow_pickle=True)
    X_test = holdout["X_test"]
    y_test = holdout["y_test"]

    split_mode = step13.get("split_strategy", {}).get("resolved_mode", "random")

    candidate_refs = [c for c in step13.get("candidates", []) if c.get("status") == "ok"]
    if not candidate_refs:
        raise ValueError("No successful candidates to evaluate.")

    evaluated: list[dict[str, Any]] = []
    for item in candidate_refs:
        name = item["model_name"]
        artifact = item.get("artifact")
        if not artifact:
            continue
        model_path = Path(artifact)
        if not model_path.exists():
            continue

        model = joblib.load(model_path)
        pred = model.predict(X_test)
        met = _metrics(y_test, pred)
        mape = _mape(y_test, pred)

        rec: dict[str, Any] = {
            "model_name": name,
            **met,
            "cv_mean_r2": float(item.get("cv_mean_r2", 0.0)),
            "cv_std_r2": float(item.get("cv_std_r2", 0.0)),
            "model_worse_than_mean_baseline": bool(met["r2"] < 0.0),
            "artifact": str(model_path.resolve()),
        }
        if mape is not None:
            rec["mape"] = mape
        evaluated.append(rec)

    if not evaluated:
        raise ValueError("No candidate models could be loaded for evaluation.")

    if len(y_test) >= 2:
        naive_pred = np.empty_like(y_test)
        naive_pred[0] = y_test[0]
        naive_pred[1:] = y_test[:-1]
    else:
        naive_pred = np.array([float(np.mean(y_test))])

    naive_metrics = _metrics(y_test, naive_pred)

    target_stats = {
        "mean": float(np.mean(y_test)),
        "median": float(np.median(y_test)),
        "std": float(np.std(y_test)),
        "min": float(np.min(y_test)),
        "max": float(np.max(y_test)),
    }

    best = max(evaluated, key=lambda c: c["r2"])
    quality_assessment = _quality_from_r2(float(best["r2"]))

    leakage_probe = {
        "triggered": False,
        "status": "not_triggered",
        "details": [],
    }

    if any(float(c["r2"]) > 0.98 for c in evaluated):
        leakage_probe = _run_leakage_probe(output_dir, split_mode)
        if leakage_probe["status"] != "pass":
            payload = {
                "step": STEP_NAME,
                "target_stats": target_stats,
                "naive_baseline": naive_metrics,
                "candidates": evaluated,
                "quality_assessment": "leakage_suspected",
                "leakage_probe": leakage_probe,
                "context": step13.get("context", {}),
            }
            return payload, True, "Leakage probe failed in step 14"

    expansion_candidates: list[dict[str, Any]] = []
    expansion_diagnosis = ""

    if quality_assessment == "subpar":
        _update_progress(
            output_dir,
            run_id=run_id,
            csv_path=_load_json(output_dir / PROGRESS_FILE).get("csv_path", ""),
            target_column=_load_json(output_dir / PROGRESS_FILE).get("target_column", ""),
            status="expansion_required",
            current_step=STEP_NAME,
        )
        expansion_diagnosis = _diagnose_subpar(evaluated, target_stats)
        expansion_candidates = _run_expansion_candidates(output_dir, split_mode)

        ok_expansion = [c for c in expansion_candidates if c.get("status") == "ok"]
        combined = evaluated + ok_expansion
        best_combined = max(combined, key=lambda c: c["r2"]) if combined else best

        if best_combined["r2"] >= 0.50:
            quality_assessment = "acceptable"
        elif best_combined["r2"] >= 0.25:
            quality_assessment = "marginal"
        else:
            quality_assessment = "subpar_after_expansion"

    payload = {
        "step": STEP_NAME,
        "target_stats": target_stats,
        "naive_baseline": naive_metrics,
        "candidates": evaluated,
        "quality_assessment": quality_assessment,
        "expansion_diagnosis": expansion_diagnosis,
        "expansion_candidates": expansion_candidates,
        "leakage_probe": leakage_probe,
        "context": step13.get("context", {}),
    }

    return payload, False, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 14 model evaluation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    progress = _load_json(output_dir / PROGRESS_FILE)
    csv_path = progress.get("csv_path", "")
    target = progress.get("target_column", "")

    _update_progress(
        output_dir,
        run_id=args.run_id,
        csv_path=csv_path,
        target_column=target,
        status="running",
        current_step=STEP_NAME,
    )

    try:
        payload, hard_stop, reason = run_step(output_dir=output_dir, run_id=args.run_id)
        _write_json(output_dir / STEP14_JSON, payload)

        if hard_stop:
            raise RuntimeError(reason or "Hard stop in step 14")

        _update_progress(
            output_dir,
            run_id=args.run_id,
            csv_path=csv_path,
            target_column=target,
            status="running",
            current_step=STEP_NAME,
            mark_completed=True,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        err = f"{exc}\n{traceback.format_exc()}"
        _update_progress(
            output_dir,
            run_id=args.run_id,
            csv_path=csv_path,
            target_column=target,
            status="failed",
            current_step=STEP_NAME,
            error=err,
        )
        print(err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
