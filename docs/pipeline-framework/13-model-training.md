# #13 Context Engineering: Model Training

This is the core step after model preselection.

## Objective

Train candidate models reproducibly, log full training history, and output artifacts for downstream evaluation/selection.

## Inputs

- `X`, `y`
- preprocessor
- model candidate definitions
- training config (`split`, `cv`, `search`, `cache`, `random_state`)

## Outputs

- trained candidate artifacts
- best estimator per candidate family
- training history (`params`, CV scores, fit time)
- updated `PipelineContext.metrics` and `artifacts`

## Recommended MVP Defaults

- Split mode:
  - `TimeSeriesSplit` if time dependency exists (which it should in most cases!)
  - otherwise `train_test_split(shuffle=True, random_state=42)`
- CV: 5 folds (or `TimeSeriesSplit(n_splits=5)`)
- Search: `RandomizedSearchCV` for non-trivial spaces, `GridSearchCV` for small spaces
- Caching: use `joblib.Memory` for sklearn pipeline caching
- Reproducibility: fixed `random_state` on split and models

## Guardrails

- No single-metric winner; track at least `r2`, `rmse`, `mae`.
- Persist search space + selected params for reproducibility.
- Log warnings if variance across folds is high.
- Do not mix temporal and random split logic.

## Copilot Prompt Snippet (High-Value)

```markdown
Implement `train_models(X, y, preprocessor, config) -> dict` in `src/pipeline/regressor.py`.
Requirements:

1. support split mode: `random`, `time_series`
2. support optional hyperparameter search per candidate model
3. log training history table with: model_name, params, cv_mean, cv_std, fit_time_sec
4. cache sklearn pipeline transformations via `joblib.Memory`
5. return serializable output containing best_model_name, best_params, and per-candidate metrics
6. include unit tests for split strategy selection and deterministic results with fixed seeds
```

## Reference Architecture

```python
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
import joblib


@dataclass
class CandidateResult:
    model_name: str
    best_params: dict
    cv_mean: float
    cv_std: float
    fit_time_sec: float


def _make_cv(split_mode: str, n_splits: int, random_state: int):
    if split_mode == "time_series":
        return TimeSeriesSplit(n_splits=n_splits)
    if split_mode == "random":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    raise ValueError(f"Unknown split_mode={split_mode}")


def train_models(X, y, preprocessor, config: dict) -> dict:
    random_state = int(config.get("random_state", 42))
    split_mode = config.get("split_mode", "random")
    n_splits = int(config.get("n_splits", 5))

    cache_dir = Path(config.get("cache_dir", ".cache/sklearn"))
    memory = joblib.Memory(location=cache_dir, verbose=0)

    cv = _make_cv(split_mode=split_mode, n_splits=n_splits, random_state=random_state)

    history: list[CandidateResult] = []
    trained_estimators: dict[str, object] = {}

    for candidate in config["candidates"]:
        model_name = candidate["name"]
        estimator = candidate["estimator"]
        param_grid = candidate.get("param_grid", {})
        search_type = candidate.get("search", "none")

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", estimator),
        ], memory=memory)

        start = perf_counter()
        if search_type == "grid" and param_grid:
            search = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring="r2", n_jobs=-1)
            search.fit(X, y)
            best_estimator = search.best_estimator_
            best_params = search.best_params_
            best_score = float(search.best_score_)
            cv_std = float(np.std(search.cv_results_["mean_test_score"]))
        elif search_type == "random" and param_grid:
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_grid,
                n_iter=int(candidate.get("n_iter", 20)),
                cv=cv,
                scoring="r2",
                n_jobs=-1,
                random_state=random_state,
            )
            search.fit(X, y)
            best_estimator = search.best_estimator_
            best_params = search.best_params_
            best_score = float(search.best_score_)
            cv_std = float(np.std(search.cv_results_["mean_test_score"]))
        else:
            # fit baseline without search
            best_estimator = pipe.fit(X, y)
            best_params = {}
            scores = []
            for tr_idx, va_idx in cv.split(X):
                Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
                ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
                fitted = pipe.fit(Xtr, ytr)
                scores.append(float(fitted.score(Xva, yva)))
            best_score = float(np.mean(scores))
            cv_std = float(np.std(scores))

        fit_time = perf_counter() - start
        trained_estimators[model_name] = best_estimator
        history.append(CandidateResult(model_name, best_params, best_score, cv_std, fit_time))

    best = max(history, key=lambda r: r.cv_mean)
    return {
        "best_model_name": best.model_name,
        "best_params": best.best_params,
        "history": [asdict(h) for h in history],
        "estimators": trained_estimators,
    }
```

## Training History Schema (for logs/artifacts)

```json
{
  "run_id": "2026-04-03T15:00:00Z",
  "split_mode": "random",
  "random_state": 42,
  "records": [
    {
      "model_name": "ridge",
      "best_params": { "model__alpha": 1.0 },
      "cv_mean": 0.81,
      "cv_std": 0.03,
      "fit_time_sec": 1.28
    }
  ]
}
```

## Test Matrix for #13

- split strategy selection (`random` vs `time_series`)
- deterministic scores when seed fixed
- search strategy path (`none/grid/random`)
- history serialization
- fallback behavior if one candidate fails (explicitly logged + failed candidate marked)
