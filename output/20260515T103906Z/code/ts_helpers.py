"""Shared helpers for time-series model wrappers.

Defines TimeSeriesPredictor — a sklearn-compatible wrapper that stores
pre-computed holdout forecasts from classical TS models (HoltWinters, ARIMA, etc.)
so that Step 14 can call .predict(X) uniformly on all candidates.

This module must NOT define classes under __main__.
"""


class TimeSeriesPredictor:
    """Sklearn-compatible wrapper for classical time-series models.

    Parameters
    ----------
    model : fitted statsmodels / pmdarima model
    holdout_preds : np.ndarray — pre-computed predictions for holdout horizon
    model_class : str — human-readable model class name
    cv_r2_scores : list[float] — backtest R² values (empty if not available)
    """

    def __init__(self, model, holdout_preds, model_class: str, cv_r2_scores=None):
        self.model = model
        self.holdout_preds = holdout_preds
        self.model_class = model_class
        self.cv_r2_scores = cv_r2_scores or []

    def predict(self, X=None):
        """Return stored holdout predictions (ignores X)."""
        import numpy as np
        return np.asarray(self.holdout_preds, dtype=float)

    def __repr__(self):
        return f"TimeSeriesPredictor(model_class={self.model_class!r})"
