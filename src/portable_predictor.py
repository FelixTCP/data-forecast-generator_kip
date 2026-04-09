class MeanPredictor:
    """Simple importable predictor that fits the training targets' mean and predicts it for any X.

    This avoids binary dependencies and is fully importable from src.portable_predictor.
    """
    def __init__(self):
        self._mean = None

    def fit(self, X, y):
        # expect y as list-like of numbers
        self._mean = float(sum(y) / len(y)) if len(y) > 0 else 0.0
        return self

    def predict(self, X):
        # return list of means with length equal to number of rows in X
        try:
            n = len(X)
        except Exception:
            # single sample
            return [self._mean]
        return [self._mean for _ in range(n)]
