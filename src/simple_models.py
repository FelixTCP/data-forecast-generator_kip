class MeanPredictor:
    def __init__(self):
        self.mean_ = None
    def fit(self, y):
        total = 0.0
        n = 0
        for v in y:
            total += v
            n += 1
        self.mean_ = total / n if n else 0.0
        return self
    def predict(self, X):
        return [self.mean_ for _ in range(len(X))]

class LastValuePredictor:
    def __init__(self):
        self.last_ = None
    def fit(self, y):
        if len(y):
            self.last_ = y[-1]
        else:
            self.last_ = 0.0
        return self
    def predict(self, X):
        return [self.last_ for _ in range(len(X))]
