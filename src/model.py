"""Model wrapper: trains a scikit-learn regressor and provides save/load helpers.
Registry is a simple filesystem folder containing model.joblib and metadata.json
"""
import joblib, json, os
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class SimpleRegressor:
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, os.path.join(path, 'model.joblib'))

    def load(self, path):
        self.model = joblib.load(os.path.join(path, 'model.joblib'))

    def save_metadata(self, path, meta):
        with open(os.path.join(path, 'metadata.json'),'w') as f:
            json.dump(meta, f, indent=2)
