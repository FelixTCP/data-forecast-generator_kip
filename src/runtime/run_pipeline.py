"""
Minimal runtime runner for a single-agent regression pipeline step.
- Loads CSV (env var CSV_PATH or default)
- Cleans minimal issues (drops NA)
- Automatic train/test split (SPLIT_MODE env or default 'auto')
- Trains a scikit-learn LinearRegression on numeric features
- Saves model (joblib) and metrics (JSON) to OUTPUT_DIR

This file is intentionally minimal and deterministic to satisfy contract-driven execution.
"""
import os
import json
from datetime import datetime

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Runtime configuration (can be overridden via environment variables)
CSV_PATH = os.environ.get("CSV_PATH", "./data/appliances_energy_prediction.csv")
TARGET_COLUMN = os.environ.get("TARGET_COLUMN", "appliances")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "artifacts/20260404T061230Z")
RUN_ID = os.environ.get("RUN_ID", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
SPLIT_MODE = os.environ.get("SPLIT_MODE", "auto")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_csv(path):
    df = pd.read_csv(path)
    return df


def prepare_data(df, target):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in CSV")
    # Drop rows with NA in features or target
    df = df.dropna()
    # Select numeric columns only (simple deterministic choice)
    numeric = df.select_dtypes(include=["number"]).copy()
    if target not in numeric.columns:
        raise ValueError(f"Target column '{target}' is not numeric")
    X = numeric.drop(columns=[target])
    y = numeric[target]
    if X.shape[1] == 0:
        raise ValueError("No numeric features available after dropping target")
    return X, y


def split_data(X, y, mode="auto"):
    # For 'auto', use 80/20 split. Other modes can be added later.
    test_size = 0.2
    return train_test_split(X, y, test_size=test_size, random_state=42)


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    return {"rmse": float(rmse), "r2": float(r2)}


def save_artifacts(model, metrics, output_dir, run_id):
    model_path = os.path.join(output_dir, f"model_{run_id}.joblib")
    metrics_path = os.path.join(output_dir, f"metrics_{run_id}.json")
    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return model_path, metrics_path


def main():
    print(f"Loading CSV from: {CSV_PATH}")
    df = load_csv(CSV_PATH)
    print(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")

    X, y = prepare_data(df, TARGET_COLUMN)
    print(f"Prepared data: {X.shape[0]} rows, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = split_data(X, y, SPLIT_MODE)
    print(f"Train/test split: {X_train.shape[0]} train, {X_test.shape[0]} test")

    model = train_model(X_train, y_train)
    print("Model training complete")

    metrics = evaluate_model(model, X_test, y_test)
    print(f"Evaluation metrics: {metrics}")

    model_path, metrics_path = save_artifacts(model, metrics, OUTPUT_DIR, RUN_ID)
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")

if __name__ == "__main__":
    main()
