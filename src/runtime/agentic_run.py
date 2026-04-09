import os
import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Minimal agentic runtime entrypoint for a single-agent regression pipeline

def main():
    # Runtime variables (fall back to defaults)
    CSV_PATH = os.environ.get("CSV_PATH", "./data/appliances_energy_prediction.csv")
    TARGET_COLUMN = os.environ.get("TARGET_COLUMN", "appliances")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "artifacts/run_output")
    RUN_ID = os.environ.get("RUN_ID", "run")

    out_dir = Path(OUTPUT_DIR) / RUN_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in CSV columns: {list(df.columns)}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Simple type coercion: drop non-numeric columns
    X = X.select_dtypes(include=["number"]).fillna(0)

    test_size = 0.2
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds, squared=False)),
        "mae": float(mean_absolute_error(y_test, preds)),
    }

    # Save model and metrics
    model_path = out_dir / "model.joblib"
    metrics_path = out_dir / "metrics.json"

    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Run complete. Artifacts saved to: {out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
