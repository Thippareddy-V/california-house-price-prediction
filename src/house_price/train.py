import json
import os

import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from .config import (
    MODEL_DIR, MODEL_PATH, FEATURE_NAMES_PATH, METRICS_PATH,
    RANDOM_STATE, TEST_SIZE, FEATURE_NAMES, XGB_PARAMS,
)
from .data import load_data, get_X_y
from .evaluate import compute_metrics


def train(params: dict | None = None):
    df = load_data()
    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model_params = {**XGB_PARAMS, **(params or {})}
    model = XGBRegressor(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    print("Test metrics:", metrics)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(FEATURE_NAMES, FEATURE_NAMES_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    return model, metrics


if __name__ == "__main__":
    train()