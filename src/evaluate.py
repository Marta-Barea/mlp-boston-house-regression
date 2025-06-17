import os
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

from src.data_loader import load_data
from src.config import MODEL_OUTPUT_PATH


def get_latest_model(directory: str | Path, pattern: str = "best_regressor_*.h5") -> str | None:
    files = sorted(
        Path(directory).glob(pattern),
        key=os.path.getmtime,
        reverse=True
    )
    return str(files[0]) if files else None


def evaluate_model(model_path: str | None = None):
    X_train, X_test, y_train, y_test = load_data()

    scaler = joblib.load(os.path.join(MODEL_OUTPUT_PATH, "scaler.save"))
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_path = model_path or get_latest_model(MODEL_OUTPUT_PATH)
    if model_path is None or not os.path.isfile(model_path):
        print("❌ Model file not found.")
        return

    print(f"🔄 Loading model: {Path(model_path).name}")
    model = tf.keras.models.load_model(model_path)

    y_train_pred = model.predict(X_train_scaled).flatten()
    y_test_pred = model.predict(X_test_scaled).flatten()

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"\n📊 Train MAE: {train_mae:.4f}")
    print(f"📊 Test MAE:  {test_mae:.4f}\n")

    print("🔎 First 10 predictions vs. actual values:")
    for pred, actual in zip(y_test_pred[:10], y_test[:10]):
        print(f"   • Predicted: {pred:.2f}, Actual: {actual:.2f}")


if __name__ == "__main__":
    evaluate_model()
