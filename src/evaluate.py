import os
import joblib
from sklearn.metrics import mean_absolute_error

from .data_loader import load_data
from .config import MODEL_OUTPUT_PATH


def evaluate_model():
    X_train, X_test, y_train, y_test = load_data()
    model_path = os.path.join(MODEL_OUTPUT_PATH, "best_mlp.pkl")

    if not os.path.isfile(model_path):
        print("❌ Model not found. Make sure to run src/train.py first.")
        return

    pipeline = joblib.load(model_path)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"\n📊Train MAE: {train_mae:.4f}")
    print(f"📊Test MAE: {test_mae:.4f}")

    print("\n🔎 First 10 predictions vs. actual values:")
    for i in range(10):
        print(
            f"   • Predicted: {int(y_test_pred[i])}, Actual: {int(y_test[i])}")


if __name__ == "__main__":
    evaluate_model()
