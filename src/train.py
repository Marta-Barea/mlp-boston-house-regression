import os
import random
import numpy as np
import tensorflow as tf

from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import joblib

from src.data_loader import load_data
from src.build_model import build_model

from .config import (
    SEED,
    UNITS_LIST,
    EPOCHS_LIST,
    BATCH_SIZE_LIST,
    LEARNING_RATE_LIST,
    RANDOM_SEARCH_ITERATIONS,
    MODEL_OUTPUT_PATH,
    CV_FOLDS
)


def train_model():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X_train, X_test, y_train, y_test = load_data()

    default_units = UNITS_LIST[0] if UNITS_LIST else 8
    default_lr = LEARNING_RATE_LIST[0] if LEARNING_RATE_LIST else 0.001

    mlp_model = KerasRegressor(
        model=build_model,
        input_dim=X_train.shape[1],
        units=default_units,
        learning_rate=default_lr)

    param_distributions = {
        'model__units': UNITS_LIST,
        'epochs': EPOCHS_LIST,
        'batch_size': BATCH_SIZE_LIST,
        'model__learning_rate': LEARNING_RATE_LIST,
    }

    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    random_search = RandomizedSearchCV(
        mlp_model,
        param_distributions=param_distributions,
        n_iter=RANDOM_SEARCH_ITERATIONS,
        cv=CV_FOLDS,
        verbose=2,
        scoring=mae_scorer,
        random_state=SEED
    )

    regression_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', random_search)
    ])

    regression_pipeline.fit(X_train, y_train)

    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

    best_model_path = os.path.join(MODEL_OUTPUT_PATH, "best_mlp.pkl")
    joblib.dump(regression_pipeline, best_model_path)

    best_params_path = os.path.join(MODEL_OUTPUT_PATH, "best_params.txt")
    with open(best_params_path, "w") as f:
        f.write(str(regression_pipeline.named_steps['mlp'].best_params_))

    print("\n✅ Training completed.")
    print(f"   • Best model saved at: {best_model_path}")
    print(f"   • Best parameters saved at: {best_params_path}")
    print("\n🔍 Best parameters found:")
    for key, value in regression_pipeline.named_steps['mlp'].best_params_.items():
        print(f"     • {key}: {value}")


if __name__ == "__main__":
    train_model()
