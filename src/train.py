import os
import datetime
import random
from pathlib import Path

import numpy as np
import joblib
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error

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
    CV_FOLDS,
    CHECKPOINTS_OUTPUT_PATH
)


def train_model():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X_train, _, y_train, _ = load_data()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_OUTPUT_PATH, "scaler.save"))

    mlp = KerasRegressor(model=build_model, input_dim=X_train.shape[1])

    param_dist = {
        "model__units": UNITS_LIST,
        "epochs": EPOCHS_LIST,
        "batch_size": BATCH_SIZE_LIST,
        "model__learning_rate": LEARNING_RATE_LIST,
    }

    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    random_search = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_dist,
        n_iter=RANDOM_SEARCH_ITERATIONS,
        cv=CV_FOLDS,
        scoring=mae_scorer,
        random_state=SEED,
        verbose=2
    )

    print("⏳ Starting RandomizedSearchCV on scaled data…")
    random_search.fit(X_train_scaled, y_train)
    best_params = random_search.best_params_
    print("🔍 Best hyperparameters:", best_params)

    run_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = Path(CHECKPOINTS_OUTPUT_PATH)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_val_mae_{run_ts}.weights.h5"

    final_model = build_model(
        input_dim=X_train.shape[1],
        units=best_params["model__units"],
        learning_rate=best_params["model__learning_rate"]
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path),
        monitor="val_mean_absolute_error",
        mode="min",
        save_weights_only=True,
        save_best_only=True,
        verbose=1
    )

    print("🚀 Training final models...")
    final_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        callbacks=[checkpoint_cb],
        verbose=1
    )

    final_model.load_weights(str(ckpt_path))

    out_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_out_dir = Path(MODEL_OUTPUT_PATH)
    model_out_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = model_out_dir / f"best_regressor_{out_ts}.h5"
    final_model.save(str(final_model_path))

    params_path = model_out_dir / f"best_params_{out_ts}.txt"
    with open(params_path, "w") as f:
        f.write(str(best_params))

    print("✅ Done.")
    print(
        f"   • Scaler saved at: {os.path.join(MODEL_OUTPUT_PATH, 'scaler.save')}")
    print(f"   • Model and weights: {final_model_path}")
    print(f"   • Hyperparameters:   {params_path}")


if __name__ == "__main__":
    train_model()
