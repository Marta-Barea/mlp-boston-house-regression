import os
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATA_PATH, SEED, TEST_SIZE


def load_data():
    boston_house_data = pd.read_csv(
        DATA_PATH, delim_whitespace=True, header=None)
    arr = boston_house_data.values

    X = arr[:, :-1]
    y = arr[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    return X_train, X_test, y_train, y_test
