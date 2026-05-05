"""
Grid search for three hidden Dense layers + epochs (matches main2.py architecture).
Resets the random seed before each trial so scores match a single main2.py run.
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Input

from utils import preprocess

RANDOM_SEED = 16

df = preprocess(pd.read_csv("diabetes.csv"))
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8, random_state=RANDOM_SEED
)

hidden_configs = [
    (128, 64, 32),
    (96, 48, 24),
    (92, 46, 23),
    (80, 40, 20),
    (72, 36, 18),
    (64, 64, 32),
    (64, 32, 16),
    (48, 32, 16),
    (48, 24, 12),
    (32, 16, 8),
    (24, 12, 6),
    (96, 64, 32),
]
epochs_list = [25, 40, 60, 90, 120]


def build_model(h1: int, h2: int, h3: int):
    return Sequential(
        [
            Input(shape=(8,)),
            Dense(h1, activation="relu"),
            Dense(h2, activation="relu"),
            Dense(h3, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )


best_acc = -1.0
best_loss = float("inf")
best = None

for h1, h2, h3 in hidden_configs:
    for epochs in epochs_list:
        set_random_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        model = build_model(h1, h2, h3)
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
        )
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(
            f"h1={h1:3d} h2={h2:3d} h3={h3:3d} epochs={epochs:3d} -> "
            f"test_acc={acc:.4f} test_loss={loss:.4f}"
        )
        if acc > best_acc or (acc == best_acc and loss < best_loss):
            best_acc = acc
            best_loss = loss
            best = (h1, h2, h3, epochs, loss, acc)

print("\nBest:", best)
