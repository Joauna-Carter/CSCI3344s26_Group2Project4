"""
Random search for hidden layer sizes and epochs.
Each sampled configuration is trained/evaluated 10 times and ranked by average test accuracy.
Search evaluates a fixed number of random configurations.
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
MAX_CONFIGS = 10
RUNS_PER_CONFIG = 10
BATCH_SIZE = 32
HIDDEN_1_RANGE = (16, 100)
HIDDEN_2_RANGE = (8, 60)
EPOCHS_RANGE = (10, 50)

df = preprocess(pd.read_csv("diabetes.csv"))
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8, random_state=RANDOM_SEED
)

def build_model(h1: int, h2: int):
    return Sequential(
        [
            Input(shape=(8,)),
            Dense(h1, activation="relu"),
            Dense(h2, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )


def sample_config(rng: np.random.Generator):
    h1 = int(rng.integers(HIDDEN_1_RANGE[0], HIDDEN_1_RANGE[1] + 1))
    # Keep second hidden layer <= first hidden layer
    h2_max = min(HIDDEN_2_RANGE[1], h1)
    h2 = int(rng.integers(HIDDEN_2_RANGE[0], h2_max + 1))
    # Bias towards lower epoch counts so more configs train quickly.
    u = float(rng.random())
    epochs = int(EPOCHS_RANGE[0] + (EPOCHS_RANGE[1] - EPOCHS_RANGE[0]) * (u**2))
    epochs = max(EPOCHS_RANGE[0], min(EPOCHS_RANGE[1], epochs))
    return h1, h2, epochs


def evaluate_config(h1: int, h2: int, epochs: int):
    accuracies = []
    losses = []
    for run_idx in range(RUNS_PER_CONFIG):
        run_seed = RANDOM_SEED + run_idx
        set_random_seed(run_seed)
        np.random.seed(run_seed)
        model = build_model(h1, h2)
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=0,
        )
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        losses.append(float(loss))
        accuracies.append(float(acc))
    return float(np.mean(losses)), float(np.mean(accuracies))


rng = np.random.default_rng(RANDOM_SEED)
seen_configs = set()
best = None
best_acc = -1.0
best_loss = float("inf")

evaluations = 0

while evaluations < MAX_CONFIGS:
    config = sample_config(rng)
    if config in seen_configs:
        continue
    seen_configs.add(config)
    h1, h2, epochs = config

    avg_loss, avg_acc = evaluate_config(h1, h2, epochs)
    evaluations += 1

    print(
        f"[{evaluations:02d}] h1={h1:3d} h2={h2:3d} epochs={epochs:3d} -> "
        f"avg_test_acc={avg_acc:.4f} avg_test_loss={avg_loss:.4f} "
        f"(progress={evaluations}/{MAX_CONFIGS})"
    )

    if avg_acc > best_acc or (avg_acc == best_acc and avg_loss < best_loss):
        best_acc = avg_acc
        best_loss = avg_loss
        best = (h1, h2, epochs, avg_loss, avg_acc)

print(f"\nCompleted {evaluations} configurations.")
print("Best:", best)
