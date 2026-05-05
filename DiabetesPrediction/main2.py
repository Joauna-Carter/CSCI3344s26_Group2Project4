import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from keras.utils import set_random_seed

RANDOM_SEED = 16

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input
from utils import preprocess

# Optimal settings from tune_hyperparams2.py (same seed/split as tuner).
HIDDEN_1, HIDDEN_2, HIDDEN_3 = 80, 40, 20
EPOCHS = 25

try:
    df = pd.read_csv("diabetes.csv")
except Exception:
    print("""
      Dataset not found in your computer.
      Please follow the instructions on how to download the dataset.
      """)
    quit()

# Perform preprocessing (imputation, standardization)
df = preprocess(df)

# Split the data into a training and testing set
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8, random_state=RANDOM_SEED
)

set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Build MLP
model = Sequential(
    [
        Input(shape=(8,)), # input layer
        Dense(HIDDEN_1, activation="relu"), # hidden layer 1
        Dense(HIDDEN_2, activation="relu"), # hidden layer 2
        Dense(HIDDEN_3, activation="relu"), # hidden layer 3
        Dense(1, activation="sigmoid"), # output layer
    ]
)

# Configure the model for training
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Train the model
model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=32,
    validation_split=0.1,
    verbose=2,
)

# Evaluate model accuracy on the testing data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
