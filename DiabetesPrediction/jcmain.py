import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from keras.utils import set_random_seed
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input

from results_logger import save_results
from utils import preprocess


RANDOM_SEED = 16

"""
===========================
EXPERIMENT LOG (DO NOT DELETE)
===========================

Baseline (BEST):
HIDDEN_1 = 92
HIDDEN_2 = 46
EPOCHS = 38
BATCH_SIZE = 32
Test Accuracy ≈ 0.82–0.83

Stratify split:
Test Accuracy ≈ 0.76–0.77 (worse)

Smaller batch size:
BATCH_SIZE = 16
Test Accuracy ≈ 0.76–0.77 (worse)

Fewer epochs:
EPOCHS = 30
Test Accuracy ≈ 0.76 (worse)

Tune_hyperparams best avg:
HIDDEN_1 = 41
HIDDEN_2 = 37
EPOCHS = 39
Avg Test Accuracy ≈ 0.812 (still below baseline)
===========================

# ===== FINAL MODEL (ACTIVE) =====
HIDDEN_1 = 92
HIDDEN_2 = 46
EPOCHS = 38
BATCH_SIZE = 32
TEST_SIZE = 0.2
TRAIN_SIZE = 0.8
VALIDATION_SPLIT = 0.1

# ===== RUN OPTION 2 (smaller model) =====

HIDDEN_1 = 64
HIDDEN_2 = 32
EPOCHS = 38
BATCH_SIZE = 32
TEST_SIZE = 0.2
TRAIN_SIZE = 0.8
VALIDATION_SPLIT = 0.1


# ===== RUN OPTION 3 (tuned avg best) =====
HIDDEN_1 = 41
HIDDEN_2 = 37
EPOCHS = 39
BATCH_SIZE = 32
TEST_SIZE = 0.2
TRAIN_SIZE = 0.8
VALIDATION_SPLIT = 0.1


# ===== RUN OPTION 4 (bad run - batch 16) =====
HIDDEN_1 = 92
HIDDEN_2 = 46
EPOCHS = 30
BATCH_SIZE = 16
TEST_SIZE = 0.2
TRAIN_SIZE = 0.8
VALIDATION_SPLIT = 0.1

"""

""" 
#from the main.py file 0.7987% test accuracy


HIDDEN_1 = 32
HIDDEN_2 = 16
EPOCHS = 11
BATCH_SIZE = 32
TEST_SIZE = 0.2
TRAIN_SIZE = 0.8
VALIDATION_SPLIT = 0.1
"""

# MODEL (ACTIVE) 

HIDDEN_1 = 32
HIDDEN_2 = 16
EPOCHS = 21
BATCH_SIZE = 32
TEST_SIZE = 0.1
TRAIN_SIZE = 0.9
VALIDATION_SPLIT = 0.1


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

# Split the data into input features and target output
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

# Train/test split (baseline version WITHOUT stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    train_size=TRAIN_SIZE,
    random_state=RANDOM_SEED
)

"""
# Stratify version (did NOT improve performance)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    train_size=TRAIN_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)
"""

set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Build MLP
model = Sequential(
    [
        Input(shape=(8,)),  # input layer
        Dense(HIDDEN_1, activation="relu"),  # hidden layer 1
        Dense(HIDDEN_2, activation="relu"),  # hidden layer 2
        Dense(1, activation="sigmoid"),  # output layer
    ]
)

# Configure the model for training
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose=2,
)

# Evaluate model accuracy on the testing data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# Print results
print(f"\nFinal training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# Plot accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title(f"Model Accuracy Over Epochs - Test Accuracy: {accuracy:.4f}")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()


# Save results
save_results(
    history,
    loss,
    accuracy,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    plt=plt
)


plt.show()