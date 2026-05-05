import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

from results_logger import save_results
from utils import preprocess

RANDOM_SEED = 16

# optimal layer sizes and epochs from tune_hyperparams.py were 96->48, 40,
# but these values were manually found to have slightly less loss:
HIDDEN_1, HIDDEN_2 = 92, 46
EPOCHS = 38

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

tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Build MLP
model = Sequential(
    [
        Dense(HIDDEN_1, activation="relu", input_shape=(8,)), # hidden layer 1
        Dense(HIDDEN_2, activation="relu"), # hidden layer 2
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
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=32,
    validation_split=0.1,
    verbose=2,
)

# Plot training and validation accuracy over each epoch
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

# Add title and axis labels for clarity
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# Show legend to distinguish the two lines
plt.legend()

# Evaluate model accuracy on the testing data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# Save graph and results table
save_results(history, accuracy, epochs=EPOCHS, batch_size=32, plt=plt)

# Show graph window
plt.show()
