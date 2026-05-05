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

# Model parameters
HIDDEN_1 = 92
HIDDEN_2 = 46
EPOCHS = 38
BATCH_SIZE = 32
TEST_SIZE = 0.2
TRAIN_SIZE = 0.8
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

# Split the data into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    train_size=TRAIN_SIZE,
    random_state=RANDOM_SEED
)

set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Build MLP
model = Sequential(
    [
        Input(shape=(8,)), # input layer
        Dense(HIDDEN_1, activation="relu"), # hidden layer 1
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
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose=2,
)

# Print final training and validation results
print(f"\nFinal training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

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
save_results(
    history,
    loss,
    accuracy,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    plt=plt
)

# Show graph window
plt.show()