import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(16)

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from utils import preprocess


try:
    df = pd.read_csv('diabetes.csv')
except:
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
    X, y, test_size=0.2, train_size=0.8, random_state = 16
)
