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


# Build MLP
model = Sequential()

# Results - Accuracy

