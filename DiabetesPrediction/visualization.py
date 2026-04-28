import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

try:
    df = pd.read_csv('diabetes.csv')
except:
    print("""
      Dataset not found in your computer.
      Please follow the instructions on how to download the dataset.
      """)
    quit()

# **ADD YOUR CODE HERE**

# print the first 5 rows of the dataset


# display histogram for each variable


# statistical summary of dataset
    

# print the number of missing values for each variable

