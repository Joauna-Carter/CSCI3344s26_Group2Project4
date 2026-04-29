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
print(df.head(5))

# display histogram for each variable
df.hist(figsize=(10, 10))
plt.show()


# statistical summary of dataset
print(df.describe())

# print the number of missing values for each variable
print((df["Glucose"] == 0).sum())
print((df["BloodPressure"] == 0).sum())
print((df["SkinThickness"] == 0).sum())
print((df["Insulin"] == 0).sum())
print((df["BMI"] == 0).sum())

