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
print("First 5 rows of the dataset:")
print(df.head(5))

# display histogram for each variable
df.hist(figsize=(10, 10))
plt.show()

# statistical summary of dataset
print("\nStatistical summary of the dataset:")
print(df.describe())

# print the number of missing values for each variable
print("\nNumber of missing values for each variable:")
print("Glucose: ", (df["Glucose"] == 0).sum(), "missing out of ", len(df)) #5 missing
print("BloodPressure: ", (df["BloodPressure"] == 0).sum(), "missing out of ", len(df)) #35 missing
print("SkinThickness: ", (df["SkinThickness"] == 0).sum(), "missing out of ", len(df)) #227 missing
print("Insulin: ", (df["Insulin"] == 0).sum(), "missing out of ", len(df)) #374 missing
print("BMI: ", (df["BMI"] == 0).sum(), "missing out of ", len(df)) #11 missing

