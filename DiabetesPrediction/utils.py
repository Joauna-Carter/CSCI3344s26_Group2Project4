import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

def preprocess(df):
    #Perform mean imputation on the dataset 
    # (replace the 0 values of the columns with themean of the non-missing values)
    df = df.copy()
    missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # Step 1 - Mean imputation
    for col in missing_cols:
        if col in df.columns:
            col_mean = df[col][df[col] != 0].mean()  # mean of non-zero values only
            df[col] = df[col].replace(0, col_mean)  # replace 0 with mean

    # Step 2 - Print out the number of 0-valued entries for each variable 
    # to ensure that your imputation worked as intended
    print("\nNumber of 0-valued entries for each variable after imputation:")
    for col in missing_cols:
        if col in df.columns:
            print(f"{col}: {(df[col] == 0).sum()} zero-valued entries")