import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

def preprocess(df):
    df = df.copy()
    missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # Step 1 - Mean imputation
    for col in missing_cols:
        if col in df.columns:
            col_mean = df[col][df[col] != 0].mean()  # mean of non-zero values only
            df[col] = df[col].replace(0, col_mean)  # replace 0 with mean

    # Step 2 - Print out the number of 0-valued entries for each variable 
    print("\nNumber of 0-valued entries for each variable after imputation:")
    for col in missing_cols:
        if col in df.columns:
            print(f"{col}: {(df[col] == 0).sum()} zero-valued entries")

    # Step 3 - Standardize the dataset (except for target variable - outcome column)
    feature_cols = df.columns.drop("Outcome")  # exclude the outcome column
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols]) #scale each feature column

    return df