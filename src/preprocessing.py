import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "hdi_data.csv")
    df = pd.read_csv(path)
    return df

def preprocess_data(df, fit_scaler=True, scaler=None, le=None):
    df = df.dropna()
    features = ["life_expectancy", "mean_schooling_years", "expected_schooling_years", "gni_per_capita"]
    X = df[features].values
    y = df["hdi_category"].values

    if le is None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = le.transform(y)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y_encoded, scaler, le

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
