import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import kagglehub
import os 

path = kagglehub.dataset_download("neeraja3/fraud-detection-preprocessed-dataset-ieee-cis")
df = pd.read_csv(os.path.join(path, "final_fraud_dataset.csv"))

X = df.drop(columns=["isFraud"])
y = df["isFraud"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"Dataset loaded successfully!")
print(f"Total rows     : {df.shape[0]}")
print(f"Total features : {X.shape[1]}")
print(f"Fraud cases    : {y.sum()} ({100*y.mean():.2f}%)")
print(f"Train size     : {X_train_s.shape}")
print(f"Test size      : {X_test_s.shape}")