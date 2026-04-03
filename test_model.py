import joblib
import pandas as pd
import kagglehub
import os

model = joblib.load("final_model.pkl")

# Example: load one row
path = kagglehub.dataset_download("neeraja3/fraud-detection-preprocessed-dataset-ieee-cis")
df = pd.read_csv(os.path.join(path, "final_fraud_dataset.csv")).iloc[:1]
df = df.fillna(0)

X = df.drop(columns=["isFraud"])

prob = model.predict_proba(X)[0][1]
prediction = 1 if prob > 0.3 else 0

print("Fraud probability:", prob)
print("Prediction:", prediction)