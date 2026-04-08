# wrap_model_art.py
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from art.estimators.classification import XGBoostClassifier
import kagglehub
import os

# ── Load data ──
path = kagglehub.dataset_download("neeraja3/fraud-detection-preprocessed-dataset-ieee-cis")
df = pd.read_csv(os.path.join(path, "final_fraud_dataset.csv"))
X = df.drop(columns=["isFraud"])
y = df["isFraud"].values

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

# ── Load P2's model ──
p2_model = joblib.load("final_model.pkl")

# ── Wrap with ART's XGBoost classifier ──
classifier = XGBoostClassifier(
    model=p2_model,
    nb_features=X_train_s.shape[1],
    nb_classes=2,
    clip_values=(float(X_train_s.min()), float(X_train_s.max())),
)

# ── Quick test ──
X_sample = scaler.inverse_transform(X_test_s[:5])  # XGBoost expects unscaled
preds = classifier.predict(X_sample)
print("Model loaded and wrapped successfully!")
print(f"Sample predictions : {np.argmax(preds, axis=1)}")
print(f"Actual labels      : {y_test[:5]}")
def compute_rrs(model, X, y):
    """
    Simple Robustness Risk Score (RRS)
    Lower FNR = better robustness
    """

    from sklearn.metrics import confusion_matrix

    y_pred = model.predict(X)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    fnr = fn / (fn + tp)

    # Convert to robustness score (higher is better)
    rrs = 1 - fnr

    return rrs