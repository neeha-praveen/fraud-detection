import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import setup_and_load_data as data

print("STARTING DEFENSE...")

# Load dataset
X_train = data.X_train
X_test = data.X_test
y_train = data.y_train
y_test = data.y_test

print("DATA LOADED!")

# Load adversarial data
X_adv = np.load("X_adv_fgsm.npy")

# 🔥 FIX: Align sizes to avoid mismatch
min_size = min(len(X_train), len(X_adv))

X_train = X_train[:min_size]
y_train = y_train[:min_size]

X_adv = X_adv[:min_size]
y_adv = y_train

# Combine clean + adversarial
X_combined = np.vstack([X_train, X_adv])
y_combined = np.hstack([y_train, y_adv])

print("TRAINING MODEL...")

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_combined, y_combined)

print("TRAINING DONE!")

# Save model
with open("hardened_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- MODEL PERFORMANCE AFTER DEFENSE ---")
print("AUROC:", roc_auc_score(y_test, y_prob))
print("F1 Score:", f1_score(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("FNR:", fn / (fn + tp))