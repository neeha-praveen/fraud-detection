import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import setup_and_load_data as data
from wrap_model_art import compute_rrs
import shap

print("STARTING DEFENSE...")

# Load data
X_train = data.X_train
X_test = data.X_test
y_train = data.y_train
y_test = data.y_test
scaler = data.scaler

print("DATA LOADED!")

# Load adversarial data
X_fgsm = np.load("X_adv_fgsm.npy")
X_pgd = np.load("X_adv_pgd.npy")
X_ga = np.load("X_adv_ga.npy")

# FIX: convert scaled → original
X_fgsm = scaler.inverse_transform(X_fgsm)
X_pgd = scaler.inverse_transform(X_pgd)
X_ga = scaler.inverse_transform(X_ga)

# Align sizes
min_size = min(len(X_train), len(X_fgsm), len(X_pgd), len(X_ga))

X_train = X_train[:min_size]
y_train = y_train[:min_size]

X_fgsm = X_fgsm[:min_size]
X_pgd = X_pgd[:min_size]
X_ga = X_ga[:min_size]

# Combine all attacks
X_adv = np.vstack([X_fgsm, X_pgd, X_ga])
y_adv = np.hstack([y_train, y_train, y_train])

X_combined = np.vstack([X_train, X_adv])
y_combined = np.hstack([y_train, y_adv])

print("TRAINING MODEL...")

model = XGBClassifier(eval_metric='logloss')
model.fit(X_combined, y_combined)

print("TRAINING DONE!")

# Save model
with open("hardened_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- MODEL PERFORMANCE ---")
print("AUROC:", roc_auc_score(y_test, y_prob))
print("F1 Score:", f1_score(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fnr = fn / (fn + tp)
print("FNR:", fnr)

# RRS
rrs = compute_rrs(model, X_test, y_test)
print("RRS Score:", rrs)

# Save RRS
df_rrs = pd.DataFrame({
    "Attack": ["FGSM", "PGD", "GA"],
    "RRS_After": [rrs, rrs, rrs]
})
df_rrs.to_csv("rrs_after_defense.csv", index=False)

# SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X_test[:100])

mean_shap = np.abs(shap_values.values).mean(axis=0)
print("Top SHAP features:", np.argsort(mean_shap)[-10:])