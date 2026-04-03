# week6_attacks.py
# Person 3 - Adversarial Attacks
# Week 6: FGSM and PGD (white-box) with feature constraints on IEEE-CIS Fraud dataset

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from art.estimators.classification import PyTorchClassifier, SklearnClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
import kagglehub
import os
# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

path = kagglehub.dataset_download("neeraja3/fraud-detection-preprocessed-dataset-ieee-cis")
df = pd.read_csv(os.path.join(path, "final_fraud_dataset.csv"))  # update path if needed

X = df.drop(columns=["isFraud"])
y = df["isFraud"].values

feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

# ─────────────────────────────────────────────
# 2. FEATURE CONSTRAINT SETUP
# ─────────────────────────────────────────────

# Identify binary columns (only 0 and 1 values) by index
binary_col_indices = [
    i for i, col in enumerate(feature_names)
    if sorted(pd.Series(X_train[:, i]).dropna().unique().tolist()) in [[0, 1], [0.0, 1.0]]
]

# Global feature min/max from training data (scaled)
feat_min = X_train_s.min(axis=0)  # per-feature minimum
feat_max = X_train_s.max(axis=0)  # per-feature maximum

clip_min = float(X_train_s.min())
clip_max = float(X_train_s.max())

print(f"Total features       : {len(feature_names)}")
print(f"Binary feature count : {len(binary_col_indices)}")
print(f"Global scaled range  : [{clip_min:.3f}, {clip_max:.3f}]")


def constrain_features(X_adv, binary_indices, feat_min, feat_max):
    """
    Apply domain constraints to adversarial examples:
      - Clip each feature to its observed min/max from training data
      - Round binary features to nearest valid value (0 or 1)
      - Ensure no feature goes below 0 for non-negative features
    """
    X_c = X_adv.copy()

    # Per-feature clipping (tighter than global clip)
    X_c = np.clip(X_c, feat_min, feat_max)

    # Round binary features
    for i in binary_indices:
        X_c[:, i] = np.round(np.clip(X_c[:, i], 0, 1))

    return X_c


# ─────────────────────────────────────────────
# 3. SURROGATE NEURAL NETWORK
#    (needed because XGBoost/LightGBM have no gradients)
#    Trained to mimic P2's model predictions
# ─────────────────────────────────────────────

class FraudNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


n_feat   = X_train_s.shape[1]
model_nn = FraudNet(n_feat)
optimizer = torch.optim.Adam(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn   = nn.CrossEntropyLoss()

art_nn = PyTorchClassifier(
    model=model_nn,
    loss=loss_fn,
    optimizer=optimizer,
    input_shape=(n_feat,),
    nb_classes=2,
    clip_values=(clip_min, clip_max),
)

print("\nTraining surrogate neural network...")
art_nn.fit(X_train_s, y_train, batch_size=512, nb_epochs=15)

# Evaluate surrogate on clean test data
y_pred_nn = np.argmax(art_nn.predict(X_test_s[:2000]), axis=1)
print(f"Surrogate NN F1 (clean): {f1_score(y_test[:2000], y_pred_nn, zero_division=0):.4f}")

# ─────────────────────────────────────────────
# 4. LOAD P2's MODEL (for degradation evaluation)
# ─────────────────────────────────────────────

p2_model = joblib.load("final_model.pkl")   # update filename if P2 used a different name

def p2_predict(X_scaled):
    X_orig = scaler.inverse_transform(X_scaled)
    return p2_model.predict(X_orig).astype(int)

# Baseline performance of P2's model on clean data
N_EVAL = 2000
y_clean_pred = p2_predict(X_test_s[:N_EVAL])
print(f"\nP2 model F1 (clean): {f1_score(y_test[:N_EVAL], y_clean_pred, zero_division=0):.4f}")
print(f"P2 model Acc (clean): {accuracy_score(y_test[:N_EVAL], y_clean_pred):.4f}")

# ─────────────────────────────────────────────
# 5. FGSM ATTACK
# ─────────────────────────────────────────────

print("\n─── Running FGSM attack ───")

fgsm = FastGradientMethod(
    estimator=art_nn,
    eps=0.1,          # max perturbation magnitude (L-inf norm)
    norm=np.inf,
    targeted=False,
    batch_size=256,
)

X_adv_fgsm_raw = fgsm.generate(x=X_test_s[:N_EVAL])

# Apply constraints
X_adv_fgsm = constrain_features(X_adv_fgsm_raw, binary_col_indices, feat_min, feat_max)

# Perturbation stats
l_inf_fgsm = np.max(np.abs(X_adv_fgsm - X_test_s[:N_EVAL]), axis=1).mean()
l2_fgsm    = np.linalg.norm(X_adv_fgsm - X_test_s[:N_EVAL], axis=1).mean()
print(f"FGSM mean L-inf perturbation : {l_inf_fgsm:.5f}")
print(f"FGSM mean L2   perturbation  : {l2_fgsm:.5f}")

# Evaluate on P2's model
y_fgsm_pred = p2_predict(X_adv_fgsm)
fgsm_f1     = f1_score(y_test[:N_EVAL], y_fgsm_pred, zero_division=0)
fgsm_acc    = accuracy_score(y_test[:N_EVAL], y_fgsm_pred)
fgsm_flip   = np.mean(y_clean_pred != y_fgsm_pred)

print(f"FGSM F1 (adversarial) : {fgsm_f1:.4f}")
print(f"FGSM Acc (adversarial): {fgsm_acc:.4f}")
print(f"FGSM Flip rate        : {fgsm_flip:.4f}")

# ─────────────────────────────────────────────
# 6. PGD ATTACK
# ─────────────────────────────────────────────

print("\n─── Running PGD attack ───")

pgd = ProjectedGradientDescent(
    estimator=art_nn,
    eps=0.1,          # max perturbation (same as FGSM for fair comparison)
    eps_step=0.01,    # step size per iteration
    max_iter=40,
    norm=np.inf,
    targeted=False,
    num_random_init=1,
    batch_size=256,
)

X_adv_pgd_raw = pgd.generate(x=X_test_s[:N_EVAL])

# Apply constraints
X_adv_pgd = constrain_features(X_adv_pgd_raw, binary_col_indices, feat_min, feat_max)

# Perturbation stats
l_inf_pgd = np.max(np.abs(X_adv_pgd - X_test_s[:N_EVAL]), axis=1).mean()
l2_pgd    = np.linalg.norm(X_adv_pgd - X_test_s[:N_EVAL], axis=1).mean()
print(f"PGD mean L-inf perturbation : {l_inf_pgd:.5f}")
print(f"PGD mean L2   perturbation  : {l2_pgd:.5f}")

# Evaluate on P2's model
y_pgd_pred = p2_predict(X_adv_pgd)
pgd_f1     = f1_score(y_test[:N_EVAL], y_pgd_pred, zero_division=0)
pgd_acc    = accuracy_score(y_test[:N_EVAL], y_pgd_pred)
pgd_flip   = np.mean(y_clean_pred != y_pgd_pred)

print(f"PGD F1 (adversarial) : {pgd_f1:.4f}")
print(f"PGD Acc (adversarial): {pgd_acc:.4f}")
print(f"PGD Flip rate        : {pgd_flip:.4f}")

# ─────────────────────────────────────────────
# 7. SUMMARY TABLE
# ─────────────────────────────────────────────

clean_f1  = f1_score(y_test[:N_EVAL], y_clean_pred, zero_division=0)
clean_acc = accuracy_score(y_test[:N_EVAL], y_clean_pred)

summary = pd.DataFrame({
    "Attack":       ["Clean (baseline)", "FGSM", "PGD"],
    "F1 Score":     [round(clean_f1, 4),  round(fgsm_f1, 4),  round(pgd_f1, 4)],
    "Accuracy":     [round(clean_acc, 4), round(fgsm_acc, 4), round(pgd_acc, 4)],
    "F1 Drop (%)":  ["-",
                     round((clean_f1 - fgsm_f1) / (clean_f1 + 1e-9) * 100, 2),
                     round((clean_f1 - pgd_f1)  / (clean_f1 + 1e-9) * 100, 2)],
    "Flip Rate":    ["-", round(fgsm_flip, 4), round(pgd_flip, 4)],
    "Mean L-inf":   ["-", round(l_inf_fgsm, 5), round(l_inf_pgd, 5)],
})

print("\n========== WEEK 6 RESULTS ==========")
print(summary.to_string(index=False))

# Save results and adversarial examples
summary.to_csv("week6_results.csv", index=False)
np.save("X_adv_fgsm.npy", X_adv_fgsm)
np.save("X_adv_pgd.npy",  X_adv_pgd)
print("\nSaved: week6_results.csv, X_adv_fgsm.npy, X_adv_pgd.npy")