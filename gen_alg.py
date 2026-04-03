# gen_alg.py
# Person 3 - Adversarial Attacks
# Week 7: Genetic Algorithm (black-box attack) + Robustness Risk Score across all three attacks

import numpy as np
import pandas as pd
import joblib
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import kagglehub
import os

# ─────────────────────────────────────────────
# 1. LOAD DATA (same setup as week 6)
# ─────────────────────────────────────────────

path = kagglehub.dataset_download("neeraja3/fraud-detection-preprocessed-dataset-ieee-cis")
df = pd.read_csv(os.path.join(path, "final_fraud_dataset.csv"))

X = df.drop(columns=["isFraud"])
y = df["isFraud"].values

feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

# Per-feature min/max from training data
feat_min = X_train_s.min(axis=0)
feat_max = X_train_s.max(axis=0)

# Binary column indices
binary_col_indices = [
    i for i, col in enumerate(feature_names)
    if sorted(pd.Series(X_train[:, i]).dropna().unique().tolist()) in [[0, 1], [0.0, 1.0]]
]

print(f"Total features       : {len(feature_names)}")
print(f"Binary feature count : {len(binary_col_indices)}")

# ─────────────────────────────────────────────
# 2. LOAD P2's MODEL
# ─────────────────────────────────────────────

p2_model = joblib.load("final_model.pkl")

def p2_predict(X_scaled):
    X_orig = scaler.inverse_transform(X_scaled)
    return p2_model.predict(X_orig).astype(int)

def p2_predict_proba(X_scaled):
    X_orig = scaler.inverse_transform(X_scaled)
    if hasattr(p2_model, "predict_proba"):
        return p2_model.predict_proba(X_orig)[:, 1]
    else:
        return p2_model.predict(X_orig).astype(float)

# ─────────────────────────────────────────────
# 3. FEATURE CONSTRAINT FUNCTION
# ─────────────────────────────────────────────

def constrain_features(X_adv, binary_indices, feat_min, feat_max):
    """
    Apply domain constraints to adversarial examples:
      - Per-feature clipping to observed training range
      - Round binary features back to 0 or 1
    """
    X_c = X_adv.copy()
    X_c = np.clip(X_c, feat_min, feat_max)
    for i in binary_indices:
        X_c[:, i] = np.round(np.clip(X_c[:, i], 0, 1))
    return X_c

# ─────────────────────────────────────────────
# 4. LOAD SAVED ADVERSARIAL EXAMPLES
#    GA already ran — load saved results directly
#    to skip re-running (saves 10+ minutes)
# ─────────────────────────────────────────────

N_GA = 500

print("\n─── Loading saved adversarial examples ───")

X_adv_ga   = np.load("X_adv_ga.npy")
X_adv_fgsm = np.load("X_adv_fgsm.npy")[:N_GA]
X_adv_pgd  = np.load("X_adv_pgd.npy")[:N_GA]

print(f"Loaded X_adv_ga   : {X_adv_ga.shape}")
print(f"Loaded X_adv_fgsm : {X_adv_fgsm.shape}")
print(f"Loaded X_adv_pgd  : {X_adv_pgd.shape}")

# ─────────────────────────────────────────────
# 5. ROBUSTNESS RISK SCORE (RRS)
#
# RRS measures how vulnerable the model is to each attack.
# Range: 0 (robust) to 1 (highly vulnerable)
#
# Formula:
#   RRS = 0.40 * normalized_f1_drop
#       + 0.30 * flip_rate
#       + 0.20 * normalized_acc_drop
#       + 0.10 * normalized_perturbation
#
# Weights prioritise F1 drop since fraud detection
# cares most about recall degradation.
# ─────────────────────────────────────────────

def compute_rrs(
    predict_fn,
    X_clean,
    X_adv,
    y_true,
    y_clean_pred=None,
    attack_name="Attack",
):
    if y_clean_pred is None:
        y_clean_pred = predict_fn(X_clean)

    y_adv_pred = predict_fn(X_adv)

    # ── Classification metrics ──
    f1_clean  = f1_score(y_true, y_clean_pred, zero_division=0)
    f1_adv    = f1_score(y_true, y_adv_pred,   zero_division=0)
    acc_clean = accuracy_score(y_true, y_clean_pred)
    acc_adv   = accuracy_score(y_true, y_adv_pred)
    prec_adv  = precision_score(y_true, y_adv_pred, zero_division=0)
    rec_adv   = recall_score(y_true, y_adv_pred, zero_division=0)

    # ── Flip rate ──
    flip_rate = float(np.mean(y_clean_pred != y_adv_pred))

    # ── Perturbation magnitude ──
    l_inf_mean = float(np.max(np.abs(X_adv - X_clean), axis=1).mean())
    l2_mean    = float(np.linalg.norm(X_adv - X_clean, axis=1).mean())

    # ── Normalized drops ──
    f1_drop_norm  = float(np.clip((f1_clean  - f1_adv)  / (f1_clean  + 1e-9), 0, 1))
    acc_drop_norm = float(np.clip((acc_clean - acc_adv) / (acc_clean + 1e-9), 0, 1))
    perturb_norm  = float(np.clip(l_inf_mean / 1.0, 0, 1))

    # ── RRS ──
    rrs = (
        0.40 * f1_drop_norm
      + 0.30 * flip_rate
      + 0.20 * acc_drop_norm
      + 0.10 * perturb_norm
    )
    rrs = float(np.clip(rrs, 0, 1))

    return {
        "Attack":           attack_name,
        "F1 (clean)":       round(f1_clean,       4),
        "F1 (adversarial)": round(f1_adv,         4),
        "F1 drop (%)":      round(f1_drop_norm  * 100, 2),
        "Acc (clean)":      round(acc_clean,      4),
        "Acc (adversarial)":round(acc_adv,        4),
        "Acc drop (%)":     round(acc_drop_norm * 100, 2),
        "Precision (adv)":  round(prec_adv,       4),
        "Recall (adv)":     round(rec_adv,        4),
        "Flip rate":        round(flip_rate,       4),
        "Mean L-inf":       round(l_inf_mean,      5),
        "Mean L2":          round(l2_mean,         5),
        "RRS":              round(rrs,             4),
    }

# ─────────────────────────────────────────────
# 6. EVALUATE ALL THREE ATTACKS
# ─────────────────────────────────────────────

print("\n─── Computing Robustness Risk Scores ───\n")

y_sub        = y_test[:N_GA]
y_clean_pred = p2_predict(X_test_s[:N_GA])   # compute once, reuse for all three

results_fgsm = compute_rrs(p2_predict, X_test_s[:N_GA], X_adv_fgsm, y_sub,
                            y_clean_pred=y_clean_pred, attack_name="FGSM")

results_pgd  = compute_rrs(p2_predict, X_test_s[:N_GA], X_adv_pgd,  y_sub,
                            y_clean_pred=y_clean_pred, attack_name="PGD")

results_ga   = compute_rrs(p2_predict, X_test_s[:N_GA], X_adv_ga,   y_sub,
                            y_clean_pred=y_clean_pred, attack_name="Genetic (GA)")

# ─────────────────────────────────────────────
# 7. PRINT SUMMARY TABLE
# ─────────────────────────────────────────────

summary_df = pd.DataFrame([results_fgsm, results_pgd, results_ga])
summary_df = summary_df.set_index("Attack")

print("=" * 70)
print("              WEEK 7 — ROBUSTNESS RISK SCORE SUMMARY")
print("=" * 70)
print(summary_df.T.to_string())
print("=" * 70)

# ── RRS ranking ──
print("\nRRS Ranking (higher = more vulnerable):")
for attack, row in summary_df[["RRS"]].sort_values("RRS", ascending=False).iterrows():
    bar = "█" * int(row["RRS"] * 40)
    print(f"  {attack.ljust(14)} RRS = {row['RRS']:.4f}  {bar}")

# ── F1 drop ranking ──
print("\nF1 Drop (%):")
for attack, row in summary_df[["F1 drop (%)"]].sort_values("F1 drop (%)", ascending=False).iterrows():
    print(f"  {attack.ljust(14)} {row['F1 drop (%)']:.2f}%")

# ── Flip rate ranking ──
print("\nFlip Rate (fraction of predictions changed):")
for attack, row in summary_df[["Flip rate"]].sort_values("Flip rate", ascending=False).iterrows():
    print(f"  {attack.ljust(14)} {row['Flip rate']:.4f}")

# ─────────────────────────────────────────────
# 8. SAVE RESULTS
# ─────────────────────────────────────────────

summary_df.to_csv("week7_rrs_results.csv")
print("\nSaved: week7_rrs_results.csv")

print("\nDone. Files to include in your Week 7 report:")
print("  week7_rrs_results.csv  — full metrics table")
print("  X_adv_fgsm.npy         — FGSM adversarial examples")
print("  X_adv_pgd.npy          — PGD adversarial examples")
print("  X_adv_ga.npy           — Genetic algorithm adversarial examples")