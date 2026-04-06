import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("hardened_model.pkl", "rb"))

st.title("💳 Fraud Detection Dashboard")

# Sidebar
page = st.sidebar.selectbox("Select Option", [
    "Live Prediction",
    "Adversarial Comparison",
])

# -------------------- LIVE PREDICTION --------------------
if page == "Live Prediction":
    st.header("🔴 Live Fraud Detection")

    input_data = st.text_input("Enter features (comma separated)")

    if input_data:
        try:
            data = np.array([float(i) for i in input_data.split(",")]).reshape(1, -1)
            prob = model.predict_proba(data)[0][1]

            st.write("Fraud Probability:", prob)

            if prob > 0.5:
                st.error("⚠ Fraud Detected")
            else:
                st.success("✅ Safe Transaction")

        except:
            st.warning("Please enter valid numeric input")


# -------------------- COMPARISON --------------------
elif page == "Adversarial Comparison":
    st.header("⚖ Clean vs Adversarial Data")

    X_clean = np.load("X_adv_fgsm.npy")
    X_adv = np.load("X_adv_pgd.npy")

    idx = st.slider("Select Transaction Index", 0, len(X_clean)-1)

    clean = X_clean[idx]
    adv = X_adv[idx]

    df = pd.DataFrame({
        "Feature": range(len(clean)),
        "Clean": clean,
        "Adversarial": adv
    })

    st.dataframe(df)

    fig, ax = plt.subplots()
    ax.plot(clean, label="Clean")
    ax.plot(adv, label="Adversarial")
    ax.legend()

    st.pyplot(fig)