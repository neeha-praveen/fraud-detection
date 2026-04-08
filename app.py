import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import shap
import setup_and_load_data as data
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Load model
model = pickle.load(open("hardened_model.pkl", "rb"))

# Sidebar
page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "🔴 Live Prediction",
    "⚖ Comparison",
    "📈 RRS Trends",
    "📊 SHAP"
])

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.title("Fraud Detection Dashboard")
    st.info("Adversarially Robust ML Model with Explainability")

# ---------------- LIVE ----------------
elif page == "🔴 Live Prediction":
    st.title("🔴 Live Fraud Detection")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())

        if st.button("Run Prediction"):
            probs = model.predict_proba(df.values)[:, 1]
            avg_prob = float(np.mean(probs))

            st.subheader("Fraud Risk Meter")

            st.progress(int(avg_prob * 100))
            st.metric("Fraud Probability", f"{avg_prob:.2f}")

            if avg_prob > 0.7:
                st.error("🔴 High Risk Transaction")
            elif avg_prob > 0.4:
                st.warning("🟡 Medium Risk Transaction")
            else:
                st.success("🟢 Low Risk Transaction")

# ---------------- COMPARISON ----------------
elif page == "⚖ Comparison":
    st.title("⚖ Clean vs Adversarial Comparison")

    # Convert to numpy (FIXED KeyError issue)
    X_clean = data.X_test.values if hasattr(data.X_test, "values") else data.X_test

    attack = st.selectbox("Select Attack", ["FGSM", "PGD", "GA"])

    if attack == "FGSM":
        X_adv = np.load("X_adv_fgsm.npy")
    elif attack == "PGD":
        X_adv = np.load("X_adv_pgd.npy")
    else:
        X_adv = np.load("X_adv_ga.npy")

    idx = st.slider("Select Transaction Index", 0, len(X_clean) - 1)

    clean = X_clean[idx]
    adv = X_adv[idx]

    feature_names = [f"Feature_{i}" for i in range(len(clean))]

    df = pd.DataFrame({
        "Feature": feature_names,
        "Clean": clean,
        "Adversarial": adv
    })

    df["Difference"] = np.abs(df["Clean"] - df["Adversarial"])
    df_top = df.sort_values("Difference", ascending=False).head(20)

    st.subheader("Top Changed Features")
    st.dataframe(df_top.style.background_gradient(cmap='Reds'))

    fig = px.bar(
        df_top,
        x="Feature",
        y="Difference",
        color="Difference",
        title="Top Feature Changes"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- RRS ----------------
elif page == "📈 RRS Trends":
    st.title("📈 Robustness Trend")

    df_before = pd.read_csv("week7_rrs_results.csv")
    df_after = pd.read_csv("rrs_after_defense.csv")

    # 🔥 AUTO-DETECT column names
    before_col = df_before.columns[-1]   # last column = RRS
    after_col = df_after.columns[-1]

    rrs_before = df_before[before_col].values
    rrs_after = df_after[after_col].values

    # Fix mismatch
    if len(rrs_after) == 1:
        rrs_after = np.repeat(rrs_after[0], len(rrs_before))

    rrs_after = rrs_after[:len(rrs_before)]

    df = pd.DataFrame({
        "Attack": df_before.iloc[:, 0],   # first column = attack name
        "Before": rrs_before,
        "After": rrs_after
    })

    df_melt = df.melt(id_vars="Attack", var_name="Type", value_name="RRS")

    fig = px.bar(
        df_melt,
        x="Attack",
        y="RRS",
        color="Type",
        barmode="group",
        title="RRS Before vs After Defense"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- SHAP ----------------
elif page == "📊 SHAP":
    st.title("📊 SHAP Feature Importance")

    X_sample = data.X_test[:50]

    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, show=False)
    st.pyplot(fig)