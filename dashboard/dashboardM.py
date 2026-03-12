import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import joblib
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

# -------------------------------
# Page config & basic theming
# -------------------------------
st.set_page_config(
    page_title="AI Operator Dashboard",
    page_icon="🏭",
    layout="wide",
)

st.markdown(
    """
    <style>
    .big-metric {font-size: 26px !important; font-weight: 600 !important;}
    .small-label {font-size: 13px !important; color: #888888 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏭 AI Operator Dashboard")
st.caption("Real-time machine failure prediction with SHAP & LIME explanations.")


# -------------------------------
# Load model & data
# -------------------------------
@st.cache_resource
def load_model():
    base_path = Path(__file__).resolve().parent
    model_path = base_path.parent / "models" / "model.pkl"
    return joblib.load(model_path)


@st.cache_resource
def load_training_data():
    base_path = Path(__file__).resolve().parent
    data_path = base_path.parent / "data" / "ai4i2020.csv"

    df = pd.read_csv(data_path)

    # Column cleaning (align with notebook)
    df.columns = (
        df.columns
        .str.replace('[', '', regex=False)
        .str.replace(']', '', regex=False)
        .str.replace('<', '', regex=False)
        .str.replace('>', '', regex=False)
        .str.replace(' ', '_')
        .str.replace('.', '_')
        .str.replace('-', '_')
        .str.strip()
    )
    df.columns = df.columns.astype(str)

    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop("Machine_failure", axis=1)
    y = df["Machine_failure"]
    return X, y, df


model = load_model()
X_train, y_train, df_full = load_training_data()

# SHAP & LIME initialization
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=["No Failure", "Failure"],
    mode="classification",
)


# -------------------------------
# Sidebar: input + controls
# -------------------------------
st.sidebar.header("🔧 Machine Parameters")

input_data = {}
for col in X_train.columns:
    default_val = float(X_train[col].mean())
    input_data[col] = st.sidebar.number_input(col, value=default_val)

input_df = pd.DataFrame([input_data])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Controls")
compare_mode = st.sidebar.checkbox("Enable comparison mode", value=False)
simulation_mode = st.sidebar.checkbox("Enable real-time simulation", value=False)


# -------------------------------
# Tabs
# -------------------------------
tab_pred, tab_shap, tab_lime, tab_global, tab_sim = st.tabs(
    ["📌 Prediction", "🔍 SHAP", "🟩 LIME", "🌍 Global View", "📡 Simulation"]
)


# -------------------------------
# Tab 1: Prediction
# -------------------------------
with tab_pred:
    st.subheader("📌 Prediction Result")

    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    col1, col2 = st.columns(2)

    with col1:
        if pred == 1:
            st.error(f"⚠️ Machine Failure Likely")
        else:
            st.success(f"✅ Machine Operating Normally")

        st.markdown(
            f"<div class='big-metric'>{proba:.2f}</div>"
            "<div class='small-label'>Failure Risk Score (0–1)</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("**Current Input Snapshot**")
        st.dataframe(input_df.T, use_container_width=True)

    if compare_mode:
        st.markdown("---")
        st.markdown("### 🔁 Compare with another configuration")

        compare_data = {}
        for col in X_train.columns:
            default_val = float(X_train[col].median())
            compare_data[col] = st.number_input(
                f"[Compare] {col}", value=default_val
            )

        compare_df = pd.DataFrame([compare_data])
        compare_proba = model.predict_proba(compare_df)[0][1]
        compare_pred = model.predict(compare_df)[0]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Baseline Configuration**")
            st.metric("Failure Risk", f"{proba:.2f}")
        with c2:
            st.markdown("**Comparison Configuration**")
            st.metric("Failure Risk", f"{compare_proba:.2f}")

        st.markdown("**Difference in risk:**")
        st.write(f"{compare_proba - proba:+.2f}")


# -------------------------------
# Tab 2: SHAP
# -------------------------------
with tab_shap:
    st.subheader("🔍 SHAP Explanation")

    # Local explanation for current input
    st.markdown("### 🎯 Local Feature Impact (Current Input)")
    shap.initjs()
    local_shap = explainer.shap_values(input_df)
    shap_html = shap.force_plot(
        explainer.expected_value,
        local_shap,
        input_df,
        matplotlib=False,
    )
    st.components.v1.html(shap_html.html(), height=300)

    # Summary plot
    st.markdown("### 🌈 SHAP Summary Plot (Global)")
    st.write("Shows how each feature contributes across the dataset.")
    shap.summary_plot(shap_values, X_train, show=False)
    st.pyplot()

    # Optional: bar plot for global importance
    st.markdown("### 📊 Global Feature Importance (Mean |SHAP|)")
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    st.pyplot()


# -------------------------------
# Tab 3: LIME
# -------------------------------
with tab_lime:
    st.subheader("🟩 LIME Local Explanation")

    lime_exp = lime_explainer.explain_instance(
        data_row=input_df.iloc[0],
        predict_fn=model.predict_proba,
    )

    st.info(
        "LIME shows how each feature contributed to this specific prediction."
    )
    st.components.v1.html(lime_exp.as_html(), height=600, scrolling=True)


# -------------------------------
# Tab 4: Global View & Comparison Tools
# -------------------------------
with tab_global:
    st.subheader("🌍 Global Model View")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("### Dataset Overview")
        st.write(f"Samples: {X_train.shape[0]}")
        st.write(f"Features: {X_train.shape[1]}")
        st.write("Target distribution:")
        st.bar_chart(y_train.value_counts())

    with col_g2:
        st.markdown("### Feature Distributions")
        feature_to_view = st.selectbox(
            "Select feature", X_train.columns
        )
        st.line_chart(df_full[feature_to_view])

    st.markdown("---")
    st.markdown("### 🔁 Compare Two Historical Samples")

    idx1 = st.number_input(
        "Index of sample A", min_value=0, max_value=len(X_train) - 1, value=0
    )
    idx2 = st.number_input(
        "Index of sample B", min_value=0, max_value=len(X_train) - 1, value=1
    )

    sample_a = X_train.iloc[[idx1]]
    sample_b = X_train.iloc[[idx2]]

    proba_a = model.predict_proba(sample_a)[0][1]
    proba_b = model.predict_proba(sample_b)[0][1]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sample A**")
        st.metric("Failure Risk", f"{proba_a:.2f}")
        st.dataframe(sample_a.T)
    with c2:
        st.markdown("**Sample B**")
        st.metric("Failure Risk", f"{proba_b:.2f}")
        st.dataframe(sample_b.T)


# -------------------------------
# Tab 5: Real-time Sensor Simulation
# -------------------------------
with tab_sim:
    st.subheader("📡 Real-time Sensor Simulation")

    st.info(
        "This simulates streaming sensor values and shows how risk changes over time."
    )

    sim_steps = st.slider("Number of steps", 10, 100, 30)
    start_button = st.button("Start Simulation")

    placeholder_metrics = st.empty()
    placeholder_chart = st.empty()

    if start_button:
        history = []
        for i in range(sim_steps):
            sim_row = input_df.copy()

            # Example: perturb a few key features if they exist
            for col in sim_row.columns:
                if "Temperature" in col or "temp" in col.lower():
                    sim_row[col] = sim_row[col] + np.random.normal(0, 1)
                if "Torque" in col or "torque" in col.lower():
                    sim_row[col] = sim_row[col] + np.random.normal(0, 0.5)

            sim_proba = model.predict_proba(sim_row)[0][1]
            history.append(sim_proba)

            with placeholder_metrics.container():
                st.markdown("### Current Simulated Risk")
                st.markdown(
                    f"<div class='big-metric'>{sim_proba:.2f}</div>"
                    "<div class='small-label'>Simulated Failure Risk</div>",
                    unsafe_allow_html=True,
                )

            with placeholder_chart.container():
                st.line_chart(history)

            time.sleep(0.1)


# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("AI Operator Dashboard — XGBoost + SHAP + LIME • Ready for Streamlit Cloud / Azure.")