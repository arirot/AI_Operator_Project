import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance

# -------------------------------
# Page config (must be first Streamlit call)
# -------------------------------
st.set_page_config(
    page_title="AI Operator Console",
    page_icon="🏭",
    layout="wide",
)

# -------------------------------
# Basic theming
# -------------------------------
st.markdown(
    """
    <style>
    .big-metric {font-size: 28px !important; font-weight: 700 !important;}
    .small-label {font-size: 13px !important; color: #888888 !important;}
    .persona-box {
        border-radius: 8px;
        padding: 10px 14px;
        border: 1px solid #44444422;
        background-color: #11111111;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Operator Console")
st.caption("Real-time machine failure prediction with interactive what‑if analysis and cloud‑safe explanations (LIME + feature importance).")

# -------------------------------
# Load model & data
# -------------------------------
@st.cache_resource
def load_model_and_data():
    base_path = Path(__file__).resolve().parent
    model_path = base_path.parent / "models" / "model.pkl"
    data_path = base_path.parent / "data" / "AI4I" / "ai4i2020.csv"

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    # Rename raw AI4I dataset columns to match model expectations
    df = df.rename(columns={
        "Air temperature [K]": "Air_temperature_K",
        "Process temperature [K]": "Process_temperature_K",
        "Rotational speed [rpm]": "Rotational_speed_rpm",
        "Torque [Nm]": "Torque_Nm",
        "Tool wear [min]": "Tool_wear_min",
        "Machine failure": "Machine_failure",
    })

    feature_cols = [
        "Air_temperature_K",
        "Process_temperature_K",
        "Rotational_speed_rpm",
        "Torque_Nm",
        "Tool_wear_min",
    ]

    target_col = "Machine_failure"

    X = df[feature_cols]
    y = df[target_col]

    return model, df, X, y, feature_cols, target_col


model, df, X, y, feature_cols, target_col = load_model_and_data()

# -------------------------------
# Global feature importance (XGBoost)
# -------------------------------
@st.cache_resource
def compute_global_importance(model, feature_cols):
    importances = model.feature_importances_
    global_importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importances}
    ).sort_values("importance", ascending=False)
    return global_importance_df


global_importance_df = compute_global_importance(model, feature_cols)

# -------------------------------
# Permutation importance
# -------------------------------
@st.cache_resource
def compute_permutation_importance(model, X, y):
    perm = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_df = pd.DataFrame(
        {"feature": X.columns, "importance": perm.importances_mean}
    ).sort_values("importance", ascending=False)
    return perm_df


perm_importance_df = compute_permutation_importance(model, X, y)

# -------------------------------
# LIME explainer
# -------------------------------
@st.cache_resource
def build_lime_explainer(X):
    explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=["Normal", "Failure"],
        mode="classification",
        discretize_continuous=True,
    )
    return explainer


lime_explainer = build_lime_explainer(X)

# -------------------------------
# Operator persona selection
# -------------------------------
persona = st.sidebar.selectbox(
    "Operator persona",
    [
        "Control Room Operator",
        "Maintenance Engineer",
        "Reliability Engineer",
    ],
)

if persona == "Control Room Operator":
    persona_text = "Focus: Is the machine safe to keep running right now?"
elif persona == "Maintenance Engineer":
    persona_text = "Focus: Root causes and failure modes. Why is the model predicting failure?"
else:
    persona_text = "Focus: Long-term risk patterns and feature impact."

st.sidebar.markdown(f"<div class='persona-box'>{persona_text}</div>", unsafe_allow_html=True)

# -------------------------------
# Scenario controls
# -------------------------------
st.subheader("Machine controls")

col1, col2, col3 = st.columns(3)

with col1:
    air_temp = st.slider("Air Temperature (K)", 290.0, 320.0, 300.0, 0.5)
with col2:
    torque = st.slider("Torque (Nm)", 0.0, 80.0, 40.0, 1.0)
with col3:
    speed = st.slider("Rotational Speed (rpm)", 500.0, 3000.0, 1500.0, 50.0)

process_temp = float(df["Process_temperature_K"].median())
tool_wear = float(df["Tool_wear_min"].median())

scenario_df = pd.DataFrame(
    {
        "Air_temperature_K": [air_temp],
        "Process_temperature_K": [process_temp],
        "Rotational_speed_rpm": [speed],
        "Torque_Nm": [torque],
        "Tool_wear_min": [tool_wear],
    }
)

# -------------------------------
# Prediction
# -------------------------------
proba = model.predict_proba(scenario_df)[0, 1]
pred_label = "Machine Operating Normally" if proba < 0.5 else "High Failure Risk"

# -------------------------------
# Layout: prediction + current input
# -------------------------------
left, middle, right = st.columns([1.2, 1.2, 1.4])

with left:
    st.markdown("### Prediction & Risk Gauge")
    st.metric("Status", pred_label)
    st.metric("Failure Risk (0–1)", f"{proba:.2f}")

    if proba < 0.3:
        action_text = "Risk is low. Continue monitoring."
    elif proba < 0.6:
        action_text = "Moderate risk. Plan inspection."
    else:
        action_text = "High risk. Schedule immediate maintenance."

    st.markdown(f"**Recommended action:** {action_text}")

with middle:
    st.markdown("### Current Input (after scenario)")
    display_df = scenario_df.copy()
    display_df.insert(0, "UDI", float(df["UDI"].median()))
    st.dataframe(display_df, use_container_width=True)

with right:
    st.markdown("### Operator context")
    if persona == "Control Room Operator":
        st.write("You care about immediate safety and alarms.")
    elif persona == "Maintenance Engineer":
        st.write("You care about root causes. Use the Feature Impact and LIME tabs.")
    else:
        st.write("You care about long-term reliability and feature impact.")

# -------------------------------
# Tabs: Feature impact, LIME, Permutation
# -------------------------------
tab1, tab2, tab3 = st.tabs(
    ["Feature Impact (Global)", "LIME Explanation (Local)", "Permutation Importance"]
)

with tab1:
    st.subheader("Global Feature Impact (XGBoost)")
    st.bar_chart(global_importance_df.set_index("feature"))

    st.markdown("**Top drivers of failure risk:**")
    for i, row in global_importance_df.head(3).iterrows():
        st.write(f"- {row['feature']}: importance {row['importance']:.3f}")

with tab2:
    st.subheader("LIME Explanation for Current Scenario")

    exp = lime_explainer.explain_instance(
        scenario_df.iloc[0].values,
        model.predict_proba,
        num_features=5,
    )

    lime_list = exp.as_list()
    lime_df = pd.DataFrame(lime_list, columns=["Feature", "Contribution"])
    st.dataframe(lime_df, use_container_width=True)

    st.markdown(
        "Positive contributions push the prediction toward **Failure**, "
        "negative contributions push it toward **Normal**."
    )

with tab3:
    st.subheader("Permutation Importance (Model-Agnostic)")
    st.bar_chart(perm_importance_df.set_index("feature"))

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(
    "Model: XGBoost | Explanations: Global feature importance, permutation importance, LIME (cloud‑safe)."
)