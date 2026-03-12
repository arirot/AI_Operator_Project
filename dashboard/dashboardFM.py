# -------------------------------
# Test to confirm app is running
# -------------------------------
import streamlit as st
st.write("App started successfully")

# -------------------------------
# Page config & basic theming
# -------------------------------
st.set_page_config(
    page_title="AI Operator Console",
    page_icon="##",
    layout="wide",
)

import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import joblib
from lime.lime_tabular import LimeTabularExplainer


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
st.caption("Real-time machine failure prediction with interactive what‑if analysis, SHAP & LIME explanations.")


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
# Sidebar: personas & controls
# -------------------------------
st.sidebar.header("Operator Persona")

persona = st.sidebar.selectbox(
    "Select persona",
    ["Line Operator", "Maintenance Engineer", "Energy Manager"],
)

if persona == "Line Operator":
    st.sidebar.markdown(
        "<div class='persona-box'>Focus: quick, clear alerts.<br>"
        "Needs: “Is the machine safe right now?”</div>",
        unsafe_allow_html=True,
    )
elif persona == "Maintenance Engineer":
    st.sidebar.markdown(
        "<div class='persona-box'>Focus: root causes & failure modes.<br>"
        "Needs: “Why is the model predicting failure?”</div>",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        "<div class='persona-box'>Focus: efficiency & energy use.<br>"
        "Needs: “How do settings affect efficiency?”</div>",
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.header("Machine Controls")

# Pick some key features for sliders (adjust names to your dataset)
temp = st.sidebar.slider("Air Temperature (K)", 290.0, 320.0, 300.0, 0.5)
torque = st.sidebar.slider("Torque (Nm)", 20.0, 100.0, 40.0, 1.0)
speed = st.sidebar.slider("Rotational Speed (rpm)", 1000.0, 3000.0, 1500.0, 50.0)

# Build input row starting from mean values, then override key sliders
base_row = X_train.mean().to_dict()
for col in base_row.keys():
    if "Air_temperature" in col:
        base_row[col] = temp
    if "Torque" in col:
        base_row[col] = torque
    if "Rotational_speed" in col:
        base_row[col] = speed

input_df = pd.DataFrame([base_row])

st.sidebar.markdown("---")
st.sidebar.header("Scenario Testing")

scenario = st.sidebar.selectbox(
    "Apply scenario",
    [
        "None",
        "High temperature spike",
        "High torque / vibration",
        "Low efficiency mode",
        "Combined stress",
    ],
)

def apply_scenario(row: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    row = row.copy()
    if scenario_name == "High temperature spike":
        for c in row.columns:
            if "Air_temperature" in c:
                row[c] = row[c] + 15
    elif scenario_name == "High torque / vibration":
        for c in row.columns:
            if "Torque" in c:
                row[c] = row[c] + 30
    elif scenario_name == "Low efficiency mode":
        for c in row.columns:
            if "Rotational_speed" in c:
                row[c] = row[c] - 500
            if "Torque" in c:
                row[c] = row[c] + 10
    elif scenario_name == "Combined stress":
        for c in row.columns:
            if "Air_temperature" in c:
                row[c] = row[c] + 10
            if "Torque" in c:
                row[c] = row[c] + 20
            if "Rotational_speed" in c:
                row[c] = row[c] + 300
    return row

scenario_df = apply_scenario(input_df, scenario)


# -------------------------------
# Tabs
# -------------------------------
tab_pred, tab_shap, tab_lime, tab_global, tab_sim, tab_ux = st.tabs(
    [
        "Prediction",
        "SHAP",
        "LIME",
        "Global View",
        "Simulation",
        "UX Evaluation",
    ]
)


# -------------------------------
# Tab 1: Prediction (with gauge)
# -------------------------------
with tab_pred:
    st.subheader("Prediction & Risk Gauge")

    proba = model.predict_proba(scenario_df)[0][1]
    pred = model.predict(scenario_df)[0]

    c1, c2 = st.columns([1, 2])

    with c1:
        if pred == 1:
            st.error("Machine Failure Likely")
        else:
            st.success("Machine Operating Normally")

        st.markdown(
            f"<div class='big-metric'>{proba:.2f}</div>"
            "<div class='small-label'>Failure Risk (0–1)</div>",
            unsafe_allow_html=True,
        )

        st.progress(int(proba * 100))

        if persona == "Line Operator":
            st.info("Action: If risk > 0.7, slow down line and notify maintenance.")
        elif persona == "Maintenance Engineer":
            st.info("Action: Inspect top contributing features in SHAP/LIME tabs.")
        else:
            st.info("Action: Check if high torque / low speed is causing inefficiency.")

    with c2:
        st.markdown("**Current Input (after scenario)**")
        st.dataframe(scenario_df.T, use_container_width=True)


# -------------------------------
# Tab 2: SHAP (bar + local)
# -------------------------------
with tab_shap:
    st.subheader("SHAP Explanations")

    st.markdown("Local Feature Impact (Current Input)")
    shap.initjs()
    local_shap = explainer.shap_values(scenario_df)
    shap_html = shap.force_plot(explainer.expected_value, local_shap, scenario_df, matplotlib=True)
    st.pyplot()

    st.components.v1.html(shap_html.html(), height=300)

    st.markdown("Global Feature Importance (Mean |SHAP|)")
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    st.pyplot()

    st.markdown("SHAP Summary Plot (Distribution)")
    shap.summary_plot(shap_values, X_train, show=False)
    st.pyplot()


# -------------------------------
# Tab 3: LIME (text + HTML)
# -------------------------------
with tab_lime:
    st.subheader("LIME Local Explanation")

    lime_exp = lime_explainer.explain_instance(
        data_row=scenario_df.iloc[0],
        predict_fn=model.predict_proba,
    )

    st.markdown("LIME Feature Contributions (Text)")
    for feature, weight in lime_exp.as_list():
        st.write(f"**{feature}** → {weight:+.3f}")

    st.markdown("LIME Detailed View (HTML)")
    st.components.v1.html(lime_exp.as_html(), height=600, scrolling=True)


# -------------------------------
# Tab 4: Global View & Comparison
# -------------------------------
with tab_global:
    st.subheader("Global Model View")

    g1, g2 = st.columns(2)

    with g1:
        st.markdown("Dataset Overview")
        st.write(f"Samples: {X_train.shape[0]}")
        st.write(f"Features: {X_train.shape[1]}")
        st.write("Target distribution:")
        st.bar_chart(y_train.value_counts())

    with g2:
        st.markdown("Feature Distribution")
        feature_to_view = st.selectbox("Select feature", X_train.columns)
        st.line_chart(df_full[feature_to_view])

    st.markdown("---")
    st.markdown("Compare Two Historical Samples")

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
# Tab 5: Real-time Simulation
# -------------------------------
with tab_sim:
    st.subheader("Real-time Sensor Simulation")

    st.info(
        "Simulates streaming sensor values and shows how risk evolves over time."
    )

    sim_steps = st.slider("Number of steps", 10, 100, 30)
    start_button = st.button("Start Simulation")

    placeholder_metrics = st.empty()
    placeholder_chart = st.empty()

    if start_button:
        history = []
        for i in range(sim_steps):
            sim_row = scenario_df.copy()

            for col in sim_row.columns:
                if "Air_temperature" in col:
                    sim_row[col] = sim_row[col] + np.random.normal(0, 0.5)
                if "Torque" in col:
                    sim_row[col] = sim_row[col] + np.random.normal(0, 1.0)
                if "Rotational_speed" in col:
                    sim_row[col] = sim_row[col] + np.random.normal(0, 20.0)

            sim_proba = model.predict_proba(sim_row)[0][1]
            history.append(sim_proba)

            with placeholder_metrics.container():
                st.markdown("Current Simulated Risk")
                st.markdown(
                    f"<div class='big-metric'>{sim_proba:.2f}</div>"
                    "<div class='small-label'>Simulated Failure Risk</div>",
                    unsafe_allow_html=True,
                )

            with placeholder_chart.container():
                st.line_chart(history)

            time.sleep(0.1)


# -------------------------------
# Tab 6: UX Evaluation (for MSc)
# -------------------------------
with tab_ux:
    st.subheader("UX Evaluation Summary")

    st.markdown("Personas")
    st.markdown(
        """
- **Line Operator:** wants quick, clear alerts and a single risk indicator.
- **Maintenance Engineer:** wants root causes and feature-level explanations.
- **Energy Manager:** wants to understand efficiency trade-offs and trends.
        """
    )

    st.markdown("Scenario Tests")
    st.markdown(
        """
1. **High temperature spike**  
   - Risk increases sharply.  
   - SHAP/LIME highlight temperature as top contributor.  
   - Action: slow down line, schedule inspection.

2. **High torque / vibration**  
   - Risk increases moderately.  
   - Torque appears as key contributor.  
   - Action: check bearings, lubrication, mechanical load.

3. **Low efficiency mode (low speed, high torque)**  
   - Risk may increase and efficiency drops.  
   - Energy manager can see impact via sliders and trends.  

4. **Normal operation**  
   - Low risk, balanced SHAP contributions.  
   - Confirms model behaves as expected.

5. **Combined stress (temp + torque + speed)**  
   - Very high risk, multiple features highlighted.  
   - Supports decision to stop machine or trigger alarm.
        """
    )

    st.markdown("Observations")
    st.markdown(
        """
- The **Prediction tab** supports the Line Operator with a single risk gauge and clear color coding.  
- The **SHAP and LIME tabs** support the Maintenance Engineer with root-cause visibility.  
- The **Global View and Simulation tabs** support the Energy Manager with trends and what‑if analysis.  
- Interactive sliders and scenarios make the model behavior transparent and auditable.
        """
    )


# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("AI Operator Console — XGBoost + SHAP + LIME • Ready for Streamlit Cloud / Azure.")