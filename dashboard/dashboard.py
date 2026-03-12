import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
from lime.lime_tabular import LimeTabularExplainer

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Operator Dashboard",
    layout="wide",
)

st.title("AI Operator Dashboard")
st.write("Real‑time machine failure prediction with SHAP & LIME explanations.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    from pathlib import Path
    base_path = Path(__file__).resolve().parent
    model_path = base_path.parent / "models" / "model.pkl"
    return joblib.load(model_path)

model = load_model()   # <-- MUST be here, at top level

# -----------------------------
# Load Training Data (for LIME)
# -----------------------------
@st.cache_resource
def load_training_data():
    from pathlib import Path
    base_path = Path(__file__).resolve().parent
    data_path = base_path.parent / "data" / "ai4i2020.csv"
    df = pd.read_csv(data_path)

    #Similar cleaning as in starterM.ipynb 
    df.columns = (
        df.columns
        .str.replace('[', '', regex=False)
        .str.replace(']', '', regex=False)
        .str.replace('<', '', regex=False)
        .str.replace('>', '', regex=False)
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .str.strip()
    )
    df.columns = df.columns.astype(str)

    # One-hot encode categorical columns (same as notebook)
    cat_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Split features/target

    X = df.drop("Machine_failure", axis=1)
    y = df["Machine_failure"]
    return X, y

X_train, y_train = load_training_data()

# -----------------------------
# SHAP Initialization
# -----------------------------
explainer = shap.TreeExplainer(model)

# -----------------------------
# LIME Initialization
# -----------------------------
lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=["No Failure", "Failure"],
    mode="classification"
)

# -----------------------------
# Sidebar Input Form
# -----------------------------
st.sidebar.header("Input Machine Parameters")

input_data = {}
for col in X_train.columns:
    default_val = float(X_train[col].mean())
    input_data[col] = st.sidebar.number_input(col, value=default_val)

input_df = pd.DataFrame([input_data])

# -----------------------------
# Prediction
# -----------------------------
st.subheader("Prediction Result")

proba = model.predict_proba(input_df)[0][1]
pred = model.predict(input_df)[0]

if pred == 1:
    st.error(f"Machine Failure Likely — Risk Score: {proba:.2f}")
else:
    st.success(f"Machine Operating Normally — Risk Score: {proba:.2f}")

# -----------------------------
# SHAP Explanation
# -----------------------------
st.subheader("SHAP Explanation")

shap_values = explainer.shap_values(input_df)

# Force plot
st.write("### SHAP Force Plot")
shap_html = shap.force_plot(
    explainer.expected_value,
    shap_values,
    input_df,
    matplotlib=False
)
st.components.v1.html(shap_html.html(), height=300)

# Summary plot
st.write("### SHAP Summary Plot")
#st.set_option('deprecation.showPyplotGlobalUse', False)
shap.summary_plot(shap_values, input_df, show=False)
st.pyplot()

# -----------------------------
# LIME Explanation
# -----------------------------
st.subheader("LIME Explanation")

lime_exp = lime_explainer.explain_instance(
    data_row=input_df.iloc[0],
    predict_fn=model.predict_proba
)

st.write("### LIME Feature Contributions")
st.components.v1.html(lime_exp.as_html(), height=600)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("AI Operator Dashboard — Powered by XGBoost, SHAP, and LIME.")