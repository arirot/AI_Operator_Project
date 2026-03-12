import streamlit as st
import pandas as pd
import pickle
import shap
from src.xai_utils import get_shap_values, get_lime_explanation

# Load model + data
model = pickle.load(open("../models/model.pkl", "rb"))
df = pd.read_csv("../data/ai4i2020.csv")
X = df.drop("Machine failure", axis=1)

st.title("AI Operator Assistant – Production Line Digital Twin")

# Select instance
idx = st.slider("Select data index", 0, len(X)-1, 0)
instance = X.iloc[idx:idx+1]

st.subheader("Model Prediction")
pred = model.predict_proba(instance)[0][1]
st.metric("Failure Risk", f"{pred:.2f}")

# SHAP explanation
st.subheader("SHAP Explanation")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(instance)
st.bar_chart(pd.DataFrame(shap_values[0], index=X.columns))

# LIME explanation
st.subheader("LIME Explanation")
lime_exp = get_lime_explanation(model, X, X.columns, instance.iloc[0])
for feature, weight in lime_exp:
    st.write(f"{feature}: {weight}")

# What-if analysis
st.subheader("What-if Analysis")
feature = st.selectbox("Feature to modify", X.columns)
new_value = st.number_input("New value", value=float(instance[feature]))

modified = instance.copy()
modified[feature] = new_value
new_pred = model.predict_proba(modified)[0][1]

st.metric("New Failure Risk", f"{new_pred:.2f}")
