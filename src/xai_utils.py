import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

def get_shap_values(model, X_train, instance):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(instance)
    return shap_values

def get_lime_explanation(model, X_train, X_columns, instance):
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_columns,
        class_names=["No Failure", "Failure"],
        mode="classification"
    )
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba
    )
    return exp.as_list()
