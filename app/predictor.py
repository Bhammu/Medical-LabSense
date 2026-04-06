"""
predictor.py
Loads the trained model and runs predictions with SHAP explanations.
"""

import joblib
import numpy as np
import pandas as pd
import shap
import os

MODEL_PATH    = os.path.join(os.path.dirname(__file__), "../model/medical_labsense_model.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "../model/medical_labsense_features.pkl")


def load_model():
    model    = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features


def predict(model, feature_row: pd.DataFrame):
    """
    Runs prediction and returns:
    - prediction: 0 (Healthy) or 1 (Iron Anemia)
    - probability: confidence score
    - label: human-readable string
    """
    pred  = model.predict(feature_row)[0]
    proba = model.predict_proba(feature_row)[0]

    label = "Iron Deficiency Anemia Detected" if pred == 1 else "No Iron Anemia Detected"
    confidence = proba[pred]

    return {
        "prediction":  int(pred),
        "label":       label,
        "confidence":  round(float(confidence) * 100, 1),
        "prob_healthy": round(float(proba[0]) * 100, 1),
        "prob_anemia":  round(float(proba[1]) * 100, 1),
    }


def get_shap_values(model, feature_row: pd.DataFrame, feature_names: list):
    """
    Returns SHAP values for a single prediction.
    Works with XGBoost and RandomForest pipelines.
    """
    try:
        # Get the actual model step from pipeline
        clf = model.named_steps["model"]

        # Apply preprocessing steps only (not resampler/model)
        from sklearn.pipeline import Pipeline as SKPipeline
        preproc_steps = [(k, v) for k, v in model.steps
                         if k not in ["model", "resampler"]]

        if preproc_steps:
            preproc = SKPipeline(preproc_steps)
            X_transformed = preproc.transform(feature_row)
        else:
            X_transformed = feature_row.values

        explainer   = shap.TreeExplainer(clf)
        shap_vals   = explainer.shap_values(X_transformed)

        # For binary classification shap_values can be list of 2 arrays
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]  # class 1 (anemia)
        else:
            vals = shap_vals[0]

        n = min(len(vals), len(feature_names))
        shap_df = pd.DataFrame({
            "feature":    feature_names[:n],
            "shap_value": vals[:n],
            "abs_shap":   np.abs(vals[:n])
        }).sort_values("abs_shap", ascending=False)

        return shap_df

    except Exception as e:
        return None


def get_interpretation(result: dict, flags: list) -> str:
    """Generates a plain-English interpretation of the result."""
    lines = []

    if result["prediction"] == 1:
        lines.append("The model indicates a **high likelihood of iron deficiency anemia**.")
        low_flags = [f for f in flags if f["status"] == "LOW"
                     and f["feature"] in ["HGB", "MCV", "MCH", "TSD", "FERRITTE"]]
        if low_flags:
            features = ", ".join(f["feature"] for f in low_flags)
            lines.append(f"Key low values: **{features}** — consistent with iron deficiency pattern.")
    else:
        lines.append("The model does **not detect iron deficiency anemia** based on these values.")
        if flags:
            lines.append("Some values are outside reference ranges — consult a physician for full evaluation.")

    lines.append("\n⚠️ *This is a screening tool only. Always confirm with a licensed clinician.*")
    return "\n\n".join(lines)
