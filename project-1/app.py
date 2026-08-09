"""
Creditworthiness Prediction App
=================================
A Streamlit web application that loads a pre-trained credit-risk model
(best_model.pkl) and predicts whether an applicant is likely to be
"Creditworthy / Approved" or "High Risk / Denied", based on the top 5
most predictive features identified during model training.

Run with:  streamlit run app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Creditworthiness Predictor",
    page_icon="💳",
    layout="centered",
)

import os

# Get the absolute path to the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the exact path to the model file
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")


# ------------------------------------------------------------------
# LOAD MODEL BUNDLE (cached so it only loads once per session)
# ------------------------------------------------------------------
@st.cache_resource
def load_model_bundle(path: str):
    """Load the pickled model bundle produced by the notebook (Section 10)."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


try:
    bundle = load_model_bundle(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find '{MODEL_PATH}'. Please run the Jupyter Notebook "
        "through Section 10 (Export Model) first, and place the resulting "
        "'best_model.pkl' file in the same folder as this app."
    )
    st.stop()

model = bundle["model"]
model_name = bundle.get("model_name", "Best Model")
scaler = bundle.get("scaler", None)
encoders = bundle.get("encoders", {})
feature_names = bundle["feature_names"]          # full ordered feature list the model expects
top_features = bundle["top_features"]             # top 5 most important features
feature_defaults = bundle["feature_defaults"]     # median/mode fallback values for the rest
feature_importances = bundle.get("feature_importances", {})

# Human-readable labels & help text for the top-5 engineered features.
# (These map 1:1 to the engineered columns produced in the notebook.)
FEATURE_META = {
    "MONTHS_SINCE_LAST_DELINQ_HIST": {
        "label": "Months Since Last Late Payment (60+ days)",
        "help": "How many months ago the applicant's last serious delinquency occurred. "
                "Larger = better (further in the past, or never).",
        "min": 0, "max": 60, "step": 1, "kind": "int",
    },
    "ONTIME_RATIO_HIST": {
        "label": "On-Time Payment Ratio",
        "help": "Share of recorded credit-history months paid on time (0 = never on time, "
                "1 = always on time).",
        "min": 0.0, "max": 1.0, "step": 0.01, "kind": "float",
    },
    "TOTAL_DEFAULTS_HIST": {
        "label": "Number of Severe Delinquencies (60+ days past due)",
        "help": "Count of months in the applicant's credit history flagged as 60+ days "
                "past due. Lower = better.",
        "min": 0, "max": 30, "step": 1, "kind": "int",
    },
    "ACCOUNT_AGE_MONTHS": {
        "label": "Length of Credit History (months)",
        "help": "How many months since the applicant's earliest credit record.",
        "min": 0, "max": 60, "step": 1, "kind": "int",
    },
    "TOTAL_RECORDS_HIST": {
        "label": "Number of Recorded Credit-History Months",
        "help": "Total number of monthly records available for this applicant "
                "(more history generally means a more reliable prediction).",
        "min": 0, "max": 60, "step": 1, "kind": "int",
    },
}


# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.title("💳 Creditworthiness Predictor")
st.write(
    f"This app uses a trained **{model_name}** model to estimate whether an "
    "applicant is likely to be creditworthy, based on the "
    f"**top {len(top_features)} most important features** identified during "
    "model training."
)

with st.expander("ℹ️ About this model"):
    st.write(
        "The model was trained on historical credit-bureau behavior "
        "(payment history) and demographic/application data, using a "
        "time-based split so that only *past* behavior is used to predict "
        "*future* risk (avoiding data leakage)."
    )
    if feature_importances:
        imp_df = (
            pd.Series(feature_importances)
            .sort_values(ascending=False)
            .head(10)
            .rename("Importance")
            .to_frame()
        )
        st.write("**Top 10 feature importances:**")
        st.bar_chart(imp_df)
    metrics = bundle.get("metrics")
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
                # Lowercase all keys in the dictionary to make lookup case-insensitive
        clean_metrics = {k.lower(): v for k, v in metrics.items()}
        
        c1.metric("Precision", f"{clean_metrics.get('precision', 0.0):.2f}" if 'precision' in clean_metrics else "N/A")
        c2.metric("Recall", f"{clean_metrics.get('recall', 0.0):.2f}" if 'recall' in clean_metrics else "N/A")
        c3.metric("F1-Score", f"{clean_metrics.get('f1', 0.0):.2f}" if 'f1' in clean_metrics else "N/A")
        c4.metric("ROC-AUC", f"{clean_metrics.get('roc_auc', 0.0):.2f}" if 'roc_auc' in clean_metrics else "N/A")


st.divider()

# ------------------------------------------------------------------
# INTERACTIVE INPUT FORM (only the top-5 features)
# ------------------------------------------------------------------
st.subheader("Enter Applicant Details")

user_inputs = {}
with st.form("credit_form"):
    for feat in top_features:
        meta = FEATURE_META.get(feat)
        default_val = feature_defaults.get(feat, 0)

        if meta is None:
            # Fallback for any feature without custom metadata
            user_inputs[feat] = st.number_input(feat, value=float(default_val))
            continue

        if meta["kind"] == "int":
            user_inputs[feat] = st.slider(
                meta["label"],
                min_value=int(meta["min"]),
                max_value=int(meta["max"]),
                value=int(np.clip(default_val, meta["min"], meta["max"])),
                step=int(meta["step"]),
                help=meta["help"],
            )
        else:
            user_inputs[feat] = st.slider(
                meta["label"],
                min_value=float(meta["min"]),
                max_value=float(meta["max"]),
                value=float(np.clip(default_val, meta["min"], meta["max"])),
                step=float(meta["step"]),
                help=meta["help"],
            )

    submitted = st.form_submit_button("🔍 Predict Creditworthiness", use_container_width=True)


# ------------------------------------------------------------------
# BUILD FEATURE VECTOR + PREDICT
# ------------------------------------------------------------------
def build_feature_vector(user_inputs: dict) -> pd.DataFrame:
    """
    Assemble a single-row DataFrame with ALL features the model expects,
    in the exact order used at training time. The 5 features the user
    provided are inserted directly; every other feature (features the
    app does not collect from the user) is filled with its training-set
    median (numeric) or mode (categorical), then categorical columns are
    label-encoded with the fitted encoders from the notebook.
    """
    row = {}
    for col in feature_names:
        if col in user_inputs:
            row[col] = user_inputs[col]
        else:
            row[col] = feature_defaults.get(col, 0)

    # dtype=object avoids pandas inferring a strict 'string' dtype for
    # text columns, which would later reject the integer-encoded values.
    df_row = pd.DataFrame([row], columns=feature_names, dtype=object)

    # Encode any categorical columns using the encoders saved from training
    for col, encoder in encoders.items():
        if col in df_row.columns:
            raw_value = df_row.at[0, col]
            try:
                df_row.at[0, col] = int(encoder.transform([raw_value])[0])
            except (ValueError, TypeError):
                # Unseen category fallback -> encode as the most frequent
                # training-time class (index 0 of the encoder's classes_)
                df_row.at[0, col] = 0

    # Ensure numeric dtype for the model
    df_row = df_row.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df_row


if submitted:
    X_input = build_feature_vector(user_inputs)

    # Random Forest / Decision Tree don't need scaling; Logistic Regression does.
    if model_name == "Logistic Regression" and scaler is not None:
        X_for_model = scaler.transform(X_input)
    else:
        X_for_model = X_input

    prediction = model.predict(X_for_model)[0]
    probability = model.predict_proba(X_for_model)[0][1]  # P(high risk)

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.error("### ❌ Denied / High Risk")
        st.write(
            f"The model estimates a **{probability:.1%} probability** that this "
            "applicant will become a high credit risk (60+ days past due) "
            "in the near term."
        )
    else:
        st.success("### ✅ Approved / Creditworthy")
        st.write(
            f"The model estimates only a **{probability:.1%} probability** of "
            "high credit risk for this applicant."
        )

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        "Note: This prediction is generated by a statistical model trained on "
        "historical data and should be used as a decision-support tool, not as "
        "the sole basis for a real lending decision."
    )

st.divider()
st.caption(
    "Model artifact: best_model.pkl · Built with scikit-learn & Streamlit"
)
