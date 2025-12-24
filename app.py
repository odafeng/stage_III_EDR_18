# app.py
# Streamlit inference-only app aligned with trained artifacts
# Uses:
#   - final_feature_columns.pkl
#   - final_knn_imputer.pkl
#   - final_model_calibrated.pkl

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Stage III Colon Cancer EDR-18 Risk Calculator",
    page_icon="🧬",
    layout="centered",
)

st.title("Stage III Colon Cancer EDR-18 Risk Calculator")
st.caption(
    "Pathology-based prediction of early distant recurrence within 18 months (EDR-18). "
    "For academic and research use."
)

# =========================
# Constants
# =========================
DEFAULT_THRESHOLD = 0.12

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR)

FEATURE_COL_PATH = os.path.join(MODEL_DIR, "final_feature_columns.pkl")
IMPUTER_PATH = os.path.join(MODEL_DIR, "final_knn_imputer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "final_model_calibrated.pkl")

AJCC_OPTIONS = ["IIIA", "IIIB", "IIIC"]
DIFF_OPTIONS = [
    ("Well / Moderate", 1),
    ("Poor / Undifferentiated", 2),
]

# =========================
# Load artifacts (cached)
# =========================
@st.cache_resource
def load_artifacts():
    feature_cols = joblib.load(FEATURE_COL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    model = joblib.load(MODEL_PATH)
    return feature_cols, imputer, model


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("About")
    st.write(
        "This calculator performs inference using a pre-trained model. "
        "If the service is temporarily unavailable, please refresh and try again."
    )
    st.write(f"Decision threshold: **{DEFAULT_THRESHOLD:.2f}**")


# =========================
# Helper functions
# =========================
def compute_lnr(ln_pos, ln_total):
    if ln_total <= 0:
        return np.nan
    return ln_pos / ln_total


def build_raw_input(
    ajcc_substage,
    lnr,
    pni,
    differentiation,
):
    """
    Build raw (pre-imputation, pre-encoding) input.
    Column names MUST match those used during training.
    """
    return pd.DataFrame(
        [
            {
                "AJCC_Substage": ajcc_substage,
                "LNR": lnr,
                "PNI": pni,
                "Differentiation": differentiation,
            }
        ]
    )


# =========================
# UI: Inputs
# =========================
st.subheader("Enter pathology variables")

col1, col2 = st.columns(2)

with col1:
    ajcc_substage = st.selectbox("AJCC substage", AJCC_OPTIONS, index=1)
    pni_present = st.selectbox("Perineural invasion (PNI)", ["Absent", "Present"], index=0)
    pni = 1 if pni_present == "Present" else 0

with col2:
    diff_label, diff_num = st.selectbox(
        "Tumor differentiation",
        DIFF_OPTIONS,
        index=0,
        format_func=lambda x: x[0],
    )

lnr_mode = st.radio(
    "Lymph node ratio (LNR)",
    ["Compute from nodes", "Enter directly"],
    horizontal=True,
)

lnr = np.nan
if lnr_mode == "Compute from nodes":
    c1, c2 = st.columns(2)
    with c1:
        ln_total = st.number_input("Total lymph nodes examined", min_value=0, value=12, step=1)
    with c2:
        ln_pos = st.number_input("Positive lymph nodes", min_value=0, value=1, step=1)
    lnr = compute_lnr(ln_pos, ln_total)
    if not np.isnan(lnr):
        st.info(f"Computed LNR = **{lnr:.3f}**")
else:
    lnr = st.number_input("LNR (0–1)", min_value=0.0, max_value=1.0, value=0.10, step=0.01)

st.divider()

# =========================
# Run inference
# =========================
if st.button("Calculate risk", type="primary", use_container_width=True):

    if np.isnan(lnr):
        st.error("Please provide a valid LNR.")
        st.stop()

    try:
        feature_cols, imputer, model = load_artifacts()
    except Exception:
        st.error("Model is temporarily unavailable. Please refresh and try again.")
        st.stop()

    # Build raw input
    X_raw = build_raw_input(
        ajcc_substage=ajcc_substage,
        lnr=lnr,
        pni=pni,
        differentiation=diff_num,
    )

    # Reindex to training feature space (CRITICAL)
    X_aligned = X_raw.reindex(columns=feature_cols)

    # Impute
    try:
        X_imputed = pd.DataFrame(
            imputer.transform(X_aligned),
            columns=feature_cols,
        )
    except Exception:
        st.error("Input processing failed. Please check inputs and retry.")
        st.stop()

    # Predict
    try:
        proba = float(model.predict_proba(X_imputed)[0, 1])
    except Exception:
        st.error("Prediction failed. Please refresh and try again.")
        st.stop()

    # =========================
    # Output
    # =========================
    st.subheader("Result")
    st.metric("Predicted probability of EDR-18", f"{proba:.3f}")

    if proba >= DEFAULT_THRESHOLD:
        st.warning(f"Risk group: **High risk** (≥ {DEFAULT_THRESHOLD:.2f})")
        st.write(
            "This classification suggests considering closer surveillance or further evaluation, "
            "depending on clinical context."
        )
    else:
        st.success(f"Risk group: **Low risk** (< {DEFAULT_THRESHOLD:.2f})")
        st.write(
            "Low-risk classification is intended to support **rule-out** decisions "
            "and conservative surveillance strategies."
        )

    with st.expander("Show processed model inputs"):
        st.dataframe(X_imputed, use_container_width=True)

    st.caption(
        "Predictions are probabilistic and should be interpreted in conjunction with clinical judgment."
    )
