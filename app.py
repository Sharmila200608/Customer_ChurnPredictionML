# ============================================================
# app.py
# Streamlit web app for Customer Churn Prediction
# Run with:  streamlit run app.py
# ============================================================

import pickle
import numpy as np
import streamlit as st

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered",
)

# ── Load Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the saved model and feature list from model.pkl"""
    with open("model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["features"]


try:
    model, feature_names = load_model()
except FileNotFoundError:
    st.error("⚠️ model.pkl not found! Please run `python train.py` first.")
    st.stop()

# ── App Header ───────────────────────────────────────────────
st.title("📉 Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict whether they will churn.")
st.divider()

# ── Input Form ───────────────────────────────────────────────
st.subheader("🧾 Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender          = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner         = st.selectbox("Has Partner", ["Yes", "No"])
    dependents      = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

with col2:
    internet_service  = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_security   = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup     = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support      = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv      = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies  = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.divider()
st.subheader("💳 Billing & Contract")

col3, col4 = st.columns(2)

with col3:
    contract         = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

with col4:
    payment_method   = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges  = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
    total_charges    = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=monthly_charges * tenure, step=1.0)

st.divider()

# ── Encoding Helpers ─────────────────────────────────────────
# These must match the LabelEncoder mappings from preprocessing.py.
# LabelEncoder sorts unique values alphabetically and assigns 0, 1, 2 ...

ENCODE = {
    # Binary Yes/No → sorted: No=0, Yes=1
    "Yes": 1, "No": 0,
    # Gender → Female=0, Male=1
    "Female": 0, "Male": 1,
    # Internet Service → DSL=0, Fiber optic=1, No=2
    "DSL": 0, "Fiber optic": 1,
    # MultipleLines / OnlineSecurity etc. have 3 values
    "No phone service": 2, "No internet service": 2,
    # Contract → Month-to-month=0, One year=1, Two year=2
    "Month-to-month": 0, "One year": 1, "Two year": 2,
    # Payment Method → alphabetical order
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3,
}

def encode(val):
    """Convert a string value to its numeric label encoding."""
    return ENCODE.get(val, 0)


# ── Predict Button ───────────────────────────────────────────
if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):

    # Build feature vector in the SAME order as training features
    input_data = {
        "gender"           : encode(gender),
        "SeniorCitizen"    : encode(senior_citizen),
        "Partner"          : encode(partner),
        "Dependents"       : encode(dependents),
        "tenure"           : tenure,
        "PhoneService"     : encode(phone_service),
        "MultipleLines"    : encode(multiple_lines),
        "InternetService"  : encode(internet_service),
        "OnlineSecurity"   : encode(online_security),
        "OnlineBackup"     : encode(online_backup),
        "DeviceProtection" : encode(device_protection),
        "TechSupport"      : encode(tech_support),
        "StreamingTV"      : encode(streaming_tv),
        "StreamingMovies"  : encode(streaming_movies),
        "Contract"         : encode(contract),
        "PaperlessBilling" : encode(paperless_billing),
        "PaymentMethod"    : encode(payment_method),
        "MonthlyCharges"   : monthly_charges,
        "TotalCharges"     : total_charges,
    }

    # Align with training feature order
    feature_vector = np.array([[input_data[f] for f in feature_names]])

    # Prediction
    prediction   = model.predict(feature_vector)[0]
    probability  = model.predict_proba(feature_vector)[0]
    churn_prob   = probability[1] * 100   # probability of churn (class=1)
    retain_prob  = probability[0] * 100   # probability of staying

    # ── Display Results ──────────────────────────────────────
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️  **This customer is likely to CHURN**")
    else:
        st.success(f"✅  **This customer is likely to STAY**")

    col_a, col_b = st.columns(2)
    col_a.metric("🔴 Churn Probability",  f"{churn_prob:.1f}%")
    col_b.metric("🟢 Retention Probability", f"{retain_prob:.1f}%")

    # Progress bar for churn risk
    st.markdown("**Churn Risk Level:**")
    st.progress(int(churn_prob))

    # Actionable insight
    if churn_prob >= 70:
        st.warning("🚨 High risk! Consider offering a discount or contract upgrade.")
    elif churn_prob >= 40:
        st.info("🔔 Medium risk. Proactive engagement recommended.")
    else:
        st.success("😊 Low risk. Customer appears satisfied.")

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with ❤️ using Scikit-learn & Streamlit | Telco Churn Dataset")
