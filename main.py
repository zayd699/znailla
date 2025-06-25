import streamlit as st
import pandas as pd
import joblib
import lightgbm

# --- CONFIGURATION ---
MODEL_PATH = "/Users/macbookpro/Desktop/DOCS/freelance/znailla_bank/models/bank_marketing_final_model.pkl"
THRESHOLD = 0.06

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --- STREAMLIT UI ---
st.title(" Znailla Bank - Term Deposit Campaign Inference")
st.write("Upload a CSV file with customer data to identify who should be contacted.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # Read uploaded CSV
    new_data = pd.read_csv(uploaded_file, sep=';')

    # Predict subscription probabilities
    probabilities = model.predict_proba(new_data)[:, 1]
    new_data['subscription_probability'] = probabilities
    new_data['should_be_contacted'] = new_data['subscription_probability'] >= THRESHOLD

    # Sort and filter
    scored = new_data.sort_values(by='subscription_probability', ascending=False)
    to_call = scored[scored['should_be_contacted']]

    st.success(f" {len(to_call)} customers should be contacted out of {len(new_data)}")

    st.subheader("List of Customers to Contact")
    st.dataframe(to_call.reset_index(drop=True))

    csv = to_call.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Contact List as CSV",
        data=csv,
        file_name="customers_to_contact.csv",
        mime="text/csv"
    )
