import streamlit as st
import pandas as pd
import joblib
import json


@st.cache_resource
def load_model():
    return joblib.load("rf_classifier.pkl")

@st.cache_resource
def load_encoder(filename):
    return joblib.load(filename)

@st.cache_data
def load_json(filename):
    with open(filename) as f:
        return json.load(f)

@st.cache_data
def load_train_columns():
    return joblib.load("train_columns.pkl")

# Load resources once
rf_classifier = load_model()
te = load_encoder("target_encoder.pkl")
ohe = load_encoder("ohe.pkl")
scaler = load_encoder("scaler.pkl")
train_columns = load_train_columns()

doctor_list = load_json('doctor.json')
hospital_list = load_json('hospital.json')

gender_options = ["Male", "Female"]
blood_type_options = ["A+","A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
medical_condition_options = ["Asthma", "Cancer", "Diabetes", "Hypertension", "Obesity"]
insurance_provider_options = ["Blue Cross", "Cigna", "Medicare", "UnitedHealthcare"]
admission_type_options = ["Emergency", "Urgent", "Elective"]
medication_options = ["Ibuprofen", "Lipitor", "Paracetamol", "Penicillin"]
test_results_options = ["Inconclusive", "Normal", "Abnormal"]


st.markdown(
    """
    <h1 style='text-align: center;'>🏥 Hospital Length of Stay Prediction</h1>
    <p style='text-align: center; font-size: 18px;'>
        This survey collects patient-related data, along with hospital and 
        information, to predict how long a patient is likely to stay. These 
        predictions help hospitals better allocate resources, schedule staff,
        and provide improved patient care.
    </p>
    """,
    unsafe_allow_html=True
)

# Numeric Inputs
age = st.number_input("Age", min_value=0, max_value=120, value=30)
billing_amount = st.number_input("Billing Amount", min_value=0.0, value=5000.0)

# Encoded Columns
doctor = st.selectbox("Doctor", doctor_list)
hospital = st.selectbox("Hospital", hospital_list)
gender = st.selectbox("Gender", gender_options)
blood_type = st.selectbox("Blood Type", blood_type_options)
medical_condition = st.selectbox("Medical Condition", medical_condition_options)
insurance_provider = st.selectbox("Insurance Provider", insurance_provider_options)
admission_type = st.selectbox("Admission Type", admission_type_options)
medication = st.selectbox("Medication", medication_options)
test_results = st.selectbox("Test Results", test_results_options)

# --------------------------
# Predict Button
# --------------------------
if st.button("Predict Stay Type"):
    # Create DataFrame with one row
    input_df = pd.DataFrame({
        "Age": [age],
        "Billing Amount": [billing_amount],
        "Doctor": [doctor],
        "Hospital": [hospital],
        "Gender": [gender],
        "Blood Type": [blood_type],
        "Medical Condition": [medical_condition],
        "Insurance Provider": [insurance_provider],
        "Admission Type": [admission_type],
        "Medication": [medication],
        "Test Results": [test_results]
    })

    # Target encode
    target_encode_cols = ['Doctor', 'Hospital']
    input_df[target_encode_cols] = te.transform(input_df[target_encode_cols])

    # One-hot encode
    categorical_cols = ['Gender', 'Blood Type', 'Medical Condition',
                         'Insurance Provider', 'Admission Type',
                         'Medication', 'Test Results']
    ohe_array = ohe.transform(input_df[categorical_cols])
    ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(categorical_cols))

    # Drop original categorical cols and merge OHE
    input_df = input_df.drop(columns=categorical_cols)
    input_df = pd.concat([input_df, ohe_df], axis=1)

    # Scale numerical cols
    numerical_cols = ['Age', 'Billing Amount']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Reindex to match training columns
    input_df = input_df.reindex(columns=train_columns, fill_value=0)

    # Predict
    prediction = rf_classifier.predict(input_df)[0]

    # ... prediction code ...
    
    stay_type_map = {
    0: "🏥 Stay Duration: up to 2 days",
    1: "🏥 Stay Duration: 3–6 days",
    2: "🏥 Stay Duration: more than 6 days"
    }

    st.markdown(f"<h3 style='text-align: center;'>{stay_type_map[prediction]}</h3>", unsafe_allow_html=True)
