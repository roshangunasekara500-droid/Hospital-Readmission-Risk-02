import streamlit as st
import pandas as pd
import joblib

# ඔබගේ model ෆයිල් එකේ නම මෙතනට ලබා දෙන්න (උදා: 'model.pkl')
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

st.title("Patient Data Prediction App 🏥")
st.write("මෙම යෙදුම හරහා රෝගීන්ගේ දත්ත ඇතුලත් කර අදාළ තොරතුරු ලබාගත හැක.")

# Input fields සැකසීම (ඔබේ model එකට අවශ්‍ය අනෙකුත් inputs ද මෙලෙස එකතු කරන්න)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (වයස)", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender (ස්ත්‍රී/පුරුෂ භාවය)", ["Male", "Female"])
    height = st.number_input("Height in cm (උස)", value=170.0)
    weight = st.number_input("Weight in kg (බර)", value=70.0)

with col2:
    blood_group = st.selectbox("Blood Group (රුධිර කාණ්ඩය)", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    diabetes = st.selectbox("Diabetes", ["Yes", "No"])

# Submit Button
if st.button("Predict (ප්‍රතිඵලය ලබා දෙන්න)"):
    # Input data Dataframe එකක් බවට පත් කිරීම
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Height_cm': [height],
        'Weight_kg': [weight],
        'Blood_Group': [blood_group],
        'Smoking_Status': [smoking_status],
        'Hypertension': [hypertension],
        'Diabetes': [diabetes]
        # Model එක පුහුණු කළ අනෙකුත් සියලුම columns මෙතනට ඇතුලත් කරන්න
    })

    try:
        prediction = model.predict(input_data)
        st.success(f"Prediction Result: {prediction[0]}")
    except Exception as e:
        st.error(f"Error: කරුණාකර සියලුම අනිවාර්ය දත්ත (features) නිවැරදිව ලබා දී ඇත්දැයි පරීක්ෂා කරන්න. ({e})")
