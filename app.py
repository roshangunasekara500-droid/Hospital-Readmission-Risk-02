import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model and scaler
try:
    with open('readmission_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Model or scaler files not found. Please run the model training cell first.")
    st.stop()

st.title('Hospital Readmission Risk Prediction')
st.write('Enter patient details to predict the likelihood of readmission.')

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Patient Age (18-90)', min_value=18, max_value=90, value=50, step=1)
    length_of_stay = st.number_input('Length of Stay (Days, 1-14)', min_value=1, max_value=14, value=5, step=1)
    num_lab_procedures = st.number_input('Number of Lab Procedures (1-80)', min_value=1, max_value=80, value=20, step=1)

with col2:
    num_medications = st.number_input('Number of Medications (1-30)', min_value=1, max_value=30, value=10, step=1)
    num_emergency_visits = st.number_input('Number of Emergency Visits (0-3)', min_value=0, max_value=3, value=0, step=1)
    chronic_conditions = st.number_input('Number of Chronic Conditions (0-3)', min_value=0, max_value=3, value=0, step=1)

# Create a button to predict
if st.button('Predict Readmission Risk', type='primary'):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame([{
        'Age': age,
        'Length_of_Stay_Days': length_of_stay,
        'Num_Lab_Procedures': num_lab_procedures,
        'Num_Medications': num_medications,
        'Num_Emergency_Visits': num_emergency_visits,
        'Chronic_Conditions': chronic_conditions
    }])

    # Scale the input data
    scaled_input_data = scaler.transform(input_data)

    # Predict readmission probability
    readmission_probability = model.predict_proba(scaled_input_data)[0][1] # Probability of class 1 (readmitted)

    # Display the result
    st.subheader(f"Predicted Readmission Probability: {readmission_probability:.2%}")

    if readmission_probability >= 0.5:
        st.error('High Readmission Risk! Consider intervention.')
    else:
        st.success('Low Readmission Risk.')
