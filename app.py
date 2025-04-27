import streamlit as st
import joblib
import pandas as pd

# Load the saved model
try:
    model = joblib.load('best_logistic_model.joblib')
except FileNotFoundError:
    st.error("Error: 'best_logistic_model.joblib' not found. Make sure it's in the same directory as this app.")
    st.stop()

st.title('Profitability Prediction')

# Define input fields based on your original features
ship_mode = st.selectbox('Ship Mode', ['Same Day', 'Second Class', 'Standard Class'])
segment = st.selectbox('Segment', ['Consumer', 'Corporate', 'Home Office'])
category = st.selectbox('Category', ['Furniture', 'Office Supplies', 'Technology'])
# ... add input fields for other categorical and numerical features

# Create a DataFrame from the input
input_data = pd.DataFrame({
    'Ship Mode': [ship_mode],
    'Segment': [segment],
    'Category': [category],
    # ... add other features here with appropriate input widgets
})

# --- Preprocessing (Adapt this to your actual preprocessing) ---
# Example for One-Hot Encoding (you might need to be more specific with columns)
categorical_cols = ['Ship Mode', 'Segment', 'Category'] # Add all your categorical columns
processed_data = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

# Ensure all columns the model expects are present (add missing ones with 0)
expected_cols = model.feature_names_in_.tolist() # Get feature names from the trained model
for col in expected_cols:
    if col not in processed_data.columns:
        processed_data[col] = 0
processed_data = processed_data[expected_cols] # Ensure correct column order

if st.button('Predict Profitability'):
    prediction = model.predict(processed_data)
    if prediction[0] == 1:
        st.success('The order is predicted to be PROFITABLE.')
    else:
        st.error('The order is predicted to be NOT PROFITABLE.')