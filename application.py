import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from xgboost import XGBClassifier

# Load the model from the Pickle file
with open('model3.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Preprocessing function
def preprocess_input(age, gender, ap_hi, ap_lo, bmi, cholesterol, gluc, smoke, alco, active):
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'ap_hi': [ap_hi],
        'ap_lo': [ap_lo],
        'bmi': [bmi],
        'cholesterol': [cholesterol],
        'gluc': [gluc],
        'smoke': [smoke],
        'alco': [alco],
        'active': [active]
    })

    # Apply the same preprocessing steps as during training
    input_data['gender'] = input_data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    input_data['smoke'] = input_data['smoke'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['alco'] = input_data['alco'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['active'] = input_data['active'].apply(lambda x: 1 if x == 'Active' else 0)

    return input_data

# Streamlit UI
st.title('Heart Disease Prediction App')
st.sidebar.header('User Input Features')

# Collect user input with Streamlit widgets
age = st.sidebar.slider('Age', 18, 100, 25)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
ap_hi = st.sidebar.slider('Systolic Blood Pressure (mm Hg)', 80, 250, 120)
ap_lo = st.sidebar.slider('Diastolic Blood Pressure (mm Hg)', 50, 150, 80)
bmi = st.sidebar.slider('Body Mass Index (BMI)', 15.0, 45.0, 25.0)
cholesterol = st.sidebar.selectbox('Cholesterol Level', [1, 2, 3])
gluc = st.sidebar.selectbox('Glucose Level', [1, 2, 3])
smoke = st.sidebar.selectbox('Smoking', ['Yes', 'No'])
alco = st.sidebar.selectbox('Alcohol Consumption', ['Yes', 'No'])
active = st.sidebar.selectbox('Physical Activity', ['Active', 'Inactive'])

# Preprocess user input
input_data = preprocess_input(age, gender, ap_hi, ap_lo, bmi, cholesterol, gluc, smoke, alco, active)

# Make predictions
if st.sidebar.button('Predict'):
    try:
        prediction = loaded_model.predict(input_data)
        if prediction[0] == 1:
            st.write('The model predicts that the individual has a high chance of heart problems.')
        else:
            st.write('The model predicts that the individual has low chance of heart problems.')
    except Exception as e:
        st.error(f"An error occurred: {e}")
