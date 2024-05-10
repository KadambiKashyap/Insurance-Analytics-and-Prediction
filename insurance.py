import streamlit as st
import pandas as pd
import pickle

# Load the model
file = open("F:\VS Code\insurance\models\gb_default_params.pkl", 'rb')
model = pickle.load(file)

# Load the data
data = pd.read_csv('./insurance_clean.csv')

# Function to get unique values for dropdowns
def get_unique_values(column_name):
    return sorted(data[column_name].unique())

# Streamlit UI
st.title('Predict Insurance Charges')

# Dropdowns for user input
age = st.slider('Age', min_value=18, max_value=64, value=30)
sex = st.selectbox('Sex', get_unique_values('sex'))
bmi = st.slider('BMI', min_value=15.0, max_value=53.0, value=25.0, step=0.1)
children = st.slider('Number of Children', min_value=0, max_value=5, value=1)
smoker = st.selectbox('Smoker', get_unique_values('smoker'))
region = st.selectbox('Region', get_unique_values('region'))

# Predict button
if st.button('Predict'):
    prediction = model.predict(pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                        columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']))
    st.write(f'Predicted Insurance Charges: {prediction[0]}')
