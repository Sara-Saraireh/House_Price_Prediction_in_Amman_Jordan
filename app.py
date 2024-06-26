import streamlit as st
import joblib
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# Define a custom transformer to map floor descriptions to numerical values
class FloorMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.floor_mapping = {'Ground Floor': 0, 'Semi-Ground Floor': -1, 'Basement': -2, 'First Floor': 1,
                              'Second Floor': 2, 'Third Floor': 3, 'Fourth Floor': 4, 'Fifth Floor': 5,
                              'Last Floor With Roof': 6}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X['floor'].map(self.floor_mapping).values.reshape(-1, 1)

# Define a custom transformer to calculate total rooms
class TotalRoomsCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Assuming X is a DataFrame with columns 'number of rooms' and 'number of bathrooms'
        total_rooms = X.iloc[:, 0] + X.iloc[:, 1]  # Access columns by position
        return total_rooms.values.reshape(-1, 1)
    

# Load the preprocessing pipeline
pipeline = joblib.load('/preprocessing_pipeline.joblib')

# Load the trained model
with open('/best_gb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a function to preprocess input features
def preprocess_input(area, age, floor, num_rooms, num_bathrooms):
    # Create a DataFrame with input features
    input_data = pd.DataFrame({
        'area': [area],
        'age': [age],
        'floor': [floor],
        'number of rooms': [num_rooms],
        'number of bathrooms': [num_bathrooms]
    })
    
    # Preprocess the input data using the pipeline
    preprocessed_data = pipeline.transform(input_data)
    # Define the list of age categories
    age_categories = ['0 - 1', '6 - 9', '1 - 5', '10 - 19', '20 - 40']

    # Create new column names for the encoded features
    age_columns = ['age_' + category.replace(' ', '_') for category in age_categories]
    preprocessed_data= pd.DataFrame(preprocessed_data, columns=['area'] + age_columns + ['floor', 'total_rooms'])
    return preprocessed_data

def predict_price(area, age, floor, num_rooms, num_bathrooms):
    # Preprocess the input features
    preprocessed_features = preprocess_input(area, age, floor, num_rooms, num_bathrooms)
    


    # Predict the house price using the loaded model
    predicted_price = model.predict(preprocessed_features)[0]  # Pass the array directly
    return predicted_price





# UI elements
st.title('Real Estate House Price Prediction')

area = st.number_input('Enter area in square feet', min_value=0)
age = st.selectbox('Select age of the house', ['0 - 1', '1 - 5', '6 - 9', '10 - 19', '20 - 40'])
floor = st.selectbox('Select floor', ['Ground Floor', 'Third Floor', 'Fourth Floor', 'First Floor',
                                      'Basement', 'Second Floor', 'Fifth Floor', 'Semi-Ground Floor',
                                      'Last Floor With Roof'])
num_rooms = st.number_input('Select number of rooms', min_value=1, max_value=6, value=1)
num_bathrooms = st.number_input('Select number of bathrooms', min_value=1, max_value=5, value=1)

if st.button('Predict Price'):
    predicted_price = predict_price(area, age, floor, num_rooms, num_bathrooms)
    st.success(f'Predicted Price: ${predicted_price:,.2f}')

