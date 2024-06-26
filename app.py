import streamlit as st
import joblib
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

# Define custom transformers
class FloorMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.floor_mapping = {'Ground Floor': 0, 'Semi-Ground Floor': -1, 'Basement': -2, 'First Floor': 1,
                              'Second Floor': 2, 'Third Floor': 3, 'Fourth Floor': 4, 'Fifth Floor': 5,
                              'Last Floor With Roof': 6}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X['floor'].map(self.floor_mapping).values.reshape(-1, 1)

class TotalRoomsCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        total_rooms = X['number of rooms'] + X['number of bathrooms']
        return total_rooms.values.reshape(-1, 1)

# Load the preprocessing pipeline and model
pipeline = joblib.load('/mnt/data/preprocessing_pipeline.joblib')
with open('/mnt/data/house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function to preprocess input features
def preprocess_input(area, age, floor, num_rooms, num_bathrooms):
    input_data = pd.DataFrame({
        'area': [area],
        'age': [age],
        'floor': [floor],
        'number of rooms': [num_rooms],
        'number of bathrooms': [num_bathrooms]
    })
    
    preprocessed_data = pipeline.transform(input_data)

    age_categories = ['0 - 1', '6 - 9', '1 - 5', '10 - 19', '20 - 40']
    age_columns = ['age_' + category.replace(' ', '_') for category in age_categories]
    preprocessed_data = pd.DataFrame(preprocessed_data, columns=['area'] + age_columns + ['floor', 'total_rooms'])
    return preprocessed_data

def predict_price(area, age, floor, num_rooms, num_bathrooms):
    preprocessed_features = preprocess_input(area, age, floor, num_rooms, num_bathrooms)
    predicted_price = model.predict(preprocessed_features)[0]
    return predicted_price

# Set the theme
st.set_page_config(page_title='Real Estate House Price Prediction', layout='wide')

# Sidebar UI elements
st.sidebar.title('House Price Prediction')
st.sidebar.write("Enter the details of the house to get the price prediction")

area = st.sidebar.number_input('Enter area in square feet', min_value=0)
age = st.sidebar.selectbox('Select age of the house', ['0 - 1', '1 - 5', '6 - 9', '10 - 19', '20 - 40'])
floor = st.sidebar.selectbox('Select floor', ['Ground Floor', 'Third Floor', 'Fourth Floor', 'First Floor',
                                              'Basement', 'Second Floor', 'Fifth Floor', 'Semi-Ground Floor',
                                              'Last Floor With Roof'])
num_rooms = st.sidebar.number_input('Select number of rooms', min_value=1, max_value=6, value=1)
num_bathrooms = st.sidebar.number_input('Select number of bathrooms', min_value=1, max_value=5, value=1)

if st.sidebar.button('Predict Price'):
    predicted_price = predict_price(area, age, floor, num_rooms, num_bathrooms)
    st.sidebar.success(f'Predicted Price: ${predicted_price:,.2f}')

    # Visualization
    st.write('### House Price Prediction Visualization')
    fig, ax = plt.subplots()
    categories = ['Area', 'Age', 'Floor', 'Total Rooms']
    values = [area, age, floor, num_rooms + num_bathrooms]
    ax.bar(categories, values, color=['blue', 'orange', 'green', 'red'])
    ax.set_ylabel('Value')
    ax.set_title('House Features')
    st.pyplot(fig)
else:
    st.sidebar.write("Click the button to predict the house price")

