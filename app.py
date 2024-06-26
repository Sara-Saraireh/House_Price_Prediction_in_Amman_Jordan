import streamlit as st
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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

# Function to preprocess input data
def preprocess_input(pipeline, area, age, floor, num_rooms, num_bathrooms):
    input_data = pd.DataFrame({
        'area': [area],
        'age': [age],
        'floor': [floor],
        'number of rooms': [num_rooms],
        'number of bathrooms': [num_bathrooms]
    })

    # Apply the preprocessing pipeline to the input data
    processed_data = pipeline.transform(input_data)
    return processed_data

# Function to make predictions
def predict_price(model, pipeline, area, age, floor, num_rooms, num_bathrooms):
    preprocessed_features = preprocess_input(pipeline, area, age, floor, num_rooms, num_bathrooms)
    return model.predict(preprocessed_features)

# Function to display the Streamlit UI
def run_ui():
    st.title('Real Estate House Price Prediction')

    pipeline_file = st.file_uploader("Upload Preprocessing Pipeline", type="joblib")
    model_file = st.file_uploader("Upload Model", type="pkl")

    if pipeline_file and model_file:
        pipeline = joblib.load(pipeline_file)
        model = joblib.load(model_file)

        area = st.number_input('Enter area in square feet', min_value=0)
        age = st.selectbox('Select age of the house', ['0 - 1', '1 - 5', '6 - 9', '10 - 19', '20 - 40'])
        floor = st.selectbox('Select floor', ['Ground Floor', 'Third Floor', 'Fourth Floor', 'First Floor',
                                              'Basement', 'Second Floor', 'Fifth Floor', 'Semi-Ground Floor',
                                              'Last Floor With Roof'])
        num_rooms = st.number_input('Select number of rooms', min_value=1, max_value=6, value=1)
        num_bathrooms = st.number_input('Select number of bathrooms', min_value=1, max_value=5, value=1)

        if st.button('Predict Price'):
            try:
                predicted_price = predict_price(model, pipeline, area, age, floor, num_rooms, num_bathrooms)
                st.success(f'Predicted Price: ${predicted_price[0]:,.2f}')

                # Visualization
                st.write('### House Price Prediction Visualization')
                fig, ax = plt.subplots()
                categories = ['Area', 'Age', 'Floor', 'Total Rooms']
                values = [area, age, floor, num_rooms + num_bathrooms]
                ax.bar(categories, values, color=['blue', 'orange', 'green', 'red'])
                ax.set_ylabel('Value')
                ax.set_title('House Features')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    else:
        st.write("Please upload both the preprocessing pipeline and the model files to proceed.")

if __name__ == "__main__":
    run_ui()
