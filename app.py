import streamlit as st
import joblib
import pandas as pd
import numpy as np
import pickle

st.title("House Price Prediction in Amman, Jordan")

try:
    # Load the preprocessing pipeline and model
    preprocessing_pipeline = joblib.load('/mnt/data/preprocessing_pipeline.joblib')
    with open('/mnt/data/house_price_model.pkl', 'rb') as f:
        model = pickle.load(f)

    st.write("""
    ### Enter the details of the house to get the price prediction
    """)

    # Input features
    def user_input_features():
        area = st.number_input("Area (sq meters)", min_value=0)
        bedrooms = st.number_input("Number of Bedrooms", min_value=0)
        bathrooms = st.number_input("Number of Bathrooms", min_value=0)
        floors = st.number_input("Number of Floors", min_value=0)
        age = st.number_input("Age of the House (years)", min_value=0)
        location = st.selectbox("Location", ['Location1', 'Location2', 'Location3']) # Replace with actual locations

        data = {
            'Area': area,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Floors': floors,
            'Age': age,
            'Location': location
        }

        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Preprocess the input data
    try:
        input_data_preprocessed = preprocessing_pipeline.transform(input_df)

        # Prediction
        prediction = model.predict(input_data_preprocessed)

        if st.button("Predict"):
            st.write(f"### Predicted Price: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

except AttributeError as e:
    st.error(f"AttributeError: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")
