import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('https://github.com/Sara-Saraireh/House_Price_Prediction_in_Amman_Jordan/blob/b13ca9d6150ac8dfa787dd1de3c8e957a5790aa1/house_price_model.pkl', 'rb'))

# Title
st.title('House Price Prediction in Amman, Jordan')

# Input features
st.sidebar.header('Input Features')
def user_input_features():
    rooms = st.sidebar.slider('Number of Rooms', 1, 10, 1)
    bathrooms = st.sidebar.slider('Number of Bathrooms', 1, 5, 1)
    area = st.sidebar.slider('Area (sq meters)', 50, 500, 50)
    floor = st.sidebar.slider('Floor', 0, 10, 0)
    age = st.sidebar.slider('Age of the Apartment', 0, 30, 0)
    data = {'rooms': rooms, 'bathrooms': bathrooms, 'area': area, 'floor': floor, 'age': age}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display input features
st.subheader('Input Features')
st.write(input_df)

# Make predictions
prediction = model.predict(input_df)

# Display prediction
st.subheader('Predicted House Price')
st.write(f"The predicted price is {prediction[0]:.2f} JD")

# To run the app, use the command: streamlit run app.py
