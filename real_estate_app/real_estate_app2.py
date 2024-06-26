import streamlit as st
import joblib
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df10 = pd.read_csv('real_estate_app/df10.csv')
df9 = pd.read_csv('real_estate_app/df9.csv')

# Print column names for debugging
st.write("df10 columns:", df10.columns)
st.write("df9 columns:", df9.columns)

# Load the preprocessing pipeline and model
pipeline = joblib.load('real_estate_app/preprocessing_pipeline.joblib')
best_gb_model = joblib.load('real_estate_app/best_gb_model.pkl')

# Function to preprocess input data
def preprocess_input(area, age, floor, num_rooms, num_bathrooms):
    input_data = pd.DataFrame({
        'area': [area],
        'age': [age],
        'floor': [floor],
        'number of rooms': [num_rooms],
        'number of bathrooms': [num_bathrooms]
    })

    # Apply the preprocessing pipeline to the input data
    processed_data = pipeline.transform(input_data)

    # Define the list of age categories
    age_categories = ['0 - 1', '6 - 9', '1 - 5', '10 - 19', '20 - 40']

    # Create new column names for the encoded features
    age_columns = ['age_' + category.replace(' ', '_') for category in age_categories]

    # Convert the NumPy array to a DataFrame
    df_pro = pd.DataFrame(processed_data, columns=['area_scaled'] + age_columns + ['floor_numeric', 'total_rooms'])

    return df_pro

# Function to make predictions
def predict_price(preprocessed_features):
    return best_gb_model.predict(preprocessed_features)

# Function to display the Streamlit UI
def run_ui():
    st.title('Real Estate House Price Prediction')

    st.sidebar.header('Input Features')
    area = st.sidebar.number_input('Enter area in square feet', min_value=0)
    age = st.sidebar.selectbox('Select age of the house', ['0 - 1', '1 - 5', '6 - 9', '10 - 19', '20 - 40'])
    floor = st.sidebar.selectbox('Select floor', ['Ground Floor', 'Third Floor', 'Fourth Floor', 'First Floor',
                                                  'Basement', 'Second Floor', 'Fifth Floor', 'Semi-Ground Floor',
                                                  'Last Floor With Roof'])
    num_rooms = st.sidebar.number_input('Select number of rooms', min_value=1, max_value=6, value=1)
    num_bathrooms = st.sidebar.number_input('Select number of bathrooms', min_value=1, max_value=5, value=1)

    if st.sidebar.button('Predict Price'):
        preprocessed_features = preprocess_input(area, age, floor, num_rooms, num_bathrooms)
        predicted_price = predict_price(preprocessed_features)
        st.success(f'Predicted Price: ${predicted_price[0]:,.2f}')

        # Visualization 1: Price Distribution
        fig1, ax1 = plt.subplots()
        sns.histplot(df9['Price'], bins=30, kde=True, ax=ax1)
        ax1.axvline(predicted_price[0], color='red', linestyle='dashed', linewidth=2)
        ax1.text(predicted_price[0], ax1.get_ylim()[1] * 0.9, f'${predicted_price[0]:,.2f}', color='red', ha='center')
        ax1.set_title('Distribution of Actual Prices with Predicted Price')
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Frequency')
        st.pyplot(fig1)

        # Ensure that df10 has all the required columns for transformation
        missing_cols = set(pipeline.named_transformers_['preprocessor'].feature_names_in_) - set(df10.columns)
        if missing_cols:
            st.error(f"The following columns are missing in df10: {missing_cols}")
        else:
            preprocessed_df10 = pipeline.transform(df10)
            df10_preprocessed = pd.DataFrame(preprocessed_df10, columns=['area_scaled'] + age_columns + ['floor_numeric', 'total_rooms'])
            df10_preprocessed['Price'] = df9['Price']  # Assuming df9 has the target 'Price' column

            df_similar_area = df10_preprocessed[df10_preprocessed['area_scaled'] == preprocessed_features.iloc[0]['area_scaled']]
            df_similar_area['Predicted Price'] = df_similar_area.apply(
                lambda row: predict_price(row.values.reshape(1, -1))[0], axis=1
            )

            # Visualization 2: Comparison with other apartments
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=df_similar_area, x='total_rooms', y='Predicted Price', hue='floor_numeric', palette='viridis', ax=ax2)
            ax2.set_title(f'Predicted Prices for Apartments with {area} sqft Area')
            ax2.set_xlabel('Number of Rooms')
            ax2.set_ylabel('Predicted Price')
            st.pyplot(fig2)

if __name__ == "__main__":
    run_ui()
