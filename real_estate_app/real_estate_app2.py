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

# Step 1: Split the Data into Training and Testing Sets
X = df10.copy()
y = df9['Price']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
best_gb_model = GradientBoostingRegressor(learning_rate=0.1,
                                          max_depth=3,
                                          min_samples_leaf=1,
                                          min_samples_split=3,
                                          n_estimators=300)

best_gb_model = best_gb_model.fit(X_train, y_train)

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
def predict_price(area, age, floor, num_rooms, num_bathrooms):
    preprocessed_features = preprocess_input(area, age, floor, num_rooms, num_bathrooms)
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
        predicted_price = predict_price(area, age, floor, num_rooms, num_bathrooms)
        st.success(f'Predicted Price: ${predicted_price[0]:,.2f}')

        # Visualization
        fig, ax = plt.subplots()
        sns.histplot(df9['Price'], bins=30, kde=True, ax=ax)
        ax.axvline(predicted_price[0], color='red', linestyle='dashed', linewidth=2)
        ax.text(predicted_price[0], ax.get_ylim()[1] * 0.9, f'${predicted_price[0]:,.2f}', color='red', ha='center')
        st.pyplot(fig)
        
        # Feature Importances
        st.subheader('Feature Importances')
        age_categories = ['0 - 1', '6 - 9', '1 - 5', '10 - 19', '20 - 40']
        age_columns = ['age_' + category.replace(' ', '_') for category in age_categories]
        features = ['area_scaled'] + age_columns + ['floor_numeric', 'total_rooms']
        feature_importances = best_gb_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        st.pyplot(fig)
        
        # Scatter Plot of Predicted vs Actual Prices
        st.subheader('Predicted vs Actual Prices')
        y_pred = best_gb_model.predict(X_test)
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        st.pyplot(fig)

if __name__ == "__main__":
    pipeline = joblib.load('real_estate_app/preprocessing_pipeline.joblib')
    model = best_gb_model
    run_ui()
