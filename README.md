# Real Estate House Price Prediction

This repository contains a Streamlit app for predicting house prices in Amman, Jordan. The app uses a Gradient Boosting Regressor model to make predictions based on several input features. The model has been trained on historical real estate data and includes custom preprocessing steps.

## Features

- **Interactive UI**: Adjust input features using sliders and dropdowns.
- **Price Prediction**: Get an estimated price based on the input features.
- **Visualizations**:
  - Histogram of house prices with the predicted price highlighted.
  - Feature importance bar plot.
  - Scatter plot comparing predicted vs. actual prices.

## Screenshots

### Main Interface
![Main Interface](https://github.com/Sara-Saraireh/House_Price_Prediction_in_Amman_Jordan/blob/c7cc398763b1f58d17d5f3ea88304dd7317419c5/app_screenshots/Screenshot%202024-06-27%20at%203.52.27%20AM.png)

### Price Prediction
![Price Prediction](https://github.com/Sara-Saraireh/House_Price_Prediction_in_Amman_Jordan/blob/9dd3d0f079ce4d475df774f792c050e8ad9618a5/app_screenshots/Screenshot%202024-06-27%20at%202.57.53%20AM.png)

### Feature Importances
![Feature Importances](https://github.com/Sara-Saraireh/House_Price_Prediction_in_Amman_Jordan/blob/9dd3d0f079ce4d475df774f792c050e8ad9618a5/app_screenshots/Screenshot%202024-06-27%20at%202.58.05%20AM.png)

### Predicted vs. Actual Prices
![Predicted vs. Actual Prices](https://github.com/Sara-Saraireh/House_Price_Prediction_in_Amman_Jordan/blob/9dd3d0f079ce4d475df774f792c050e8ad9618a5/app_screenshots/Screenshot%202024-06-27%20at%202.58.14%20AM.png)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Sara-Saraireh/House_Price_Prediction_in_Amman_Jordan.git
    cd House_Price_Prediction_in_Amman_Jordan
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate   # On Windows, use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have the datasets (`df10.csv` and `df9.csv`) in the `real_estate_app` directory.
2. Load the preprocessing pipeline:
    ```python
    pipeline = joblib.load('real_estate_app/preprocessing_pipeline.joblib')
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Project Structure

- `app.py`: The main Streamlit application script.
- `requirements.txt`: List of dependencies required to run the app.
- `real_estate_app/`: Directory containing datasets and the preprocessing pipeline.

## Custom Transformers

### FloorMapper
Maps floor levels to numerical values.

### TotalRoomsCalculator
Calculates the total number of rooms (sum of rooms and bathrooms).

## How to Contribute

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature_branch
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Your commit message"
    ```
4. Push to the branch:
    ```bash
    git push origin feature_branch
    ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
