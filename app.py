import joblib
import pickle

# Load and inspect the preprocessing pipeline
try:
    preprocessing_pipeline = joblib.load('/mnt/data/preprocessing_pipeline.joblib')
    print("Preprocessing pipeline loaded successfully.")
    print(preprocessing_pipeline)
except Exception as e:
    print(f"Error loading preprocessing pipeline: {e}")

# Load and inspect the model
try:
    with open('/mnt/data/house_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
    print(model)
except Exception as e:
    print(f"Error loading model: {e}")
