from typing import Union
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import os
import logging

app = FastAPI()

# Initialize the Service
admin_password = ""

# Define the model paths for the best models
model_paths = {
    'lab_pH': 'best_model_lab_pH.joblib',
    'lab_N': 'best_model_lab_N.joblib',
    'lab_P': 'best_model_lab_P.joblib',
    'lab_K': 'best_model_lab_K.joblib',
    'lab_EC': 'best_model_lab_EC.joblib'
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load all models into a dictionary
models = {}
try:
    for target, path in model_paths.items():
        model_path = os.path.join("/app/model", path)
        logger.info(f"Loading model for {target} from {model_path}")
        models[target] = joblib.load(model_path)
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise e

# Default Section ==============================================================

@app.get("/")
def read_root():
    # Return Report or status
    return {"Hello": "World"}

# Training Model Section ==============================================================

@app.post("/add_sample", tags=["Training Section"])
def add_sample(var1: float, var2: float, var3: float, var4: float):
    # Add samples to the training dataset
    return {"message": f"add {var1} {var2} {var3} {var4}"}

@app.post("/train", tags=["Training Section"])
def train():
    # Populate data and re fitting
    # Get performance metric (RMSE etc.)
    return {"message": f"Model trained successfully with RMSE: {4.332}"}

@app.post("/commit", tags=["Training Section"])
def commit():
    # Populate - Re-train Model - Save to file
    # Retrieve Model
    return {"message": f"Model has been updated"}

# Prediction Section ==============================================================

@app.get("/predict", tags=["Prediction Section"])
def predict(
    soiltype: str,
    test_temp: float,
    test_humid: float,
    test_PH: float,
    test_N: float,
    test_P: float,
    Test_K: float,
    test_Conductivity: float
):
    features = np.array([[test_temp, test_humid, test_PH, test_N, test_P, Test_K, test_Conductivity]])
    
    predictions = {}
    for target, model in models.items():
        predictions[target] = model.predict(features)[0]

    return predictions
