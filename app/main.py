from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import pandas as pd
import os
import logging

app = FastAPI()

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic model for input data
class SoilData(BaseModel):
    soiltype: str
    test_temp: float
    test_humid: float
    test_PH: float
    test_N: float
    test_P: float
    Test_K: float
    test_Conductivity: float

# Directory where models are stored
MODEL_DIR = "/model"

# Function to predict soil parameters for new input data
def predict_new_data(input_data: dict):
    # Ensure input_data is a DataFrame
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data], columns=['soiltype', 'test_temp', 'test_humid', 'test_PH', 'test_N', 'test_P', 'Test_K', 'test_Conductivity'])
    
    # Dictionary to store predictions
    predictions = {}
    
    # Predict each target using the corresponding best model
    for target in ['lab_pH', 'lab_N', 'lab_P', 'lab_K', 'lab_EC']:
        model_filename = os.path.join(MODEL_DIR, f'best_model_{target}.joblib')
        if not os.path.exists(model_filename):
            logger.error(f"Model file {model_filename} not found")
            raise HTTPException(status_code=404, detail=f"Model file {model_filename} not found")
        model = load(model_filename)
        predictions[target] = model.predict(input_data)[0]
    
    return predictions

@app.post("/predict")
def predict_post(soil_data: SoilData):
    try:
        input_data = soil_data.dict()
        predictions = predict_new_data(input_data)
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/predict")
def predict_get(
    soiltype: str,
    test_temp: float,
    test_humid: float,
    test_PH: float,
    test_N: float,
    test_P: float,
    Test_K: float,
    test_Conductivity: float
):
    try:
        input_data = {
            'soiltype': soiltype,
            'test_temp': test_temp,
            'test_humid': test_humid,
            'test_PH': test_PH,
            'test_N': test_N,
            'test_P': test_P,
            'Test_K': Test_K,
            'test_Conductivity': test_Conductivity
        }
        predictions = predict_new_data(input_data)
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
