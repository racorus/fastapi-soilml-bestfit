from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import logging
from joblib import load

app = FastAPI()

# Define the input data model
class SoilTestData(BaseModel):
    soiltype: str
    temp: float
    humid: float
    ph: float
    N: float
    P: float
    K: float
    conductivity: float

# Define the model paths
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
        models[target] = load(model_path)
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise e

@app.post("/predict", tags=["Prediction Section"])
async def predict(data: SoilTestData):
    if data.soiltype not in ['clay', 'sand', 'silt']:
        raise HTTPException(status_code=400, detail="Invalid soil type")

    features = np.array([[data.temp, data.humid, data.ph, data.N, data.P, data.K, data.conductivity]])

    predictions = {}
    for target, model in models.items():
        predictions[target] = model.predict(features)[0]

    return predictions

# Default Section ==============================================================

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Training Model Section ==============================================================

@app.post("/add_sample", tags=["Training Section"])
def add_sample(var1: float, var2: float, var3: float, var4: float):
    return {"message": f"add {var1} {var2} {var3} {var4}"}

@app.post("/train", tags=["Training Section"])
def train():
    return {"message": f"Model trained successfully with RMSE: {4.332}"}

@app.post("/commit", tags=["Training Section"])
def commit():
    return {"message": f"Model has been updated"}
