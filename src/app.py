import pandas as pd
from logger import logging
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# --- Pydantic Model for Input Validation ---
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --- FastAPI App Initialization ---
app = FastAPI(title="Iris Classifier API", version="1.0")

# --- Load Model and Scaler at Startup ---
# These files are now local to the container, loaded from the 'saved_model' directory
MODEL_PATH = os.path.join("saved_model", "model.joblib")
SCALER_PATH = os.path.join("saved_model", "scaler.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the Iris Classifier API. Go to /docs for documentation."}

@app.post("/predict", tags=["Prediction"])
def predict_species(features: IrisFeatures):
    logging.info(f"Received prediction request: {features.dict()}")
    # Convert input data to a pandas DataFrame
    input_data = pd.DataFrame([features.dict()])
    
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    # Make a prediction
    prediction_label = model.predict(scaled_data)[0]
    prediction_proba = model.predict_proba(scaled_data)[0]

    # Map the numeric label to the species name
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    species_name = species_map.get(int(prediction_label), "Unknown")
    
    return {
        "predicted_species": species_name,
        "predicted_label": int(prediction_label),
        "prediction_confidence": float(max(prediction_proba))
    }