# src/app.py

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import joblib
import os
import requests # Needed for retraining trigger
import logging # For logging requests and responses
from datetime import datetime

# --- Logging Configuration ---
# Create a logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging to write to a file
log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}_api.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler() # Also log to console
    ]
)

# --- Pydantic Model with Enhanced Validation ---
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm, must be positive")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm, must be positive")
    petal_length: float = Field(..., gt=0, description="Petal length in cm, must be positive")
    petal_width: float = Field(..., gt=0, description="Petal width in cm, must be positive")

# --- FastAPI App Initialization ---
app = FastAPI(title="Iris Classifier API", version="1.0")

# --- Prometheus Integration  ---
from starlette_prometheus import metrics, PrometheusMiddleware
from prometheus_client import Counter

app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)
PREDICTION_COUNTER = Counter("prediction_requests_total", "Total number of prediction requests received")

# --- Load Model and Scaler at Startup ---
MODEL_PATH = os.path.join("saved_model", "model.joblib")
SCALER_PATH = os.path.join("saved_model", "scaler.joblib")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- API Security ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
RETRAIN_API_KEY = os.environ.get("RETRAIN_API_KEY", "your-secret-key")

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == RETRAIN_API_KEY:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the Iris Classifier API. Go to /docs for documentation."}

@app.post("/predict", tags=["Prediction"])
def predict_species(features: IrisFeatures):
    # Increment the prediction counter for monitoring
    PREDICTION_COUNTER.inc()
    
    try:
        # Log the incoming request data
        logging.info(f"Prediction request received with data: {features.dict()}")
        
        input_data = pd.DataFrame([features.dict()])
        scaled_data = scaler.transform(input_data)
        prediction_label = model.predict(scaled_data)[0]
        prediction_proba = model.predict_proba(scaled_data)[0]

        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        species_name = species_map.get(int(prediction_label), "Unknown")
        
        response = {
            "predicted_species": species_name,
            "predicted_label": int(prediction_label),
            "prediction_confidence": float(max(prediction_proba))
        }
        
        # Log the model's output
        logging.info(f"Prediction successful. Output: {response}")
        
        return response

    except Exception as e:
        # Log any errors that occur during prediction
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during model prediction.",
        )

# --- Model Re-training Trigger Endpoint ---
@app.post("/retrain", tags=["Admin"], dependencies=[Depends(get_api_key)])
def trigger_retraining():
    """
    Triggers the model retraining workflow on GitHub Actions.
    Requires a valid API key in the 'X-API-Key' header.
    """
    github_owner = "Ashwani484" # <-- CHANGE THIS according your repo
    github_repo = "MLOPS" # <-- CHANGE THIS according your repo
    # for Personal Access Token
    github_token = os.environ.get("GH_PAT") 

    if not github_token:
        logging.error("GH_PAT is not configured on the server.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GitHub Personal Access Token not configured on the server.",
        )

    url = f"https://api.github.com/repos/{github_owner}/{github_repo}/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {github_token}",
    }
    data = {"event_type": "retrain-model"}

    logging.info("Triggering model retraining workflow...")
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 204:
        logging.info("Model retraining workflow successfully triggered.")
        return {"message": "Model retraining workflow successfully triggered."}
    else:
        logging.error(f"Failed to trigger workflow. Status: {response.status_code}, Response: {response.text}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger workflow: {response.text}",
        )
