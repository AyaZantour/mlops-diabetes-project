import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.config import MODEL_DIR


# Initialize FastAPI
app = FastAPI(
    title="Diabetes Prediction API",
    description="Predict diabetes disease progression",
    version="1.0"
)

# Load model at startup
MODEL_PATH = os.getenv("MODEL_PATH", str(MODEL_DIR / "baseline.pkl"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Request model
class PredictionRequest(BaseModel):
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.038, 0.050, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]
            }
        }

# Response model
class PredictionResponse(BaseModel):
    prediction: float
    model_version: str

@app.get("/")
def root():
    return {
        "message": "Diabetes Prediction API",
        "version": MODEL_VERSION,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_version": MODEL_VERSION}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate input
        if len(request.features) != 10:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 10 features, got {len(request.features)}"
            )
        
        # Make prediction
        features = np.array(request.features).reshape(1, -1)
        prediction = float(model.predict(features)[0])
        
        return PredictionResponse(
            prediction=prediction,
            model_version=MODEL_VERSION
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)