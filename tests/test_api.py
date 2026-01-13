import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import app

client = TestClient(app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code in [200, 503]  # 503 if model not loaded

def test_predict():
    """Test prediction endpoint"""
    payload = {
        "features": [0.038, 0.050, 0.061, 0.021, -0.044, 
                    -0.034, -0.043, -0.002, 0.019, -0.017]
    }
    response = client.post("/predict", json=payload)
    
    if response.status_code == 200:
        assert "prediction" in response.json()
        assert "model_version" in response.json()
        assert isinstance(response.json()["prediction"], float)

def test_predict_invalid_features():
    """Test prediction with wrong number of features"""
    payload = {"features": [0.1, 0.2]}  # Only 2 features instead of 10
    response = client.post("/predict", json=payload)
    assert response.status_code in [400, 503]