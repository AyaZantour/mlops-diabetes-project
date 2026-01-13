import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import load_and_save_data, prepare_data, train_model
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR

def test_load_data():
    """Test data loading"""
    df = load_and_save_data()
    assert df.shape == (442, 11)  # 10 features + target
    assert 'target' in df.columns
    assert (RAW_DATA_DIR / "diabetes.csv").exists()

def test_prepare_data():
    """Test data preparation"""
    X_train, X_test, y_train, y_test = prepare_data()
    assert len(X_train) == 353  # 80% of 442
    assert len(X_test) == 89    # 20% of 442
    assert (PROCESSED_DATA_DIR / "train.csv").exists()
    assert (PROCESSED_DATA_DIR / "test.csv").exists()

def test_train_model():
    """Test model training"""
    model, rmse, r2 = train_model(alpha=0.1, model_name="test_model")
    assert model is not None
    assert rmse > 0
    assert 0 <= r2 <= 1
    assert (MODEL_DIR / "test_model.pkl").exists()

def test_model_prediction():
    """Test model can make predictions"""
    import pickle
    
    # Ensure model exists
    model_path = MODEL_DIR / "baseline.pkl"
    if not model_path.exists():
        train_model(model_name="baseline")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Test prediction
    sample = np.array([[0.038, 0.050, 0.061, 0.021, -0.044, 
                        -0.034, -0.043, -0.002, 0.019, -0.017]])
    prediction = model.predict(sample)
    
    assert prediction.shape == (1,)
    assert isinstance(prediction[0], (float, np.floating))