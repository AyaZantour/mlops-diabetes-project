import pytest
import sys
import os

# Test that basic Python works
def test_python():
    assert 1 + 1 == 2

def test_imports():
    """Test that we can import main modules"""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import Ridge
        import mlflow
        import fastapi
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_directories():
    """Test that project directories exist"""
    import pathlib
    
    base_dir = pathlib.Path(__file__).parent.parent
    assert (base_dir / "src").exists()
    assert (base_dir / "data").exists()
    assert (base_dir / "models").exists()
    assert (base_dir / "tests").exists()

def test_model_files():
    """Test that model files exist"""
    import pathlib
    import glob
    
    models_dir = pathlib.Path(__file__).parent.parent / "models"
    model_files = list(models_dir.glob("*.pkl"))
    
    # At least one model should exist
    assert len(model_files) > 0, f"No .pkl files found in {models_dir}"
    print(f"Found {len(model_files)} model files: {[f.name for f in model_files]}")