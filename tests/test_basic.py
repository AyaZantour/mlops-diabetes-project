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