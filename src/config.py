from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"

# Create directories
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Model parameters (baseline)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Default hyperparameters
DEFAULT_PARAMS = {
    'alpha': 0.1,
    'max_iter': 1000,
    'tol': 1e-3
}

# MLflow
EXPERIMENT_NAME = "diabetes-prediction"