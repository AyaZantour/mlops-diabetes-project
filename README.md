# MLOps Diabetes Prediction Project

Complete MLOps pipeline for diabetes disease progression prediction using scikit-learn.

## 📋 Project Overview

- **Dataset**: Scikit-learn Diabetes (442 samples, 10 features)
- **Model**: Ridge Regression
- **Metric**: RMSE (Root Mean Squared Error), R² score
- **MLOps Stack**: Git, Docker, DVC, MLflow, ZenML, Optuna

## 🏗️ Project Structure

```
mlops-diabetes-project/
├── data/
│   ├── raw/              # Raw data (DVC tracked)
│   └── processed/        # Train/test splits (DVC tracked)
├── models/               # Trained models (DVC tracked)
├── src/
│   ├── train.py          # Training script
│   ├── predict.py        # FastAPI inference API
│   ├── pipeline.py       # ZenML pipeline
│   ├── optimize.py       # Optuna hyperparameter tuning
│   └── config.py         # Configuration
├── tests/                # Pytest tests
├── Dockerfile            # Training container
├── Dockerfile.api        # API container
├── docker-compose.yml    # Multi-service orchestration
├── requirements.txt      # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker Desktop
- Git

### 1. Clone Repository
```bash
git clone https://github.com/AyaZantour/mlops-diabetes-project.git
cd mlops-diabetes-project
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Train Baseline Model
```bash
python src/train.py --name baseline
```

### 4. View MLflow Experiments
```bash
mlflow ui
# Open http://localhost:5000
```

### 5. Run ZenML Pipeline
```bash
zenml init
python src/pipeline.py

# View ZenML dashboard
zenml up
# Open http://localhost:8237
```

### 6. Hyperparameter Optimization
```bash
python src/optimize.py
```

### 7. Start API
```bash
# Local
python src/predict.py

# Or with Docker
docker-compose up -d api
```

### 8. Test API
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.038, 0.050, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]}'
```

## 🔄 Data Version Control (DVC)

### Local Storage Setup
We use DVC with **local file storage** for simplicity and demonstration. The actual data files are stored separately from Git, while DVC tracks them with metadata files.

```bash
# Initialize DVC
dvc init

# Configure local storage (default: .dvc/cache)
# Already configured to use local storage

# Track data files with DVC
dvc add data/raw/diabetes.csv
dvc add data/processed/train.csv
dvc add data/processed/test.csv
dvc add models/baseline.pkl
dvc add models/optuna_best.pkl

# These create .dvc files (small metadata files)
ls -la data/raw/*.dvc
ls -la models/*.dvc

# Commit DVC metadata to Git
git add data/.gitignore data/raw/*.dvc data/processed/*.dvc models/.gitignore models/*.dvc
git commit -m "feat: track data and models with DVC"
git push
```

## 🐳 Docker Usage

### Build Images
```bash
docker-compose build
```

### Run Training in Docker
```bash
docker-compose up training
```

### Run Full Stack (MLflow + API)
```bash
docker-compose up -d
```

### Test Deployment v1 → v2 → Rollback
```bash
# Start v1
docker-compose up -d api
curl http://localhost:8000/health

# Update to v2 (manually change MODEL_PATH in docker-compose.yml)
# Change: MODEL_PATH=/app/models/model_v2.pkl
docker-compose up -d api
curl http://localhost:8000/health

# Rollback to v1
# Change back: MODEL_PATH=/app/models/baseline.pkl
docker-compose up -d api
curl http://localhost:8000/health
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_train.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Experiment Tracking

### MLflow Runs
- `baseline`: Initial Ridge model (α=0.1)
- `variation1`: Ridge with α=0.5
- `variation2`: Ridge with α=1.0
- `optuna_best`: Best model from Optuna optimization

### ZenML Pipelines
- Pipeline runs visible in ZenML dashboard
- Each run tracked with data/model versioning

## 🎯 Hyperparameter Optimization (Optuna)

Optuna study searches over:
- `alpha`: [0.001, 10.0] (log scale)
- `max_iter`: [500, 2000]

Best parameters saved to `models/optuna_best.pkl`

## 📈 Model Performance

| Model | Test RMSE | Test R² | Notes |
|-------|-----------|---------|-------|
| Baseline (α=0.1) | ~55.75 | ~0.45 | Initial model |
| Variation (α=0.5) | ~56.20 | ~0.44 | Higher regularization |
| Optuna Best | ~54.50 | ~0.47 | Optimized hyperparameters |

## 🔐 API Endpoints

- `GET /`: API information
- `GET /health`: Health check
- `POST /predict`: Make predictions

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.038, 0.050, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]
  }'
```

### Example Response
```json
{
  "prediction": 206.32,
  "model_version": "v1"
}
```

## 🎓 Project Requirements Checklist

- ✅ Git version control with branches and tags
- ✅ Docker containerization (training + API)
- ✅ Docker Compose for multi-service orchestration
- ✅ DVC for data versioning with remote storage
- ✅ MLflow for experiment tracking
- ✅ ZenML pipeline orchestration
- ✅ Optuna hyperparameter optimization
- ✅ FastAPI inference service
- ✅ Deployment v1 → v2 → rollback demonstration
- ✅ CI/CD pipeline (GitHub Actions)
- ✅ Comprehensive tests (pytest)



## 👤 Author

- GitHub: [@AyaZantour](https://github.com/AyaZantour)
- Project: [mlops-diabetes-project](https://github.com/AyaZantour/mlops-diabetes-project)



