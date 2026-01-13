from zenml import pipeline, step
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from typing import Tuple
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RANDOM_STATE, TEST_SIZE

# ----------------- Steps -----------------

@step
def data_loader() -> pd.DataFrame:
    """Load diabetes dataset"""
    print("📊 Loading data...")
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target

    # Save raw data
    raw_path = RAW_DATA_DIR / "diabetes.csv"
    df.to_csv(raw_path, index=False)
    print(f"✅ Data saved: {df.shape}")

    return df

@step
def data_splitter(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test"""
    print("🔧 Splitting data...")
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

@step
def model_trainer(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 0.1) -> object:
    """Train Ridge model"""
    print(f"🚀 Training model with alpha={alpha}...")
    model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    print("✅ Model trained")
    return model

@step
def model_evaluator(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model"""
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"rmse": rmse, "r2": r2}
    print(f"✅ RMSE: {rmse:.2f}, R²: {r2:.4f}")

    return metrics

@step
def model_saver(model: object, metrics: dict, model_name: str = "zenml_model") -> str:
    """Save model"""
    print(f"💾 Saving model as {model_name}...")
    model_path = MODEL_DIR / f"{model_name}.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ Model saved to {model_path}")
    return str(model_path)

# ----------------- Pipeline -----------------

@pipeline
def training_pipeline(alpha: float = 0.1, model_name: str = "zenml_model"):
    """Complete training pipeline"""
    df = data_loader()
    X_train, X_test, y_train, y_test = data_splitter(df)
    model = model_trainer(X_train, y_train, alpha=alpha)
    metrics = model_evaluator(model, X_test, y_test)
    model_path = model_saver(model, metrics, model_name=model_name)

# ----------------- Run pipeline -----------------

if __name__ == "__main__":
    # Run baseline
    print("\n" + "="*50)
    print("Running ZenML Pipeline - Baseline")
    print("="*50)
    training_pipeline(alpha=0.1, model_name="zenml_baseline")

    # Run variation
    print("\n" + "="*50)
    print("Running ZenML Pipeline - Variation")
    print("="*50)
    training_pipeline(alpha=0.5, model_name="zenml_variation")
