import os
import pickle
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR,
    RANDOM_STATE, TEST_SIZE, DEFAULT_PARAMS, EXPERIMENT_NAME
)

def load_and_save_data():
    """Load diabetes dataset from sklearn and save as CSV"""
    print("📊 Loading diabetes dataset...")
    
    # Load from sklearn
    diabetes = load_diabetes()
    
    # Create DataFrame
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    
    # Save raw data
    raw_path = RAW_DATA_DIR / "diabetes.csv"
    df.to_csv(raw_path, index=False)
    print(f"✅ Raw data saved to {raw_path}")
    
    # Basic statistics
    print("\n📈 Dataset Info:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Features: {len(diabetes.feature_names)}")
    print(f"   - Target range: [{df['target'].min():.1f}, {df['target'].max():.1f}]")
    
    return df

def prepare_data():
    """Split and save train/test data"""
    print("\n🔧 Preparing train/test split...")
    
    # Load raw data
    df = pd.read_csv(RAW_DATA_DIR / "diabetes.csv")
    
    # Features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Save processed data
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    
    print(f"✅ Train set: {len(X_train)} samples")
    print(f"✅ Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def plot_results(y_true, y_pred, model_name="model"):
    """Create prediction vs actual plot"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title('Predictions vs Actual')
    
    # Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    
    plt.tight_layout()
    plot_path = MODEL_DIR / f"{model_name}_results.png"
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def train_model(alpha=None, max_iter=None, tol=None, model_name="baseline"):
    """Train Ridge regression model with MLflow tracking"""
    
    # Use defaults if not provided
    alpha = alpha or DEFAULT_PARAMS['alpha']
    max_iter = max_iter or DEFAULT_PARAMS['max_iter']
    tol = tol or DEFAULT_PARAMS['tol']
    
    # Set MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=model_name):
        print(f"\n🚀 Training model: {model_name}")
        
        # Load processed data
        train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
        test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
        
        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']
        
        # Log parameters
        params = {
            'alpha': alpha,
            'max_iter': max_iter,
            'tol': tol,
            'model_type': 'Ridge'
        }
        mlflow.log_params(params)
        
        # Train model
        model = Ridge(alpha=alpha, max_iter=max_iter, tol=tol, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Log metrics
        mlflow.log_metrics({
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2
        })
        
        # Save model locally
        model_path = MODEL_DIR / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Create and log plot
        plot_path = plot_results(y_test, y_pred_test, model_name)
        mlflow.log_artifact(plot_path)
        
        print(f"\n📊 Results for {model_name}:")
        print(f"   Train RMSE: {train_rmse:.2f}")
        print(f"   Test RMSE:  {test_rmse:.2f}")
        print(f"   Test MAE:   {test_mae:.2f}")
        print(f"   Test R²:    {test_r2:.4f}")
        print(f"✅ Model saved to {model_path}")
        
        return model, test_rmse, test_r2

def main():
    parser = argparse.ArgumentParser(description='Train diabetes prediction model')
    parser.add_argument('--alpha', type=float, default=DEFAULT_PARAMS['alpha'])
    parser.add_argument('--max-iter', type=int, default=DEFAULT_PARAMS['max_iter'])
    parser.add_argument('--tol', type=float, default=DEFAULT_PARAMS['tol'])
    parser.add_argument('--name', type=str, default='baseline')
    args = parser.parse_args()
    
    # Step 1: Load and save data (only if not exists)
    if not (RAW_DATA_DIR / "diabetes.csv").exists():
        load_and_save_data()
    
    # Step 2: Prepare data (only if not exists)
    if not (PROCESSED_DATA_DIR / "train.csv").exists():
        prepare_data()
    
    # Step 3: Train model
    train_model(
        alpha=args.alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        model_name=args.name
    )

if __name__ == "__main__":
    main()