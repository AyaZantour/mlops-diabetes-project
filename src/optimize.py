import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
import mlflow
from config import PROCESSED_DATA_DIR, MODEL_DIR, RANDOM_STATE, EXPERIMENT_NAME
import pickle

def objective(trial):
    """Optuna objective function"""
    
    # Load data
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    
    # Suggest hyperparameters
    alpha = trial.suggest_float('alpha', 0.001, 10.0, log=True)
    max_iter = trial.suggest_int('max_iter', 500, 2000)
    
    # Train and evaluate with cross-validation
    model = Ridge(alpha=alpha, max_iter=max_iter, random_state=RANDOM_STATE)
    
    # Use negative MSE for scoring (higher is better)
    scores = cross_val_score(model, X_train, y_train, cv=5, 
                             scoring='neg_mean_squared_error')
    
    # Return RMSE (lower is better, so Optuna minimizes)
    rmse = np.sqrt(-scores.mean())
    
    return rmse

def run_optimization(n_trials=10):
    """Run Optuna study"""
    
    print(f"\n🔍 Starting Optuna optimization with {n_trials} trials...")
    
    # Create study
    study = optuna.create_study(
        study_name="diabetes_optimization",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    print("\n✅ Optimization complete!")
    print(f"   Best RMSE: {study.best_value:.2f}")
    print(f"   Best params: {study.best_params}")
    
    # Train final model with best params
    print("\n🚀 Training final model with best parameters...")
    
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # Train with best params
    best_model = Ridge(**study.best_params, random_state=RANDOM_STATE)
    best_model.fit(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    y_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n📊 Final Model Performance:")
    print(f"   Test RMSE: {test_rmse:.2f}")
    print(f"   Test R²:   {test_r2:.4f}")
    print(f"   Test MAE:  {test_mae:.2f}")
    
    # Save best model
    model_path = MODEL_DIR / "optuna_best.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"✅ Best model saved to {model_path}")
    
    # Log to MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="optuna_best"):
        mlflow.log_params(study.best_params)
        mlflow.log_metrics({
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "cv_rmse": study.best_value
        })
        mlflow.sklearn.log_model(best_model, "model")
    
    return study, best_model

if __name__ == "__main__":
    study, best_model = run_optimization(n_trials=10)
    
    # Print trial history
    print("\n📋 Top 5 trials:")
    df_trials = study.trials_dataframe().sort_values('value').head()
    print(df_trials[['number', 'value', 'params_alpha', 'params_max_iter']])