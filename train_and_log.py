import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import mlflow
import mlflow.catboost
import joblib
import os

# 1. SETUP: We assume you have the 'labeled_features' DataFrame from your notebook.
# For this script to run standalone, I will generate dummy data matching your notebook's shape.
# REPLACE THIS BLOCK with your actual data loading logic if running locally.
print("Generating dummy data for demonstration...")
columns = [
    'voltmean_3h', 'temperaturemean_3h', 'pressuremean_3h', 'vibrationmean_3h',
    'voltsd_3h', 'temperaturesd_3h', 'pressuresd_3h', 'vibrationsd_3h',
    'voltmean_24h', 'temperaturemean_24h', 'pressuremean_24h', 'vibrationmean_24h',
    'voltsd_24h', 'temperaturesd_24h', 'pressuresd_24h', 'vibrationsd_24h',
    'error1count', 'error2count', 'error3count', 'error4count', 'error5count',
    'comp1', 'comp2', 'comp3', 'comp4', 'age',
    'model_model1', 'model_model2', 'model_model3', 'model_model4'
]
X_train = pd.DataFrame(np.random.rand(100, 30), columns=columns)
# Target: 0=None, 1=Comp1, 2=Comp2, 3=Comp3, 4=Comp4
y_train = np.random.randint(0, 5, 100) 

# 2. MLFLOW SETUP
mlflow.set_experiment("Predictive_Maintenance_CatBoost")

print("Starting Training...")
with mlflow.start_run():
    # Define Model (Matching your notebook params)
    cat_model = CatBoostClassifier(
        iterations=100, # Reduced for demo speed
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        random_seed=42,
        verbose=False,
        auto_class_weights='Balanced'
    )
    
    # Train
    cat_model.fit(X_train, y_train)
    
    # Log Parameters
    mlflow.log_param("iterations", 100)
    mlflow.log_param("depth", 6)
    
    # Log Model using MLflow
    mlflow.catboost.log_model(cat_model, "model")
    
    # 3. SAVE FOR DOCKER
    # We save to a specific folder so Docker can find it
    os.makedirs("model", exist_ok=True)
    model_path = "model/catboost_maintenance_model.pkl"
    joblib.dump(cat_model, model_path)
    
    print(f" Model saved to {model_path}")
    print(" Run 'mlflow ui' to view experiments.")