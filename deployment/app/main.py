from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import traceback

app = FastAPI(title="Predictive Maintenance API")

#  CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  GLOBAL VARIABLES 
MODEL_PATH = "model/catboost_maintenance_model.pkl"
DATA_PATH = "data/labeled_features.csv"
model = None
historical_data = None 

#  SCHEMA 
class MachineData(BaseModel):
    #  FIX: machineID must be the first column
    machineID: int 
    voltmean_3h: float
    temperaturemean_3h: float
    pressuremean_3h: float
    vibrationmean_3h: float
    voltsd_3h: float
    temperaturesd_3h: float
    pressuresd_3h: float
    vibrationsd_3h: float
    voltmean_24h: float
    temperaturemean_24h: float
    pressuremean_24h: float
    vibrationmean_24h: float
    voltsd_24h: float
    temperaturesd_24h: float
    pressuresd_24h: float
    vibrationsd_24h: float
    error1count: float
    error2count: float
    error3count: float
    error4count: float
    error5count: float
    comp1: float
    comp2: float
    comp3: float
    comp4: float
    age: int
    model_model1: int
    model_model2: int
    model_model3: int
    model_model4: int

#  STARTUP 
@app.on_event("startup")
def startup_event():
    global model, historical_data
    
    # 1. Load Model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f" Model loaded from {MODEL_PATH}")
        else:
            print(f" Model not found at {MODEL_PATH}")
    except Exception as e:
        print(f" Error loading model: {e}")

    # 2. Load CSV Data
    try:
        if os.path.exists(DATA_PATH):
            #  FIX: Handle missing values immediately
            df = pd.read_csv(DATA_PATH)
            df['failure'] = df['failure'].fillna('none')
            df['model'] = df['model'].fillna('model1')
            df = df.fillna(0)
            
            historical_data = df
            historical_data['datetime'] = historical_data['datetime'].astype(str)
            print(f" Data loaded: {len(historical_data)} rows")
        else:
            print(f" CSV Data not found at {DATA_PATH}")
    except Exception as e:
        print(f" Error loading CSV: {e}")

#  ROUTES 

@app.get("/machine/{machine_id}")
def get_machine_history(machine_id: int):
    if historical_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    subset = historical_data[historical_data['machineID'] == machine_id]
    if subset.empty:
        raise HTTPException(status_code=404, detail="Machine ID not found")
    
    return {"machine_id": machine_id, "timestamps": subset['datetime'].tolist()}

@app.get("/machine/{machine_id}/{timestamp_str}")
def get_machine_data_at_time(machine_id: int, timestamp_str: str):
    if historical_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    row = historical_data[
        (historical_data['machineID'] == machine_id) & 
        (historical_data['datetime'] == timestamp_str)
    ]
    
    if row.empty:
        raise HTTPException(status_code=404, detail="Data point not found")
    
    data = row.iloc[0].to_dict()

    # Transform 'model' string (e.g. "model3") to dummy columns
    model_cols = {'model_model1': 0, 'model_model2': 0, 'model_model3': 0, 'model_model4': 0}
    model_str = data.get('model', '')
    target_col = f"model_{model_str}"
    if target_col in model_cols:
        model_cols[target_col] = 1
        
    features = {k: v for k, v in data.items() if k not in ['machineID', 'datetime', 'failure', 'model']}
    features.update(model_cols)
    
    return {
        "features": features,
        "actual_failure": data.get('failure', 'none')
    }

@app.post("/predict")
def predict(data: MachineData):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_df = pd.DataFrame([data.dict()])
    
    try:
        #  FIX: Explicitly convert model columns to boolean to satisfy CatBoost
        model_cols = [c for c in input_df.columns if "model_" in c]
        for col in model_cols:
            input_df[col] = input_df[col].astype(bool)

        probs = model.predict_proba(input_df)[0]
    except Exception as e:
        print(" PREDICTION FAILED!")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model Error: {str(e)}")

    pred_class = int(np.argmax(probs))
    labels = {0: "none", 1: "comp1", 2: "comp2", 3: "comp3", 4: "comp4"}
    
    return {
        "prediction_id": pred_class,
        "status": labels.get(pred_class, "Unknown"),
        "confidence": float(np.max(probs)),
        "probabilities": {
            "none": float(probs[0]),
            "comp1": float(probs[1]),
            "comp2": float(probs[2]),
            "comp3": float(probs[3]),
            "comp4": float(probs[4]),
        }
    }

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/ui")
async def read_index():
    return FileResponse('frontend/index.html')