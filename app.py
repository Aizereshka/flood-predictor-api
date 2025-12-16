from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from catboost import Pool
from datetime import datetime

app = FastAPI(title="Flood Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "catboost_flood_cv_final.joblib")
DATA_PATH = os.path.join(BASE_DIR, "flood-kz.csv")

model = joblib.load(MODEL_PATH)
df_history = pd.read_csv(DATA_PATH, low_memory=False)

EXPECTED_FEATURES = model.feature_names_
if 'Unnamed: 0' in EXPECTED_FEATURES:
    EXPECTED_FEATURES.remove('Unnamed: 0')
df_history["date"] = pd.to_datetime(df_history["date"], errors="coerce")

NUM_STATIC_COLS = ["latitude", "longitude", "basin_mean_elevation", "basin_mean_slope", "basin_twi", "urban_area_pct"]
CAT_STATIC_COLS = ["region", "major_city"]
MEDIAN_VALUES = df_history[NUM_STATIC_COLS].median(numeric_only=True).to_dict()
static_map = df_history.groupby("basin")[NUM_STATIC_COLS + CAT_STATIC_COLS].first().to_dict(orient="index")
CATEGORICAL_COLS = ["region", "major_city", "basin", "season"]

class InputData(BaseModel):
    date: str
    basin: str
    water_level: float
    soil_moisture_avg: float
    temp_avg: float
    precip_sum: float

def create_input_data(data: InputData):
    df = pd.DataFrame([{
        "water_level": data.water_level,
        "soil_moisture_avg": data.soil_moisture_avg,
        "temp_avg": data.temp_avg,
        "precip_sum": data.precip_sum,
        "basin": data.basin
    }])
    static = static_map.get(data.basin, {})
    for col in NUM_STATIC_COLS:
        df[col] = static.get(col, MEDIAN_VALUES.get(col, 0))
    df["region"] = static.get("region", "Unknown")
    df["major_city"] = static.get("major_city", "Unknown")
    date_obj = pd.to_datetime(data.date, errors='coerce')
    month = date_obj.month if not pd.isna(date_obj) else 4
    df["season"] = (month % 12 // 3) + 1
    return df

@app.post("/predict")
def predict(data: InputData):
    df_input = create_input_data(data)
    try:
        df_input = df_input[EXPECTED_FEATURES]
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing feature: {e}. Check EXPECTED_FEATURES.")
    
    for col in df_input.columns:
        if col not in CATEGORICAL_COLS:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(MEDIAN_VALUES.get(col, 0))
    
    try:
        pool = Pool(df_input, cat_features=CATEGORICAL_COLS)
        prob = model.predict_proba(pool)[0][1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: Check logs for details. Error: {e}")

    return {
        "prob_flood": float(prob),
        "prediction": int(prob > 0.5)
    }
