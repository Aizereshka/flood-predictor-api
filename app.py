from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from catboost import Pool
from datetime import datetime

app = FastAPI(title="Flood Risk Prediction API")

# ================= LOAD FILES =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "catboost_flood_cv_final.joblib")
DATA_PATH = os.path.join(BASE_DIR, "flood-kz.csv")

try:
    model = joblib.load(MODEL_PATH)
    df_history = pd.read_csv(DATA_PATH, low_memory=False)
except FileNotFoundError as e:
    print(f"CRITICAL FILE ERROR: {e}")
    raise e

# Извлекаем ожидаемый порядок и имена столбцов из модели
EXPECTED_FEATURES = model.feature_names_
# На этот раз мы НЕ удаляем 'Unnamed: 0' из списка ожидаемых признаков.
# Мы просто убедимся, что он есть в списке, чтобы его порядок был соблюден.
# Если он не удаляется (как мы видели), то он остается в EXPECTED_FEATURES.
    
df_history["date"] = pd.to_datetime(df_history["date"], errors="coerce")

# ================= FEATURES =================
NUM_STATIC_COLS = [
    "latitude", "longitude", "basin_mean_elevation",
    "basin_mean_slope", "basin_twi", "urban_area_pct"
]

CAT_STATIC_COLS = ["region", "major_city"]

MEDIAN_VALUES = (
    df_history[NUM_STATIC_COLS]
    .median(numeric_only=True)
    .to_dict()
)

static_map = (
    df_history
    .groupby("basin")[NUM_STATIC_COLS + CAT_STATIC_COLS]
    .first()
    .to_dict(orient="index")
)

CATEGORICAL_COLS = ["region", "major_city", "basin", "season"]

# ================= INPUT =================
class InputData(BaseModel):
    date: str
    basin: str
    water_level: float
    soil_moisture_avg: float
    temp_avg: float
    precip_sum: float

# ================= FEATURE ENGINEERING =================
def create_input_data(data: InputData):
    df = pd.DataFrame([{
        # !!! КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Добавляем Unnamed: 0 на позицию 0 !!!
        "Unnamed: 0": 0,
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
    if pd.isna(date_obj):
        month = 4
    else:
        month = date_obj.month
        
    df["season"] = (month % 12 // 3) + 1

    return df

# ================= ENDPOINT =================
@app.post("/predict")
def predict(data: InputData):
    
    # 1. Создаем признаки
    df_input = create_input_data(data)
    
    # 2. !!! ИСПРАВЛЕНИЕ: Оставляем столбцы в порядке, который ожидает CatBoost, 
    # включая Unnamed: 0
    try:
        df_input = df_input[EXPECTED_FEATURES]
    except KeyError as e:
        # Эта ошибка должна показать, что мы забыли создать какой-то признак
        raise HTTPException(status_code=500, detail=f"Missing feature: {e}. Check EXPECTED_FEATURES.")
    
    # 3. Очистка и преобразование типов
    for col in df_input.columns:
        if col not in CATEGORICAL_COLS and col != 'Unnamed: 0': # Игнорируем Unnamed: 0 при проверке
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(MEDIAN_VALUES.get(col, 0))
    
    # 4. Прогноз
    try:
        pool = Pool(df_input, cat_features=CATEGORICAL_COLS)
        prob = model.predict_proba(pool)[0][1]
    
    except Exception as e:
        print(f"CRITICAL ERROR IN CATBOOST PREDICTION: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: Check logs for details. Error: {e}")

    # 5. Возвращаем результат
    return {
        "prob_flood": float(prob),
        "prediction": int(prob > 0.5)
    }