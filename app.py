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
    # Загрузка модели и исторических данных
    model = joblib.load(MODEL_PATH)
    df_history = pd.read_csv(DATA_PATH, low_memory=False)
except FileNotFoundError as e:
    print(f"CRITICAL FILE ERROR: {e}")
    # Поднимаем исключение, если файлы не найдены
    raise e

# Извлекаем ожидаемый порядок и имена столбцов из модели
EXPECTED_FEATURES = model.feature_names_

# ================= FEATURES LISTS =================

# ❗ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Список всех признаков, которые модель ожидает, но которых нет в API-запросе
MISSING_FEATURES = [
    'region_postcode', 'water_discharge', 'ice_thickness', 'temp_max', 
    'temp_min', 'snow_cover', 'height_cm', 'snow_cover_extent_pct', 
    'snow_water_equivalent_avg', 'water_level_lag_1', 'water_discharge_lag_1', 
    'water_level_lag_3', 'water_discharge_lag_3', 'water_level_lag_7', 
    'water_discharge_lag_7', 'water_level_roll_mean_3', 'precip_roll_sum_3', 
    'water_level_roll_mean_7', 'precip_roll_sum_7', 'level_change_1d', 
    'level_change_3d', 'discharge_per_level'
]

NUM_STATIC_COLS = [
    "latitude", "longitude", "basin_mean_elevation",
    "basin_mean_slope", "basin_twi", "urban_area_pct"
]

CAT_STATIC_COLS = ["region", "major_city"]

# ... (Остальная часть инициализации данных) ...
df_history["date"] = pd.to_datetime(df_history["date"], errors="coerce")

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

# ================= INPUT Pydantic Model =================
class InputData(BaseModel):
    date: str
    basin: str
    water_level: float
    soil_moisture_avg: float
    temp_avg: float
    precip_sum: float

# ================= FEATURE ENGINEERING =================
def create_input_data(data: InputData):
    
    # 1. Сбор обязательных данных из запроса
    df_data = {
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 1: Добавляем Unnamed: 0 на позицию 0
        "Unnamed: 0": 0, 
        "water_level": data.water_level,
        "soil_moisture_avg": data.soil_moisture_avg,
        "temp_avg": data.temp_avg,
        "precip_sum": data.precip_sum,
        "basin": data.basin
    }
    
    # ❗ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 2: Добавляем ВСЕ отсутствующие признаки и заполняем их 0.0
    # Это решает проблему KeyError, которую мы видели в последнем логе.
    for feature in MISSING_FEATURES:
        df_data[feature] = 0.0 
        
    df = pd.DataFrame([df_data])

    # 2. Добавление статических признаков бассейна
    static = static_map.get(data.basin, {})

    for col in NUM_STATIC_COLS:
        df[col] = static.get(col, MEDIAN_VALUES.get(col, 0))

    df["region"] = static.get("region", "Unknown")
    df["major_city"] = static.get("major_city", "Unknown")

    # 3. Создание признака 'season'
    date_obj = pd.to_datetime(data.date, errors='coerce')
    if pd.isna(date_obj):
        month = 4
    else:
        month = date_obj.month
        
    # Преобразование месяца в сезон (1:Зима, 2:Весна, 3:Лето, 4:Осень)
    df["season"] = (month % 12 // 3) + 1

    return df

# ================= ENDPOINT =================
@app.post("/predict")
def predict(data: InputData):
    
    # 1. Создаем признаки
    df_input = create_input_data(data)
    
    # 2. Переставляем столбцы в ТОЧНОМ порядке, который ожидает CatBoost (включая все добавленные нулями)
    try:
        df_input = df_input[EXPECTED_FEATURES]
    except KeyError as e:
        # Если не хватает какого-то ожидаемого признака
        raise HTTPException(status_code=500, detail=f"Missing feature expected by model: {e}. Check FEATURE ENGINEERING.")
    
    # 3. Очистка и преобразование типов 
    for col in df_input.columns:
        if col not in CATEGORICAL_COLS and col != 'Unnamed: 0': 
            # Принудительное преобразование в число с заполнением медианами
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(MEDIAN_VALUES.get(col, 0))
    
    # 4. Прогноз
    try:
        pool = Pool(df_input, cat_features=CATEGORICAL_COLS)
        prob = model.predict_proba(pool)[0][1]
    
    except Exception as e:
        # Захват и вывод любой ошибки CatBoost в логах Render
        print(f"CRITICAL ERROR IN CATBOOST PREDICTION: {e}")
        # Возвращаем 500 с деталями клиенту
        raise HTTPException(status_code=500, detail=f"Prediction failed: Check logs for details. Error: {e}")

    # 5. Возвращаем результат
    return {
        "prob_flood": float(prob),
        "prediction": int(prob > 0.5)
    }