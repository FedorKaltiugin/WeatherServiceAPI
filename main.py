from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import gdown


load_dotenv()

files = {
    'models/model_lstm.keras': os.getenv("model_id"),
    'models/X_scaler.pkl': os.getenv("X_scaler_id"),
    'models/y_scaler.pkl': os.getenv("y_scaler_id"),
    'models/minmax_scaler.pkl': os.getenv("minmax_scaler_id"),
    'models/scaler_precipitation.pkl': os.getenv("scaler_precipitation_id"),
    'models/scaler_temp_max.pkl': os.getenv("scaler_temp_max_id"),
    'models/scaler_temp_min.pkl': os.getenv("scaler_temp_min_id"),
    'models/scaler_wind.pkl': os.getenv("scaler_wind_id"),
    'data/X_last.csv': os.getenv("X_id"),
}

for path, file_id in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        gdown.download(url, path, quiet=False)

model = load_model("models/model_lstm.keras")
X_scaler = joblib.load("models/X_scaler.pkl")
y_scaler = joblib.load("models/y_scaler.pkl")
minmax_scaler = joblib.load("models/minmax_scaler.pkl")
scaler_precipitation = joblib.load("models/scaler_precipitation.pkl")
scaler_temp_max = joblib.load("models/scaler_temp_max.pkl")
scaler_temp_min = joblib.load("models/scaler_temp_min.pkl")
scaler_wind = joblib.load("models/scaler_wind.pkl")
X = pd.read_csv("data/X_last.csv")

app = FastAPI()

LOOKBACK = 7
HORIZON = 30
FEATURES = X_scaler.feature_names_in_

class WeatherInput(BaseModel):
    precipitation: float
    temp_min: float
    wind: float
    weather: int
    year: int
    month: int
    day: int


@app.post("/predict")
def predict(input_data: WeatherInput):
    user_df = pd.DataFrame([input_data.dict()])

    user_df['wind'] = scaler_wind.transform(user_df[['wind']])
    user_df['temp_min'] = scaler_temp_min.transform(user_df[['temp_min']])
    user_df['precipitation'] = scaler_precipitation.transform(user_df[['precipitation']])

    user_df['weather'] = user_df['weather'].astype(int)
    user_df['year'] = user_df['year'].astype(int)
    user_df['month'] = user_df['month'].astype(int)
    user_df['day'] = user_df['day'].astype(int)

    # Получение последнего контекста
    last_context = X.iloc[-(LOOKBACK - 1):].copy()
    full_context = pd.concat([last_context, user_df], ignore_index=True)

    full_context_scaled = X_scaler.transform(full_context)

    X_input = np.expand_dims(full_context_scaled, axis=0)

    y_pred_scaled = model.predict(X_input)
    y_pred = scaler_temp_max.inverse_transform(y_pred_scaled)

    return {
        f"day_{i+1}": round(float(temp), 2)
        for i, temp in enumerate(y_pred[0])
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
