from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


app = FastAPI()

model = load_model("models/model_lstm.keras")
X_scaler = joblib.load("models/X_scaler.pkl")
y_scaler = joblib.load("models/y_scaler.pkl")
minmax_scaler = joblib.load("models/minmax_scaler.pkl")
scaler_precipitation = joblib.load("models/scaler_precipitation.pkl")
scaler_temp_max = joblib.load("models/scaler_temp_max.pkl")
scaler_temp_min = joblib.load("models/scaler_temp_min.pkl")
scaler_wind = joblib.load("models/scaler_wind.pkl")
X = pd.read_csv("data/X_last.csv")


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
