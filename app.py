from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from abc import ABC, abstractmethod

app = FastAPI()
templates = Jinja2Templates(directory="templates")
@app.get("/history", response_class=HTMLResponse)
async def view_history(request: Request):
    history_data = []
    if os.path.exists("aqi_history.txt"):
        with open("aqi_history.txt", "r") as f:
            history_data = f.readlines()
    
    return templates.TemplateResponse("history.html", {
        "request": request, 
        "history": history_data
    })

# --- YOUR OOPS LOGIC ---

class BasePredictor(ABC):
    @abstractmethod
    def train_model(self, data_path: str): pass
    @abstractmethod
    def predict(self, features: list): pass

class AirSensor:
    def __init__(self, pm25, pm10, no2, co):
        self.__pm25 = pm25  # Encapsulation
        self.__pm10 = pm10
        self.__no2 = no2
        self.__co = co

    def get_data(self):
        return [[self.__pm25, self.__pm10, self.__no2, self.__co]]

class AQIPredictor(BasePredictor):
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False

    def train_model(self, data_path):
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'AQI']
            df = df[cols].fillna(df[cols].mean())
            self.model.fit(df[['PM2.5', 'PM10', 'NO2', 'CO']], df['AQI'])
            self.is_trained = True

    def predict(self, sensor_obj):
        if not self.is_trained: return "Model Loading..."
        return round(self.model.predict(sensor_obj.get_data())[0], 2)

# Global Predictor Instance
predictor = AQIPredictor()
predictor.train_model("air_quality_data.csv")

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def predict_aqi(
    request: Request, 
    pm25: float = Form(...), 
    pm10: float = Form(...), 
    no2: float = Form(...), 
    co: float = Form(...)
):
    sensor = AirSensor(pm25, pm10, no2, co)
    result = predictor.predict(sensor)
    
    status = "Satisfactory" if result <= 100 else "Moderate" if result <= 200 else "Poor"
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "result": result, 
        "status": status
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)    