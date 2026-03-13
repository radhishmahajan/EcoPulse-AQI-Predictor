from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import os
import requests
from sklearn.ensemble import RandomForestRegressor
from abc import ABC, abstractmethod

# --- CONFIGURATION & DATA ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

LOCATION_DATA = {
    "India": {
        "Delhi": ["Central Delhi", "East Delhi", "New Delhi", "North Delhi", "South Delhi"],
        "Punjab": ["Amritsar", "Ludhiana", "Jalandhar", "Patiala", "Bathinda"],
        "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Thane", "Nashik"],
        "Karnataka": ["Bengaluru", "Mysuru", "Hubballi", "Mangaluru"],
        "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem"],
        "Uttar Pradesh": ["Lucknow", "Kanpur", "Noida", "Agra", "Varanasi"],
        "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Siliguri"]
    }
}

# --- OOPS LOGIC ---
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
        if not self.is_trained: return 0.0
        return round(float(self.model.predict(sensor_obj.get_data())[0]), 2)

# Initialize and Train Model
predictor = AQIPredictor()
predictor.train_model("air_quality_data.csv")

# --- UTILITY FUNCTIONS ---
def get_precautions(aqi):
    if aqi <= 50: return "Enjoy your outdoor activities! The air is fresh."
    elif aqi <= 100: return "Air quality is acceptable. No major precautions needed."
    elif aqi <= 200: return "⚠️ Sensitive groups should reduce prolonged outdoor exertion."
    elif aqi <= 300: return "🚨 Health alert: Everyone may experience effects. Wear a mask (N95)."
    else: return "💀 HEALTH EMERGENCY: Stay indoors. Keep windows closed."

def get_live_aqi(city):
    TOKEN = "e7c7398fdc76c87a1192df42563e3e835862b478" 
    try:
        url = f"https://api.waqi.info/feed/{city}/?token={TOKEN}"
        r = requests.get(url, timeout=5).json()
        if r['status'] == 'ok':
            return r['data']['aqi']
    except:
        return "Not Available"
    return "Not Available"

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "locations": LOCATION_DATA,
        "result": None
    })

@app.get("/history", response_class=HTMLResponse)
async def view_history(request: Request):
    history_data = []
    if os.path.exists("aqi_history.txt"):
        with open("aqi_history.txt", "r") as f:
            history_data = f.readlines()
    return templates.TemplateResponse("history.html", {"request": request, "history": history_data})

@app.post("/", response_class=HTMLResponse)
async def predict(
    request: Request, 
    country: str = Form(...),
    state: str = Form(...),
    city: str = Form(...),
    pm25: float = Form(...),
    pm10: float = Form(80.0), # Providing defaults so app doesn't crash
    no2: float = Form(30.0),
    co: float = Form(1.5)
):
    # 1. Use OOPS to predict
    sensor = AirSensor(pm25, pm10, no2, co)
    aqi_result = predictor.predict(sensor)
    
    # 2. Get Precautions and Live Data
    precaution = get_precautions(aqi_result)
    live_aqi_val = get_live_aqi(city)

    # 3. Save to History
    with open("aqi_history.txt", "a") as f:
        f.write(f"City: {city} | Predicted: {aqi_result} | Status: {precaution}\n")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "locations": LOCATION_DATA,
        "result": aqi_result,
        "status": precaution,
        "city": city,
        "live_aqi": live_aqi_val
    })

if __name__ == "__main__":
    import uvicorn
    # Use port 8000 to avoid the 'Address already in use' error
    uvicorn.run(app, host="127.0.0.1", port=8000)