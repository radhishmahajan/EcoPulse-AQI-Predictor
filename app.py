import streamlit as st
import pandas as pd
import os
import requests
from sklearn.ensemble import RandomForestRegressor
from abc import ABC, abstractmethod

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="EcoPulse AQI Predictor", page_icon="🌿", layout="wide")

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
        self.__pm25 = pm25
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

# Cache the model training so it doesn't retrain on every button click
@st.cache_resource
def load_model():
    predictor = AQIPredictor()
    predictor.train_model("air_quality_data.csv")
    return predictor

predictor = load_model()

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

# --- STREAMLIT UI ---
st.title("🌿 EcoPulse: AI-Driven Air Quality Predictor")
st.markdown("Monitor and predict urban air quality using Machine Learning and real-time sensor data.")

tab1, tab2 = st.tabs(["Prediction Dashboard", "Prediction History"])

with tab1:
    st.header("Enter Environmental Data")
    
    col1, col2 = st.columns(2)
    with col1:
        country = st.selectbox("Select Country", list(LOCATION_DATA.keys()))
        state = st.selectbox("Select State", list(LOCATION_DATA[country].keys()))
        city = st.selectbox("Select City", LOCATION_DATA[country][state])
        
    with col2:
        pm25 = st.number_input("PM2.5 Level", min_value=0.0, value=50.0)
        pm10 = st.number_input("PM10 Level", min_value=0.0, value=80.0)
        no2 = st.number_input("NO2 Level", min_value=0.0, value=30.0)
        co = st.number_input("CO Level", min_value=0.0, value=1.5)

    if st.button("Predict AQI"):
        # 1. Use OOPS to predict
        sensor = AirSensor(pm25, pm10, no2, co)
        aqi_result = predictor.predict(sensor)
        
        # 2. Get Precautions and Live Data
        precaution = get_precautions(aqi_result)
        live_aqi_val = get_live_aqi(city)
        
        # 3. Save to History
        with open("aqi_history.txt", "a", encoding="utf-8") as f:
            f.write(f"City: {city} | Predicted: {aqi_result} | Status: {precaution}\n")
            
        # 4. Display Results
        st.divider()
        st.subheader("Results")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Predicted AQI (ML Model)", aqi_result)
        res_col2.metric("Live AQI (WAQI API)", live_aqi_val)
        st.info(f"**Health Advice:** {precaution}")

with tab2:
    st.header("Prediction History")
    if os.path.exists("aqi_history.txt"):
        with open("aqi_history.txt", "r", encoding="utf-8") as f:
            history_data = f.readlines()
            if history_data:
                for line in reversed(history_data): # Show newest first
                    st.text(line.strip())
            else:
                st.write("No history available yet.")
    else:
        st.write("No history available yet.")