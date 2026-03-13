import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from abc import ABC, abstractmethod

# 1. ABSTRACTION: Blueprint for the Predictor
class BasePredictor(ABC):
    @abstractmethod
    def train_model(self, data_path): pass
    @abstractmethod
    def predict(self, features): pass

# 2. ENCAPSULATION: Hiding Raw Sensor Readings
class AirSensor:
    def __init__(self, pm25, pm10, no2, co):
        self.__pm25 = pm25  # Private attributes
        self.__pm10 = pm10
        self.__no2 = no2
        self.__co = co

    def get_data_as_list(self):
        return [[self.__pm25, self.__pm10, self.__no2, self.__co]]

# 3. INHERITANCE: AQI Predictor with Data Cleaning
class AQIPredictor(BasePredictor):
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False

    def train_model(self, data_path):
        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError("Dataset not found! Check the file path.")
            
            # --- DATA CLEANING PIPELINE ---
            df = pd.read_csv(data_path)
            relevant_cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'AQI']
            df = df[relevant_cols]
            
            # Imputation: Filling missing values with Column Mean
            for col in relevant_cols:
                df[col] = df[col].fillna(df[col].mean())
            
            df.dropna(subset=['AQI'], inplace=True)
            
            X = df[['PM2.5', 'PM10', 'NO2', 'CO']]
            y = df['AQI']

            self.model.fit(X, y)
            self.is_trained = True

        except Exception as e:
            st.error(f"Training failed: {e}")

    def predict(self, sensor_obj):
        if not self.is_trained: return "Model not ready."
        return round(self.model.predict(sensor_obj.get_data_as_list())[0], 2)

# 4. FILE & EXCEPTION HANDLING
class ReportManager:
    @staticmethod
    def save_report(station, result):
        try:
            with open("aqi_history.txt", "a") as f:
                f.write(f"Station: {station} | Predicted AQI: {result}\n")
        except IOError:
            st.error("File handling error occurred.")

# --- WEB APP INTERFACE ---
def run_web_app():
    st.set_page_config(page_title="EcoPulse AQI Predictor", layout="wide", page_icon="🌱")
    
    st.title("🌱 EcoPulse: SDG 11 Air Quality Monitor")
    st.markdown("### Predicting Urban Air Quality using Classical Machine Learning")
    st.info("This project supports UN SDG 11: Sustainable Cities and Communities.")

    # Sidebar for User Inputs
    st.sidebar.header("📍 Enter Sensor Data")
    station = st.sidebar.text_input("City/Station Name", "New Delhi")
    pm25 = st.sidebar.slider("PM2.5 Level", 0, 500, 50)
    pm10 = st.sidebar.slider("PM10 Level", 0, 500, 80)
    no2 = st.sidebar.slider("NO2 Level", 0, 200, 30)
    co = st.sidebar.slider("CO Level", 0, 50, 2)

    # Cache the model so it doesn't retrain on every slider movement
    @st.cache_resource
    def load_predictor():
        p = AQIPredictor()
        p.train_model("./air_quality_data.csv")
        return p

    predictor = load_predictor()

    if st.button("Calculate Impact"):
        sensor = AirSensor(pm25, pm10, no2, co)
        result = predictor.predict(sensor)
        
        # VISUAL EFFECT: Metric Cards
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Predicted AQI Value", f"{result}")
        
        # VISUAL EFFECT: Color-coded Health Alerts
        with col2:
            if isinstance(result, float):
                if result <= 100:
                    st.success(f"✅ Status: Satisfactory. Safe for outdoor activities.")
                elif result <= 200:
                    st.warning(f"⚠️ Status: Moderate. Sensitive groups should stay indoors.")
                else:
                    st.error(f"🚨 Status: POOR/SEVERE. High health risk (SDG 11 Alert).")
            else:
                st.error(result)

        # VISUAL EFFECT: Comparison Chart
        st.subheader("Pollutant Breakdown")
        chart_data = pd.DataFrame({
            'Pollutant': ['PM2.5', 'PM10', 'NO2'],
            'Current Level': [pm25, pm10, no2],
            'Safe Limit': [60, 100, 80]
        })
        st.bar_chart(chart_data.set_index('Pollutant'))

        # File Handling
        ReportManager.save_report(station, result)
        st.toast("Report saved to aqi_history.txt!", icon='💾')
