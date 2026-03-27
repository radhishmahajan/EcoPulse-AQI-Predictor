import streamlit as st
import pandas as pd
import os
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from abc import ABC, abstractmethod
import plotly.graph_objects as go

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="EcoPulse AI | Global Dashboard", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for an Enterprise Look (Dark/Light Mode Safe)
st.markdown("""
    <style>
    .stMetric { background-color: rgba(128, 128, 128, 0.1); padding: 20px; border-radius: 12px; border-left: 5px solid #007bff;}
    h1 { font-weight: 800; }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZE MEMORY (SESSION STATE) ---
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

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

@st.cache_resource
def load_model():
    predictor = AQIPredictor()
    predictor.train_model("air_quality_data.csv")
    return predictor

predictor = load_model()

# --- UTILITY FUNCTIONS ---
def get_precautions(aqi):
    if aqi <= 50: return "Air is fresh. Ideal for outdoor activities."
    elif aqi <= 100: return "Acceptable air quality. Safe for most."
    elif aqi <= 200: return "⚠️ Poor Quality. Sensitive groups take care."
    elif aqi <= 300: return "🚨 Hazardous. Wear N95 masks outdoors."
    else: return "💀 EMERGENCY: Extreme pollution. Stay indoors."

def get_aqi_color(aqi):
    if aqi <= 50: return "#22c55e" # Emerald Green
    elif aqi <= 100: return "#eab308" # Warning Yellow
    elif aqi <= 200: return "#f97316" # Alert Orange
    else: return "#ef4444" # Critical Red

def get_live_data(city):
    TOKEN = "e7c7398fdc76c87a1192df42563e3e835862b478" 
    try:
        url = f"https://api.waqi.info/feed/{city}/?token={TOKEN}"
        r = requests.get(url, timeout=5).json()
        if r['status'] == 'ok':
            return r['data']['aqi'], r['data']['city']['geo'][0], r['data']['city']['geo'][1]
    except:
        return "N/A", 20.5937, 78.9629 
    return "N/A", 20.5937, 78.9629

# --- SIDEBAR (CONTROL PANEL) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208985.png", width=60)
    st.title("⚙️ Control Panel")
    st.markdown("Configure location and IoT sensor data for ML inference.")
    
    st.header("📍 Region Selector")
    country = st.selectbox("Select Country", list(LOCATION_DATA.keys()))
    state = st.selectbox("Select State", list(LOCATION_DATA[country].keys()))
    city = st.selectbox("Select City", LOCATION_DATA[country][state])
    
    st.header("🔬 Sensor Telemetry")
    pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, 50.0)
    pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 80.0)
    no2 = st.slider("NO2 (ppb)", 0.0, 200.0, 30.0)
    co = st.slider("CO (ppm)", 0.0, 50.0, 1.5)
    
    st.divider()
    predict_clicked = st.button("🚀 Run AI Analysis", use_container_width=True, type="primary")

# --- MAIN DASHBOARD ---
st.title("🌍 EcoPulse Enterprise Dashboard")
st.markdown("Real-time telemetry and predictive analytics for urban environmental monitoring.")

tab1, tab2 = st.tabs(["📊 Analytics Overview", "📑 System Logs"])

with tab1:
    live_aqi_val, live_lat, live_lon = get_live_data(city)
    
    if predict_clicked:
        sensor = AirSensor(pm25, pm10, no2, co)
        aqi_result = predictor.predict(sensor)
        precaution = get_precautions(aqi_result)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("aqi_history.txt", "a", encoding="utf-8") as f:
            f.write(f"[{current_time}] Location: {city}, {state} | Predicted AQI: {aqi_result} | Status: {precaution}\n")
            
        st.session_state.prediction_made = True
        st.session_state.aqi_result = aqi_result
        st.session_state.precaution = precaution
        st.session_state.city = city
        st.session_state.state = state
        st.session_state.live_aqi_val = live_aqi_val
        st.session_state.lat = live_lat
        st.session_state.lon = live_lon

    if st.session_state.prediction_made:
        # Load memory variables
        mem_aqi = st.session_state.aqi_result
        mem_city = st.session_state.city
        mem_live = st.session_state.live_aqi_val
        mem_prec = st.session_state.precaution
        lat, lon = st.session_state.lat, st.session_state.lon

        # TOP ROW METRICS
        st.markdown("### Key Performance Indicators")
        col1, col2, col3 = st.columns(3)
        
        bg_color = get_aqi_color(mem_aqi)
        text_color = "black" if mem_aqi > 50 and mem_aqi <= 100 else "white"

        with col1:
            st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 20px; border-radius: 12px; text-align: left; color: {text_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h5 style="margin: 0; font-size: 16px; opacity: 0.9;">AI Predicted AQI</h5>
                    <h1 style="margin: 0; font-size: 48px; font-weight: bold;">{mem_aqi}</h1>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.metric(label="📡 Live Station API (WAQI)", value=mem_live, delta="Real-Time" if mem_live != "N/A" else "Offline", delta_color="normal")
            
        with col3:
            st.metric(label="📍 Target Location", value=mem_city, delta=st.session_state.state, delta_color="off")

        # HEALTH STATUS ALERTS
        st.write("")
        if mem_aqi > 150:
            st.error(f"**Action Required:** {mem_prec}", icon="🚨")
        elif mem_aqi > 50:
            st.warning(f"**Monitoring Advised:** {mem_prec}", icon="⚠️")
        else:
            st.success(f"**Status Optimal:** {mem_prec}", icon="✅")

        # 3D GLOBE VISUALIZATION
        st.divider()
        st.markdown(f"### 🌐 Geospatial Mapping: {mem_city}")
        
        fig = go.Figure(go.Scattergeo(
            lat=[lat], lon=[lon],
            mode='markers+text',
            marker=dict(size=18, color=bg_color, symbol="circle", line=dict(width=2, color="white")),
            text=[f"{mem_city}"],
            textfont=dict(color="white", size=14, family="Inter, sans-serif"),
            textposition="top center"
        ))

        fig.update_geos(
            projection_type="orthographic",
            showocean=True, oceancolor="#020617",
            showland=True, landcolor="#1e293b",
            showcountries=True, countrycolor="#334155",
            showcoastlines=True, coastlinecolor="#475569",
            projection_rotation=dict(lon=lon, lat=lat, roll=0)
        )

        fig.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)", geo_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("👈 Please configure your parameters in the Control Panel and click 'Run AI Analysis' to view data.")

# --- HISTORY / LOGS TAB ---
with tab2:
    st.markdown("### 📑 System Inference Logs")
    if os.path.exists("aqi_history.txt"):
        data = []
        with open("aqi_history.txt", "r", encoding="utf-8") as f:
            for line in f:
                try:
                    # Cleanly parsing your text file into a dataframe
                    time_str = line.split("] ")[0].replace("[", "")
                    rest = line.split("] ")[1]
                    loc = rest.split(" | Predicted")[0].replace("Location: ", "")
                    aqi = float(rest.split("Predicted AQI: ")[1].split(" |")[0])
                    status = rest.split("Status: ")[1].strip()
                    data.append({"Timestamp": time_str, "Location": loc, "AI AQI": aqi, "System Recommendation": status})
                except:
                    pass
        if data:
            # Convert to Pandas DataFrame for a professional table look
            df = pd.DataFrame(data).iloc[::-1] # Show newest first
            
            # --- ENTERPRISE DATA STYLING ---
            def highlight_aqi(val):
                # Apply professional, high-contrast colors based on AQI severity
                if val <= 50: 
                    return 'background-color: rgba(34, 197, 94, 0.2); color: #22c55e; font-weight: bold;'
                elif val <= 100: 
                    return 'background-color: rgba(234, 179, 8, 0.2); color: #eab308; font-weight: bold;'
                elif val <= 200: 
                    return 'background-color: rgba(249, 115, 22, 0.2); color: #f97316; font-weight: bold;'
                else: 
                    return 'background-color: rgba(239, 68, 68, 0.2); color: #ef4444; font-weight: bold;'

            try:
                # pandas >= 2.1.0
                styled_df = df.style.map(highlight_aqi, subset=['AI AQI'])
            except AttributeError:
                # pandas < 2.1.0
                styled_df = df.style.applymap(highlight_aqi, subset=['AI AQI'])

            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.write("No logs recorded yet.")
    else:
        st.write("System logs are currently empty.")