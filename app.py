import os
import time
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from abc import ABC, abstractmethod


# -----------------------------
# PAGE CONFIGURATION & CSS
# -----------------------------
st.set_page_config(
    page_title="EcoPulse AI | Global Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stMetric {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #007bff;
    }
    h1 { font-weight: 800; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

# -----------------------------
# LOCATION DATA
# -----------------------------
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


# -----------------------------
# OOP LOGIC
# -----------------------------
class BasePredictor(ABC):
    @abstractmethod
    def train_model(self, data_path: str):
        pass

    @abstractmethod
    def predict(self, features: list):
        pass


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
            cols = ["PM2.5", "PM10", "NO2", "CO", "AQI"]
            df = df[cols].fillna(df[cols].mean())
            self.model.fit(df[["PM2.5", "PM10", "NO2", "CO"]], df["AQI"])
            self.is_trained = True

    def predict(self, sensor_obj):
        if not self.is_trained:
            return 0.0
        return round(float(self.model.predict(sensor_obj.get_data())[0]), 2)


@st.cache_resource
def load_model():
    predictor = AQIPredictor()
    predictor.train_model("air_quality_data.csv")
    return predictor


predictor = load_model()


# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def get_precautions(aqi):
    if aqi <= 50:
        return "Air is fresh. Ideal for outdoor activities."
    elif aqi <= 100:
        return "Satisfactory air quality. Safe for most people."
    elif aqi <= 150:
        return "Sensitive groups should reduce prolonged outdoor exertion."
    elif aqi <= 200:
        return "Poor air quality. Sensitive groups should avoid outdoor activity."
    elif aqi <= 300:
        return "Very poor air quality. Wear a mask outdoors and limit exposure."
    elif aqi <= 400:
        return "Severe pollution. Stay indoors if possible."
    else:
        return "💀 HEALTH EMERGENCY: Stay indoors. Keep windows closed."


def get_aqi_color(aqi):
    if aqi <= 50:
        return "#22c55e"   # green
    elif aqi <= 100:
        return "#eab308"   # yellow
    elif aqi <= 150:
        return "#f59e0b"   # amber
    elif aqi <= 200:
        return "#f97316"   # orange
    elif aqi <= 300:
        return "#ef4444"   # red
    elif aqi <= 400:
        return "#8b5cf6"   # purple
    else:
        return "#7f1d1d"   # dark maroon


def get_live_data(city):
    TOKEN = "e7c7398fdc76c87a1192df42563e3e835862b478"
    try:
        url = f"https://api.waqi.info/feed/{city}/?token={TOKEN}"
        r = requests.get(url, timeout=5).json()
        if r.get("status") == "ok":
            return r["data"]["aqi"], r["data"]["city"]["geo"][0], r["data"]["city"]["geo"][1]
    except:
        return "N/A", 20.5937, 78.9629
    return "N/A", 20.5937, 78.9629


def parse_history_line(line):
    try:
        time_str = line.split("] ")[0].replace("[", "")
        rest = line.split("] ")[1]
        loc = rest.split(" | Predicted")[0].replace("Location: ", "")
        aqi = float(rest.split("Predicted AQI: ")[1].split(" |")[0])
        status = rest.split("Status: ")[1].strip() if "Status: " in rest else ""
        return {
            "Timestamp": pd.to_datetime(time_str),
            "Location": loc,
            "AQI": aqi,
            "Status": status
        }
    except:
        return None


def generate_pdf_report(df_trend, city_name, state_name, mem_aqi, mem_live, mem_precaution):
    try:
        from io import BytesIO
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        x_left = 40
        y = height - 50

        c.setFont("Helvetica-Bold", 18)
        c.drawString(x_left, y, "EcoPulse AQI Report")

        y -= 30
        c.setFont("Helvetica", 11)
        c.drawString(x_left, y, f"Location: {city_name}, {state_name}")
        y -= 18
        c.drawString(x_left, y, f"AI Predicted AQI: {mem_aqi}")
        y -= 18
        c.drawString(x_left, y, f"Live AQI: {mem_live}")
        y -= 18
        c.drawString(x_left, y, f"Recommendation: {mem_precaution}")
        y -= 25

        c.setFont("Helvetica-Bold", 12)
        c.drawString(x_left, y, "Historical Trend (latest entries)")
        y -= 18

        c.setFont("Helvetica", 9)
        max_rows = min(len(df_trend), 20)
        for _, row in df_trend.tail(max_rows).iterrows():
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 9)
            line = f"{row['Timestamp']} | AQI: {row['AQI']}"
            c.drawString(x_left, y, line[:110])
            y -= 14

        c.showPage()
        c.save()
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
    except Exception:
        return None


# -----------------------------
# SIDEBAR
# -----------------------------
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
    auto_refresh = st.checkbox("🔄 Auto Refresh (5 sec)")


# -----------------------------
# MAIN DASHBOARD
# -----------------------------
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
        mem_aqi = st.session_state.aqi_result
        mem_city = st.session_state.city
        mem_live = st.session_state.live_aqi_val
        mem_prec = st.session_state.precaution
        lat, lon = st.session_state.lat, st.session_state.lon

        # KPI CARDS
        st.markdown("### Key Performance Indicators")
        col1, col2, col3 = st.columns(3)

        bg_color = get_aqi_color(mem_aqi)
        text_color = "black" if 50 < mem_aqi <= 100 else "white"

        with col1:
            st.markdown(
                f"""
                <div style="background-color: {bg_color}; padding: 20px; border-radius: 12px; text-align: left; color: {text_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h5 style="margin: 0; font-size: 16px; opacity: 0.9;">AI Predicted AQI</h5>
                    <h1 style="margin: 0; font-size: 48px; font-weight: bold;">{mem_aqi}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.metric(
                label="📡 Live Station API (WAQI)",
                value=mem_live,
                delta="Real-Time" if mem_live != "N/A" else "Offline",
                delta_color="normal"
            )

        with col3:
            st.metric(
                label="📍 Target Location",
                value=mem_city,
                delta=st.session_state.state,
                delta_color="off"
            )

        # STATUS ALERTS
        st.write("")
        if mem_aqi > 150:
            st.error(f"**Action Required:** {mem_prec}", icon="🚨")
        elif mem_aqi > 50:
            st.warning(f"**Monitoring Advised:** {mem_prec}", icon="⚠️")
        else:
            st.success(f"**Status Optimal:** {mem_prec}", icon="✅")

        # GEO VISUALIZATION
        st.divider()
        st.markdown(f"### 🌐 Geospatial Mapping: {mem_city}")

        fig = go.Figure(go.Scattergeo(
            lat=[lat],
            lon=[lon],
            mode="markers+text",
            marker=dict(size=18, color=bg_color, symbol="circle", line=dict(width=2, color="white")),
            text=[f"{mem_city}"],
            textfont=dict(color="white", size=14, family="Inter, sans-serif"),
            textposition="top center"
        ))

        fig.update_geos(
            projection_type="orthographic",
            showocean=True,
            oceancolor="#020617",
            showland=True,
            landcolor="#1e293b",
            showcountries=True,
            countrycolor="#334155",
            showcoastlines=True,
            coastlinecolor="#475569",
            projection_rotation=dict(lon=lon, lat=lat, roll=0)
        )

        fig.update_layout(
            height=500,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            geo_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # AQI TREND
        st.divider()
        st.markdown("### 📈 AQI Trend Analysis (Past Data)")

        df_trend = None

        if os.path.exists("aqi_history.txt"):
            trend_data = []

            with open("aqi_history.txt", "r", encoding="utf-8") as f:
                for line in f:
                    row = parse_history_line(line)
                    if row and mem_city in row["Location"]:
                        trend_data.append({
                            "Timestamp": row["Timestamp"],
                            "AQI": row["AQI"]
                        })

            if trend_data:
                df_trend = pd.DataFrame(trend_data).sort_values("Timestamp")
                df_trend["MA"] = df_trend["AQI"].rolling(window=3).mean()

                threshold = df_trend["AQI"].mean() + 1.5 * df_trend["AQI"].std()
                spikes = df_trend[df_trend["AQI"] > threshold]

                trend_fig = go.Figure()

                trend_fig.add_trace(go.Scatter(
                    x=df_trend["Timestamp"],
                    y=df_trend["AQI"],
                    mode="lines+markers",
                    line=dict(color="#3b82f6", width=4, shape="spline", smoothing=1.3),
                    marker=dict(
                        size=10,
                        color=df_trend["AQI"],
                        colorscale="RdYlGn_r",
                        showscale=True,
                        colorbar=dict(title="AQI Level")
                    ),
                    fill="tozeroy",
                    fillcolor="rgba(59,130,246,0.15)",
                    hovertemplate="""
                    <b>%{x}</b><br>
                    AQI: <b>%{y}</b><br>
                    Status: %{customdata}
                    <extra></extra>
                    """,
                    customdata=[
                        "Good" if v <= 50 else
                        "Moderate" if v <= 100 else
                        "Poor" if v <= 200 else
                        "Hazardous"
                        for v in df_trend["AQI"]
                    ],
                    name="AQI"
                ))

                trend_fig.add_trace(go.Scatter(
                    x=df_trend["Timestamp"],
                    y=df_trend["MA"],
                    mode="lines",
                    line=dict(color="cyan", width=3, dash="dash"),
                    name="Moving Avg"
                ))

                trend_fig.add_trace(go.Scatter(
                    x=spikes["Timestamp"],
                    y=spikes["AQI"],
                    mode="markers",
                    marker=dict(color="red", size=14, symbol="x"),
                    name="Anomaly"
                ))

                trend_fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.05, line_width=0)
                trend_fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.05, line_width=0)
                trend_fig.add_hrect(y0=100, y1=200, fillcolor="orange", opacity=0.05, line_width=0)
                trend_fig.add_hrect(y0=200, y1=500, fillcolor="red", opacity=0.05, line_width=0)

                trend_fig.update_layout(
                    height=450,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        title="Date & Time",
                        showgrid=False,
                        tickformat="%H:%M:%S\n%b %d"
                    ),
                    yaxis=dict(
                        title="AQI",
                        gridcolor="rgba(255,255,255,0.1)"
                    ),
                    hoverlabel=dict(
                        bgcolor="#111",
                        font_size=13
                    ),
                    legend=dict(orientation="h")
                )

                with st.container():
                    st.plotly_chart(trend_fig, use_container_width=True)

                # DOWNLOADS
                csv = df_trend.to_csv(index=False).encode("utf-8")
                pdf_bytes = generate_pdf_report(
                    df_trend=df_trend,
                    city_name=mem_city,
                    state_name=st.session_state.state,
                    mem_aqi=mem_aqi,
                    mem_live=mem_live,
                    mem_precaution=mem_prec
                )

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "📥 Download CSV Report",
                        csv,
                        "aqi_report.csv",
                        "text/csv",
                        use_container_width=True
                    )
                with dl2:
                    if pdf_bytes:
                        st.download_button(
                            "📄 Download PDF Report",
                            pdf_bytes,
                            "aqi_report.pdf",
                            "application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.info("PDF report needs reportlab installed.")
            else:
                st.info("No historical data available for this location yet.")
        else:
            st.warning("AQI history file not found.")

        # LIVE AQI VS AI PREDICTION
        st.divider()
        st.markdown("### ⚖️ Live AQI vs AI Prediction")

        try:
            live_aqi_num = float(mem_live)
        except:
            live_aqi_num = 0

        comparison_df = pd.DataFrame({
            "Type": ["Live AQI", "AI Predicted AQI"],
            "Value": [live_aqi_num, mem_aqi]
        })

        fig_bar = px.bar(
            comparison_df,
            x="Type",
            y="Value",
            color="Type",
            text="Value"
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        # MULTI-CITY COMPARISON
        st.divider()
        st.markdown("### 🌆 Multi-City AQI Comparison")

        options = LOCATION_DATA[country][state]

        # Fix default safely
        default_city = [mem_city] if mem_city in options else [options[0]]

        cities_to_compare = st.multiselect(
            "Select Cities",
            options,
            default=default_city
        )

        if cities_to_compare:
            multi_data = []

            if os.path.exists("aqi_history.txt"):
                with open("aqi_history.txt", "r", encoding="utf-8") as f:
                    for line in f:
                        row = parse_history_line(line)
                        if row:
                            for c in cities_to_compare:
                                if c in row["Location"]:
                                    multi_data.append({
                                        "Timestamp": row["Timestamp"],
                                        "AQI": row["AQI"],
                                        "City": c
                                    })

            if multi_data:
                df_multi = pd.DataFrame(multi_data)

                fig_multi = px.line(
                    df_multi,
                    x="Timestamp",
                    y="AQI",
                    color="City",
                    markers=True
                )
                fig_multi.update_layout(
                    height=420,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )

                st.plotly_chart(fig_multi, use_container_width=True)
            else:
                st.info("No historical data available for the selected cities yet.")
        else:
            st.info("Select at least one city to compare.")
    else:
        st.info("👈 Please configure your parameters in the Control Panel and click 'Run AI Analysis' to view data.")


# -----------------------------
# HISTORY / LOGS TAB
# -----------------------------
with tab2:
    st.markdown("### 📑 System Inference Logs")

    if os.path.exists("aqi_history.txt"):
        data = []
        with open("aqi_history.txt", "r", encoding="utf-8") as f:
            for line in f:
                row = parse_history_line(line)
                if row:
                    data.append({
                        "Timestamp": row["Timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                        "Location": row["Location"],
                        "AI AQI": row["AQI"],
                        "System Recommendation": row["Status"]
                    })

        if data:
            df = pd.DataFrame(data).iloc[::-1]

            def highlight_aqi(val):
                if val <= 50:
                    return "background-color: rgba(34, 197, 94, 0.2); color: #22c55e; font-weight: bold;"
                elif val <= 100:
                    return "background-color: rgba(234, 179, 8, 0.2); color: #eab308; font-weight: bold;"
                elif val <= 200:
                    return "background-color: rgba(249, 115, 22, 0.2); color: #f97316; font-weight: bold;"
                else:
                    return "background-color: rgba(239, 68, 68, 0.2); color: #ef4444; font-weight: bold;"

            try:
                styled_df = df.style.map(highlight_aqi, subset=["AI AQI"])
            except AttributeError:
                styled_df = df.style.applymap(highlight_aqi, subset=["AI AQI"])

            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.write("No logs recorded yet.")
    else:
        st.write("System logs are currently empty.")


# -----------------------------
# AUTO REFRESH
# -----------------------------
if auto_refresh:
    time.sleep(5)
    st.rerun()