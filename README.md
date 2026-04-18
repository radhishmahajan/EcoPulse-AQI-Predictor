# 🌿 EcoPulse AI: Intelligent Air Quality Analytics Platform

**EcoPulse AI** is a next-generation, enterprise-grade web application for **real-time air quality monitoring, predictive analytics, and environmental intelligence**.
It integrates **Machine Learning, IoT sensor simulation, and live API data** to provide actionable insights for healthier urban living.

---

## 🎯 Sustainable Development Goals (SDG)

This project aligns with **United Nations SDG 11.6**:

> *Reduce the environmental impact of cities by improving air quality and sustainable urban management.*

---

## 🚀 Core Capabilities

### 🔬 Predictive Intelligence

* Uses **Random Forest Regressor (Scikit-Learn)** to predict AQI from:

  * PM2.5
  * PM10
  * NO₂
  * CO
* Real-time inference using simulated IoT sensor inputs

---

### 📡 Live Environmental Data

* Integrated with **WAQI (World Air Quality Index API)**
* Fetches:

  * Real-time AQI
  * Geolocation (Latitude & Longitude)
* Enables **comparison between predicted vs actual AQI**

---

### 🌍 Advanced Data Visualization

* **3D Interactive Globe (Plotly)**

  * Orthographic projection
  * Real-time city mapping

* **AQI Trend Analysis**

  * Smooth curve visualization
  * Historical data tracking

* **Multi-City Comparison Dashboard**

  * Compare AQI trends across multiple cities
  * Dynamic selection via UI

---

### 📊 Advanced Analytics Features

* 📉 **Moving Average (Trend Smoothing)**
  Detects long-term patterns in AQI data

* ⚠️ **Anomaly Detection (Spike Detection)**
  Identifies abnormal pollution spikes using statistical thresholds

* ⚖️ **Live vs AI Comparison**

  * Bar chart comparing:

    * Live AQI (API)
    * Predicted AQI (ML Model)

---

### 🔄 Real-Time System Behavior

* Auto-refresh dashboard every 5 seconds (optional)
* Dynamic UI updates using **Streamlit session state**

---

### 📑 Reporting & Data Export

* 📥 Export AQI data as:

  * CSV Report
  * PDF Report (auto-generated summary)

* 🧾 System Logs:

  * Persistent logging of predictions
  * Styled DataFrame with AQI-based color coding

---

## 🏗️ Software Engineering & OOPS Design

### 🔒 Encapsulation

* Private attributes in `AirSensor` class for sensor integrity

### 🧩 Abstraction

* `BasePredictor` abstract class standardizes ML pipeline

### 🔁 Modular Architecture

* Separation of:

  * Data ingestion
  * ML model
  * Visualization
  * UI logic

### ⚙️ Data Pipeline

* Converts:

  * API JSON → Structured DataFrame
  * Log file → Analytical dataset

---

## 🛠️ Tech Stack

| Category         | Technology      |
| ---------------- | --------------- |
| Framework        | Streamlit       |
| Language         | Python 3.9+     |
| Machine Learning | Scikit-Learn    |
| Data Processing  | Pandas          |
| Visualization    | Plotly          |
| API              | WAQI API        |
| Deployment       | Streamlit Cloud |

---

## 🧠 System Architecture

```text
IoT Sensor Input → ML Model (Random Forest) → AQI Prediction
                                 ↓
                      Data Logging (TXT)
                                 ↓
              Data Processing (Pandas DataFrame)
                                 ↓
   Visualization (Trend, Comparison, Globe, Analytics)
```

---

## 🚀 Setup & Execution

### 1️⃣ Clone Repository

```bash
git clone https://github.com/radhishmahajan/EcoPulse-AQI-Predictor.git
cd EcoPulse-AQI-Predictor
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run app.py
```

---

## 🌐 Live Deployment

👉 https://ecopulse-aqi-predictor-sxsrynk5skaw3yuvawxmnf.streamlit.app/

---

## 📸 Key Highlights

* Enterprise Dashboard UI
* Real-time + Predicted AQI comparison
* Interactive 3D Globe
* AI-based anomaly detection
* Multi-city analytics
* Exportable reports

---

## 👨‍💻 Developer

**Radhish Mahajan**
Computer Science & Engineering

🔗 GitHub: https://github.com/radhishmahajan

---

## 💡 Future Enhancements

* Deep Learning (LSTM for time-series AQI prediction)
* Mobile app integration
* Alert system (SMS/Email notifications)
* Cloud database (Firebase / MongoDB)
* AI-based pollution forecasting

---

## ⭐ Project Impact

EcoPulse AI bridges the gap between **data science and environmental sustainability**, enabling smarter decisions for urban air quality management.
