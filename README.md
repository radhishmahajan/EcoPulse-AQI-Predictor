# 🌿 EcoPulse: AI-Driven Air Quality Predictor

**EcoPulse** is a high-performance web application designed to monitor and predict urban air quality. By leveraging **Machine Learning** and real-time **IoT sensor data**, it empowers users with health-centric insights to mitigate the impact of environmental pollution.

---

## 🎯 Sustainable Development Goals (SDG)
This project is aligned with **SDG Target 11.6**:
> *By 2030, reduce the adverse per capita environmental impact of cities, including by paying special attention to air quality and municipal and other waste management.*

---

## ✨ Key Features
* **Predictive Analytics:** Implements a **Random Forest Regressor** to calculate AQI from multi-pollutant sensor data (PM2.5, PM10, NO2, CO).
* **Object-Oriented Architecture:** Strictly adheres to OOPS principles for modular and scalable code.
* **Live API Integration:** Synchronizes with the **WAQI Global API** to provide real-time air quality metrics for thousands of cities.
* **Interactive Mapping:** Utilizes **Leaflet.js** for spatial visualization of air quality data.
* **Persistent Logging:** Maintains a local history of predictions for trend analysis.
* **Responsive UI:** A modern, mobile-friendly dashboard built with **Bootstrap 5**.

---

## 🏗️ OOPS & Engineering Principles
* **Encapsulation:** Used private attributes in the `AirSensor` class to protect internal sensor states.
* **Abstraction:** Implemented a `BasePredictor` abstract class to standardize ML model interfaces.
* **Robustness:** Features a Unicode-safe (UTF-8) logging system to handle complex environmental emojis and statuses.
* **CI/CD:** Automated deployment pipeline via **GitHub** and **Render**.

---

## 🛠️ Tech Stack
* **Backend:** FastAPI (Python)
* **Machine Learning:** Scikit-Learn, Pandas
* **Frontend:** Jinja2, Bootstrap 5, Leaflet.js
* **API:** WAQI (World Air Quality Index)
* **Server:** Uvicorn

---

## 🚀 Setup & Execution
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/your-username/EcoPulse-AQI-Predictor.git](https://github.com/your-username/EcoPulse-AQI-Predictor.git)
    ```
2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Locally:**
    ```bash
    uvicorn app:app --reload --port 8001
    ```
4.  **View App:** Navigate to `https://ecopulse-aqi-predictor-sxsrynk5skaw3yuvawxmnf.streamlit.app/`

---

## 👤 Developer
**Radhish Mahajan** *Computer Science & Engineering*
