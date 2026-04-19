import random
from datetime import datetime, timedelta

cities_data = {
    "Delhi": ["Central Delhi", "East Delhi", "New Delhi", "North Delhi", "South Delhi"],
    "Punjab": ["Amritsar", "Ludhiana", "Jalandhar", "Patiala", "Bathinda"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Thane", "Nashik"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hubballi", "Mangaluru"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Noida", "Agra", "Varanasi"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Siliguri"]
}

def get_status(aqi):
    if aqi <= 50:
        return "Air is fresh. Ideal for outdoor activities."
    elif aqi <= 100:
        return "Acceptable air quality. Safe for most."
    elif aqi <= 200:
        return "⚠️ Poor Quality. Sensitive groups take care."
    elif aqi <= 300:
        return "🚨 Hazardous. Wear N95 masks outdoors."
    else:
        return "💀 EMERGENCY: Extreme pollution. Stay indoors."

start_date = datetime(2026, 3, 10)
lines = []

for state, cities in cities_data.items():
    for city in cities:
        base_aqi = random.randint(70, 150)

        for i in range(10):  # 🔥 10 entries per city (TOTAL 200+)
            dt = start_date + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            fluctuation = random.randint(-40, 80)
            aqi = max(20, base_aqi + fluctuation)

            # spike (for anomaly detection)
            if random.random() < 0.2:
                aqi += random.randint(100, 250)

            aqi = min(aqi, 450)

            line = f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] Location: {city}, {state} | Predicted AQI: {round(aqi,2)} | Status: {get_status(aqi)}"
            lines.append(line)

# sort by time
lines.sort()

with open("aqi_history.txt", "w", encoding="utf-8") as f:
    for l in lines:
        f.write(l + "\n")

print("✅ Dataset generated successfully (200+ entries)")