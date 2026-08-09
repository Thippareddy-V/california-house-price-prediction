import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:5000")

st.title("California House Price Predictor")

med_inc = st.number_input("Median Income (in $10,000s)", 0.5, 15.0, 3.5)
house_age = st.number_input("House Age (years)", 1, 52, 20)
ave_rooms = st.number_input("Average Rooms", 1.0, 20.0, 5.0)
ave_bedrms = st.number_input("Average Bedrooms", 0.5, 5.0, 1.0)
population = st.number_input("Population", 3, 40000, 1000)
ave_occup = st.number_input("Average Occupancy", 0.5, 20.0, 3.0)
latitude = st.number_input("Latitude", 32.0, 42.0, 34.0)
longitude = st.number_input("Longitude", -125.0, -114.0, -118.0)

if st.button("Predict Price"):
    payload = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            price = response.json()["predicted_price_usd"]
            st.success(f"Predicted Median House Price: ${price:,.0f}")
        else:
            st.error(response.json().get("errors", "Prediction failed"))
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach API: {e}")