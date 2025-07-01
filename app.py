import streamlit as st
import numpy as np
import pickle
from datetime import datetime

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🚕 NYC Taxi Fare Predictor")

# User inputs
pickup_date = st.date_input("Pickup Date", datetime.now().date())
pickup_time = st.time_input("Pickup Time", datetime.now().time())
pickup_datetime = datetime.combine(pickup_date, pickup_time)

pickup_lat = st.number_input("Pickup Latitude", value=40.7128)
pickup_long = st.number_input("Pickup Longitude", value=-74.0060)
dropoff_lat = st.number_input("Dropoff Latitude", value=40.7769)
dropoff_long = st.number_input("Dropoff Longitude", value=-73.9813)
passenger_count = st.slider("Passenger Count", 1, 6, 1)

# Haversine formula to calculate distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

distance = haversine(pickup_lat, pickup_long, dropoff_lat, dropoff_long)

# Predict button
if st.button("Predict Fare"):
    hour = pickup_datetime.hour
    features = np.array([[hour, passenger_count, distance]])
    fare = model.predict(features)[0]
    st.success(f"Estimated Fare: ${fare:.2f}")
