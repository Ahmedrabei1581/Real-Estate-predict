import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import joblib
import os
import pandas as pd
from PIL import Image
import base64

BASE_DIR = os.path.dirname(__file__)

# --- Load model architecture ---
json_path = os.path.join(BASE_DIR, "model.json")
if not os.path.exists(json_path):
    st.error(f"model.json not found at {json_path}")
    st.stop()
with open(json_path, "r") as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

# --- Load weights and scaler ---
weights_path = os.path.join(BASE_DIR, "model.weights.h5")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
if not os.path.exists(weights_path):
    st.error(f"model.weights.h5 not found at {weights_path}")
    st.stop()
if not os.path.exists(scaler_path):
    st.error(f"scaler.pkl not found at {scaler_path}")
    st.stop()
model.load_weights(weights_path)
scaler = joblib.load(scaler_path)

# --- Load Egypt city data ---
csv_path = os.path.join(BASE_DIR, "eg.csv")
if not os.path.exists(csv_path):
    st.error("Error: eg.csv not found. Please add it to the app directory.")
    st.stop()
try:
    city_data = pd.read_csv(csv_path, encoding="utf-8")
except Exception as e:
    st.error(f"Error loading eg.csv: {e}")
    st.stop()

# --- Load and encode image (optional) ---
def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
img_path = os.path.join(BASE_DIR, "images.png")
img_base64 = get_base64_image(img_path)

# --- Button styling ---
st.markdown(
    """
    <style>
    .stButton>button {
        color: black !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- App title ---
st.title("🏡 Real Estate Price Predictor (Egypt)")

# --- Input form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.header("Property Details")
        beds = st.selectbox("Bedrooms", options=range(1, 11), index=2)
        baths = st.selectbox("Bathrooms", options=range(1, 11), index=1)
        house_size_m2 = st.number_input("House Size (m²)", min_value=50, max_value=1400, value=200, step=10)
    with col2:
        st.header("Location")
        # City dropdown from eg.csv
        cities = sorted(city_data['city'].dropna().unique())
        city = st.selectbox("City", cities)
        neighborhood = st.text_input("Neighborhood (حي)", "")
        street = st.text_input("Street (شارع)", "")
        postal_code = st.text_input("Postal Code (الرمز البريدي)", "")
        currency = st.radio("Currency", ("EGP", "USD"), horizontal=True)
    submitted = st.form_submit_button("Predict Price")

if submitted:
    # Convert m² to sqft for the model (if needed)
    house_size_sqft = house_size_m2 * 10.7639

    # --- Prepare features for model ---
    # Update this as per your model's expected input order and features!
    # Here, we use dummy values for fields not collected from the user.
    features = np.array([[beds, baths, house_size_sqft, 0.12, 1962661, 0, 920]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)

    # Format prediction
    if currency == "USD":
        formatted_price = f"${prediction[0][0]:,.2f}"
    else:
        formatted_price = f"EGP {prediction[0][0]:,.0f}"

    # Build location string and dynamic Google Maps link
    location_parts = [street, neighborhood, city, postal_code, "Egypt"]
    location_text = ", ".join([part for part in location_parts if part])
    maps_url = f"https://www.google.com/maps/search/?api=1&query={location_text.replace(' ', '+')}"

    # Display results
    st.markdown("### Prediction Result")
    st.markdown(f"""
|Estimated Value|Property Size|Location|
|--|--|--|
|{formatted_price}|{house_size_m2:,} m²|{location_text}|
""")
    st.markdown(f"[View on Map]({maps_url})")
