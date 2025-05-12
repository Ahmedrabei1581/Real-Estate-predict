import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import joblib
import os
import pandas as pd
from PIL import Image
import base64

# Get the directory where this script is located
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

# --- Load zip code data ---
csv_path = os.path.join(BASE_DIR, "uszips.csv")
if not os.path.exists(csv_path):
    st.error("Error: uszips.csv not found. Make sure it's in the same directory as this script.")
    st.stop()
try:
    zip_data = pd.read_csv(csv_path, encoding="utf-8")
except Exception as e:
    st.error(f"Error loading uszips.csv: {e}")
    st.stop()

# --- Create zip-to-address dictionary ---
try:
    zip_data = zip_data.rename(columns={'zip': 'zip_code'})
    zip_to_address = zip_data.set_index('zip_code')[['city', 'state_name', 'county_name']].to_dict('index')
except KeyError as e:
    st.error(f"KeyError: {e}. Please check that required columns exist in uszips.csv.")
    st.stop()

# --- Load and encode image ---
def get_base64_image(image_path):
    if not os.path.exists(image_path):
        st.error(f"Image not found at {image_path}")
        st.stop()
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_path = os.path.join(BASE_DIR, "images.png")
img_base64 = get_base64_image(img_path)

# --- Apply background, overlay, and styling ---
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),
                          url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }}
    h1, h2, h3, h4, h5, h6, label, .stRadio > div > label {{
        color: white !important;
    }}
    .stTextInput > div > input,
    .stNumberInput > div > input,
    .stSelectbox > div > div,
    .stRadio > div {{
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: black !important;
    }}
    .stButton > button {{
        background-color: red;
        color: white;
        font-weight: bold;
    }}
    .result-table {{
        color: white;
        border-collapse: collapse;
        width: 100%;
        margin-top: 10px;
    }}
    .result-table td, .result-table th {{
        border: 1px solid white;
        padding: 8px;
    }}
    .result-table th {{
        background-color: #333;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- App title ---
st.title("🏡 Real Estate Price Predictor")

# --- Input form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.header("Property Details")
        beds = st.selectbox("Bedrooms", options=range(1, 11), index=2)
        baths = st.selectbox("Bathrooms", options=range(1, 11), index=1)
        house_size = st.number_input("House Size (sqft)", min_value=500, max_value=15000, value=2000, step=500)

    with col2:
        st.header("Location")
        zip_code = st.selectbox(
            "Zip Code",
            options=sorted([str(z) for z in zip_data['zip_code'].unique()]),
            index=0
        )
        currency = st.radio("Currency", ("USD", "NTD"), horizontal=True)

    submitted = st.form_submit_button("Predict Price")

    if submitted:
        try:
            address_info = zip_to_address[int(zip_code)]
            city = address_info['city']
            state = address_info['state_name']
            county = address_info['county_name']
        except KeyError:
            st.error("Invalid Zip Code")
            st.stop()

        # Preprocess features
        features = np.array([[beds, baths, house_size, 0.12, 1962661, int(zip_code), 920]])
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)

        # Format prediction
        formatted_price = f"${prediction[0][0]:,.2f}" if currency == "USD" else f"NT${prediction[0][0]:,.0f}"
        location_text = f"{city}, {state} County: {county}, ZIP {zip_code}"

        # Display results using styled HTML table
        st.markdown("### Prediction Result")
        st.markdown(f"""
            <table class="result-table">
                <tr>
                    <th>Estimated Value</th>
                    <th>Property Size</th>
                    <th>Location</th>
                </tr>
                <tr>
                    <td>{formatted_price}</td>
                    <td>{house_size:,} sqft</td>
                    <td>{location_text}</td>
                </tr>
            </table>
        """, unsafe_allow_html=True)
