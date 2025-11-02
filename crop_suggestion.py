import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os

# --- List of your custom fruits and their image files ---
IMAGE_MAP = {
    "Ramphal": "Ram phala.jpg",
    "Lakshmanphal": "Lakshman phala.jpg",
    "Wood Apple": "wood apple.jpg"
}

# --- Load trained model and dataset ---
@st.cache_resource
def load_model():
    return joblib.load("crop_recommendation_model.joblib")   # adjust filename if needed

@st.cache_data
def load_data():
    return pd.read_csv("Crop_recommendation.csv")

model = load_model()
df = load_data()

st.title("🌾 Smart Crop Recommendation & Similar Crop Finder")

# --- Input form ---
st.header("Enter Environmental Parameters")

N = st.number_input('Nitrogen (N)', 0, 200, 90)
P = st.number_input('Phosphorus (P)', 0, 200, 40)
K = st.number_input('Potassium (K)', 0, 250, 40)
temperature = st.number_input('Temperature (°C)', -10.0, 50.0, 25.0, step=0.1)
humidity = st.number_input('Humidity (%)', 0.0, 100.0, 70.0, step=0.1)
ph = st.number_input('pH Value', 0.0, 14.0, 6.5, step=0.1)
rainfall = st.number_input('Rainfall (mm)', 0.0, 1500.0, 100.0, step=0.1)

# --- When button pressed ---
if st.button("🌱 Recommend Crop"):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N','P','K','temperature','humidity','ph','rainfall'])

    # Predict primary crop
    prediction = model.predict(input_data)[0]
    st.success(f"✅ **Recommended Crop:** {prediction.capitalize()}")

    # Show image if exists
    crop_lower = prediction.lower()
    if crop_lower in IMAGE_MAP and os.path.exists(IMAGE_MAP[crop_lower]):
        st.image(IMAGE_MAP[crop_lower], caption=f"Recommended Crop: {prediction.capitalize()}", use_container_width=True)

    # --- Find similar crops ---
    st.subheader("🌿 Other Crops with Similar Conditions")
    target = input_data.iloc[0]

    # Compute a simple similarity score (lower = more similar)
    df["score"] = np.sqrt(
        (df["N"] - target["N"])**2 +
        (df["P"] - target["P"])**2 +
        (df["K"] - target["K"])**2 +
        (df["temperature"] - target["temperature"])**2 +
        (df["humidity"] - target["humidity"])**2 +
        (df["ph"] - target["ph"])**2 +
        (df["rainfall"] - target["rainfall"])**2
    )

    # Average per crop type
    crop_scores = df.groupby("label")["score"].mean().reset_index()
    crop_scores = crop_scores.sort_values(by="score", ascending=True)

    # Convert to similarity percentage
    crop_scores["suitability (%)"] = 100 * (1 - (crop_scores["score"] / crop_scores["score"].max()))

    # Show top 5 similar crops
    top_similar = crop_scores.head(5)
    for _, row in top_similar.iterrows():
        st.info(f"**{row['label'].capitalize()}** — Suitability: {row['suitability (%)']:.2f}%")

        # show image if available
        label_lower = row["label"].lower()
        if label_lower in IMAGE_MAP and os.path.exists(IMAGE_MAP[label_lower]):
            st.image(IMAGE_MAP[label_lower], caption=f"{row['label'].capitalize()}", use_container_width=True)
