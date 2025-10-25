# 2_crop_suggestion_app.py
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

def predict_soil_from_image(image):
    soil_types = ["Alluvial Soil", "Black Soil", "Red Soil", "Laterite Soil"]
    return np.random.choice(soil_types)

def suggest_crop(soil_type):
    crop_suggestions = {
        "Alluvial Soil": ["Ramphala", "Lakshmanaphala"],
        "Black Soil": ["Wood Apple"],
        "Red Soil": ["Ramphala", "Wood Apple"],
        "Laterite Soil": ["Lakshmanaphala"]
    }
    return crop_suggestions.get(soil_type, ["No specific suggestion for this soil type."])

st.title("🌱 Crop Suggestion for Indigenous Fruits")
st.header("Input Soil Information")

input_method = st.radio("Choose input method:", ("Manual Input", "Upload Image"))
predicted_soil = None

if input_method == "Manual Input":
    soil_type_manual = st.selectbox(
        "Select Soil Type:",
        ("Alluvial Soil", "Black Soil", "Red Soil", "Laterite Soil")
    )
    if st.button("Get Crop Suggestion"):
        predicted_soil = soil_type_manual

elif input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose a soil image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Soil Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        predicted_soil = predict_soil_from_image(image)
        st.success(f"Predicted Soil Type: **{predicted_soil}**")

if predicted_soil:
    st.header("Recommended Crops")
    suggestions = suggest_crop(predicted_soil)
    for crop in suggestions:
        st.subheader(f"-> {crop}")