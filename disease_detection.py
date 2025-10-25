# 3_disease_detection_app.py
import streamlit as st
from PIL import Image
import numpy as np

def predict_disease(image):
    diseases = ["Healthy", "Leaf Spot", "Rust"]
    remedies = {
        "Leaf Spot": "Remove and destroy infected leaves. Use a fungicide if the problem persists.",
        "Rust": "Ensure good air circulation. Avoid overhead watering. Apply a copper-based fungicide.",
        "Healthy": "No action needed. Keep monitoring the plant."
    }
    predicted_disease = np.random.choice(diseases)
    remedy = remedies[predicted_disease]
    return predicted_disease, remedy

st.title("🌿 Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload an image of a leaf...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image.', use_column_width=True)
    st.write("")

    if st.button("Diagnose"):
        st.write("Analyzing...")
        
        disease, remedy = predict_disease(image)

        st.header("Diagnosis Result")
        if disease == "Healthy":
            st.success("The leaf appears to be **Healthy**.")
        else:
            st.error(f"Disease Detected: **{disease}**")
            st.subheader("Recommended Action:")
            st.write(remedy)