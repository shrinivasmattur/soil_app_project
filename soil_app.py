import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ---------------------------
# Soil Image Analysis Function
# ---------------------------
def analyze_soil(image_bytes):
    # Convert uploaded file into OpenCV format
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess
    resized = cv2.resize(img, (256, 256))
    blurred = cv2.GaussianBlur(resized, (5,5), 0)

    # Extract features
    mean_color = cv2.mean(blurred)[:3]
    hist_r = cv2.calcHist([blurred], [0], None, [256], [0,256])
    hist_g = cv2.calcHist([blurred], [1], None, [256], [0,256])
    hist_b = cv2.calcHist([blurred], [2], None, [256], [0,256])

    # Visualization
    fig, ax = plt.subplots(1,3, figsize=(15,4))
    ax[0].imshow(img)
    ax[0].set_title("Original Soil Image")
    ax[0].axis("off")

    ax[1].imshow([[mean_color]])
    ax[1].set_title(f"Mean Color: {np.round(mean_color,2)}")
    ax[1].axis("off")

    ax[2].plot(hist_r, color='red')
    ax[2].plot(hist_g, color='green')
    ax[2].plot(hist_b, color='blue')
    ax[2].set_title("RGB Histogram")
    ax[2].set_xlabel("Pixel Intensity")
    ax[2].set_ylabel("Frequency")

    st.pyplot(fig)

    # Dummy soil type detection (replace with ML later)
    avg_intensity = np.mean(mean_color)
    if avg_intensity > 150:
        soil_type = "Sandy/Light Soil"
    elif avg_intensity > 100:
        soil_type = "Loamy Soil"
    else:
        soil_type = "Clay/Black Soil"

    # Dummy pH estimation
    ph_value = round(avg_intensity/50, 2)

    return soil_type, ph_value

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("🌱 Smart Soil Testing with Map Visualization")

uploaded_file = st.file_uploader("Upload a soil image", type=["jpg","png","jpeg"])

st.subheader("🗺 Select Location on Map")

# Default map centered on Karnataka
m = folium.Map(location=[12.2958, 76.6394], zoom_start=7)

# Add instruction marker
folium.Marker(
    [12.2958, 76.6394],
    popup="Click anywhere on the map to select location"
).add_to(m)

# Capture map clicks
map_data = st_folium(m, width=700, height=500)

if uploaded_file is not None and map_data and map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    # Analyze soil
    soil_type, ph_value = analyze_soil(uploaded_file)

    st.success(f"Detected Soil Type: **{soil_type}** | Estimated pH: **{ph_value}**")
    st.write(f"📍 Location chosen: Latitude = {lat:.4f}, Longitude = {lon:.4f}")

    # Show updated map with marker
    result_map = folium.Map(location=[lat, lon], zoom_start=7)
    popup_text = f"""
    <b>Soil Analysis Result</b><br>
    Soil: {soil_type}<br>
    pH: {ph_value}<br>
    Lat: {lat:.4f}, Lon: {lon:.4f}
    """
    folium.Marker([lat, lon], popup=popup_text).add_to(result_map)
    st_folium(result_map, width=700, height=500)
