import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from geopy.distance import geodesic
from datetime import datetime

# ---------------------------
# Karnataka Soil Data
# ---------------------------
karnataka_soils = {
    "Bangalore": {"coords": [12.9716, 77.5946], "soil_type": "Red Sandy Soil"},
    "Mysore": {"coords": [12.2958, 76.6394], "soil_type": "Red Sandy Soil"},
    "Bidar": {"coords": [17.9149, 77.5046], "soil_type": "Black Soil"},
    "Gulbarga": {"coords": [17.3297, 76.8343], "soil_type": "Black Soil"},
    "Udupi": {"coords": [13.3409, 74.7421], "soil_type": "Laterite Soil"},
    "Dakshina Kannada": {"coords": [12.8438, 75.2473], "soil_type": "Laterite Soil"},
    "Tumkur": {"coords": [13.3422, 77.1010], "soil_type": "Loamy Soil"},
    "Chitradurga": {"coords": [14.2306, 76.3980], "soil_type": "Loamy Soil"}
}

# ---------------------------
# Plant Recommendations Data
# ---------------------------
plants = {
    "Ramphal (Annona reticulata)": {
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/7/79/Annona_reticulata_Blanco1.197-cropped.jpg",
        "soil_types": ["Sandy/Light Soil", "Loamy Soil"],
        "ph_range": (5.5, 7.5),
        "moisture_range": (40, 80),
        "flowering_months": (1, 3),  # January–March
        "fruiting_months": (4, 6),   # April–June
        "harvest_info": "When slightly soft and color changes",
        "description": "Prefers well-drained loamy or sandy soils with pH 5.5-7.5 and moderate moisture. Flowers Jan–Mar, fruits Apr–Jun."
    },
    "Lakshmanaphala (Annona muricata, Soursop)": {
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/4/43/Soursop%2C_Annona_muricata.jpg",
        "soil_types": ["Sandy/Light Soil", "Loamy Soil"],
        "ph_range": (5.0, 6.5),
        "moisture_range": (50, 90),
        "flowering_months": (1, 6),  # January–June (all year possible)
        "fruiting_months": (6, 9),   # June–September
        "harvest_info": "When greenish-yellow and yields to pressure",
        "description": "Thrives in well-drained sandy or fertile soils with pH 5-6.5 and high humidity. Flowers Jan–Jun (all year possible), fruits Jun–Sep."
    },
    "Wood Apple (Limonia acidissima)": {
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/3/3b/Wood_Apple_-_Limonia_acidissima.jpg",
        "soil_types": ["Sandy/Light Soil", "Loamy Soil", "Clay/Black Soil"],
        "ph_range": (5.5, 6.5),
        "moisture_range": (30, 70),
        "flowering_months": (2, 4),  # February–April
        "fruiting_months": (5, 7),   # May–July
        "harvest_info": "When pulp inside becomes sweet",
        "description": "Tolerant to light, loamy, and clay soils, drought-resistant, pH 5.5-6.5. Flowers Feb–Apr, fruits May–Jul."
    }
}

# ---------------------------
# Disease Data for Plants
# ---------------------------
diseases = {
    "Ramphal (Annona reticulata)": {
        "Anthracnose": {
            "symptoms": "Black spots on leaves and fruits",
            "cure": "Spray with Bavistin or Dithane M-45 fungicides. Remove infected parts and ensure good drainage."
        },
        "Leaf Spot": {
            "symptoms": "Greyish-black spots on leaves",
            "cure": "Use copper-based fungicides. Improve air circulation."
        },
        "Black Canker": {
            "symptoms": "Cankers on branches",
            "cure": "Prune affected branches and apply fungicides like Carbendazim."
        },
        "Root Rot": {
            "symptoms": "Wilting and yellowing leaves",
            "cure": "Improve soil drainage, use fungicides like Metalaxyl."
        }
    },
    "Lakshmanaphala (Annona muricata, Soursop)": {
        "Anthracnose": {
            "symptoms": "Dark spots on fruits and leaves",
            "cure": "Apply fungicides such as Mancozeb. Remove and destroy infected parts."
        },
        "Phytophthora Fruit Rot": {
            "symptoms": "Soft rot on fruits",
            "cure": "Use Phosphorous acid or Metalaxyl fungicides. Avoid overwatering."
        },
        "Powdery Mildew": {
            "symptoms": "White powdery coating on leaves",
            "cure": "Spray with sulfur-based fungicides. Increase air flow."
        },
        "Diplodia Fruit Rot": {
            "symptoms": "Brown rot on fruits",
            "cure": "Prune infected areas, apply Copper oxychloride."
        }
    },
    "Wood Apple (Limonia acidissima)": {
        "Anthracnose": {
            "symptoms": "Spots on leaves and fruits",
            "cure": "Fungicides like Carbendazim. Maintain tree health."
        },
        "Leaf Spot": {
            "symptoms": "Circular spots on leaves",
            "cure": "Copper fungicides. Remove debris."
        },
        "Root Rot": {
            "symptoms": "Wilting due to root decay",
            "cure": "Improve drainage, use Trichoderma-based biofungicides."
        },
        "Fungal Rot": {
            "symptoms": "Rotting fruits",
            "cure": "Sanitation, fungicides like Mancozeb."
        }
    }
}

# ---------------------------
# Soil Image Analysis Function
# ---------------------------
@st.cache_data
def analyze_soil(image_bytes):
    try:
        # Convert uploaded file into OpenCV format
        file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if img is None:
            st.error("Invalid image file. Please upload a valid JPG, PNG, or JPEG file.")
            return None, None, None, None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocess
        resized = cv2.resize(img, (256, 256))
        blurred = cv2.GaussianBlur(resized, (5,5), 0)

        # Convert to HSV for more accurate pH estimation
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        mean_S = np.mean(hsv[:,:,1])  # Mean Saturation (0-255)

        # Extract features
        mean_color = cv2.mean(blurred)[:3]  # RGB mean
        hist_r = cv2.calcHist([blurred], [0], None, [256], [0,256])
        hist_g = cv2.calcHist([blurred], [1], None, [256], [0,256])
        hist_b = cv2.calcHist([blurred], [2], None, [256], [0,256])

        # Visualization
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].imshow(img)
        ax[0].set_title("Original Soil Image")
        ax[0].axis("off")

        ax[1].imshow([[mean_color]])
        ax[1].set_title(f"Mean Color: {np.round(mean_color, 2)}")
        ax[1].axis("off")

        ax[2].plot(hist_r, color='red', label='Red')
        ax[2].plot(hist_g, color='green', label='Green')
        ax[2].plot(hist_b, color='blue', label='Blue')
        ax[2].set_title("RGB Histogram")
        ax[2].set_xlabel("Pixel Intensity")
        ax[2].set_ylabel("Frequency")
        ax[2].legend()
        ax[2].grid(True, alpha=0.3)

        st.pyplot(fig)

        # Dummy soil type detection
        avg_intensity = np.mean(mean_color)
        if avg_intensity > 150:
            soil_type = "Sandy/Light Soil"
        elif avg_intensity > 100:
            soil_type = "Loamy Soil"
        else:
            soil_type = "Clay/Black Soil"

        # Improved pH estimation using saturation from HSV (mapped to 5.0-7.0)
        ph_value = round(5.0 + (mean_S / 255) * 2.0, 2)  # Maps 0-255 to 5.0-7.0
        ph_value = np.clip(ph_value, 5.0, 7.0)  # Ensure within 5.0-7.0

        # Dummy soil moisture (based on blue channel intensity)
        moisture = round((mean_color[2] / 255) * 100, 2)  # Blue channel as proxy (0-100%)
        
        # Dummy temperature (based on red channel intensity)
        temperature = round(15 + (mean_color[0] / 255) * 20, 2)  # Scale to 15-35°C
        
        # Dummy NPK percentages (based on RGB channels)
        nitrogen = round((mean_color[1] / 255) * 100, 2)   # Green for nitrogen (0-100%)
        phosphorus = round((mean_color[2] / 255) * 50, 2)  # Blue for phosphorus (0-50%)
        potassium = round((mean_color[0] / 255) * 50, 2)   # Red for potassium (0-50%)

        return soil_type, ph_value, moisture, temperature, (nitrogen, phosphorus, potassium)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None, None, None

# ---------------------------
# Leaf Disease Analysis Function
# ---------------------------
@st.cache_data
def analyze_leaf(leaf_bytes, plant_name):
    try:
        # Convert uploaded file into OpenCV format
        file_bytes = np.asarray(bytearray(leaf_bytes.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if img is None:
            st.error("Invalid leaf image file. Please upload a valid JPG, PNG, or JPEG file.")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocess
        resized = cv2.resize(img, (256, 256))
        blurred = cv2.GaussianBlur(resized, (5,5), 0)

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

        # Simple heuristic for disease detection based on color thresholds
        # Detect brown/black spots (anthracnose, spots) - low saturation, low value
        # Detect white powdery (powdery mildew) - high value, low saturation
        # Detect yellowing (root rot) - high hue in yellow range

        # Mask for brown spots (anthracnose/leaf spot)
        lower_brown = np.array([10, 50, 20])
        upper_brown = np.array([30, 255, 100])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        brown_ratio = np.sum(mask_brown) / (256*256*255)

        # Mask for white powdery (powdery mildew)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        white_ratio = np.sum(mask_white) / (256*256*255)

        # Mask for yellowing (root rot)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_ratio = np.sum(mask_yellow) / (256*256*255)

        # Determine disease based on highest ratio
        if brown_ratio > 0.1:
            disease = "Anthracnose" if "Anthracnose" in diseases[plant_name] else "Leaf Spot"
        elif white_ratio > 0.1:
            disease = "Powdery Mildew"
        elif yellow_ratio > 0.1:
            disease = "Root Rot"
        else:
            disease = "No Disease Detected"

        # Visualization
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(img)
        ax.set_title("Original Leaf Image")
        ax.axis("off")
        st.pyplot(fig)

        return disease

    except Exception as e:
        st.error(f"Error processing leaf image: {str(e)}")
        return None

# ---------------------------
# Plant Recommendation Function
# ---------------------------
def recommend_plants(soil_type, ph_value, moisture):
    current_month = datetime.now().month  # Current month (e.g., 9 for September)
    recommendations = []
    scores = []
    
    for plant_name, plant_data in plants.items():
        # Soil type score
        soil_score = 1.0 if soil_type in plant_data["soil_types"] else 0.0
        
        # pH score: proximity to midpoint of plant's pH range
        ph_midpoint = (plant_data["ph_range"][0] + plant_data["ph_range"][1]) / 2
        ph_max_deviation = max((plant_data["ph_range"][1] - plant_data["ph_range"][0]) / 2, 1.0)
        ph_score = 1.0 - abs(ph_value - ph_midpoint) / ph_max_deviation
        ph_score = max(0.0, ph_score)  # Ensure non-negative
        
        # Moisture score: proximity to midpoint of plant's moisture range
        moisture_midpoint = (plant_data["moisture_range"][0] + plant_data["moisture_range"][1]) / 2
        moisture_max_deviation = max((plant_data["moisture_range"][1] - plant_data["moisture_range"][0]) / 2, 10.0)
        moisture_score = 1.0 - abs(moisture - moisture_midpoint) / moisture_max_deviation
        moisture_score = max(0.0, moisture_score)  # Ensure non-negative
        
        # Season score: proximity to flowering period
        flowering_start = plant_data["flowering_months"][0]
        months_until_flowering = (flowering_start - current_month) % 12
        season_score = 1.0 - (months_until_flowering / 12.0)
        if plant_name == "Lakshmanaphala (Annona muricata, Soursop)" and months_until_flowering > 6:
            season_score = max(season_score, 0.5)  # Baseline for all-year flowering
        
        # Total score
        total_score = soil_score + ph_score + moisture_score + season_score
        suitable = (
            soil_score > 0 and
            plant_data["ph_range"][0] <= ph_value <= plant_data["ph_range"][1] and
            plant_data["moisture_range"][0] <= moisture <= plant_data["moisture_range"][1]
        )
        
        if suitable:
            recommendations.append((plant_name, plant_data["description"], total_score))
        scores.append((plant_name, total_score))
    
    # Sort recommendations by score (descending)
    recommendations.sort(key=lambda x: x[2], reverse=True)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # If no suitable plants, return default message
    if not recommendations:
        return [("None highly suitable. Consider soil amendments.", 
                 "Soil conditions or current season may not match the preferred requirements for Ramphal, Lakshmanaphala, or Wood Apple.", 0.0)], scores
    
    return recommendations, scores

# ---------------------------
# Find Nearest District
# ---------------------------
def find_nearest_district(lat, lon):
    min_distance = float('inf')
    nearest_district = None
    for district, data in karnataka_soils.items():
        district_coords = data["coords"]
        distance = geodesic((lat, lon), district_coords).kilometers
        if distance < min_distance:
            min_distance = distance
            nearest_district = district
    return nearest_district, karnataka_soils.get(nearest_district, {}).get("soil_type", "Unknown")

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("🌱 Smart Soil Testing with Plant Recommendations and Leaf Disease Detection")
st.write("Upload a soil image to analyze its properties, compare with Karnataka districts, and get the best plant recommendation for Ramphal, Lakshmanaphala, or Wood Apple based on soil and season (September 2025). Note: This is a preliminary analysis based on image color; consult a professional for accurate soil testing.")

# Display Karnataka soil types table
st.subheader("🌍 Typical Soil Types in Karnataka")
soil_data = [[district, data["soil_type"], f"{data['coords'][0]:.4f}, {data['coords'][1]:.4f}"] for district, data in karnataka_soils.items()]
st.table({"District": [row[0] for row in soil_data], "Typical Soil Type": [row[1] for row in soil_data], "Coordinates (Lat, Lon)": [row[2] for row in soil_data]})

# Display plant info
st.subheader("🌿 Plants for Recommendation")
for plant_name, plant_data in plants.items():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(plant_data["image_url"], caption=plant_name, width=100)
    with col2:
        st.write(f"**{plant_name}**")
        st.write(f"{plant_data['description']}")
        st.write(f"- **Flowering**: {plant_data['flowering_months'][0]}–{plant_data['flowering_months'][1]}")
        st.write(f"- **Fruiting**: {plant_data['fruiting_months'][0]}–{plant_data['fruiting_months'][1]}")
        st.write(f"- **Best Harvest**: {plant_data['harvest_info']}")

uploaded_soil_file = st.file_uploader("Upload a soil image", type=["jpg", "png", "jpeg"])

st.subheader("🗺 Select Location on Map")

# Default map centered on Karnataka
m = folium.Map(location=[15.3173, 75.7139], zoom_start=7)
marker_cluster = MarkerCluster().add_to(m)

# Add markers for Karnataka districts
for district, data in karnataka_soils.items():
    folium.Marker(
        location=data["coords"],
        popup=f"{district}: {data['soil_type']}",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(marker_cluster)

# Add instruction marker
folium.Marker(
    [15.3173, 75.7139],
    popup="Click anywhere on the map to select location",
    icon=folium.Icon(color="green", icon="info-sign")
).add_to(m)

# Capture map clicks
map_data = st_folium(m, width=700, height=500)

if uploaded_soil_file is not None and map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    # Analyze soil
    with st.spinner("Analyzing soil image..."):
        soil_type, ph_value, moisture, temperature, npk = analyze_soil(uploaded_soil_file)

    if soil_type is not None:
        nitrogen, phosphorus, potassium = npk
        # Find nearest district
        nearest_district, district_soil_type = find_nearest_district(lat, lon)
        
        # Display analysis results
        st.success(
            f"**Detected Soil Properties**\n\n"
            f"- Soil Type: **{soil_type}**\n"
            f"- pH: **{ph_value}**\n"
            f"- Moisture: **{moisture}%**\n"
            f"- Temperature: **{temperature}°C**\n"
            f"- Nitrogen: **{nitrogen}%**\n"
            f"- Phosphorus: **{phosphorus}%**\n"
            f"- Potassium: **{potassium}%**\n"
            f"- Nearest District: **{nearest_district}** (Typical Soil: **{district_soil_type}**)"
        )
        st.write(f"📍 Location chosen: Latitude = {lat:.4f}, Longitude = {lon:.4f}")
        if soil_type != district_soil_type:
            st.info(f"Note: The detected soil type ({soil_type}) differs from the typical soil type in {nearest_district} ({district_soil_type}). This could be due to local variations or the preliminary nature of the analysis.")

        # Plant Recommendations
        st.subheader("🌱 Plant Recommendation")
        recommendations, scores = recommend_plants(soil_type, ph_value, moisture)
        top_plant = recommendations[0][0]
        top_description = recommendations[0][1]
        top_score = recommendations[0][2]
        
        if "None" in top_plant:
            st.warning(f"**No Recommended Plant**: {top_description}")
        else:
            st.success(f"**Top Recommended Plant: {top_plant}** (Score: {top_score:.2f}/4.0)\n\n{top_description}")
            st.image(plants[top_plant]["image_url"], width=200)
            if len(recommendations) > 1:
                st.write("**Other Suitable Plants**:")
                for plant_name, desc, score in recommendations[1:]:
                    st.info(f"✅ **{plant_name}** (Score: {score:.2f}/4.0) - {desc}")
                    st.image(plants[plant_name]["image_url"], width=150)

        # Show updated map with markers
        result_map = folium.Map(location=[lat, lon], zoom_start=7)
        marker_cluster = MarkerCluster().add_to(result_map)
        
        # Add district markers
        for district, data in karnataka_soils.items():
            folium.Marker(
                location=data["coords"],
                popup=f"{district}: {data['soil_type']}",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(marker_cluster)
        
        # Add user location marker with recommendations
        popup_text = f"""
        <b>Soil Analysis Result</b><br>
        Soil: {soil_type}<br>
        pH: {ph_value}<br>
        Moisture: {moisture}%<br>
        Temperature: {temperature}°C<br>
        Nitrogen: {nitrogen}%<br>
        Phosphorus: {phosphorus}%<br>
        Potassium: {potassium}%<br>
        Nearest District: {nearest_district} ({district_soil_type})<br>
        Lat: {lat:.4f}, Lon: {lon:.4f}<br><br>
        <b>Top Recommended Plant:</b><br>
        """
        if "None" not in top_plant:
            popup_text += f'<img src="{plants[top_plant]["image_url"]}" width="100" height="100"><br>{top_plant} (Score: {top_score:.2f}/4.0)<br>'
            popup_text += f'Flowering: {plants[top_plant]["flowering_months"][0]}–{plants[top_plant]["flowering_months"][1]}<br>'
            popup_text += f'Fruiting: {plants[top_plant]["fruiting_months"][0]}–{plants[top_plant]["fruiting_months"][1]}<br>'
            popup_text += f'Harvest: {plants[top_plant]["harvest_info"]}<br>'
            if len(recommendations) > 1:
                popup_text += "<b>Other Suitable Plants:</b><br>"
                for plant_name, _, score in recommendations[1:]:
                    popup_text += f'<img src="{plants[plant_name]["image_url"]}" width="80" height="80"><br>{plant_name} (Score: {score:.2f}/4.0)<br>'
        else:
            popup_text += f"{top_plant}<br>"
        folium.Marker(
            [lat, lon],
            popup=popup_text,
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(marker_cluster)
        
        # Add plant marker (only for top recommended plant)
        if "None" not in top_plant:
            plant_position = (lat + 0.01, lon)  # Slightly offset from user location
            plant_popup = f"""
            <b>Top Recommended: {top_plant}</b><br>
            <img src="{plants[top_plant]["image_url"]}" width="150" height="150"><br>
            Flowering: {plants[top_plant]["flowering_months"][0]}–{plants[top_plant]["flowering_months"][1]}<br>
            Fruiting: {plants[top_plant]["fruiting_months"][0]}–{plants[top_plant]["fruiting_months"][1]}<br>
            Harvest: {plants[top_plant]["harvest_info"]}
            """
            folium.Marker(
                plant_position,
                popup=plant_popup,
                icon=folium.Icon(color="green", icon="leaf")
            ).add_to(marker_cluster)

        st_folium(result_map, width=700, height=500)
else:
    if uploaded_soil_file is None:
        st.warning("Please upload a soil image.")
    if not map_data or not map_data.get("last_clicked"):
        st.warning("Please click on the map to select a location.")

# ---------------------------
# Leaf Disease Detection Section
# ---------------------------
st.subheader("🍃 Leaf Disease Detection")
st.write("Upload a leaf image from one of the three plants to detect possible diseases and get cure recommendations. Select the plant type first.")

plant_choice = st.selectbox("Select Plant", list(plants.keys()))

uploaded_leaf_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"], key="leaf_uploader")

if uploaded_leaf_file is not None:
    with st.spinner("Analyzing leaf image..."):
        disease = analyze_leaf(uploaded_leaf_file, plant_choice)

    if disease is not None:
        if disease == "No Disease Detected":
            st.success("No disease detected in the leaf image.")
        else:
            st.warning(f"Detected Disease: **{disease}**")
            if disease in diseases[plant_choice]:
                symptoms = diseases[plant_choice][disease]["symptoms"]
                cure = diseases[plant_choice][disease]["cure"]
                st.write(f"**Symptoms**: {symptoms}")
                st.write(f"**Cure**: {cure}")
            else:
                st.info("No specific cure information available for this disease.")