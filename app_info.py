# app.py (Modified to fix deprecation warning)
import streamlit as st
from PIL import Image # Import the Python Imaging Library
import os

st.title("📚 Fruit Information")
st.markdown("Detailed information on the featured indigenous fruits.")

# --- Load Local Images ---
# --- (Ensure these files are in the same folder as this script!) ---

# Define file names exactly as you provided
RAMPHALA_IMG_FILE = "Ram phala.jpg"
LAKSHMANPHALA_IMG_FILE = "Lakshman phala.jpg"
WOOD_APPLE_IMG_FILE = "wood apple.jpg"

# Function to load an image, handling errors
def load_image(file_name):
    if os.path.exists(file_name):
        try:
            return Image.open(file_name)
        except Exception as e:
            st.warning(f"Could not load image {file_name}: {e}")
            return None
    return None

ram_img = load_image(RAMPHALA_IMG_FILE)
lak_img = load_image(LAKSHMANPHALA_IMG_FILE)
wood_img = load_image(WOOD_APPLE_IMG_FILE)

st.markdown("---")

# Ramphala (Custard Apple/Bull's Heart)
st.header("Ramphala (Bull's Heart / *Annona reticulata*)")
col1_ram, col2_ram = st.columns([2, 2])
with col1_ram:
    if ram_img:
        # Changed use_column_width to use_container_width
        st.image(ram_img, caption="Ramphala", use_container_width=True)
    else:
        st.info(f"Image not found. Make sure '{RAMPHALA_IMG_FILE}' is in the folder.")
with col2_ram:
    st.subheader("Cultivation Factors")
    st.markdown("""
    * **Climate:** Thrives in warm, humid, tropical, and subtropical climates.
    * **Temperature:** Prefers temperatures between 20°C to 35°C. It is sensitive to frost, especially when young.
    * **Sunlight:** Requires bright, direct sunlight (at least 6 hours).
    * **Soil:** Flourishes in rich, well-draining loamy soil. It can tolerate various soil types, including shallow or sandy soil, but good drainage is crucial to prevent root rot.
    * **Water:** Needs regular watering to keep the soil moist, but dislikes waterlogging.
    * **Propagation:** Can be propagated by seeds or grafting. Grafted plants bear fruit faster.
    """)
    
    st.subheader("Nutritional Value (per 100g)")
    st.markdown("""
    * **Energy:** ~101 Kcal
    * **Vitamin C:** ~19.2 mg (approx. 21-23% DV)
    * **Vitamin B6:** ~0.221 mg (approx. 17% DV)
    * **Potassium:** ~382 mg (approx. 13% DV)
    * **Dietary Fiber:** ~2.4 g
    * **Protein:** ~1.7 g
    * **Also Contains:** Magnesium, Calcium, and antioxidants (polyphenols, flavonoids).
    """)
    
    st.subheader("Medicinal & Economic Uses")
    st.markdown("""
    * **Culinary:** Eaten fresh, or used in juices, smoothies, desserts, and ice cream.
    * **Immunity:** High Vitamin C content helps boost the immune system.
    * **Digestive Health:** Fiber aids in digestion and helps regulate bowel movements.
    * **Heart Health:** Potassium helps regulate blood pressure, and fiber can help lower cholesterol.
    * **Skin Health:** Vitamin C and antioxidants help protect skin and boost collagen production.
    * **Traditional Medicine:** Unripe fruit used for diarrhea/dysentery; leaves and bark used for various ailments like ulcers, boils, and fever.
    """)

st.markdown("---")

# Lakshmanphala (Soursop/Guanabana)
st.header("Lakshmanphala (Soursop / *Annona muricata*)")
col1_lak, col2_lak = st.columns([2, 2])
with col1_lak:
    if lak_img:
        # Changed use_column_width to use_container_width
        st.image(lak_img, caption="Lakshmanphala", use_container_width=True)
    else:
        st.info(f"Image not found. Make sure '{LAKSHMANPHALA_IMG_FILE}' is in the folder.")
with col2_lak:
    st.subheader("Cultivation Factors")
    st.markdown("""
    * **Climate:** Requires a wet, tropical or subtropical climate with high humidity.
    * **Temperature:** Prefers warm temperatures (25°C - 30°C). It is highly sensitive to frost and cannot withstand temperatures below 5°C.
    * **Sunlight:** Needs full, direct sun exposure (at least 8 hours).
    * **Soil:** Prefers deep, well-drained, sandy loam soil rich in organic matter.
    * **Soil pH:** Ideal pH is slightly acidic, between 5.5 to 6.5.
    * **Water:** Needs consistently moist soil; avoid waterlogging, which causes root rot.
    * **Propagation:** Grown from seeds or grafting. Grafted trees bear fruit earlier (3-5 years).
    """)
    
    st.subheader("Nutritional Value (per 100g)")
    st.markdown("""
    * **Energy:** ~66 Kcal
    * **Vitamin C:** ~21 mg (approx. 23% DV). (Excellent source overall).
    * **Potassium:** ~278 mg (approx. 8% DV).
    * **Dietary Fiber:** ~3.3 g (approx. 13% DV). (Very high fiber content).
    * **Protein:** ~1.0 g
    * **Also Contains:** Good source of B vitamins (B1, B2, B3, B6, Folate), Magnesium, and Copper.
    """)
    
    st.subheader("Medicinal & Economic Uses")
    st.markdown("""
    * **Culinary:** Very popular for juices, smoothies, sorbets, and desserts. Also eaten fresh.
    * **Economic:** High market demand both locally and internationally, increasing profitability.
    * **Immunity:** Excellent source of Vitamin C, a known immune booster.
    * **Digestive Health:** Very high fiber content promotes regular bowel movements.
    * **Antioxidant/Anti-inflammatory:** Rich in antioxidants (phenolic compounds, acetogenins).
    * **Traditional Medicine:** Leaves are famously brewed into a tea used to treat a variety of ailments, including inflammation, pain, and infections.
    """)

st.markdown("---")

# Wood Apple (Bael)
st.header("Wood Apple (Bael / *Limonia acidissima*)")
col1_wood, col2_wood = st.columns([2, 2])
with col1_wood:
    if wood_img:
        # Changed use_column_width to use_container_width
        st.image(wood_img, caption="Wood Apple", use_container_width=True)
    else:
        st.info(f"Image not found. Make sure '{WOOD_APPLE_IMG_FILE}' is in the folder.")
with col2_wood:
    st.subheader("Cultivation Factors")
    st.markdown("""
    * **Climate:** Extremely hardy; thrives in seasonally dry tropical or subtropical monsoon climates. Prefers a distinct dry season.
    * **Temperature:** Prefers mean annual temperatures from 20°C to 35°C.
    * **Sunlight:** Grows best in full sun.
    * **Soil:** Not particular. Tolerates a very wide range of soils, including poor, sandy, or light soils. Best in well-drained sandy or deep loam soils.
    * **Tolerance:** Highly drought-tolerant once established.
    * **Propagation:** Commonly grown from seeds, but budded plants bear fruit much earlier.
    """)
    
    st.subheader("Nutritional Value (per 100g)")
    st.markdown("""
    * **Energy:** ~134-137 Kcal
    * **Carbohydrates:** ~18-31.8 g
    * **Protein:** ~1.8-7.0 g
    * **Fiber:** ~2.9-5.0 g
    * **Vitamins:** Rich in B vitamins, especially Riboflavin (B2) and Thiamine (B1). Contains Vitamin C and Beta-carotene (Vitamin A precursor).
    * **Minerals:** Good source of Calcium (85-130 mg) and Potassium (600 mg).
    """)
    
    st.subheader("Medicinal & Economic Uses")
    st.markdown("""
    * **Culinary:** Pulp is famously processed into a refreshing juice (sherbet), jams (murabba), and chutneys.
    * **Digestive Health:** Highly valued in Ayurveda. The *unripe* fruit is used to treat diarrhea and dysentery. The *ripe* fruit pulp is an excellent laxative.
    * **Medicinal:** Considered a natural antioxidant and cardio-protective fruit. Used in traditional medicine for respiratory issues, to purify blood, and to manage blood sugar.
    * **Other Uses:** The hard shell is used for crafts, and gum from the trunk is used in inks and dyes.
    * **Cosmetic:** Antimicrobial properties help treat scalp infections and dandruff; antioxidants aid skin health.
    """)

st.markdown("---")
st.info("Information sourced from various agricultural and nutritional web resources.")
