# disease_detection_app.py (Simplified: No Suggestions)        .\venv\Scripts\python.exe -m streamlit run disease_detection.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import json
import re

MODEL_PATH_KERAS = 'leaf_disease_model.keras'
MODEL_PATH_H5 = 'leaf_disease_model.h5'
CLASSES_PATH = 'class_names.txt'
IMG_SIZE = 224

HEALTHY_RAMPHALA_LEAF_IMG = "healthy_ramphala_leaf.jpg"
HEALTHY_LAKSHMANPHALA_LEAF_IMG = "healthy_lakshmanphala_leaf.jpg"
HEALTHY_WOOD_APPLE_LEAF_IMG = "healthy_wood_apple_leaf.jpg"

VULNERABILITY_DATA = {
    "Anthracnose ,Alternaria_Leaf_blight ,Cercospora_Leaf_Spot": ["Ramphala", "Lakshmanphala", "Wood Apple"],
    "Phoma_Leaf_Spot": ["Ramphala"],
    #"Alternaria_Leaf_Blight": ["Ramphala", "Lakshmanphala", "Wood Apple"],
    "Alternaria_Leaf_Spot": ["Wood Apple"],
    "Powdery_Mildew": ["Ramphala", "Lakshmanphala", "Wood Apple"],
    #"Cercospora_Leaf_Spot": ["Lakshmanphala", "Wood Apple"],
    #"Cercospora_Leaf_Blight": ["Wood Apple"],
    #"Sooty_Mold": ["Wood Apple"],
    "Leaf Crinkle": ["Wood Apple"],
    "Healthy": ["Ramphala", "Lakshmanphala", "Wood Apple"],
    "Yellow_Mosaic": ["Wood Apple"],
    
}

@st.cache_resource
def load_trained_model_internal():
    model_path_to_load = None
    loaded_format = None
    if os.path.exists(MODEL_PATH_KERAS):
        model_path_to_load = MODEL_PATH_KERAS
        loaded_format = ".keras"
    elif os.path.exists(MODEL_PATH_H5):
        model_path_to_load = MODEL_PATH_H5
        loaded_format = ".h5"

    if model_path_to_load:
        try:
            model = load_model(model_path_to_load)
            return model, model_path_to_load, loaded_format, None
        except Exception as e:
            return None, model_path_to_load, None, str(e)
    else:
        return None, None, None, "Model file (.keras or .h5) not found."

model, loaded_path, loaded_format, load_error = load_trained_model_internal()

if model is not None:
    st.toast(f"Model loaded successfully from {os.path.basename(loaded_path)}")
    if loaded_format == ".h5":
        st.warning("Model loaded from legacy .h5 format.", icon="⚠️")
elif load_error:
    if loaded_path:
         st.error(f"Error loading model from {os.path.basename(loaded_path)}: {load_error}", icon="❌")
    else:
         st.error(f"Model load error: {load_error}", icon="❌")


@st.cache_data
def load_class_names():
    if not os.path.exists(CLASSES_PATH):
        st.error(f"Class names file '{CLASSES_PATH}' not found.", icon="❌")
        return None
    try:
        with open(CLASSES_PATH, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except Exception as e:
        st.error(f"Error loading class names: {e}", icon="❌")
        return None

@st.cache_data
def load_comparison_image(file_name):
    if os.path.exists(file_name):
        try:
            return Image.open(file_name)
        except Exception as e:
            st.warning(f"Could not load comparison image {file_name}: {e}")
            return None
    return None

def preprocess_image(image):
    try:
        image = image.convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}", icon="❌")
        return None

healthy_ram_leaf = load_comparison_image(HEALTHY_RAMPHALA_LEAF_IMG)
healthy_lak_leaf = load_comparison_image(HEALTHY_LAKSHMANPHALA_LEAF_IMG)
healthy_wood_leaf = load_comparison_image(HEALTHY_WOOD_APPLE_LEAF_IMG)

class_names = load_class_names()

st.title("🌿 Simple Leaf Disease Detection")

if model is None or class_names is None:
    st.error("Cannot proceed: Model or class names failed to load. Check errors above and run `2_train_model.py` if needed.")
    st.stop()

uploaded_file = st.file_uploader("Upload leaf image...", type=["jpg", "jpeg", "png"], key="detector_uploader")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Leaf Image.', use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying image: {e}", icon="❌")
        st.stop()

    st.write("")

    diagnose_key = f"diagnose_btn_{uploaded_file.id if hasattr(uploaded_file, 'id') else uploaded_file.name}"

    if st.button("Diagnose", key=diagnose_key):
        with st.spinner("Analyzing..."):
            processed_image = preprocess_image(image)

            if processed_image is None:
                 st.error("Image processing failed.", icon="❌")
                 st.stop()

            try:
                prediction = model.predict(processed_image)
                score = tf.nn.softmax(prediction[0])
                predicted_class_index = np.argmax(score)
                predicted_class_name = class_names[predicted_class_index] 
                display_name = predicted_class_name.replace('_', ' ')

                st.header("Diagnosis Result")
                if "healthy" in display_name.lower():
                     st.success(f"Diagnosis: **{display_name}**", icon="✅")
                else:
                     st.error(f"Diagnosis: **{display_name}**", icon="❌")
                
                
                st.subheader("Healthy Leaf Comparison")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if healthy_ram_leaf:
                        st.image(healthy_ram_leaf, caption="Healthy Ramphala Leaf", use_container_width=True)
                    else:
                        st.info(f"Missing '{HEALTHY_RAMPHALA_LEAF_IMG}'")
                
                with col2:
                    if healthy_lak_leaf:
                        st.image(healthy_lak_leaf, caption="Healthy Lakshmanphala Leaf", use_container_width=True)
                    else:
                        st.info(f"Missing '{HEALTHY_LAKSHMANPHALA_LEAF_IMG}'")
                
                with col3:
                    if healthy_wood_leaf:
                        st.image(healthy_wood_leaf, caption="Healthy Wood Apple Leaf", use_container_width=True)
                    else:
                        st.info(f"Missing '{HEALTHY_WOOD_APPLE_LEAF_IMG}'")
                
                st.markdown("---")
                
                st.subheader("Vulnerable Fruits:")
                if predicted_class_name in VULNERABILITY_DATA:
                    fruit_list = VULNERABILITY_DATA[predicted_class_name]
                    if fruit_list and fruit_list[0].lower() != 'n/a':
                        st.write(", ".join(fruit_list))
                    else:
                        st.info("This class does not have specific fruit vulnerabilities listed (e.g., 'Healthy').")
                else:
                    st.warning("No vulnerability data found for this disease in the code.")


            except Exception as e:
                 st.error(f"Prediction Error: {e}", icon="❌")
                 st.exception(e)
# --- VULNERABILITY DATA (Extracted from your images) ---




