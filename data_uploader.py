# 1_data_uploader.py (Simplified: Disease Only, No Suggestions/Symptoms)
import streamlit as st
import os
from PIL import Image
import json
import re

DATASET_BASE_PATH = "my_plant_dataset_disease_only"

if not os.path.exists(DATASET_BASE_PATH):
    os.makedirs(DATASET_BASE_PATH)

if 'messages' in st.session_state and st.session_state.messages:
    for msg_type, msg_text in st.session_state.messages:
        if msg_type == "success": st.success(msg_text, icon="✅")
        elif msg_type == "info": st.info(msg_text, icon="ℹ️")
        elif msg_type == "warning": st.warning(msg_text, icon="⚠️")
        elif msg_type == "error": st.error(msg_text, icon="❌")
    st.session_state.messages = []

st.title("📝 Simple Dataset Uploader")
st.markdown("Add images. Images will be saved into folders named after the **Disease Name**.")
#st.info("Remember: Re-run `train_model.py` (update DATASET_PATH if needed) after adding enough images!")
st.markdown("---")

st.subheader("1. Define Disease Label")
disease_name_input = st.text_input("Enter Disease Name (e.g., Leaf_Spot, Healthy) - THIS DEFINES THE FOLDER", key="disease_input_widget")

st.markdown("---")

st.subheader("2. Upload Images")
uploaded_files_list = st.file_uploader(
    "Choose images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="file_uploader_widget"
)

st.markdown("---")

if st.button("Save Images", key="save_button"):

    current_disease = disease_name_input
    current_files = uploaded_files_list

    current_run_messages = []

    current_disease_class_name = ""
    current_save_path = ""
    if current_disease:
        safe_disease = re.sub(r'[ ,/]+', '_', current_disease.strip())
        safe_disease = re.sub(r'[^a-zA-Z0-9_]', '', safe_disease)
        if safe_disease:
             current_disease_class_name = safe_disease
             current_save_path = os.path.join(DATASET_BASE_PATH, current_disease_class_name)

    if not current_disease:
        current_run_messages.append(("error", "Please fill in the Disease Name."))
    elif not current_files:
        current_run_messages.append(("error", "Please upload at least one image."))
    elif not current_disease_class_name:
         current_run_messages.append(("error", "Invalid Disease name. Use letters, numbers, underscores."))
    else:
        current_run_messages.append(("info", f"Processing save for disease class: '{current_disease_class_name}'..."))
        os.makedirs(current_save_path, exist_ok=True)

        saved_count = 0
        skipped_count = 0
        error_count = 0

        for uploaded_file in current_files:
            try:
                base, ext = os.path.splitext(uploaded_file.name)
                safe_base = re.sub(r'[^a-zA-Z0-9_.-]', '_', base)
                safe_filename = safe_base + ext if safe_base else f"image_{hash(uploaded_file.getvalue())}{ext}"
                file_path = os.path.join(current_save_path, safe_filename)

                if not os.path.exists(file_path):
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                current_run_messages.append(("error", f"Error saving file {uploaded_file.name}: {e}"))
                error_count += 1

        if saved_count > 0:
            current_run_messages.append(("success", f"{saved_count} new image(s) saved to: `{current_save_path}`"))
        if skipped_count > 0:
            current_run_messages.append(("warning", f"{skipped_count} image(s) skipped (already existed)."))

    st.session_state.messages = current_run_messages
    st.rerun()