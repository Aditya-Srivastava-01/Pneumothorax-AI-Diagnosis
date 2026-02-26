import streamlit as st
import pydicom
import numpy as np
import cv2
import os
from engine import DiagnosisEngine

st.set_page_config(page_title="Pneumothorax AI", page_icon="🫁", layout="wide")
st.title("🫁 Pneumothorax Diagnostic Assistant")

@st.cache_resource
def get_engine():
    return DiagnosisEngine("best_efficientnet_model.pth")

if os.path.exists("best_efficientnet_model.pth"):
    engine = get_engine()
    uploaded_file = st.file_uploader("Upload Chest X-ray (DICOM)", type=["dcm"])
    
    if uploaded_file:
        try:
            ds = pydicom.dcmread(uploaded_file)
            img = ds.pixel_array.astype(float)
            
            # --- FIX 1: REMOVE EXTRA DIMENSIONS (SQUEEZE) ---
            # This turns (1, 1024, 1024) into (1024, 1024)
            img = np.squeeze(img)
            
            # If it's a 3D volume, just take the first slice
            if len(img.shape) > 2:
                img = img[0]

            # 1. Handle Monochrome Inversion
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                img = np.max(img) - img
                
            # 2. Normalize to 0-255 safely
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
            img = (img * 255).astype(np.uint8)
            
            # --- FIX 2: APPLY CLAHE ON CLEAN 2D ARRAY ---
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
            # 3. BULLETPROOF RGB CONVERSION
            img_rgb = np.stack([img]*3, axis=-1)

            # 4. Run AI Inference
            with st.spinner('AI is analyzing imagery...'):
                diag, prob, heat = engine.predict(img_rgb)
            
            # 5. UI Display
            col1, col2 = st.columns(2)
            with col1: 
                st.subheader("Original Radiograph")
                st.image(img_rgb, use_container_width=True)
            with col2: 
                st.subheader("AI Localization (Grad-CAM)")
                st.image(heat, use_container_width=True)
            
            if "DETECTED" in diag:
                st.error(f"**{diag}** (Confidence: {prob:.2%})")
            else:
                st.success(f"**{diag}** (Confidence: {1-prob:.2%})")
                
        except Exception as e:
            st.error(f"Error processing DICOM: {e}")
else:
    st.error("Model weights not found. Please upload best_efficientnet_model.pth to the repo.")