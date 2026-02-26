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
            
            # 1. Handle Monochrome Inversion
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                img = np.max(img) - img
                
            # 2. Normalize to 0-255 safely
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
            img = (img * 255).astype(np.uint8)
            
            # 3. Apply CLAHE (Contrast Enhancement)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
            # 4. BULLETPROOF RGB CONVERSION (Replacing cv2.cvtColor)
            # This stacks the grayscale image 3 times to create RGB. 
            # It never crashes, even if the DICOM shape is weird.
            if len(img.shape) == 2:
                img_rgb = np.stack([img]*3, axis=-1)
            else:
                # If it's already got channels, just take the first 3
                img_rgb = img[:, :, :3]

            # 5. Run AI Inference
            diag, prob, heat = engine.predict(img_rgb)
            
            # 6. UI Display
            col1, col2 = st.columns(2)
            with col1: 
                st.image(img_rgb, caption="Original X-ray", use_container_width=True)
            with col2: 
                st.image(heat, caption="AI Localization (Grad-CAM)", use_container_width=True)
            
            if "DETECTED" in diag:
                st.error(f"**{diag}** (Confidence: {prob:.2%})")
            else:
                st.success(f"**{diag}** (Confidence: {1-prob:.2%})")
                
        except Exception as e:
            st.error(f"Error processing DICOM: {e}")
else:
    st.error("Model weights not found. Please upload best_efficientnet_model.pth to the repo.")