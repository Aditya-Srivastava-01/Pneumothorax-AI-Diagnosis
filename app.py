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
        ds = pydicom.dcmread(uploaded_file)
        img = ds.pixel_array
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = np.max(img) - img
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6) * 255
        img_rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        diag, prob, heat = engine.predict(img_rgb)
        
        col1, col2 = st.columns(2)
        with col1: st.image(img_rgb, caption="Original X-ray", use_container_width=True)
        with col2: st.image(heat, caption="AI Localization (Grad-CAM)", use_container_width=True)
        
        if "DETECTED" in diag:
            st.error(f"**{diag}** (Confidence: {prob:.2%})")
        else:
            st.success(f"**{diag}** (Confidence: {1-prob:.2%})")
else:
    st.error("Model weights not found. Please upload best_efficientnet_model.pth to the repo.")