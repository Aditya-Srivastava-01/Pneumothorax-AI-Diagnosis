import streamlit as st
import pydicom
import numpy as np
import cv2
import os
import gc
from engine import DiagnosisEngine

st.set_page_config(page_title="Pneumothorax AI", page_icon="🫁", layout="wide")

@st.cache_resource
def get_engine():
    return DiagnosisEngine("best_efficientnet_model.pth")

# Clear memory from previous runs
gc.collect()

if os.path.exists("best_efficientnet_model.pth"):
    engine = get_engine()
    st.title("🫁 Pneumothorax Diagnostic Assistant")
    uploaded_file = st.file_uploader("Upload DICOM", type=["dcm"])
    
    if uploaded_file:
        try:
            ds = pydicom.dcmread(uploaded_file)
            img = ds.pixel_array.astype(float)
            img = np.squeeze(img)
            if len(img.shape) > 2: img = img[0]

            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                img = np.max(img) - img
                
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
            img = (img * 255).astype(np.uint8)
            
            # Lower UI resolution to save browser and server RAM
            img = cv2.resize(img, (448, 448)) 
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            img_rgb = np.stack([img]*3, axis=-1)

            with st.spinner('AI analyzing...'):
                diag, prob, heat = engine.predict(img_rgb)
            
            col1, col2 = st.columns(2)
            with col1: st.image(img_rgb, caption="Original", use_container_width=True)
            with col2: st.image(heat, caption="AI Localization", use_container_width=True)
            
            if "DETECTED" in diag:
                st.error(f"**{diag}** ({prob:.2%})")
            else:
                st.success(f"**{diag}** ({1-prob:.2%})")
            
            # Cleanup for next run
            del img_rgb, img, ds
            gc.collect()
                
        except Exception as e:
            st.error(f"Error: {e}")