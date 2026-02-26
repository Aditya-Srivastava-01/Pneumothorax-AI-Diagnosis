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
            img = ds.pixel_array
            
            # --- ROBUST DIMENSION FIX (Fixes the Streaks) ---
            # 1. Handle "Fake" DICOMs that might be RGB already or have extra dimensions
            if len(img.shape) == 3:
                # If it's (Rows, Cols, 3), convert to grayscale first
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    # If it's (Slices, Rows, Cols), take the middle slice
                    img = img[img.shape[0] // 2]
            elif len(img.shape) == 4:
                # Handle Video DICOMs
                img = img[0, :, :, 0]
            
            img = np.squeeze(img).astype(float)

            # 2. Handle Monochrome Inversion
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                img = np.max(img) - img
                
            # 3. Safe Normalization (Fixes the Grey Box)
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
            img = (img * 255).astype(np.uint8)
            
            # 4. Resize to a clean 512x512 square BEFORE stacking
            # This is the most important step to stop the "Horizontal Streaks"
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            
            # 5. Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
            # 6. Final RGB Stack
            img_rgb = np.stack([img]*3, axis=-1)

            # 7. Run AI Inference
            with st.spinner('AI is analyzing imagery...'):
                diag, prob, heat = engine.predict(img_rgb)
            
            # 8. UI Display
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