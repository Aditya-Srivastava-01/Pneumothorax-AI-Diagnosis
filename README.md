# 🫁 Pneumothorax AI Diagnostic Assistant

An end-to-end Deep Learning system to detect lung collapse (Pneumothorax) from DICOM and standard radiographs. This project demonstrates an engineering evolution from a ResNet-50 baseline to a high-resolution EfficientNet-B4 production model.

## 🔗 Project Links
- **Live Demo:** [Link to your Hugging Face Space]
- **Technical Presentation:** [Link to your Grad-CAM results]

## 📊 Technical Performance (ResNet vs. EfficientNet)
| Metric | Baseline (ResNet-50) | Production (EfficientNet-B4) |
| :--- | :--- | :--- |
| **Input Resolution** | 224 x 224 | **512 x 512** |
| **Recall (Sensitivity)** | 65% | **83%** |
| **Weighted F1-Score** | 0.84 | **0.89** |
| **Healthy Precision** | 88% | **94%** |

## 🛠️ Engineering Highlights
- **High-Resolution Pipeline:** Upgraded to 512px to capture fine visceral pleural lines that are lost in standard 224px downsampling.
- **Explainable AI (XAI):** Integrated **Grad-CAM** to visualize pathology localization, ensuring the model focuses on the pleural space rather than medical artifacts.
- **Format Agnostic:** Developed a robust preprocessing pipeline using **Pydicom** and **OpenCV** to handle DICOM metadata, MONOCHROME1 inversions, and standard JPG/PNG uploads.
- **Cloud Deployment:** Optimized for **Hugging Face Spaces** using memory-hardening techniques (garbage collection and single-thread execution) to run high-parameter models on CPU-basic instances.

## 💻 Tech Stack
- **AI/ML:** PyTorch, TIMM, Albumentations
- **Medical Imaging:** Pydicom, OpenCV (CLAHE, LANCZOS4)
- **Deployment:** Streamlit, Hugging Face, Docker

## 📂 Project Structure
- `/weights`: Contains the trained EfficientNet-B4 model checkpoints.
- `app.py`: Main Streamlit dashboard code.
- `engine.py`: AI inference and Grad-CAM logic.
