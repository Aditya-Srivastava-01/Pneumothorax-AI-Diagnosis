import gc
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

torch.set_num_threads(1) # Crucial for Streamlit Cloud stability

class DiagnosisEngine:
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        self.model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.target_layers = [self.model.conv_head]

    def predict(self, img_rgb):
        img_res = cv2.resize(img_rgb, (384, 384)) # 384px for memory safety
        img_input = (img_res / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(img_input).permute(2, 0, 1).float().unsqueeze(0)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(tensor)).item()
        
        diagnosis = "PNEUMOTHORAX DETECTED" if prob > 0.62 else "NORMAL / HEALTHY"
        
        try:
            cam = GradCAM(model=self.model, target_layers=self.target_layers)
            grayscale_cam = cam(input_tensor=tensor)[0, :]
            visualization = show_cam_on_image(img_res / 255.0, grayscale_cam, use_rgb=True)
            del cam
        except Exception:
            visualization = img_res

        del tensor
        gc.collect() 
        return diagnosis, prob, visualization