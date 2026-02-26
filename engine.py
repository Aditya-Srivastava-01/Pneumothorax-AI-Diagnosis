import gc
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. LIMIT CPU THREADS (Prevents memory spikes)
torch.set_num_threads(1)

class DiagnosisEngine:
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        # Initialize B4
        self.model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.target_layers = [self.model.conv_head]

    def predict(self, img_rgb):
        # 2. REDUCE RESOLUTION TO 384px (Uses 40% less RAM than 512px)
        img_res = cv2.resize(img_rgb, (384, 384))
        
        # Preprocessing
        img_input = (img_res / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(img_input).permute(2, 0, 1).float().unsqueeze(0)

        # 3. USE INFERENCE MODE (Faster and lighter than no_grad)
        with torch.inference_mode():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits).item()
        
        diagnosis = "PNEUMOTHORAX DETECTED" if prob > 0.62 else "NORMAL / HEALTHY"
        
        # 4. GRAD-CAM WITH AUTO-CLEANUP
        try:
            cam = GradCAM(model=self.model, target_layers=self.target_layers)
            grayscale_cam = cam(input_tensor=tensor)[0, :]
            visualization = show_cam_on_image(img_res / 255.0, grayscale_cam, use_rgb=True)
            # Kill the CAM object immediately to free RAM
            del cam
            del grayscale_cam
        except Exception:
            visualization = img_res

        # 5. AGGRESSIVE GARBAGE COLLECTION
        del tensor
        gc.collect() 
        
        return diagnosis, prob, visualization