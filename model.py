# model.py
import torch
import torch.nn as nn
from transformers import ViTModel

class ViTForKeypointRegression(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224", num_keypoints=9):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.head = nn.Linear(self.vit.config.hidden_size, num_keypoints * 2)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]  # Token [CLS]
        return self.head(cls_token)


