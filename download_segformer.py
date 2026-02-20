import torch
import numpy as np
import segmentation_models_pytorch as smp

from PIL import Image

device = "cpu"

# Load pretrained model and preprocessing function
checkpoint = "smp-hub/segformer-b5-1024x1024-city-160k"
print(f"Loading model from {checkpoint}...")
model = smp.from_pretrained(checkpoint).eval().to(device)


save_path = "./segformer_b5_city_160k.pth"

# 保存模型权重 (state_dict)
torch.save(model.state_dict(), save_path)

print(f"模型权重已保存至: {save_path}")