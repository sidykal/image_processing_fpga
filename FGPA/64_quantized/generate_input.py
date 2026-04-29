import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from CPU.test.cpu_inference_64_color import crop_sign_region

# 🔧 CHANGE THIS to your image
IMAGE_PATH = "./CPU/image_testing/images/chat_stop.png"

# --- SAME preprocessing as your model ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# load image
img = crop_sign_region(IMAGE_PATH, debug=True)

# preprocess
x = transform(img)              # [1, 64, 64]
x = x.unsqueeze(0)              # [1, 1, 64, 64]

# --- convert to int8 ---
# PyTorch normalized range is roughly [-1, 1]
# map to [-127, 127]
x_int8 = torch.round(x * 127).clamp(-128, 127).to(torch.int8)

# flatten to 4096 values
flat = x_int8.squeeze().cpu().numpy().flatten()

# --- pack into 32-bit words (4 int8 per word) ---
packed = []

for i in range(0, len(flat), 4):
    word = 0
    for b in range(4):
        val = int(flat[i + b]) & 0xFF  # convert to unsigned byte
        word |= (val << (8 * b))       # pack into 32-bit
    packed.append(word)

# --- write to file ---
with open("input.txt", "w") as f:
    for word in packed:
        f.write(f"{word}\n")

print("✅ input.txt generated successfully")
print(f"Total words: {len(packed)} (should be 1024)")