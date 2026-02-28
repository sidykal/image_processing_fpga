import torch
import time
import sys
import os
import platform
from PIL import Image
import torchvision.transforms as transforms
from model import ModifiedLeNet5
from collections import defaultdict
import torch.nn.functional as F

# --- CONFIGURATION ---
CROPPED_DIR = "./cropped_images"  # folder for preprocessed/cropped images
MODEL_PATH = "lenet5_traffic.pth"

# New 4-class dataset
CLASSES = {
    0: 'Traffic Light',
    1: 'Stop',
    2: 'Speed Limit',
    3: 'Crosswalk'
}
NUM_CLASSES = len(CLASSES)

def load_system():
    # Detect System
    sys_info = f"{platform.system()} {platform.machine()} ({platform.processor()})"
    print(f"System Detected: {sys_info}")

    # Load Model
    device = torch.device("cpu")  # Force CPU
    model = ModifiedLeNet5(num_classes=NUM_CLASSES).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Please place it in this directory.")
        sys.exit(1)
        
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        sys.exit(1)
        
    return model, device

def preprocess_image(image_path, device, save_cropped=False):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        sys.exit(1)

    # Open image
    img = Image.open(image_path).convert('RGB')
    """
    # Center-crop to square
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    img = img.crop((left, top, right, bottom))
    print(f"Cropped image size: {img.size}")
    """

    # Save cropped image if needed
    if save_cropped:
        os.makedirs(CROPPED_DIR, exist_ok=True)
        cropped_path = os.path.join(CROPPED_DIR, os.path.basename(image_path))
        img.save(cropped_path)
        print(f"Cropped image saved to: {cropped_path}")

    # Transform: Resize, Grayscale, Tensor, Normalize
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    return input_tensor

def get_image_paths(images_dir):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir))
                   if f.lower().endswith(valid_exts)]
    return image_paths

def run_folder_inference():
    model, device = load_system()
    image_paths = get_image_paths(CROPPED_DIR)

    if len(image_paths) == 0:
        print(f"No images found in {CROPPED_DIR}.")
        return

    print("\n" + "="*50)
    print(f" RUNNING FOLDER INFERENCE ({len(image_paths)} images)")
    print("="*50)

    with torch.no_grad():
        for image_path in image_paths:
            input_tensor = preprocess_image(image_path, device, save_cropped=False)

            start_time = time.perf_counter()
            output = model(input_tensor)
            end_time = time.perf_counter()

            probs = F.softmax(output, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_id].item()

            label = CLASSES.get(pred_id, "Unknown")
            latency_ms = (end_time - start_time) * 1000

            print(f"\nImage           : {os.path.basename(image_path)}")
            print(f"Prediction      : Class {pred_id} -> {label}")
            print(f"Confidence      : {confidence:.4f}")
            print(f"Latency         : {latency_ms:.4f} ms")
            print("-" * 50)

    print("\n" + "="*50)
    print(" FOLDER INFERENCE COMPLETE")
    print("="*50)

if __name__ == "__main__":
    run_folder_inference()