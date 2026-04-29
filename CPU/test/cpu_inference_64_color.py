import torch
import time
import sys
import os
import platform
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from CPU.models.model_64 import LeNet5_64
from collections import defaultdict
import torch.nn.functional as F

DEBUG_DIR = "debug_outputs"
os.makedirs(DEBUG_DIR, exist_ok=True)

# images folder 
IMAGES_DIR = "./CPU/image_testing/images"

# --- CONFIGURATION ---
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "weights", "64_lenet5_traffic.pth")
)
IMAGE_PATH = "chat_stop.png"  # Change this to your image filename

# GTSRB Class Labels
CLASSES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left',
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory',
    41: 'End of no passing', 42: 'End of no passing by vehicles over 3.5 metric tons'
}

def forward_debug(model, x):
    x = model.pool(F.relu(model.conv1(x)))
    print("s1:", x.min().item(), x.max().item(), x.mean().item())

    x = model.pool(F.relu(model.conv2(x)))
    print("s2:", x.min().item(), x.max().item(), x.mean().item())

    x = model.pool2(x)
    print("s3:", x.min().item(), x.max().item(), x.mean().item())

    x = x.view(x.size(0), -1)

    x = F.relu(model.fc1(x))
    print("f1:", x.min().item(), x.max().item(), x.mean().item())

    x = F.relu(model.fc2(x))
    print("f2:", x.min().item(), x.max().item(), x.mean().item())

    x = model.fc3(x)
    print("f3:", x.min().item(), x.max().item(), x.mean().item())

    return x

def load_system():
    # Detect System
    sys_info = f"{platform.system()} {platform.machine()} ({platform.processor()})"
    print(f"System Detected: {sys_info}")

    # Load Model
    device = torch.device("cpu") # Force CPU for fair comparison
    model = LeNet5_64(num_classes=43).to(device)
    
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

def crop_sign_region(image_path, debug=False):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: could not read {image_path}")
        sys.exit(1)

    # OpenCV loads BGR, convert to RGB for easier logic
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    # Simple traffic-sign color masks
    R = img_rgb[:, :, 0].astype(np.int16)
    G = img_rgb[:, :, 1].astype(np.int16)
    B = img_rgb[:, :, 2].astype(np.int16)

    max_channel = np.maximum(np.maximum(R, G), B)
    min_channel = np.minimum(np.minimum(R, G), B)

    red_mask = (
        (R > 120) &
        (R > G * 1.35) &
        (R > B * 1.35) &
        ((max_channel - min_channel) > 50)
    )

    blue_mask = (B > 100) & (B > R + 30) & (B > G + 20)

    yellow_mask = (R > 120) & (G > 100) & (B < 120) & (R > B + 40) & (G > B + 40)

    mask = red_mask | blue_mask | yellow_mask
    mask = mask.astype(np.uint8) * 255

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print(f"No sign-like colored region found in {image_path}. Using full image.")
        return Image.fromarray(img_rgb)

    # Pick largest colored region
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    pad = 20
    size = max(w, h) + 2 * pad

    cx = x + w // 2
    cy = int(y + h * 0.40)   # shift upward to avoid pole

    x1 = cx - size // 2
    y1 = cy - size // 2
    x2 = x1 + size
    y2 = y1 + size

    crop = img_rgb[y1:y2, x1:x2]

    # Reject tiny regions
    if w * h < 300:
        print(f"Detected region too small in {image_path}. Using full image.")
        return Image.fromarray(img_rgb)

    # Add padding around crop
    pad = 20
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img_rgb.shape[1])
    y2 = min(y + h + pad, img_rgb.shape[0])

    crop = img_rgb[y1:y2, x1:x2]

    if debug:
        debug_img = img_rgb.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        if debug:
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            detection_path = os.path.join(DEBUG_DIR, f"{base_name}_detect.jpg")
            crop_path = os.path.join(DEBUG_DIR, f"{base_name}_crop.jpg")

            debug_img = img_rgb.copy()
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            Image.fromarray(debug_img).save(detection_path)
            Image.fromarray(crop).save(crop_path)

    return Image.fromarray(crop)


def preprocess_image(image_path, device):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        sys.exit(1)

    img = crop_sign_region(image_path, debug=True)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)
    return input_tensor



def benchmark():
    model, device = load_system()
    input_tensor = preprocess_image(IMAGE_PATH, device)

    print("\n" + "="*50)
    print(f" RUNNING BENCHMARK ON {IMAGE_PATH}")
    print("="*50)

    # 1. Warmup
    print("Warming up CPU (10 runs)...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # 2. Run Inference Loop
    ITERATIONS = 1000
    print(f"Running {ITERATIONS} inference loops for averaging...")

    prediction_counts = defaultdict(int)
    confidence_sums = defaultdict(float)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(ITERATIONS):
            output = model(input_tensor)

            print("\nLogits:")
            for i, v in enumerate(output[0]):
                print(f"class {i}: {float(v):.4f}")

            # Softmax for confidence
            probs = F.softmax(output, dim=1)

            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_id].item()

            prediction_counts[pred_id] += 1
            confidence_sums[pred_id] += confidence
    end_time = time.perf_counter()

    # 3. Calculate Stats
    total_time = end_time - start_time
    avg_latency_ms = (total_time / ITERATIONS) * 1000
    fps = ITERATIONS / total_time

    # 4. Sort and get Top-5 predictions
    top5 = sorted(
        prediction_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    # 5. Print Report
    print("\n" + "="*50)
    print(f" RESULTS: CPU INFERENCE")
    print("="*50)

    print("Top-5 Predictions:")
    for class_id, count in top5:
        label = CLASSES.get(class_id, "Unknown")
        avg_conf = confidence_sums[class_id] / count
        print(
            f"  Class {class_id:2d} -> {label:45s} "
            f": {count:4d} times | Avg Confidence: {avg_conf:.4f}"
        )

    print("-" * 50)
    print(f"Avg Latency     : {avg_latency_ms:.4f} ms")
    print(f"Throughput      : {fps:.2f} FPS")
    print("="*50)


# get images
def get_image_paths(images_dir):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths = []

    for filename in sorted(os.listdir(images_dir)):
        if filename.lower().endswith(valid_exts):
            image_paths.append(os.path.join(images_dir, filename))

    return image_paths

def run_folder_inference():
    model, device = load_system()
    image_paths = get_image_paths(IMAGES_DIR)

    if len(image_paths) == 0:
        print("No images found.")
        return

    print("\n" + "="*50)
    print(f" RUNNING FOLDER INFERENCE ({len(image_paths)} images)")
    print("="*50)

    with torch.no_grad():
        for image_path in image_paths:
            input_tensor = preprocess_image(image_path, device)

            # Time a single inference
            '''start_time = time.perf_counter()
            output = model(input_tensor)
            end_time = time.perf_counter()'''


            # original 
            start_time = time.perf_counter()


            output = model(input_tensor)
            #output = forward_debug(model, input_tensor)

            '''
            prints logits
            print("\nLogits:")
            for i, v in enumerate(output[0]):
                print(f"class {i}: {float(v):.4f}")
            '''
            end_time = time.perf_counter()

            probs = F.softmax(output, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_id].item()

            label = CLASSES.get(pred_id, "Unknown")
            latency_ms = (end_time - start_time) * 1000

            # Report (benchmark-style)
            # Report (benchmark-style)
            print(f"\nImage           : {os.path.basename(image_path)}")
            print(f"Prediction      : Class {pred_id:2d} -> {label}")
            print(f"Confidence      : {confidence:.4f}")
            print(f"Latency         : {latency_ms:.4f} ms")
            print("-" * 50)

        

    print("\n" + "="*50)
    print(" FOLDER INFERENCE COMPLETE")
    print("="*50)



if __name__ == "__main__":
    # benchmark()
    run_folder_inference()