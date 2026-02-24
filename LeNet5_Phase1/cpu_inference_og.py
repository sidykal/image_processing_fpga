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

# images folder 
IMAGES_DIR = "./images"

# --- CONFIGURATION ---
MODEL_PATH = "lenet5_traffic.pth"
IMAGE_PATH = "jacey2.jpg"  # Change this to your image filename

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

def load_system():
    # Detect System
    sys_info = f"{platform.system()} {platform.machine()} ({platform.processor()})"
    print(f"System Detected: {sys_info}")

    # Load Model
    device = torch.device("cpu") # Force CPU for fair comparison
    model = ModifiedLeNet5().to(device)
    
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

def preprocess_image(image_path, device):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        sys.exit(1)

    # Identical preprocessing to Training and FPGA
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension
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