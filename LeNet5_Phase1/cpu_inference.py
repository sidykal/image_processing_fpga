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

"""def preprocess_image(image_path, device):
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
    return input_tensor"""

"""def preprocess_image(image_path, device):
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
    tensor = transform(img)  # Shape: [1, 32, 32]

    # Split into left and right halves (each 32x16)
    left_half  = tensor[:, :, 0:16]
    right_half = tensor[:, :, 16:32]

    # Add batch dimension and move to device
    halves = [
        h.unsqueeze(0).to(device)
        for h in [left_half, right_half]
    ]

    return halves  # List of 2 tensors"""

# this version is splitting into 2 (left, right) - didn't work well
"""def preprocess_image(image_path, device):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        sys.exit(1)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img = Image.open(image_path).convert('RGB')
    width, height = img.size

    # Split original image first
    left_img = img.crop((0, 0, width // 2, height))
    right_img = img.crop((width // 2, 0, width, height))

    # Now resize each half to 32x32
    left_tensor = transform(left_img).unsqueeze(0).to(device)
    right_tensor = transform(right_img).unsqueeze(0).to(device)

    return [left_tensor, right_tensor]"""

def preprocess_image(image_path, device, save_splits=True):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        sys.exit(1)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img = Image.open(image_path).convert('RGB')
    width, height = img.size

    # Split into 4 quadrants
    top_left     = img.crop((0, 0, width // 2, height // 2))
    top_right    = img.crop((width // 2, 0, width, height // 2))
    bottom_left  = img.crop((0, height // 2, width // 2, height))
    bottom_right = img.crop((width // 2, height // 2, width, height))

    crops = [top_left, top_right, bottom_left, bottom_right]

    # Create folder to save split images
    if save_splits:
        split_folder = "split_images"
        os.makedirs(split_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        for i, crop in enumerate(crops, start=1):
            crop_path = os.path.join(split_folder, f"{base_name}_q{i}.png")
            crop.save(crop_path)
            # Optional: print confirmation
            # print(f"Saved quadrant {i} to {crop_path}")

    # Apply transforms and move to device
    quadrants = [transform(crop).unsqueeze(0).to(device) for crop in crops]

    return quadrants  # List of 4 tensors




"""def benchmark():
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
    print("="*50)"""

def benchmark():
    model, device = load_system()
    input_tensors = preprocess_image(IMAGE_PATH, device)  # List of 2 tensors

    left_tensor, right_tensor = input_tensors

    print("\n" + "="*50)
    print(f" RUNNING BENCHMARK ON {IMAGE_PATH}")
    print("="*50)

    # 1. Warmup
    print("Warming up CPU (10 runs)...")
    with torch.no_grad():
        for _ in range(10):
            out_left = model(left_tensor)
            out_right = model(right_tensor)
            _ = (out_left + out_right) / 2  # Combine logits

    # 2. Run Inference Loop
    ITERATIONS = 1000
    print(f"Running {ITERATIONS} inference loops for averaging...")

    prediction_counts = defaultdict(int)
    confidence_sums = defaultdict(float)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(ITERATIONS):
            out_left = model(left_tensor)
            out_right = model(right_tensor)

            # Average logits
            output = (out_left + out_right) / 2

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
            #input_tensor = preprocess_image(image_path, device)

            # Time a single inference
            '''start_time = time.perf_counter()
            output = model(input_tensor)
            end_time = time.perf_counter()'''

            #left_tensor, right_tensor = input_tensor

            # original 
            """start_time = time.perf_counter()

            out_left = model(left_tensor)
            out_right = model(right_tensor)

            output = (out_left + out_right) / 2  # Average logits

            end_time = time.perf_counter()

            probs = F.softmax(output, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_id].item()

            label = CLASSES.get(pred_id, "Unknown")
            latency_ms = (end_time - start_time) * 1000

            # Report (benchmark-style)
            print(
                f"Prediction      : Class {pred_id:2d} -> {label}"
            )
            print(
                f"Confidence      : {confidence:.4f}"
            )
            print(
                f"Latency         : {latency_ms:.4f} ms"
            )"""

            #left, right image
            """left_tensor, right_tensor = input_tensor

            start_time = time.perf_counter()

            out_left = model(left_tensor)
            out_right = model(right_tensor)

            end_time = time.perf_counter()

            # Softmax separately
            probs_left = F.softmax(out_left, dim=1)
            probs_right = F.softmax(out_right, dim=1)

            pred_left = torch.argmax(probs_left, dim=1).item()
            pred_right = torch.argmax(probs_right, dim=1).item()

            conf_left = probs_left[0, pred_left].item()
            conf_right = probs_right[0, pred_right].item()

            latency_ms = (end_time - start_time) * 1000

            print(f"Left Half       : Class {pred_left:2d} -> {CLASSES.get(pred_left)} | Conf: {conf_left:.4f}")
            print(f"Right Half      : Class {pred_right:2d} -> {CLASSES.get(pred_right)} | Conf: {conf_right:.4f}")
            print(f"Latency         : {latency_ms:.4f} ms")"""

            # 4 images
            quadrants = preprocess_image(image_path, device)  # List of 4 tensors

            print("\n" + "-"*50)
            print(f" IMAGE: {os.path.basename(image_path)}")
            print("-"*50)

            start_time = time.perf_counter()

            for i, tensor in enumerate(quadrants):
                output = model(tensor)
                probs = F.softmax(output, dim=1)
                pred_id = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_id].item()

                print(f" Quadrant {i+1:1d} -> Class {pred_id:2d} : {CLASSES.get(pred_id, 'Unknown')} | Confidence: {confidence:.4f}")

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            print(f" Inference time for 4 quadrants: {latency_ms:.4f} ms")

    print("\n" + "="*50)
    print(" FOLDER INFERENCE COMPLETE")
    print("="*50)



if __name__ == "__main__":
    # benchmark()
    run_folder_inference()