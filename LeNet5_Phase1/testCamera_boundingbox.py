import torch
import time
import sys
import os
import platform
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model_64 import LeNet5_64
import torch.nn.functional as F

# --- CONFIGURATION ---
MODEL_PATH = "64_lenet5_traffic.pth"

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
    sys_info = f"{platform.system()} {platform.machine()} ({platform.processor()})"
    print(f"System Detected: {sys_info}")

    device = torch.device("cpu")
    model = LeNet5_64(num_classes=43).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        sys.exit(1)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        sys.exit(1)

    return model, device


def crop_sign_region_from_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    R = img_rgb[:, :, 0].astype(np.int16)
    G = img_rgb[:, :, 1].astype(np.int16)
    B = img_rgb[:, :, 2].astype(np.int16)

    max_channel = np.maximum(np.maximum(R, G), B)
    min_channel = np.minimum(np.minimum(R, G), B)

    red_mask = (
        (R > 80) &
        (R > G * 1.10) &
        (R > B * 1.10) &
        ((max_channel - min_channel) > 25)
    )

    blue_mask = (B > 100) & (B > R + 30) & (B > G + 20)

    yellow_mask = (
        (R > 120) &
        (G > 100) &
        (B < 120) &
        (R > B + 40) &
        (G > B + 40)
    )

    mask = red_mask | blue_mask | yellow_mask
    mask = mask.astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug_frame = frame.copy()

    if len(contours) == 0:
        return Image.fromarray(img_rgb), debug_frame, mask

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    if w * h < 300:
        return Image.fromarray(img_rgb), debug_frame, mask

    pad = 20
    size = max(w, h) + 2 * pad

    cx = x + w // 2

    # shift up slightly to avoid including pole/bottom
    cy = int(y + h * 0.40)

    x1 = cx - size // 2
    y1 = cy - size // 2
    x2 = x1 + size
    y2 = y1 + size

    # clamp to image bounds
    img_h, img_w = img_rgb.shape[:2]

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img_w)
    y2 = min(y2, img_h)

    crop = img_rgb[y1:y2, x1:x2]

    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return Image.fromarray(crop), debug_frame, mask


def preprocess_frame(frame, device):
    cropped_img, debug_frame, mask = crop_sign_region_from_frame(frame)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    tensor = transform(cropped_img).unsqueeze(0).to(device)

    return tensor, debug_frame, mask


def show_transformed_image(tensor):
    img = tensor.squeeze().cpu().numpy()

    img = (img * 0.5 + 0.5) * 255
    img = img.astype("uint8")

    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Preprocessed Grayscale 64x64", img)


def run_camera_inference():
    model, device = load_system()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        return

    print("\n" + "=" * 50)
    print(" REAL-TIME CAMERA INFERENCE STARTED")
    print(" Press 'q' to quit")
    print("=" * 50)

    last_inference_time = 0
    inference_interval = 0.1

    with torch.no_grad():
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            current_time = time.time()

            if current_time - last_inference_time > inference_interval:
                input_tensor, debug_frame, mask = preprocess_frame(frame, device)

                show_transformed_image(input_tensor)

                start = time.perf_counter()
                output = model(input_tensor)
                end = time.perf_counter()

                probs = F.softmax(output, dim=1)
                pred_id = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_id].item()

                label = CLASSES.get(pred_id, "Unknown")
                latency_ms = (end - start) * 1000

                print(f"Prediction: Class {pred_id} -> {label}")
                print(f"Confidence: {confidence:.4f}")
                print(f"Latency: {latency_ms:.2f} ms")
                print("-" * 40)

                last_inference_time = current_time

                cv2.imshow("Bounding Box Detection", debug_frame)
                cv2.imshow("Color Mask", mask)

            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera stopped.")


if __name__ == "__main__":
    run_camera_inference()