import torch
import time
import sys
import os
import platform
import cv2
from PIL import Image
import torchvision.transforms as transforms
from model_64 import LeNet5_64
import torch.nn.functional as F

# --- CONFIGURATION ---
MODEL_PATH = "64_lenet5_traffic.pth"

CLASSES = {
    0: 'Traffic Light',
    1: 'Stop',
    2: 'Speed Limit',
    3: 'Crosswalk'
}
NUM_CLASSES = len(CLASSES)

def show_transformed_image(tensor):
    img = tensor.squeeze().cpu().numpy()

    # Unnormalize
    img = (img * 0.5 + 0.5) * 255
    img = img.astype('uint8')

    # 🔥 Resize so you can actually SEE it
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Transformed (64x64)", img)

# --- LOAD MODEL ---
def load_system():
    sys_info = f"{platform.system()} {platform.machine()} ({platform.processor()})"
    print(f"System Detected: {sys_info}")

    device = torch.device("cpu")
    model = LeNet5_64(num_classes=NUM_CLASSES).to(device)

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

# --- PREPROCESS CAMERA FRAME ---
def preprocess_frame(frame, device):
    # Convert BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL
    img = Image.fromarray(frame_rgb)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    tensor = transform(img).unsqueeze(0).to(device)
    return tensor

# --- MAIN CAMERA LOOP ---
def run_camera_inference():
    model, device = load_system()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        return

    print("\n" + "="*50)
    print(" REAL-TIME CAMERA INFERENCE STARTED")
    print(" Press 'q' to quit")
    print("="*50)

    last_inference_time = 0
    inference_interval = 0.5  # seconds

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            current_time = time.time()

            # Only run inference every 0.5 sec
            if current_time - last_inference_time > inference_interval:
                input_tensor = preprocess_frame(frame, device)

                show_transformed_image(input_tensor)
                
                start = time.perf_counter()
                output = model(input_tensor)
                end = time.perf_counter()

                probs = F.softmax(output, dim=1)
                pred_id = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_id].item()

                label = CLASSES.get(pred_id, "Unknown")
                latency_ms = (end - start) * 1000

                # 🔥 TERMINAL OUTPUT
                print(f"Prediction: {label}")
                print(f"Confidence: {confidence:.4f}")
                print(f"Latency: {latency_ms:.2f} ms")
                print("-" * 40)

                last_inference_time = current_time
            

            # Show camera feed
            cv2.imshow("Camera", frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()
    print("Camera stopped.")

# --- RUN ---
if __name__ == "__main__":
    run_camera_inference()