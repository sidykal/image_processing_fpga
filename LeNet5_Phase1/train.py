import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import ModifiedLeNet5
import os

# --- Configuration ---
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
EPOCHS = 35
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = './data'

def get_dataloaders():
    # FPGA Requirement: Inputs must be strictly controlled (Grayscale, 32x32)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    print("Downloading/Loading GTSRB Dataset...")
    try:
        train_set = torchvision.datasets.GTSRB(
            root=DATA_PATH, split='train', download=True, transform=transform
        )
        test_set = torchvision.datasets.GTSRB(
            root=DATA_PATH, split='test', download=True, transform=transform
        )
    except RuntimeError:
        print("Automatic download failed. Ensure data is in ./data manually.")
        return None, None

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

def train():
    train_loader, test_loader = get_dataloaders()
    if not train_loader: return

    model = ModifiedLeNet5(num_classes=43).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # --- Validation phase (CORRECTED) ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Accuracy: {accuracy:.2f}%")

    print("Saving Golden Model...")
    torch.save(model.state_dict(), 'lenet5_traffic.pth')
    print("Saved as 'lenet5_traffic.pth'")

if __name__ == "__main__":
    train()