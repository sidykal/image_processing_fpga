import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from model_64 import LeNet5_64
import torchvision


# --- Configuration ---
BATCH_SIZE = 128
#LEARNING_RATE = 0.0001
#EPOCHS = 100
LEARNING_RATE = 0.0002
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "./data"
IMAGES_DIR = os.path.join(DATA_PATH, "images")
ANNOTATIONS_DIR = os.path.join(DATA_PATH, "annotations")



def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.RandomCrop((64, 64)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Downloading/Loading GTSRB Dataset...")

    train_set = torchvision.datasets.GTSRB(
        root=DATA_PATH, split='train', download=True, transform=transform
    )

    test_set = torchvision.datasets.GTSRB(
        root=DATA_PATH, split='test', download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train():
    train_loader, test_loader = get_dataloaders()

    model = LeNet5_64(num_classes=43).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {running_loss/len(train_loader):.4f} "
              f"| Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "64_lenet5_traffic.pth")
    print("Model saved as 64_lenet5_traffic.pth")


if __name__ == "__main__":
    train()