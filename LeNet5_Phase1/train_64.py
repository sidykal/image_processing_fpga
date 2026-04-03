import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import xml.etree.ElementTree as ET
from model_64 import LeNet5_64


# --- Configuration ---
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "./data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")


# -----------------------------
# Dataset for XML Annotations
# -----------------------------
class RoadSignDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform

        self.xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]

        # Build class index mapping dynamically
        self.class_to_idx = self._build_class_index()

    def _build_class_index(self):
        classes = set()
        for xml_file in self.xml_files:
            tree = ET.parse(os.path.join(self.annotations_dir, xml_file))
            root = tree.getroot()
            for obj in root.findall("object"):
                classes.add(obj.find("name").text)
        classes = sorted(list(classes))
        return {cls_name: idx for idx, cls_name in enumerate(classes)}

    def __len__(self):
        return len(self.xml_files)

    def __getitem__(self, idx):
        xml_path = os.path.join(self.annotations_dir, self.xml_files[idx])
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image filename
        filename = root.find("filename").text
        img_path = os.path.join(self.images_dir, filename)
        image = Image.open(img_path).convert("RGB")

        # Use first object (single-sign assumption)
        obj = root.find("object")
        label_name = obj.find("name").text
        bbox = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Crop to bounding box
        image = image.crop((xmin, ymin, xmax, ymax))

        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = RoadSignDataset(IMAGES_DIR, ANNOTATIONS_DIR, transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train():
    train_loader, test_loader = get_dataloaders()

    model = LeNet5_64(num_classes=len(train_loader.dataset.dataset.class_to_idx)).to(DEVICE)

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