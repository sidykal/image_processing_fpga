import torch
import torch.nn as nn
import torch.nn.functional as F

class ModifiedLeNet5(nn.Module):
    def __init__(self, num_classes=43):
        super(ModifiedLeNet5, self).__init__()
        
        # --- Layer 1: Convolution ---
        # Input: 1 x 32 x 32 (Grayscale)
        # Output: 6 x 28 x 28
        # Kernel: 5x5, Stride: 1
        # FPGA Note: 5x5 kernels are efficiently handled by Xilinx DSP48 slices.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        
        # --- Layer 2: Subsampling (Max Pooling) ---
        # Input: 6 x 28 x 28
        # Output: 6 x 14 x 14
        # Kernel: 2x2, Stride: 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Layer 3: Convolution ---
        # Input: 6 x 14 x 14
        # Output: 16 x 10 x 10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # --- Layer 4: Subsampling (Max Pooling) ---
        # Input: 16 x 10 x 10
        # Output: 16 x 5 x 5 (via self.pool reuse)
        
        # --- Flattening ---
        # 16 channels * 5 * 5 = 400 features
        
        # --- Layer 5: Fully Connected ---
        # Input: 400
        # Output: 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        
        # --- Layer 6: Fully Connected ---
        # Input: 120
        # Output: 84
        self.fc2 = nn.Linear(120, 84)
        
        # --- Layer 7: Output Layer ---
        # Input: 84
        # Output: num_classes (43 for GTSRB)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # C1: Convolution -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        
        # C2: Convolution -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten: Prepare data for dense layers
        # view(-1) handles the batch size dimension automatically
        x = x.view(-1, 16 * 5 * 5)
        
        # FC1: Linear -> ReLU
        x = F.relu(self.fc1(x))
        
        # FC2: Linear -> ReLU
        x = F.relu(self.fc2(x))
        
        # Output: Raw logits (CrossEntropyLoss handles Softmax internally)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # Sanity Check
    model = ModifiedLeNet5()
    test_input = torch.randn(1, 1, 32, 32) # Batch=1, Channel=1, H=32, W=32
    output = model(test_input)
    print(f"Model Output Shape: {output.shape} (Expected: [1, 43])")