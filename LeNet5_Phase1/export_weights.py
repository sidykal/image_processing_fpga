import torch
import numpy as np
import os
from model import ModifiedLeNet5

# Output file name
OUTPUT_HEADER = "parameters.h"
MODEL_PATH = "lenet5_traffic.pth"

def write_header(f, name, tensor):
    # Flatten the tensor to 1D for easier C++ handling (optional, but easier for streams later)
    # OR keep dimensions. Let's keep dimensions for readability in the first pass.
    data = tensor.detach().numpy()
    
    # C++ Variable Declaration
    # e.g., const float conv1_w[6][1][5][5] = { ... };
    
    dims = "[" + "][".join(map(str, data.shape)) + "]"
    f.write(f"// Layer: {name} | Shape: {data.shape}\n")
    f.write(f"const float {name}{dims} = {{\n")
    
    # recursively write data
    # To avoid complex recursion for writing, we flatten, write, and let C++ fill the dims
    # But C++ requires braces {} for multidimensional init unless we treat it as 1D.
    # STRATEGY: Flatten to 1D array in C++. It makes HLS memory mapping much easier.
    
    flat_data = data.flatten()
    f.write("\t")
    for i, val in enumerate(flat_data):
        f.write(f"{val:.8f}, ")
        if (i + 1) % 10 == 0: # Newline every 10 numbers for readability
            f.write("\n\t")
            
    f.write("\n};\n\n")
    print(f"Exported {name} -> Size: {len(flat_data)} elements")

def export():
    print(f"Loading {MODEL_PATH}...")
    device = torch.device("cpu")
    model = ModifiedLeNet5(num_classes=43)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f"Writing to {OUTPUT_HEADER}...")
    
    with open(OUTPUT_HEADER, "w") as f:
        f.write("#ifndef PARAMETERS_H\n")
        f.write("#define PARAMETERS_H\n\n")
        f.write("// Automatic Export from PyTorch\n")
        f.write("// Float32 weights for HLS verification\n\n")

        # --- Layer 1 ---
        write_header(f, "conv1_weights", model.conv1.weight)
        write_header(f, "conv1_bias",    model.conv1.bias)

        # --- Layer 2 ---
        write_header(f, "conv2_weights", model.conv2.weight)
        write_header(f, "conv2_bias",    model.conv2.bias)

        # --- FC 1 ---
        write_header(f, "fc1_weights",   model.fc1.weight)
        write_header(f, "fc1_bias",      model.fc1.bias)

        # --- FC 2 ---
        write_header(f, "fc2_weights",   model.fc2.weight)
        write_header(f, "fc2_bias",      model.fc2.bias)

        # --- FC 3 ---
        write_header(f, "fc3_weights",   model.fc3.weight)
        write_header(f, "fc3_bias",      model.fc3.bias)

        f.write("#endif\n")

    print("Done! parameters.h is ready.")

if __name__ == "__main__":
    export()