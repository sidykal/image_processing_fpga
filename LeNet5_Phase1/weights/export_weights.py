import torch
import numpy as np
from model import ModifiedLeNet5

# Output file
OUTPUT_HEADER = "parameters_64.h"
MODEL_PATH = "64_lenet5_traffic.pth"

# ----------------------------
# WRITE ONE LAYER
# ----------------------------
def write_header(f, name, tensor):
    data = tensor.detach().cpu().numpy()

    dims = "[" + "][".join(map(str, data.shape)) + "]"

    f.write(f"// Layer: {name} | Shape: {data.shape}\n")
    f.write(f"const float {name}{dims} = {{\n")

    flat = data.flatten()

    f.write("\t")
    for i, val in enumerate(flat):
        f.write(f"{val:.8f}, ")

        if (i + 1) % 10 == 0:
            f.write("\n\t")

    f.write("\n};\n\n")

    print(f"Exported {name} -> {len(flat)} elements")

# ----------------------------
# EXPORT MODEL
# ----------------------------
def export():
    print(f"Loading {MODEL_PATH}...")

    device = torch.device("cpu")
    model = ModifiedLeNet5(num_classes=43)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f"Writing {OUTPUT_HEADER}...")

    with open(OUTPUT_HEADER, "w") as f:
        f.write("#ifndef PARAMETERS_64_H\n")
        f.write("#define PARAMETERS_64_H\n\n")

        f.write("// -------------------------------------------------\n")
        f.write("// Auto-generated PyTorch → HLS parameter file\n")
        f.write("// Format: float32 (verification version)\n")
        f.write("// -------------------------------------------------\n\n")

        # ---------------- Conv1 ----------------
        write_header(f, "conv1_weights", model.conv1.weight)
        write_header(f, "conv1_bias",    model.conv1.bias)

        # ---------------- Conv2 ----------------
        write_header(f, "conv2_weights", model.conv2.weight)
        write_header(f, "conv2_bias",    model.conv2.bias)

        # ---------------- FC1 ----------------
        write_header(f, "fc1_weights", model.fc1.weight)
        write_header(f, "fc1_bias",    model.fc1.bias)

        # ---------------- FC2 ----------------
        write_header(f, "fc2_weights", model.fc2.weight)
        write_header(f, "fc2_bias",    model.fc2.bias)

        # ---------------- FC3 ----------------
        write_header(f, "fc3_weights", model.fc3.weight)
        write_header(f, "fc3_bias",    model.fc3.bias)

        f.write("#endif\n")

    print("Done → parameters_64.h created")

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    export()